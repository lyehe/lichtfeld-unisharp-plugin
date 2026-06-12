from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from uuid import uuid4

import lichtfeld as lf

from ..core import downloads, pipeline_loader, repo_policy
from ..core.camera import normalize_camera_mode
from ..core.job import JobConfig, JobResult, UnisharpJob

PLUGIN_NAME = repo_policy.PLUGIN_LINK_NAME
PLUGIN_ROOT = repo_policy.plugin_root()
_CLIPBOARD_DIR = repo_policy.clipboard_dir()
_MIN_SCALE = 0.001

_DIRTY_PREP = (
    "model_downloading",
    "model_error",
    "prep_progress_value",
    "prep_progress_pct",
    "prep_bytes_line",
    "prep_error_text",
    "model_status_line",
    "dependency_status_line",
    "model_loaded",
    "can_run",
    "can_gen_clipboard",
)
_DIRTY_RUN = ("stage_text", "progress_value", "progress_pct", "progress_status")
_DIRTY_RUNNING = ("show_idle", "show_running", "can_run", "can_gen_clipboard")
_DIRTY_LOG = ("show_logs", "live_log_text")
_DIRTY_RESULT = (
    "show_results",
    "show_error",
    "error_text",
    "result_count",
    "result_time",
    "result_camera",
    "result_output_dir",
    "has_node",
)
_DIRTY_PLACE = (
    "tx",
    "ty",
    "tz",
    "rx",
    "ry",
    "rz",
    "scl",
    "gizmo_mode",
    "is_translate",
    "is_rotate",
    "is_scale",
    "gizmo_active",
    "edit_target",
)


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _open_json_dialog(default_path: str) -> str | None:
    try:
        host_dialog = getattr(lf.ui, "open_file_dialog", None)
        if host_dialog is not None:
            picked = host_dialog("Select camera JSON", default_path or "")
            return picked or None
    except Exception:
        pass

    initial_dir = os.path.dirname(default_path) if default_path else os.path.expanduser("~")
    try:
        if sys.platform == "win32":
            script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = "Select camera JSON"
$dialog.InitialDirectory = "{initial_dir.replace(chr(92), chr(92) + chr(92))}"
$dialog.Filter = "JSON (*.json)|*.json|All files (*.*)|*.*"
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    Write-Output $dialog.FileName
}}
"""
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return result.stdout.strip() or None
        for cmd in (
            ["zenity", "--file-selection", "--title", "Select camera JSON", "--filename", initial_dir + "/"],
            ["kdialog", "--getopenfilename", initial_dir, "*.json"],
        ):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip() or None
                return None
            except FileNotFoundError:
                continue
    except Exception as exc:  # noqa: BLE001
        lf.log.warn(f"[unisharp] camera JSON dialog failed: {exc}")
    return None


class UnisharpPanel(lf.ui.Panel):
    id = "unisharp.main"
    label = repo_policy.PLUGIN_DISPLAY_NAME
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 227
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 150

    def __init__(self):
        self._doc = None
        self._handle = None
        self.image_path = ""
        self.camera_mode = "auto"
        self.camera_json_path = ""
        self.camera_intrinsics = ""
        self.camera_params = ""
        self.low_pass_filter_eps = 0.0
        self.splat_scale = 1.25
        self.append_mode = True
        self.tx = self.ty = self.tz = 0.0
        self.rx = self.ry = self.rz = 0.0
        self.scl = 1.0
        self.gizmo_mode = "translate"
        self._job: UnisharpJob | None = None
        self._last_result: JobResult | None = None
        self._clipboard_path: Path | None = None
        self._node_name = ""
        self._generated_nodes: list[str] = []
        self._last_selection = None
        self._selection_unsub = None
        self._gizmo = None
        self._collapsed = {"calibration", "precise"}
        self._last_prep = None
        self._last_stage = ""
        self._last_progress = -1.0
        self._last_status = ""
        self._last_log = ""
        self._last_running = False
        self._last_result_key = None
        self._last_can_paste = None

    def _dirty(self, *fields):
        if not self._handle:
            return
        for name in fields:
            self._handle.dirty(name)

    def on_mount(self, doc):
        self._doc = doc
        self._sync_section_states()
        if self._selection_unsub is None:
            try:
                from lfs_plugins.ui.state import AppState

                self._selection_unsub = AppState.selection_generation.subscribe_as(
                    repo_policy.PLUGIN_LINK_NAME, self._on_selection_generation
                )
            except Exception as exc:  # noqa: BLE001
                lf.log.warn(f"[unisharp] selection-follow subscribe failed: {exc}")

    def on_unmount(self, doc):
        if self._selection_unsub is not None:
            try:
                self._selection_unsub()
            except Exception:
                pass
            self._selection_unsub = None
        if self._job and self._job.is_running():
            self._job.cancel()
        self._detach_gizmo()
        self._gizmo = None
        _safe_unlink(self._clipboard_path)
        self._clipboard_path = None
        try:
            lf.ui.free_plugin_textures(PLUGIN_NAME)
        except Exception:
            pass
        doc.remove_data_model("unisharp")
        self._doc = None
        self._handle = None

    def on_scene_changed(self, doc):
        del doc
        return self._prune_dead_nodes()

    def draw(self, ui):
        del ui

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("unisharp")
        if model is None:
            return
        model.bind("image_path", lambda: self.image_path, self._set_image_path)
        model.bind_func("image_name", lambda: Path(self.image_path).name if self.image_path else "No image selected")
        model.bind_func("can_paste", self._can_paste_clipboard)
        model.bind("camera_mode", lambda: self.camera_mode, self._set_camera_mode)
        model.bind("camera_json_path", lambda: self.camera_json_path, self._set_camera_json_path)
        model.bind_func(
            "camera_json_name",
            lambda: Path(self.camera_json_path).name if self.camera_json_path else "No camera JSON selected",
        )
        model.bind("camera_intrinsics", lambda: self.camera_intrinsics, lambda v: self._set_text("camera_intrinsics", v))
        model.bind("camera_params", lambda: self.camera_params, lambda v: self._set_text("camera_params", v))
        model.bind(
            "low_pass_filter_eps",
            lambda: f"{self.low_pass_filter_eps:.4f}",
            lambda v: self._set_float("low_pass_filter_eps", v, 0.0, 0.10),
        )
        model.bind(
            "splat_scale",
            lambda: f"{self.splat_scale:.2f}",
            lambda v: self._set_float("splat_scale", v, 0.25, 4.0),
        )
        model.bind("append_mode", lambda: self.append_mode, lambda v: self._set_bool("append_mode", v))
        model.bind_func("model_downloading", lambda: downloads.get_state()["stage"] in {"checking", "cloning", "downloading"})
        model.bind_func("model_error", lambda: downloads.get_state()["stage"] == "error")
        model.bind_func("prep_progress_value", lambda: downloads.get_state()["progress"])
        model.bind_func("prep_progress_pct", lambda: f"{int(downloads.get_state()['progress'] * 100)}%")
        model.bind_func("prep_bytes_line", self._prep_bytes_line)
        model.bind_func("prep_error_text", lambda: downloads.get_state()["error"])
        model.bind_func("model_status_line", self._model_status_line)
        model.bind_func("dependency_status_line", self._dependency_status_line)
        model.bind_func("model_loaded", pipeline_loader.is_loaded)
        model.bind_func("can_run", self._can_run)
        model.bind_func("can_gen_clipboard", self._can_gen_clipboard)
        model.bind_func("show_idle", lambda: not self._is_running())
        model.bind_func("show_running", self._is_running)
        model.bind_func("stage_text", lambda: self._job.stage.value if self._job else "")
        model.bind_func("progress_value", lambda: self._job.progress if self._job else 0.0)
        model.bind_func("progress_pct", lambda: f"{int((self._job.progress if self._job else 0) * 100)}%")
        model.bind_func("progress_status", lambda: self._job.status if self._job else "")
        model.bind_func("show_logs", lambda: bool(self._job and self._job.log_text))
        model.bind_func("live_log_text", lambda: self._job.log_text if self._job else "")
        model.bind_func("show_results", self._show_results)
        model.bind_func("show_error", self._show_error)
        model.bind_func("error_text", lambda: self._last_result.error if self._last_result else "")
        model.bind_func("result_count", self._result_count_text)
        model.bind_func("result_time", lambda: f"{self._last_result.elapsed_s:.1f}s" if self._last_result else "")
        model.bind_func("result_camera", lambda: self._last_result.camera_kind if self._last_result else "")
        model.bind_func("result_output_dir", lambda: self._last_result.output_dir if self._last_result else "")
        model.bind_func("has_node", lambda: bool(self._node_name))
        model.bind_func("edit_target", lambda: self._node_name or "(none)")
        model.bind_func("gizmo_active", self._gizmo_active)
        model.bind_func("is_translate", lambda: self.gizmo_mode == "translate")
        model.bind_func("is_rotate", lambda: self.gizmo_mode == "rotate")
        model.bind_func("is_scale", lambda: self.gizmo_mode == "scale")
        for ax in ("tx", "ty", "tz", "rx", "ry", "rz", "scl"):
            model.bind(ax, (lambda a=ax: f"{getattr(self, a):.3f}"), (lambda v, a=ax: self._set_transform_field(a, v)))
        model.bind_event("browse_image", self._on_browse_image)
        model.bind_event("paste_image", self._on_paste_image)
        model.bind_event("generate_from_clipboard", self._on_generate_from_clipboard)
        model.bind_event("browse_camera_json", self._on_browse_camera_json)
        model.bind_event("clear_camera_json", self._on_clear_camera_json)
        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_event("retry_download", lambda *_: downloads.start_background_prepare())
        model.bind_event("unload_model", lambda *_: threading.Thread(target=pipeline_loader.unload, daemon=True).start())
        model.bind_event("do_start", self._on_start)
        model.bind_event("do_cancel", self._on_cancel)
        model.bind_event("finalize_placement", self._on_finalize_placement)
        model.bind_event("start_placement", self._on_start_placement)
        model.bind_event("set_gizmo_mode", self._on_set_gizmo_mode)
        self._handle = model.get_handle()

    def on_update(self, doc):
        del doc
        dirty = False
        st = downloads.get_state()
        prep_key = (
            st["stage"],
            round(st["progress"], 3),
            st.get("checkpoint_cached"),
            st.get("unik3d_source_cached"),
            st.get("threedgeer_source_cached"),
            st.get("threedgeer_built"),
            pipeline_loader.is_loaded(),
        )
        if prep_key != self._last_prep:
            self._last_prep = prep_key
            self._dirty(*_DIRTY_PREP)
            dirty = True
        job = self._job
        if job:
            triple = (job.stage.value, job.progress, job.status)
            if triple != (self._last_stage, self._last_progress, self._last_status):
                self._last_stage, self._last_progress, self._last_status = triple
                self._dirty(*_DIRTY_RUN)
                dirty = True
            if job.log_text != self._last_log:
                self._last_log = job.log_text
                self._dirty(*_DIRTY_LOG)
                dirty = True
            if job.is_running() != self._last_running:
                self._last_running = job.is_running()
                self._dirty(*_DIRTY_RUNNING)
                dirty = True
            rk = self._result_key(job.result)
            if rk is not None and rk != self._last_result_key:
                self._last_result = job.result
                self._last_result_key = rk
                self._on_job_finished(job)
                self._dirty(*_DIRTY_RESULT, *_DIRTY_PLACE)
                dirty = True
        if self._sync_selection():
            dirty = True
        cp = self._can_paste_clipboard()
        if cp != self._last_can_paste:
            self._last_can_paste = cp
            self._dirty("can_paste", "can_gen_clipboard")
            dirty = True
        try:
            if self._gizmo_active() or (self._generated_nodes and lf.has_selection()):
                lf.ui.request_redraw()
        except Exception:
            pass
        return dirty

    @staticmethod
    def _result_key(r):
        if r is None:
            return None
        return (r.success, r.num_gaussians, round(r.elapsed_s, 2), r.node_name, r.camera_kind, r.error)

    def _is_running(self):
        return self._job is not None and self._job.is_running()

    def _can_run(self):
        return bool(self.image_path) and downloads.is_ready() and not self._is_running()

    def _can_gen_clipboard(self):
        return self._can_paste_clipboard() and downloads.is_ready() and not self._is_running()

    def _show_results(self):
        return self._last_result is not None and self._last_result.success

    def _show_error(self):
        return self._last_result is not None and not self._last_result.success and self._last_result.error != "Cancelled"

    def _result_count_text(self):
        if not self._last_result:
            return ""
        return str(self._last_result.num_gaussians) if self._last_result.num_gaussians else "unknown"

    def _prep_bytes_line(self):
        s = downloads.get_state()
        total = int(s.get("bytes_total") or 0)
        done = int(s.get("bytes_downloaded") or 0)
        if total <= 0:
            return ""
        return f"{done // 1_000_000} / {total // 1_000_000} MB"

    def _model_status_line(self):
        st = downloads.get_state()
        if st["stage"] in {"checking", "cloning", "downloading"}:
            return st.get("message") or "Preparing UniSHARP assets..."
        if st["stage"] == "error":
            return "Asset preparation failed - retry after checking the log."
        if not downloads.is_ready():
            return "Checkpoint and UniK3D prepare on first use."
        if pipeline_loader.is_loaded():
            return "Ready - model in VRAM"
        return "Ready - checkpoint cached"

    def _dependency_status_line(self):
        s = downloads.get_state()
        parts = [
            "checkpoint ok" if s["checkpoint_cached"] else "checkpoint missing",
            "UniK3D ok" if s["unik3d_source_cached"] else "UniK3D missing",
        ]
        if s["threedgeer_built"]:
            parts.append("fisheye ok")
        elif s["threedgeer_source_cached"]:
            parts.append("fisheye source present; rasterizer not built")
        else:
            parts.append("fisheye source missing")
        return " / ".join(parts)

    def _set_image_path(self, v):
        self._use_image_path(v, dirty_path=False)

    def _use_image_path(self, path, *, dirty_path: bool) -> None:
        self.image_path = str(path)
        downloads.start_background_prepare()
        fields = ["image_name", "can_run"]
        if dirty_path:
            fields.insert(0, "image_path")
        self._dirty(*fields)

    def _set_camera_json_path(self, v):
        self.camera_json_path = str(v)
        self._dirty("camera_json_path", "camera_json_name")

    def _on_browse_image(self, *_):
        path = lf.ui.open_image_dialog("")
        if path:
            self._use_image_path(path, dirty_path=True)

    @staticmethod
    def _can_paste_clipboard():
        try:
            return bool(lf.ui.has_clipboard_image())
        except Exception:
            return False

    def _paste_clipboard_to_image(self) -> bool:
        path = _CLIPBOARD_DIR / f"clipboard_{uuid4().hex}.png"
        try:
            _CLIPBOARD_DIR.mkdir(parents=True, exist_ok=True)
            ok = lf.ui.save_clipboard_image(str(path))
        except Exception as exc:  # noqa: BLE001
            lf.log.warn(f"[unisharp] clipboard paste failed: {exc}")
            return False
        if not ok:
            lf.log.warn("[unisharp] No image on the clipboard to paste.")
            return False
        _safe_unlink(self._clipboard_path)
        self._clipboard_path = path
        self._use_image_path(path, dirty_path=True)
        return True

    def _on_paste_image(self, *_):
        self._paste_clipboard_to_image()

    def _on_generate_from_clipboard(self, *_):
        if self._paste_clipboard_to_image():
            self._on_start()

    def _on_browse_camera_json(self, *_):
        path = _open_json_dialog(self.camera_json_path)
        if path:
            self._set_camera_json_path(path)

    def _on_clear_camera_json(self, *_):
        self._set_camera_json_path("")

    def _new_job(self):
        cfg = JobConfig(
            image_path=self.image_path,
            camera_mode=self.camera_mode,
            camera_json_path=self.camera_json_path,
            camera_intrinsics=self.camera_intrinsics,
            camera_params=self.camera_params,
            low_pass_filter_eps=self.low_pass_filter_eps,
            splat_scale=self.splat_scale,
            append=self.append_mode,
        )
        self._last_result = None
        self._last_result_key = None
        self._last_log = ""
        self._job = UnisharpJob(cfg)
        self._job.start()
        self._dirty(*_DIRTY_RUNNING, *_DIRTY_LOG, *_DIRTY_RESULT, *_DIRTY_RUN)

    def _on_start(self, *_):
        if not self._can_run():
            lf.log.warn("[unisharp] Cannot run: need an image and prepared UniSHARP assets.")
            downloads.start_background_prepare()
            return
        self._new_job()

    def _on_cancel(self, *_):
        if self._job and self._job.is_running():
            self._job.cancel()

    def _on_job_finished(self, job):
        if job.result and job.result.success:
            self._node_name = job.result.node_name
            if self._node_name and self._node_name not in self._generated_nodes:
                self._generated_nodes.append(self._node_name)
            self._reset_placement_fields()
            self._attach_gizmo()

    def _set_camera_mode(self, v):
        try:
            self.camera_mode = normalize_camera_mode(str(v))
        except ValueError:
            return

    def _set_text(self, name, v):
        setattr(self, name, str(v))

    def _set_bool(self, name, v):
        setattr(self, name, bool(v))

    def _set_float(self, name, v, lo, hi):
        try:
            setattr(self, name, max(lo, min(hi, float(v))))
        except (TypeError, ValueError):
            pass

    def _on_toggle_section(self, handle, event, args):
        del handle, event
        name = args[0] if args else ""
        if name in self._collapsed:
            self._collapsed.discard(name)
        else:
            self._collapsed.add(name)
        self._sync_section_states()

    def _sync_section_states(self):
        if not self._doc:
            return
        for name in ("calibration", "precise"):
            content = self._doc.get_element_by_id(f"sec-{name}")
            arrow = self._doc.get_element_by_id(f"arrow-{name}")
            if content:
                content.set_class("collapsed", name in self._collapsed)
            if arrow:
                arrow.set_class("is-expanded", name not in self._collapsed)

    def _prune_dead_nodes(self):
        if self._is_running():
            return False
        alive = []
        for name in self._generated_nodes:
            try:
                exists = lf.get_node_visualizer_world_transform(name) is not None
            except Exception:
                exists = True
            if exists:
                alive.append(name)
        changed = len(alive) != len(self._generated_nodes)
        self._generated_nodes = alive
        if self._node_name and self._node_name not in alive:
            self._detach_gizmo()
            self._node_name = ""
            self._last_selection = None
            self._dirty(*_DIRTY_PLACE, "has_node")
            changed = True
        return changed

    def _ensure_gizmo(self):
        if self._gizmo is None:
            self._gizmo = lf.TransformGizmo()
            self._gizmo.set_on_change(self._on_gizmo_change)
            self._gizmo.set_on_end(self._on_gizmo_end)
        return self._gizmo

    def _attach_gizmo(self):
        if not self._node_name:
            return
        try:
            g = self._ensure_gizmo()
            g.operation = self.gizmo_mode
            g.attach_to_node(self._node_name)
            self._on_gizmo_change()
            lf.ui.request_redraw()
        except Exception as exc:  # noqa: BLE001
            lf.log.warn(f"[unisharp] gizmo attach failed: {exc}")

    def _detach_gizmo(self):
        if self._gizmo is not None:
            try:
                self._gizmo.detach()
            except Exception:
                pass
            lf.ui.request_redraw()

    def _gizmo_active(self):
        try:
            return self._gizmo is not None and self._gizmo.attached
        except Exception:
            return False

    def _on_finalize_placement(self, *_):
        self._detach_gizmo()
        self._dirty(*_DIRTY_PLACE)

    def _on_start_placement(self, *_):
        self._attach_gizmo()
        self._dirty(*_DIRTY_PLACE)

    def _on_selection_generation(self, _gen):
        self._sync_selection()

    def _sync_selection(self):
        try:
            sel = lf.get_selected_node_name() if lf.has_selection() else ""
        except Exception:
            return False
        if sel == self._last_selection:
            return False
        self._last_selection = sel
        if sel and sel != self._node_name and sel in self._generated_nodes:
            self._node_name = sel
            self._attach_gizmo()
            self._dirty(*_DIRTY_PLACE, "has_node")
            return True
        return False

    def _on_gizmo_change(self, *_):
        try:
            d = lf.decompose_transform(self._gizmo.matrix)
            self.tx, self.ty, self.tz = d["translation"]
            self.rx, self.ry, self.rz = d["rotation_euler_deg"]
            self.scl = max(abs(d["scale"][0]), _MIN_SCALE)
            self._dirty(*_DIRTY_PLACE)
        except Exception:
            pass

    def _on_gizmo_end(self, *_):
        self._on_gizmo_change()

    def _set_gizmo_mode(self, v):
        self.gizmo_mode = str(v)
        if self._gizmo is not None:
            try:
                self._gizmo.operation = self.gizmo_mode
            except Exception:
                pass

    def _on_set_gizmo_mode(self, handle, event, args):
        del handle, event
        mode = args[0] if args else ""
        if mode in ("translate", "rotate", "scale"):
            self._set_gizmo_mode(mode)
            self._dirty("gizmo_mode", "is_translate", "is_rotate", "is_scale")

    def _set_transform_field(self, axis, v):
        try:
            val = float(v)
        except (TypeError, ValueError):
            return
        if axis == "scl":
            val = max(abs(val), _MIN_SCALE)
        setattr(self, axis, val)
        if not self._node_name:
            return
        try:
            m = list(lf.get_node_visualizer_world_transform(self._node_name))
        except Exception:
            return
        if not m or len(m) != 16:
            return
        if axis == "scl":
            target = abs(val)
            for c in range(3):
                ox, oy, oz = m[c * 4], m[c * 4 + 1], m[c * 4 + 2]
                length = (ox * ox + oy * oy + oz * oz) ** 0.5
                if length > 1e-8:
                    k = target / length
                    m[c * 4], m[c * 4 + 1], m[c * 4 + 2] = ox * k, oy * k, oz * k
        elif axis in ("tx", "ty", "tz"):
            m[12 + {"tx": 0, "ty": 1, "tz": 2}[axis]] = val
        elif axis in ("rx", "ry", "rz"):
            try:
                d = lf.decompose_transform(m)
                t, e, s = list(d["translation"]), list(d["rotation_euler_deg"]), list(d["scale"])
            except Exception:
                return
            e[{"rx": 0, "ry": 1, "rz": 2}[axis]] = val
            m = lf.compose_transform(t, e, s)
        lf.set_node_visualizer_world_transform(self._node_name, m)
        lf.ui.request_redraw()

    def _reset_placement_fields(self):
        self.tx = self.ty = self.tz = 0.0
        self.rx = self.ry = self.rz = 0.0
        self.scl = 1.0
