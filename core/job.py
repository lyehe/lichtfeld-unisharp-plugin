from __future__ import annotations

import os
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from uuid import uuid4

try:
    import lichtfeld as lf
except Exception:  # pragma: no cover - outside LFS host
    lf = None  # type: ignore[assignment]

from . import downloads, inference, insertion, pipeline_loader, repo_policy
from .camera import normalize_camera_mode

_INSERT_TIMEOUT_S = 120.0
_DIRECT_LFS_INSERT_ENV = "UNISHARP_USE_DIRECT_LFS_INSERT"


class JobStage(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    LOADING_MODEL = "loading_model"
    RUNNING_MODEL = "running_model"
    INSERTING = "inserting"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


_RUNNING = {JobStage.PREPARING, JobStage.LOADING_MODEL, JobStage.RUNNING_MODEL, JobStage.INSERTING}


def _use_direct_lfs_insert() -> bool:
    return os.environ.get(_DIRECT_LFS_INSERT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class JobConfig:
    image_path: str
    camera_mode: str = "auto"
    camera_json_path: str = ""
    camera_intrinsics: str = ""
    camera_params: str = ""
    low_pass_filter_eps: float = 0.0
    splat_scale: float = 1.25
    append: bool = True


@dataclass
class JobResult:
    success: bool
    elapsed_s: float = 0.0
    num_gaussians: int = 0
    node_name: str = ""
    ply_path: str = ""
    output_dir: str = ""
    camera_kind: str = ""
    error: str = ""


class _Cancelled(Exception):
    pass


class UnisharpJob:
    def __init__(self, cfg: JobConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancelled = False
        self._stage = JobStage.IDLE
        self._progress = 0.0
        self._status = ""
        self._result: JobResult | None = None
        self._log: deque[str] = deque(maxlen=64)

    @property
    def stage(self):
        with self._lock:
            return self._stage

    @property
    def progress(self):
        with self._lock:
            return self._progress

    @property
    def status(self):
        with self._lock:
            return self._status

    @property
    def result(self):
        with self._lock:
            return self._result

    @property
    def log_text(self):
        with self._lock:
            return "\n".join(self._log)

    def is_running(self) -> bool:
        return self.stage in _RUNNING

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("Job already started")
            self._thread = threading.Thread(target=self._run, daemon=True)
            thread = self._thread
        thread.start()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            self._status = "Cancelling after the current UniSHARP step..."

    def _set(self, stage, progress, status):
        with self._lock:
            self._stage, self._progress, self._status = stage, progress, status

    def _log_line(self, msg):
        line = str(msg).rstrip()
        if not line:
            return
        with self._lock:
            self._log.append(line)
        if lf is not None:
            lf.log.info(f"[unisharp] {line}")

    def _is_cancelled(self):
        with self._lock:
            return self._cancelled

    def _check_cancel(self):
        if self._is_cancelled():
            raise _Cancelled()

    def _run(self):
        t0 = time.time()
        try:
            self._run_pipeline(t0)
        except _Cancelled:
            with self._lock:
                progress = self._progress
                self._result = JobResult(False, error="Cancelled", elapsed_s=time.time() - t0)
            self._set(JobStage.CANCELLED, progress, "Cancelled")
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            self._log_line(f"ERROR: {msg}")
            self._log_line(traceback.format_exc())
            with self._lock:
                progress = self._progress
                self._result = JobResult(False, error=msg, elapsed_s=time.time() - t0)
            self._set(JobStage.ERROR, progress, msg)
        finally:
            self._trim_cuda_cache()

    @staticmethod
    def _trim_cuda_cache() -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _run_pipeline(self, t0):
        cfg = self.cfg
        image_path = Path(cfg.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        camera_mode = normalize_camera_mode(cfg.camera_mode)
        if camera_mode == "fisheye":
            downloads.assert_fisheye_ready()

        self._set(JobStage.PREPARING, 0.05, "Preparing UniSHARP assets...")
        downloads.start_background_prepare()
        while not downloads.is_ready():
            self._check_cancel()
            st = downloads.get_state()
            if st["stage"] == "error":
                raise RuntimeError(st.get("error") or "UniSHARP asset preparation failed.")
            downloads.join(timeout=0.5)

        self._set(JobStage.LOADING_MODEL, 0.18, "Loading UniSHARP model...")
        self._check_cancel()
        pipe = pipeline_loader.get_pipeline(low_pass_filter_eps=float(cfg.low_pass_filter_eps))

        out_root = repo_policy.jobs_dir() / uuid4().hex
        out_root.mkdir(parents=True, exist_ok=True)

        direct_lfs_insert = _use_direct_lfs_insert()

        self._set(JobStage.RUNNING_MODEL, 0.35, "Running UniSHARP inference...")
        self._check_cancel()
        run = inference.run_single_image(
            pipeline=pipe,
            config=inference.InferenceConfig(
                image_path=image_path,
                out_root=out_root,
                checkpoint_path=downloads.checkpoint_path(),
                camera=inference.CameraOptions(
                    mode=camera_mode,
                    json_path=cfg.camera_json_path,
                    intrinsics=cfg.camera_intrinsics,
                    params=cfg.camera_params,
                ),
                low_pass_filter_eps=float(cfg.low_pass_filter_eps),
                splat_scale=float(cfg.splat_scale),
                return_gaussians=direct_lfs_insert,
            ),
        )
        self._check_cancel()

        self._set(JobStage.INSERTING, 0.90, "Inserting Gaussian into scene...")
        self._check_cancel()
        node_holder: dict = {}
        done = threading.Event()

        def _ui_insert():
            try:
                if direct_lfs_insert and run.gaussians is not None:
                    try:
                        node_holder["name"] = insertion.insert_gaussians(
                            run.gaussians,
                            append=cfg.append,
                            log=self._log_line,
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._log_line(f"insertion: direct LFS insert failed; falling back to PLY load: {exc}")
                        node_holder["name"] = insertion.insert_ply(run.ply_path, append=cfg.append, log=self._log_line)
                else:
                    node_holder["name"] = insertion.insert_ply(run.ply_path, append=cfg.append, log=self._log_line)
            except Exception as exc:  # noqa: BLE001
                node_holder["error"] = exc
            finally:
                done.set()

        lf.ui.schedule_on_ui_thread(_ui_insert)
        if not done.wait(timeout=_INSERT_TIMEOUT_S):
            raise RuntimeError(
                f"Scene insertion did not complete within {_INSERT_TIMEOUT_S:.0f}s "
                "(UI thread busy)."
            )
        if "error" in node_holder:
            raise node_holder["error"]

        with self._lock:
            self._result = JobResult(
                True,
                elapsed_s=time.time() - t0,
                num_gaussians=run.num_gaussians,
                node_name=node_holder.get("name") or "",
                ply_path=str(run.ply_path),
                output_dir=str(run.sample_dir),
                camera_kind=run.camera_kind,
            )
        self._set(JobStage.DONE, 1.0, "Done")
