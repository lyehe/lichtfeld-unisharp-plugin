"""Plugin-local UniSHARP checkpoint and source dependency preparation."""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import repo_policy
from .unik3d_patch import patch_unik3d_for_inference

PLUGIN_DIR = repo_policy.plugin_root()
MODELS_DIR = repo_policy.models_dir()
CKPTS_DIR = repo_policy.checkpoints_dir()
CHECKPOINT_NAME = repo_policy.CHECKPOINT_NAME
CHECKPOINT_REPO = repo_policy.CHECKPOINT_REPO
APPROX_BYTES = repo_policy.CHECKPOINT_APPROX_BYTES
MIN_CHECKPOINT_BYTES = repo_policy.CHECKPOINT_MIN_BYTES

UNIK3D_DIR = repo_policy.unik3d_dir()
THREEDGEER_DIR = repo_policy.threedgeer_dir()
GEER_RASTERIZER_DIR = repo_policy.geer_rasterizer_dir()

SOURCE_REPOS = {
    dep.name: dep.repo_url for dep in repo_policy.SOURCE_DEPENDENCIES
}

_lock = threading.Lock()
_state = {
    "stage": "idle",  # idle | checking | cloning | downloading | ready | error
    "progress": 0.0,
    "message": "",
    "error": "",
    "cancelled": False,
    "bytes_downloaded": 0,
    "bytes_total": 0,
    "source": "",
}
_thread: threading.Thread | None = None


def _noop_log(_msg: str) -> None:
    return None


_log_fn: Callable[[str], None] = _noop_log


def set_logger(fn: Callable[[str], None]) -> None:
    global _log_fn
    _log_fn = fn


def get_state() -> dict:
    with _lock:
        state = dict(_state)
    state.update(dependency_status())
    return state


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def _is_cancelled() -> bool:
    with _lock:
        return bool(_state["cancelled"])


def checkpoint_path() -> Path:
    return CKPTS_DIR / CHECKPOINT_NAME


def is_checkpoint_cached() -> bool:
    path = checkpoint_path()
    return path.is_file() and path.stat().st_size > MIN_CHECKPOINT_BYTES


def is_unik3d_source_cached() -> bool:
    return (UNIK3D_DIR / "unik3d" / "__init__.py").is_file()


def is_3dgeer_source_cached() -> bool:
    return GEER_RASTERIZER_DIR.is_dir()


def _try_import_geer() -> tuple[bool, str]:
    if not is_3dgeer_source_cached():
        return False, "3DGEER source is missing."
    root = str(GEER_RASTERIZER_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        importlib.import_module("diff_gaussian_rasterization")
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def is_3dgeer_built() -> bool:
    if not is_3dgeer_source_cached():
        return False
    patterns = ("*.pyd", "*.so", "*.dll", "*.dylib")
    for pattern in patterns:
        for _path in GEER_RASTERIZER_DIR.rglob(pattern):
            return True
    return False


def dependency_status() -> dict:
    geer_built = is_3dgeer_built()
    return {
        "checkpoint_cached": is_checkpoint_cached(),
        "unik3d_source_cached": is_unik3d_source_cached(),
        "threedgeer_source_cached": is_3dgeer_source_cached(),
        "threedgeer_built": geer_built,
        "threedgeer_error": "",
    }


def is_ready() -> bool:
    return is_checkpoint_cached() and is_unik3d_source_cached()


def assert_fisheye_ready() -> None:
    if not is_3dgeer_source_cached():
        raise RuntimeError(
            "Fisheye mode requires 3DGEER. Let UniSHARP preparation clone it, "
            f"or clone {repo_policy.THREEDGEER_SOURCE.repo_url} into the plugin's 3dgeer/ folder."
        )
    ok, err = _try_import_geer()
    if not ok:
        raise RuntimeError(
            "Fisheye mode requires the 3DGEER CUDA rasterizer to be built. "
            f"Run: cd \"{GEER_RASTERIZER_DIR}\" && python setup.py build_ext --inplace. "
            f"Import error: {err}"
        )


def start_background_prepare() -> None:
    """Clone source deps and download the UniSHARP checkpoint if needed."""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        if _state["stage"] == "ready" and is_ready():
            _patch_unik3d_for_inference()
            return
    if is_ready():
        _patch_unik3d_for_inference()
        _set(stage="ready", progress=1.0, message="UniSHARP assets ready", error="", source="")
        return
    _set(
        stage="checking",
        progress=0.0,
        message="Checking UniSHARP assets...",
        error="",
        cancelled=False,
        bytes_downloaded=0,
        bytes_total=0,
        source="",
    )
    _thread = threading.Thread(target=_run, name="unisharp-prepare", daemon=True)
    _thread.start()


def cancel_prepare() -> None:
    _set(cancelled=True)


def join(timeout: float = 2.0) -> None:
    t = _thread
    if t and t.is_alive():
        t.join(timeout=timeout)


def delete_models() -> None:
    cancel_prepare()
    join(timeout=3.0)
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR, ignore_errors=True)
    _set(stage="idle", progress=0.0, message="", error="", bytes_downloaded=0, bytes_total=0, source="")


def _run() -> None:
    try:
        _ensure_unik3d()
        _ensure_3dgeer_source()
        _download_checkpoint()
        if _is_cancelled():
            _set(stage="error", error="Cancelled", message="Preparation cancelled")
            return
        _set(stage="ready", progress=1.0, message="UniSHARP assets ready", source="")
        _log_fn("UniSHARP checkpoint and source dependencies are ready.")
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        _log_fn(f"Preparation failed: {msg}")
        _set(stage="error", error=msg, message="Preparation failed")


def _ensure_unik3d() -> None:
    if not is_unik3d_source_cached():
        _clone_source("UniK3D", UNIK3D_DIR, SOURCE_REPOS["UniK3D"], recursive=False)
    _patch_unik3d_for_inference()


def _patch_unik3d_for_inference() -> None:
    if patch_unik3d_for_inference(UNIK3D_DIR):
        _log_fn("Patched UniK3D optional imports for inference-only use.")


def _ensure_3dgeer_source() -> None:
    if is_3dgeer_source_cached():
        return
    try:
        _clone_source("3dgeer", THREEDGEER_DIR, SOURCE_REPOS["3dgeer"], recursive=True)
    except Exception as exc:  # noqa: BLE001
        # Perspective and panorama still work without GEER. Fisheye raises a clear
        # error later through assert_fisheye_ready().
        _log_fn(f"3DGEER clone failed; fisheye mode will remain unavailable: {exc}")


def _clone_source(name: str, target: Path, url: str, *, recursive: bool) -> None:
    if _is_cancelled():
        return
    _set(stage="cloning", progress=0.05, message=f"Cloning {name}...", source=name)
    _log_fn(f"Cloning {name} into plugin-local source directory...")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    cmd = ["git", "clone", "--depth", "1"]
    if recursive:
        cmd += ["--recursive", "--shallow-submodules"]
    cmd += [url, str(target)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _download_checkpoint() -> None:
    if is_checkpoint_cached():
        _set(stage="ready", progress=1.0, message="Checkpoint ready", source="")
        return
    CKPTS_DIR.mkdir(parents=True, exist_ok=True)
    _set(
        stage="downloading",
        message=f"Downloading UniSHARP checkpoint (~{APPROX_BYTES // 1_000_000_000} GB)...",
        progress=0.10,
        bytes_total=APPROX_BYTES,
        source="checkpoint",
    )
    _log_fn("Downloading UniSHARP checkpoint to plugin-local cache...")
    stop_flag = threading.Event()

    def watch() -> None:
        while not stop_flag.is_set():
            try:
                total = sum(f.stat().st_size for f in CKPTS_DIR.rglob("*") if f.is_file())
            except OSError:
                total = 0
            frac = 0.10 + (0.89 * min(1.0, total / APPROX_BYTES))
            _set(progress=min(0.99, frac), bytes_downloaded=total, bytes_total=APPROX_BYTES)
            if _is_cancelled():
                return
            time.sleep(0.5)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            filename=CHECKPOINT_NAME,
            local_dir=str(CKPTS_DIR),
            local_dir_use_symlinks=False,
        )
    finally:
        stop_flag.set()
        watcher.join(timeout=1.0)
