"""UniSHARP - universal-camera single-image Gaussian generation for LichtFeld Studio."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# LFS embedded Python (Windows) can expose a sys.stderr whose flush() raises
# OSError(EINVAL). tqdm/HF call flush() early, so redirect only that broken handle.
if sys.stderr is not None:
    try:
        sys.stderr.flush()
    except OSError:
        sys.stderr = open(os.devnull, "w", buffering=1)

_PLUGIN_DIR = Path(__file__).resolve().parent

from .core import dependency_policy as _dependency_policy  # noqa: E402
from .core import repo_policy as _repo_policy  # noqa: E402

_dependency_policy.apply_runtime_environment(_PLUGIN_DIR)
_repo_policy.apply_cache_environment(_PLUGIN_DIR)
_repo_policy.prepend_repo_import_roots(_PLUGIN_DIR)

try:
    import lichtfeld as lf  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - test/DX path outside LFS host
    from types import ModuleType, SimpleNamespace

    class _Panel:
        pass

    lf = ModuleType("lichtfeld")
    lf.ui = SimpleNamespace(
        Panel=_Panel,
        PanelSpace=SimpleNamespace(MAIN_PANEL_TAB="MAIN_PANEL_TAB"),
        PanelHeightMode=SimpleNamespace(CONTENT="CONTENT"),
        free_plugin_textures=lambda _plugin_name: None,
        schedule_on_ui_thread=lambda fn: fn(),
    )
    lf.log = SimpleNamespace(
        info=lambda _msg: None,
        warn=lambda _msg: None,
        error=lambda _msg: None,
    )
    lf.register_class = lambda _cls: None
    lf.unregister_class = lambda _cls: None
    lf.stop_training = lambda: None
    sys.modules["lichtfeld"] = lf

from .core import downloads, pipeline_loader  # noqa: E402
from .panels.main_panel import UnisharpPanel  # noqa: E402

_classes = [UnisharpPanel]
_last_training_state = False


def _on_training_state_changed(new):
    global _last_training_state
    import threading

    is_now = bool(new)
    rising_edge = is_now and not _last_training_state
    _last_training_state = is_now
    if rising_edge and pipeline_loader.is_loaded():
        lf.log.info("[unisharp] Training started - unloading model to free VRAM.")
        threading.Thread(target=pipeline_loader.unload, daemon=True).start()


def on_load():
    downloads.set_logger(lambda msg: lf.log.info(f"[unisharp] {msg}"))
    for cls in _classes:
        lf.register_class(cls)
    try:
        global _last_training_state
        from lfs_plugins.ui.state import AppState

        _last_training_state = bool(AppState.is_training.value)
        AppState.is_training.subscribe_as(_repo_policy.PLUGIN_LINK_NAME, _on_training_state_changed)
    except Exception as exc:  # noqa: BLE001
        lf.log.warn(f"{_repo_policy.PLUGIN_LINK_NAME}: couldn't subscribe to is_training ({exc}).")
    # Do not clone/download assets during host startup. LFS plugin loading runs
    # inside the embedded Python host; keeping startup side-effect-free avoids
    # native crashes caused by long-running network/subprocess work at launch.
    # The panel starts preparation when an image is selected or Retry is clicked.
    lf.log.info(f"{_repo_policy.PLUGIN_LINK_NAME} loaded")


def on_unload():
    import gc
    import time

    try:
        from lfs_plugins.ui.state import AppState

        if getattr(AppState, "is_training", None) is not None and AppState.is_training.value:
            try:
                lf.stop_training()
            except Exception:
                pass
            for _ in range(20):
                if not AppState.is_training.value:
                    break
                time.sleep(0.1)
    except Exception as exc:  # noqa: BLE001
        lf.log.warn(f"{_repo_policy.PLUGIN_LINK_NAME}: stop_training on unload failed: {exc}")
    try:
        downloads.cancel_prepare()
        downloads.join(timeout=2.0)
    except Exception:
        pass
    try:
        pipeline_loader.unload()
    except Exception as exc:  # noqa: BLE001
        lf.log.warn(f"{_repo_policy.PLUGIN_LINK_NAME}: pipeline_loader.unload() failed: {exc}")
    for _ in range(2):
        gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info(f"{_repo_policy.PLUGIN_LINK_NAME} unloaded")
