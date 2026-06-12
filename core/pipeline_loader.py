"""Cached, GPU-resident UniSHARP pipeline singleton."""
from __future__ import annotations

import threading
from dataclasses import dataclass

from . import dependency_policy, downloads, inference

_lock = threading.Lock()
_pipeline = None  # type: ignore[var-annotated]


@dataclass
class UnisharpPipeline:
    model: object
    renderer: object
    panorama_renderer: object
    step: int
    device: object
    low_pass_filter_eps: float


def is_loaded() -> bool:
    return _pipeline is not None


def _assert_cuda() -> None:
    dependency_policy.apply_runtime_environment()
    dependency_policy.assert_repo_owned_torch_not_preloaded()
    import torch

    dependency_policy.assert_canonical_torch_loaded(torch)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "UniSHARP requires a CUDA-capable NVIDIA GPU in this plugin; no CUDA device is available."
        )


def _apply_perf_flags() -> None:
    import torch

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def get_pipeline(low_pass_filter_eps: float = 0.0) -> UnisharpPipeline:
    global _pipeline
    eps = float(low_pass_filter_eps)
    if _pipeline is not None and abs(float(_pipeline.low_pass_filter_eps) - eps) < 1e-9:
        return _pipeline
    with _lock:
        if _pipeline is not None and abs(float(_pipeline.low_pass_filter_eps) - eps) < 1e-9:
            return _pipeline
        if not downloads.is_ready():
            raise RuntimeError("UniSHARP checkpoint and UniK3D source are not ready yet.")
        _assert_cuda()
        import torch

        _apply_perf_flags()
        inference.configure_torchhub_cache()
        device = torch.device("cuda:0")
        model, step = inference.load_model(downloads.checkpoint_path(), device=device)
        renderer = inference.create_renderer(device=device, low_pass_filter_eps=eps)
        panorama_renderer = inference.create_panorama_renderer(renderer=renderer)
        _pipeline = UnisharpPipeline(
            model=model,
            renderer=renderer,
            panorama_renderer=panorama_renderer,
            step=int(step),
            device=device,
            low_pass_filter_eps=eps,
        )
        return _pipeline


def unload() -> None:
    global _pipeline
    with _lock:
        _pipeline = None
    import gc

    for _ in range(2):
        gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
