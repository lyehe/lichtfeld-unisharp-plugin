"""Canonical repo ownership policy for plugin assets and runtime paths."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from . import dependency_policy

PLUGIN_LINK_NAME = "unisharp_plugin"
PLUGIN_DISPLAY_NAME = "UniSHARP"
DATA_MODEL_NAME = "unisharp"
SCENE_GROUP_BASE_NAME = "UniSHARP"

MODELS_DIR_NAME = "models"
CACHE_DIR_NAME = "cache"
CHECKPOINTS_DIR_NAME = "ckpts"
CHECKPOINT_NAME = "pretained_model.pt"
CHECKPOINT_REPO = "Insta360-Research/Unisharp"
CHECKPOINT_APPROX_BYTES = 9_000_000_000
CHECKPOINT_MIN_BYTES = 100_000_000


@dataclass(frozen=True)
class SourceDependency:
    name: str
    directory_name: str
    repo_url: str
    recursive: bool = False

    def path(self, root: Path | None = None) -> Path:
        return plugin_root(root) / self.directory_name


UNIK3D_SOURCE = SourceDependency("UniK3D", "UniK3D", "https://github.com/lpiccinelli-eth/UniK3D.git")
THREEDGEER_SOURCE = SourceDependency(
    "3dgeer",
    "3dgeer",
    "https://github.com/boschresearch/3dgeer.git",
    recursive=True,
)
SOURCE_DEPENDENCIES = (UNIK3D_SOURCE, THREEDGEER_SOURCE)


def plugin_root(root: Path | None = None) -> Path:
    return Path(root).resolve() if root is not None else dependency_policy.plugin_root()


def models_dir(root: Path | None = None) -> Path:
    return plugin_root(root) / MODELS_DIR_NAME


def cache_dir(root: Path | None = None) -> Path:
    return plugin_root(root) / CACHE_DIR_NAME


def checkpoints_dir(root: Path | None = None) -> Path:
    return models_dir(root) / CHECKPOINTS_DIR_NAME


def checkpoint_path(root: Path | None = None) -> Path:
    return checkpoints_dir(root) / CHECKPOINT_NAME


def unik3d_dir(root: Path | None = None) -> Path:
    return UNIK3D_SOURCE.path(root)


def threedgeer_dir(root: Path | None = None) -> Path:
    return THREEDGEER_SOURCE.path(root)


def geer_rasterizer_dir(root: Path | None = None) -> Path:
    return threedgeer_dir(root) / "submodules" / "geer-rasterizer"


def clipboard_dir(root: Path | None = None) -> Path:
    return cache_dir(root) / "clipboard"


def jobs_dir(root: Path | None = None) -> Path:
    return cache_dir(root) / "jobs"


def huggingface_home(root: Path | None = None) -> Path:
    return models_dir(root) / "huggingface"


def huggingface_hub_cache(root: Path | None = None) -> Path:
    return huggingface_home(root) / "hub"


def torch_home(root: Path | None = None) -> Path:
    return models_dir(root) / "torch"


def torch_compile_cache(root: Path | None = None) -> Path:
    return cache_dir(root) / "torch_compile"


def triton_cache(root: Path | None = None) -> Path:
    return cache_dir(root) / "triton"


def runtime_cache_dirs(root: Path | None = None) -> tuple[Path, ...]:
    return (
        huggingface_home(root),
        huggingface_hub_cache(root),
        torch_home(root),
        torch_compile_cache(root),
        triton_cache(root),
    )


def repo_import_roots(root: Path | None = None) -> tuple[Path, ...]:
    return (
        plugin_root(root),
        unik3d_dir(root),
        geer_rasterizer_dir(root),
    )


def apply_cache_environment(root: Path | None = None) -> None:
    """Point runtime caches at repo-owned directories."""
    root = plugin_root(root)
    for path in runtime_cache_dirs(root):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(huggingface_home(root))
    os.environ["HF_HUB_CACHE"] = str(huggingface_hub_cache(root))
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(huggingface_hub_cache(root))
    os.environ.setdefault("TORCH_HOME", str(torch_home(root)))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torch_compile_cache(root)))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache(root)))


def prepend_repo_import_roots(root: Path | None = None) -> None:
    """Prefer repo-owned source checkouts when optional dependencies are present."""
    for path in repo_import_roots(root):
        if path.exists():
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)

