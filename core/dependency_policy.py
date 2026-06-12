"""Canonical dependency policy for the UniSHARP LFS plugin."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

PYTHON_REQUIRES = ">=3.12,<3.13"

TORCH_VERSION = "2.8.0"
TORCHVISION_VERSION = "0.23.0"
TORCH_CUDA_TAG = "cu128"
TORCH_CUDA_VERSION = "12.8"
PYTORCH_INDEX_NAME = "pytorch-cu128"
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"

GSPAT_VERSION = "1.5.3"

DISABLE_XFORMERS_ENV = "UNISHARP_DISABLE_XFORMERS"
XFORMERS_DISABLED_DEFAULT = "1"
XFORMERS_DISABLED_REASON = (
    "xFormers is disabled because LFS loads plugins in one Python process, "
    "and mixed Torch DLLs from other plugin venvs can break xFormers native extensions."
)

_DLL_DIR_HANDLES = []


def plugin_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_dependency_pins() -> dict[str, str]:
    return {
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
        "gsplat": GSPAT_VERSION,
    }


def venv_site_packages(root: Path | None = None) -> Path:
    root = Path(root or plugin_root())
    if os.name == "nt":
        return root / ".venv" / "Lib" / "site-packages"
    return root / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def torch_lib_dir(root: Path | None = None) -> Path:
    return venv_site_packages(root) / "torch" / "lib"


def apply_runtime_environment(root: Path | None = None) -> None:
    """Prefer repo-owned Python packages and apply native-extension safeguards."""
    root = Path(root or plugin_root()).resolve()
    os.environ.setdefault(DISABLE_XFORMERS_ENV, XFORMERS_DISABLED_DEFAULT)

    site_packages = venv_site_packages(root)
    if site_packages.is_dir():
        site = str(site_packages)
        sys.path[:] = [p for p in sys.path if _normcase(p) != _normcase(site)]
        sys.path.insert(0, site)

    if os.name == "nt":
        lib_dir = torch_lib_dir(root)
        if lib_dir.is_dir():
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(lib_dir)))


def assert_repo_owned_torch_not_preloaded(root: Path | None = None) -> None:
    """Fail early if a different plugin already imported Torch into this process."""
    module = sys.modules.get("torch")
    if module is None:
        return
    assert_repo_owned_module("torch", module, root)


def assert_repo_owned_module(name: str, module: ModuleType, root: Path | None = None) -> None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return
    root = Path(root or plugin_root()).resolve()
    path = Path(module_file).resolve()
    if _is_relative_to(path, root):
        return
    raise RuntimeError(
        f"UniSHARP requires repo-owned {name} from '{root}', but {name} is already loaded from "
        f"'{path}'. LFS plugins share one Python process, so unload other ML plugins or align "
        f"their Torch stack to torch=={TORCH_VERSION}+{TORCH_CUDA_TAG}."
    )


def assert_canonical_torch_loaded(torch_module, root: Path | None = None) -> None:
    assert_repo_owned_module("torch", torch_module, root)
    version = str(getattr(torch_module, "__version__", ""))
    expected = f"{TORCH_VERSION}+{TORCH_CUDA_TAG}"
    if version != expected:
        raise RuntimeError(f"UniSHARP requires torch {expected}, but loaded torch {version}.")


def _normcase(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
