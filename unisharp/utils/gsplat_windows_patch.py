from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_OLD_FLAG_BLOCK = '''        opt_level = "-O0" if FAST_COMPILE else "-O3"
        extra_cflags = [opt_level, "-Wno-attributes"]
        extra_cuda_cflags = [opt_level]
'''

_NEW_FLAG_BLOCK = '''        opt_level = "-O0" if FAST_COMPILE else "-O3"
        cxx_opt_level = "/Od" if FAST_COMPILE else "/O2"
        extra_cflags = [cxx_opt_level] if os.name == "nt" else [opt_level, "-Wno-attributes"]
        extra_cuda_cflags = [opt_level]
'''


def patch_gsplat_windows_backend() -> bool:
    if sys.platform != "win32":
        return False

    backend_path = _gsplat_backend_path()
    if backend_path is None:
        return False

    try:
        source = backend_path.read_text(encoding="utf-8")
    except OSError:
        return False

    if _NEW_FLAG_BLOCK in source:
        return True
    if _OLD_FLAG_BLOCK not in source:
        return False

    try:
        backend_path.write_text(source.replace(_OLD_FLAG_BLOCK, _NEW_FLAG_BLOCK), encoding="utf-8")
    except OSError:
        return False
    return True


def _gsplat_backend_path() -> Path | None:
    spec = importlib.util.find_spec("gsplat")
    if spec is None or spec.origin is None:
        return None
    package_dir = Path(spec.origin).resolve().parent
    backend_path = package_dir / "cuda" / "_backend.py"
    return backend_path if backend_path.is_file() else None
