from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


def ensure_msvc_build_env() -> bool:
    """Make MSVC's command-line tools visible to PyTorch extension builds."""
    if sys.platform != "win32":
        return False
    if shutil.which("cl"):
        _ensure_standard_msvc_preprocessor()
        _ensure_python_import_lib_on_lib_path()
        _skip_torch_compiler_abi_check()
        return True

    for vsdevcmd in _candidate_vsdevcmd_paths():
        env = _capture_vsdevcmd_env(vsdevcmd)
        if not env:
            continue
        os.environ.update(env)
        if shutil.which("cl"):
            _ensure_standard_msvc_preprocessor()
            _ensure_python_import_lib_on_lib_path()
            _skip_torch_compiler_abi_check()
            return True
    return False


def _candidate_vsdevcmd_paths() -> Iterable[Path]:
    seen: set[Path] = set()

    for raw_path in (
        os.environ.get("VSDEVCMD_PATH"),
        _vsdevcmd_from_install_dir(os.environ.get("VSINSTALLDIR")),
    ):
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.is_file() and path not in seen:
            seen.add(path)
            yield path

    for root_env in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(root_env)
        if not root:
            continue
        for path in Path(root).glob("Microsoft Visual Studio/*/*/Common7/Tools/VsDevCmd.bat"):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def _vsdevcmd_from_install_dir(raw_install_dir: str | None) -> Path | None:
    if not raw_install_dir:
        return None
    return Path(raw_install_dir) / "Common7" / "Tools" / "VsDevCmd.bat"


def _ensure_standard_msvc_preprocessor() -> None:
    flag = "/Zc:preprocessor"
    current = os.environ.get("CL", "")
    if flag.lower() not in current.lower():
        os.environ["CL"] = f"{current} {flag}".strip()


def _ensure_python_import_lib_on_lib_path() -> None:
    lib_name = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    for base in _python_base_prefixes():
        for lib_dir in (base / "libs", base.parent.parent / "lib"):
            if (lib_dir / lib_name).is_file():
                _prepend_env_path("LIB", lib_dir)
                return


def _skip_torch_compiler_abi_check() -> None:
    os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")


def _python_base_prefixes() -> Iterable[Path]:
    seen: set[Path] = set()
    for raw_path in (sys.base_prefix, sys.base_exec_prefix, sys.exec_prefix):
        path = Path(raw_path)
        if path not in seen:
            seen.add(path)
            yield path


def _prepend_env_path(name: str, path: Path) -> None:
    path_text = str(path)
    current = os.environ.get(name, "")
    parts = [part for part in current.split(os.pathsep) if part]
    if any(part.lower() == path_text.lower() for part in parts):
        return
    os.environ[name] = os.pathsep.join([path_text, *parts])


def _capture_vsdevcmd_env(vsdevcmd: Path) -> dict[str, str]:
    command = (
        "prompt=\r\n"
        f'call "{vsdevcmd}" -arch=x64 -host_arch=x64 >nul\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        "set\r\n"
        "exit /b 0\r\n"
    )
    try:
        completed = subprocess.run(
            ["cmd.exe", "/d", "/q"],
            input=command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}

    env: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key.startswith("="):
            continue
        env[key] = value
    return env
