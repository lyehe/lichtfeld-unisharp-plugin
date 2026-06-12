from types import SimpleNamespace

from unisharp.utils import windows_build_env


def test_capture_vsdevcmd_env_parses_set_output(monkeypatch, tmp_path):
    vsdevcmd = tmp_path / "VsDevCmd.bat"
    vsdevcmd.write_text("@echo off\n")

    def fake_run(args, input, check, capture_output, text):
        assert args == ["cmd.exe", "/d", "/q"]
        assert f'call "{vsdevcmd}"' in input
        assert "prompt=" in input
        assert check is True
        assert capture_output is True
        assert text is True
        return SimpleNamespace(stdout="PATH=C:\\VC\\bin\nINCLUDE=C:\\VC\\include\n=C:=C:\\ignored\n")

    monkeypatch.setattr(windows_build_env.subprocess, "run", fake_run)

    env = windows_build_env._capture_vsdevcmd_env(vsdevcmd)

    assert env == {"PATH": "C:\\VC\\bin", "INCLUDE": "C:\\VC\\include"}


def test_ensure_msvc_build_env_loads_vsdevcmd_when_cl_missing(monkeypatch, tmp_path):
    vsdevcmd = tmp_path / "VsDevCmd.bat"
    vsdevcmd.write_text("@echo off\n")
    which_results = iter([None, "C:\\VC\\bin\\cl.exe"])

    monkeypatch.setattr(windows_build_env.sys, "platform", "win32")
    monkeypatch.setattr(windows_build_env.shutil, "which", lambda name: next(which_results))
    monkeypatch.setattr(windows_build_env, "_candidate_vsdevcmd_paths", lambda: [vsdevcmd])
    monkeypatch.setattr(
        windows_build_env,
        "_capture_vsdevcmd_env",
        lambda path: {"PATH": "C:\\VC\\bin", "INCLUDE": "C:\\VC\\include"},
    )
    monkeypatch.setenv("CL", "/O2")

    assert windows_build_env.ensure_msvc_build_env() is True
    assert windows_build_env.os.environ["INCLUDE"] == "C:\\VC\\include"
    assert windows_build_env.os.environ["CL"] == "/O2 /Zc:preprocessor"
    assert windows_build_env.os.environ["TORCH_DONT_CHECK_COMPILER_ABI"] == "1"


def test_ensure_python_import_lib_on_lib_path_adds_vcpkg_lib(monkeypatch, tmp_path):
    base = tmp_path / "vcpkg_installed" / "x64-windows" / "tools" / "python3"
    lib_dir = tmp_path / "vcpkg_installed" / "x64-windows" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / f"python{windows_build_env.sys.version_info.major}{windows_build_env.sys.version_info.minor}.lib").write_text("")

    monkeypatch.setattr(windows_build_env.sys, "base_prefix", str(base))
    monkeypatch.setattr(windows_build_env.sys, "base_exec_prefix", str(base))
    monkeypatch.setattr(windows_build_env.sys, "exec_prefix", str(base))
    monkeypatch.delenv("LIB", raising=False)

    windows_build_env._ensure_python_import_lib_on_lib_path()

    assert windows_build_env.os.environ["LIB"] == str(lib_dir)
