from types import SimpleNamespace

from unisharp.utils import gsplat_windows_patch


def test_patch_gsplat_windows_backend_rewrites_msvc_flags(monkeypatch, tmp_path):
    package_dir = tmp_path / "gsplat"
    backend_path = package_dir / "cuda" / "_backend.py"
    backend_path.parent.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    backend_path.write_text(
        "prefix\n"
        '        opt_level = "-O0" if FAST_COMPILE else "-O3"\n'
        '        extra_cflags = [opt_level, "-Wno-attributes"]\n'
        "        extra_cuda_cflags = [opt_level]\n"
        "suffix\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gsplat_windows_patch.sys, "platform", "win32")
    monkeypatch.setattr(
        gsplat_windows_patch.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(package_dir / "__init__.py")),
    )

    assert gsplat_windows_patch.patch_gsplat_windows_backend() is True
    patched = backend_path.read_text(encoding="utf-8")
    assert 'cxx_opt_level = "/Od" if FAST_COMPILE else "/O2"' in patched
    assert 'extra_cflags = [cxx_opt_level] if os.name == "nt" else [opt_level, "-Wno-attributes"]' in patched


def test_patch_gsplat_windows_backend_is_idempotent(monkeypatch, tmp_path):
    package_dir = tmp_path / "gsplat"
    backend_path = package_dir / "cuda" / "_backend.py"
    backend_path.parent.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    backend_path.write_text(gsplat_windows_patch._NEW_FLAG_BLOCK, encoding="utf-8")

    monkeypatch.setattr(gsplat_windows_patch.sys, "platform", "win32")
    monkeypatch.setattr(
        gsplat_windows_patch.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(package_dir / "__init__.py")),
    )

    assert gsplat_windows_patch.patch_gsplat_windows_backend() is True
    assert backend_path.read_text(encoding="utf-8") == gsplat_windows_patch._NEW_FLAG_BLOCK
