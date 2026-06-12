from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from core import dependency_policy


def _pyproject() -> dict:
    root = Path(__file__).resolve().parent.parent
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_uses_canonical_dependency_versions():
    data = _pyproject()
    deps = data["project"]["dependencies"]

    for package, version in dependency_policy.canonical_dependency_pins().items():
        assert f"{package}=={version}" in deps

    assert not any(dep.lower().startswith("xformers") for dep in deps)


def test_pyproject_uses_canonical_pytorch_index():
    data = _pyproject()

    sources = data["tool"]["uv"]["sources"]
    for package in ("torch", "torchvision"):
        assert sources[package]["index"] == dependency_policy.PYTORCH_INDEX_NAME

    indexes = {index["name"]: index["url"] for index in data["tool"]["uv"]["index"]}
    assert indexes[dependency_policy.PYTORCH_INDEX_NAME] == dependency_policy.PYTORCH_INDEX_URL


def test_apply_runtime_environment_prefers_repo_venv_and_disables_xformers(tmp_path, monkeypatch):
    site_packages = tmp_path / ".venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", ["existing"])
    monkeypatch.delenv(dependency_policy.DISABLE_XFORMERS_ENV, raising=False)

    dependency_policy.apply_runtime_environment(tmp_path)

    assert sys.path[0] == str(site_packages)
    assert sys.path[1:] == ["existing"]
    assert os.environ[dependency_policy.DISABLE_XFORMERS_ENV] == dependency_policy.XFORMERS_DISABLED_DEFAULT
    assert dependency_policy.XFORMERS_DISABLED_DEFAULT == "1"


def test_assert_repo_owned_torch_not_preloaded_rejects_foreign_module(tmp_path, monkeypatch):
    fake = ModuleType("torch")
    fake.__file__ = str(tmp_path.parent / "other_plugin" / ".venv" / "Lib" / "site-packages" / "torch" / "__init__.py")
    monkeypatch.setitem(sys.modules, "torch", fake)

    with pytest.raises(RuntimeError, match="already loaded"):
        dependency_policy.assert_repo_owned_torch_not_preloaded(tmp_path)


def test_assert_canonical_torch_loaded_accepts_repo_owned_module(tmp_path):
    fake = ModuleType("torch")
    torch_init = tmp_path / ".venv" / "Lib" / "site-packages" / "torch" / "__init__.py"
    torch_init.parent.mkdir(parents=True)
    torch_init.write_text("", encoding="utf-8")
    fake.__file__ = str(torch_init)
    fake.__version__ = f"{dependency_policy.TORCH_VERSION}+{dependency_policy.TORCH_CUDA_TAG}"

    dependency_policy.assert_canonical_torch_loaded(fake, tmp_path)
