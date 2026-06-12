from __future__ import annotations

import os
import sys

from core import downloads, repo_policy


def test_repo_policy_owns_asset_paths(tmp_path):
    assert repo_policy.models_dir(tmp_path) == tmp_path / repo_policy.MODELS_DIR_NAME
    assert repo_policy.cache_dir(tmp_path) == tmp_path / repo_policy.CACHE_DIR_NAME
    assert (
        repo_policy.checkpoint_path(tmp_path)
        == tmp_path / repo_policy.MODELS_DIR_NAME / repo_policy.CHECKPOINTS_DIR_NAME / repo_policy.CHECKPOINT_NAME
    )
    assert repo_policy.unik3d_dir(tmp_path) == tmp_path / repo_policy.UNIK3D_SOURCE.directory_name
    assert (
        repo_policy.geer_rasterizer_dir(tmp_path)
        == tmp_path / repo_policy.THREEDGEER_SOURCE.directory_name / "submodules" / "geer-rasterizer"
    )
    assert repo_policy.clipboard_dir(tmp_path) == tmp_path / repo_policy.CACHE_DIR_NAME / "clipboard"
    assert repo_policy.jobs_dir(tmp_path) == tmp_path / repo_policy.CACHE_DIR_NAME / "jobs"


def test_repo_policy_applies_cache_environment(tmp_path, monkeypatch):
    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME"):
        monkeypatch.delenv(key, raising=False)

    repo_policy.apply_cache_environment(tmp_path)

    assert os.environ["HF_HOME"] == str(tmp_path / "models" / "huggingface")
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "models" / "huggingface" / "hub")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(tmp_path / "models" / "huggingface" / "hub")
    assert os.environ["TORCH_HOME"] == str(tmp_path / "models" / "torch")
    for path in repo_policy.runtime_cache_dirs(tmp_path):
        assert path.is_dir()


def test_repo_policy_prepend_import_roots_uses_existing_repo_paths(tmp_path, monkeypatch):
    (tmp_path / "UniK3D").mkdir()
    (tmp_path / "3dgeer" / "submodules" / "geer-rasterizer").mkdir(parents=True)
    monkeypatch.setattr(sys, "path", ["existing"])

    repo_policy.prepend_repo_import_roots(tmp_path)

    assert sys.path[:3] == [
        str(tmp_path / "3dgeer" / "submodules" / "geer-rasterizer"),
        str(tmp_path / "UniK3D"),
        str(tmp_path),
    ]
    assert sys.path[3:] == ["existing"]


def test_downloads_uses_repo_policy_identity():
    assert downloads.PLUGIN_DIR == repo_policy.plugin_root()
    assert downloads.CHECKPOINT_NAME == repo_policy.CHECKPOINT_NAME
    assert downloads.CHECKPOINT_REPO == repo_policy.CHECKPOINT_REPO
    assert downloads.APPROX_BYTES == repo_policy.CHECKPOINT_APPROX_BYTES
    assert downloads.MIN_CHECKPOINT_BYTES == repo_policy.CHECKPOINT_MIN_BYTES
    assert downloads.SOURCE_REPOS[repo_policy.UNIK3D_SOURCE.name] == repo_policy.UNIK3D_SOURCE.repo_url
    assert downloads.SOURCE_REPOS[repo_policy.THREEDGEER_SOURCE.name] == repo_policy.THREEDGEER_SOURCE.repo_url
