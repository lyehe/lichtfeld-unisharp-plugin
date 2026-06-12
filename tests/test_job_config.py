from types import SimpleNamespace

import pytest

from core import job
from core.camera import normalize_camera_mode, parse_float_list
from core.job import JobConfig


def test_jobconfig_defaults_match_panel_defaults():
    cfg = JobConfig(image_path="image.jpg")
    assert cfg.camera_mode == "auto"
    assert cfg.append is True
    assert cfg.low_pass_filter_eps == 0.0
    assert cfg.splat_scale == 1.25


def test_parse_float_list_accepts_spaces_and_commas():
    assert parse_float_list("") is None
    assert parse_float_list("1 2,3.5") == [1.0, 2.0, 3.5]


def test_parse_float_list_rejects_text():
    with pytest.raises(ValueError, match="Expected numeric"):
        parse_float_list("1 nope")


def test_normalize_camera_mode_aliases_and_rejects_unknown():
    assert normalize_camera_mode("AUTO") == "auto"
    assert normalize_camera_mode("pinhole") == "pinhole"
    assert normalize_camera_mode("erp") == "erp"
    with pytest.raises(ValueError, match="Unsupported camera mode"):
        normalize_camera_mode("orthographic")


def test_direct_lfs_insert_env_overrides_auto(monkeypatch):
    monkeypatch.setattr(job, "lf", SimpleNamespace())

    monkeypatch.setenv("UNISHARP_USE_DIRECT_LFS_INSERT", "1")
    assert job._use_direct_lfs_insert() is True

    monkeypatch.setenv("UNISHARP_USE_DIRECT_LFS_INSERT", "off")
    assert job._use_direct_lfs_insert() is False


def test_direct_lfs_insert_auto_requires_tensor_bridge(monkeypatch):
    monkeypatch.delenv("UNISHARP_USE_DIRECT_LFS_INSERT", raising=False)

    monkeypatch.setattr(job, "lf", SimpleNamespace(Tensor=SimpleNamespace(from_dlpack=lambda tensor: tensor)))
    assert job._use_direct_lfs_insert() is True

    monkeypatch.setattr(job, "lf", SimpleNamespace(Tensor=SimpleNamespace()))
    assert job._use_direct_lfs_insert() is False
