from core import downloads, repo_policy


def test_checkpoint_path_uses_upstream_typo(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "CKPTS_DIR", tmp_path)
    assert downloads.checkpoint_path() == tmp_path / repo_policy.CHECKPOINT_NAME


def test_is_checkpoint_cached_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "CKPTS_DIR", tmp_path)
    assert downloads.is_checkpoint_cached() is False


def test_is_checkpoint_cached_true_when_large_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "CKPTS_DIR", tmp_path)
    path = tmp_path / downloads.CHECKPOINT_NAME
    with path.open("wb") as f:
        f.seek(101_000_000 - 1)
        f.write(b"\0")
    assert downloads.is_checkpoint_cached() is True


def test_source_readiness_checks_expected_local_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "UNIK3D_DIR", tmp_path / "UniK3D")
    monkeypatch.setattr(downloads, "THREEDGEER_DIR", tmp_path / "3dgeer")
    monkeypatch.setattr(downloads, "GEER_RASTERIZER_DIR", tmp_path / "3dgeer" / "submodules" / "geer-rasterizer")
    assert downloads.is_unik3d_source_cached() is False
    assert downloads.is_3dgeer_source_cached() is False

    (tmp_path / "UniK3D" / "unik3d").mkdir(parents=True)
    (tmp_path / "UniK3D" / "unik3d" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "3dgeer" / "submodules" / "geer-rasterizer").mkdir(parents=True)
    assert downloads.is_unik3d_source_cached() is True
    assert downloads.is_3dgeer_source_cached() is True


def test_fisheye_ready_error_mentions_build_when_source_exists(tmp_path, monkeypatch):
    root = tmp_path / "geer-rasterizer"
    root.mkdir()
    monkeypatch.setattr(downloads, "GEER_RASTERIZER_DIR", root)
    try:
        downloads.assert_fisheye_ready()
    except RuntimeError as exc:
        assert "build_ext --inplace" in str(exc)
    else:
        raise AssertionError("expected missing GEER build to raise")
