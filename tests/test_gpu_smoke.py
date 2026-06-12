import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA", allow_module_level=True)
pytest.importorskip("einops")
pytest.importorskip("gsplat")


@pytest.mark.gpu
def test_unisharp_pipeline_loads_when_assets_are_present():
    from core import downloads, pipeline_loader

    if not downloads.is_ready():
        pytest.skip("UniSHARP checkpoint or UniK3D source not prepared")
    pipe = pipeline_loader.get_pipeline()
    assert pipe.model is not None
    assert pipe.renderer is not None
    assert pipe.panorama_renderer is not None
    assert int(pipe.step) >= 0
