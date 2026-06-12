from unisharp.utils.unik3d_adapter import _with_unik3d_losses_skipped


def test_with_unik3d_losses_skipped_restores_builder():
    calls = []

    class _FakeUniK3D:
        def __init__(self):
            self.build_losses({"training": True})

        def build_losses(self, config):
            calls.append(config)
            raise AssertionError("training losses should not be built for inference")

    original = _FakeUniK3D.build_losses

    model = _with_unik3d_losses_skipped(_FakeUniK3D, _FakeUniK3D)

    assert model.losses == {}
    assert calls == []
    assert _FakeUniK3D.build_losses is original
