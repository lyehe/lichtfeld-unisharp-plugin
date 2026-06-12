from __future__ import annotations

import json
from types import SimpleNamespace

from core import inference, repo_policy


def test_run_single_image_builds_upstream_args_and_reads_result(monkeypatch, tmp_path):
    seen = {}

    class _FakeInfer:
        @staticmethod
        def load_camera_json(path):
            return {"loaded": str(path)}

        @staticmethod
        def slug_from_path(path):
            return f"slug_{path.stem}"

        @staticmethod
        def process_one(*, model, renderer, panorama_renderer, image_path, out_root, step, args):
            seen.update(
                model=model,
                renderer=renderer,
                panorama_renderer=panorama_renderer,
                image_path=image_path,
                out_root=out_root,
                step=step,
                args=args,
            )
            sample = out_root / _FakeInfer.slug_from_path(image_path)
            sample.mkdir(parents=True)
            (sample / "gaussians.ply").write_text("not a real ply", encoding="utf-8")
            (sample / "metadata.json").write_text(json.dumps({"camera_kind": "perspective"}), encoding="utf-8")
            return {"gaussians": "in-memory-gaussians"}

    monkeypatch.setattr(inference, "_infer_module", lambda: _FakeInfer)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpg")
    camera_json = tmp_path / "camera.json"
    camera_json.write_text("{}", encoding="utf-8")

    pipe = SimpleNamespace(model="m", renderer="r", panorama_renderer="p", step=7, device="cuda:0")
    result = inference.run_single_image(
        pipeline=pipe,
        config=inference.InferenceConfig(
            image_path=image,
            out_root=tmp_path / "out",
            checkpoint_path=tmp_path / repo_policy.CHECKPOINT_NAME,
            camera=inference.CameraOptions(
                mode="perspective",
                json_path=str(camera_json),
                intrinsics="1 2 3 4",
                params="",
            ),
            splat_scale=1.5,
            return_gaussians=True,
        ),
    )

    assert seen["model"] == "m"
    assert seen["step"] == 7
    assert seen["args"].camera == "perspective"
    assert seen["args"].camera_intrinsics == [1.0, 2.0, 3.0, 4.0]
    assert seen["args"].camera_json == camera_json
    assert seen["args"]._camera_json_data == {"loaded": str(camera_json)}
    assert seen["args"].splat_scale == 1.5
    assert seen["args"].return_gaussians is True
    assert result.camera_kind == "perspective"
    assert result.ply_path.name == "gaussians.ply"
    assert result.gaussians == "in-memory-gaussians"
