from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from core import insertion, repo_policy
from unisharp.utils.gaussians import Gaussians3D


class _FakeScene:
    def __init__(self, existing):
        self._names = set(existing)

    def get_node(self, name):
        return object() if name in self._names else None


def test_next_group_name_replace_mode_returns_base():
    scene = _FakeScene({repo_policy.SCENE_GROUP_BASE_NAME, f"{repo_policy.SCENE_GROUP_BASE_NAME}_01"})
    assert insertion.next_group_name(scene, append=False) == repo_policy.SCENE_GROUP_BASE_NAME


def test_next_group_name_append_picks_first_free_slot():
    scene = _FakeScene({repo_policy.SCENE_GROUP_BASE_NAME, f"{repo_policy.SCENE_GROUP_BASE_NAME}_01"})
    assert insertion.next_group_name(scene, append=True) == f"{repo_policy.SCENE_GROUP_BASE_NAME}_02"


def test_next_group_name_append_from_empty_scene():
    scene = _FakeScene(set())
    assert insertion.next_group_name(scene, append=True) == f"{repo_policy.SCENE_GROUP_BASE_NAME}_01"


def test_insert_gaussians_uses_scene_add_splat(monkeypatch):
    class FakeTensor:
        @staticmethod
        def from_dlpack(obj):
            return obj.detach().clone()

        @staticmethod
        def from_numpy(arr, copy=True):
            return torch.from_numpy(arr.copy() if copy else arr)

    class FakeInsertScene:
        def __init__(self):
            self.added = None
            self.notified = False

        def is_valid(self):
            return True

        def get_node(self, _name):
            return None

        def add_group(self, name):
            self.group_name = name
            return 7

        def add_splat(self, **kwargs):
            self.added = kwargs
            return 8

        def notify_changed(self):
            self.notified = True

    scene = FakeInsertScene()
    fake_lf = SimpleNamespace(Tensor=FakeTensor, get_scene=lambda: scene)
    monkeypatch.setitem(sys.modules, "lichtfeld", fake_lf)

    gaussians = Gaussians3D(
        mean_vectors=torch.zeros((1, 2, 3)),
        singular_values=torch.ones((1, 2, 3)),
        quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
        colors=torch.ones((1, 2, 3)),
        opacities=torch.full((1, 2), 0.5),
    )

    name = insertion.insert_gaussians(gaussians, append=True)

    assert name == f"{repo_policy.SCENE_GROUP_BASE_NAME}_01 / splats"
    assert scene.group_name == f"{repo_policy.SCENE_GROUP_BASE_NAME}_01"
    assert scene.added["name"] == name
    assert scene.added["parent"] == 7
    assert scene.added["means"].shape == (2, 3)
    assert scene.added["shN"].shape == (2, 0, 3)
    assert scene.added["sh_degree"] == 0
    assert scene.added["scene_scale"] == 1.0
    assert scene.notified is True
