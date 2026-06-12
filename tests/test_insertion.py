from core import insertion, repo_policy


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
