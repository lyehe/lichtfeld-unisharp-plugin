"""Insert a UniSHARP-generated PLY into the current LFS scene."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import repo_policy

if TYPE_CHECKING:
    from . import lfs_splat_adapter

BASE_NAME = repo_policy.SCENE_GROUP_BASE_NAME


def next_group_name(scene, append: bool) -> str:
    if not append:
        return BASE_NAME
    i = 1
    while True:
        candidate = f"{BASE_NAME}_{i:02d}"
        if scene.get_node(candidate) is None:
            return candidate
        i += 1


def _get_valid_scene(lf, *, log) -> object | None:
    try:
        scene = lf.get_scene()
    except Exception as exc:  # noqa: BLE001
        log(f"insertion: lf.get_scene() failed: {exc}")
        return None
    if scene is None or not scene.is_valid():
        log("insertion: no valid scene; skipping.")
        return None
    return scene


def _create_splat_parent(scene, *, append: bool) -> tuple[int, str]:
    group_name = next_group_name(scene, append=append)
    if not append:
        try:
            scene.remove_node(group_name, keep_children=False)
        except Exception:
            pass

    parent_id = scene.add_group(group_name)
    return parent_id, f"{group_name} / splats"


def _add_splat_data(scene, splat_name: str, parent_id: int, sd) -> None:
    scene.add_splat(
        name=splat_name,
        means=sd.means_raw,
        sh0=sd.sh0_raw,
        shN=sd.shN_raw,
        scaling=sd.scaling_raw,
        rotation=sd.rotation_raw,
        opacity=sd.opacity_raw,
        sh_degree=sd.active_sh_degree,
        scene_scale=sd.scene_scale,
        parent=parent_id,
    )


def _notify_scene(scene, *, log) -> None:
    try:
        scene.notify_changed()
    except Exception as exc:  # noqa: BLE001
        log(f"insertion: notify_changed failed: {exc}")


def insert_ply(ply_path: Path | str, *, append: bool, log=None) -> str | None:
    """Insert a PLY into lf.get_scene(); returns the splat node name or None."""
    import lichtfeld as lf

    _log = log or (lambda _m: None)
    path = Path(ply_path)
    if not path.is_file():
        raise FileNotFoundError(f"UniSHARP PLY not found: {path}")
    scene = _get_valid_scene(lf, log=_log)
    if scene is None:
        return None

    parent_id, splat_name = _create_splat_parent(scene, append=append)

    result = lf.io.load(str(path))
    sd = result.splat_data
    if sd is None:
        _log("insertion: lf.io.load returned no splat_data; skipping.")
        return None
    _add_splat_data(scene, splat_name, parent_id, sd)
    _notify_scene(scene, log=_log)
    _log(f"insertion: added '{splat_name}'.")
    return splat_name


def insert_gaussians(
    gaussians,
    *,
    append: bool,
    log=None,
    color_space: lfs_splat_adapter.ColorSpace = "linearRGB",
    scene_scale: float = 1.0,
) -> str | None:
    """Insert activated UniSHARP Gaussians using the raw LFS tensor encoding."""
    import lichtfeld as lf

    from . import lfs_splat_adapter

    _log = log or (lambda _m: None)
    scene = _get_valid_scene(lf, log=_log)
    if scene is None:
        return None

    parent_id, splat_name = _create_splat_parent(scene, append=append)
    sd = lfs_splat_adapter.gaussians_to_lfs_splat_data(
        gaussians,
        lf=lf,
        color_space=color_space,
        scene_scale=scene_scale,
    )
    _add_splat_data(scene, splat_name, parent_id, sd)
    _notify_scene(scene, log=_log)
    _log(f"insertion: added '{splat_name}' from in-memory Gaussians.")
    return splat_name
