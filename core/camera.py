from __future__ import annotations

_VALID_CAMERA_MODES = {"auto", "perspective", "pinhole", "fisheye", "panorama", "erp"}


def normalize_camera_mode(mode: str) -> str:
    value = str(mode or "auto").strip().lower()
    if value not in _VALID_CAMERA_MODES:
        raise ValueError(f"Unsupported camera mode: {mode!r}")
    return value


def parse_float_list(raw: str) -> list[float] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    parts = text.replace(",", " ").split()
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Expected numeric values, got {raw!r}") from exc
