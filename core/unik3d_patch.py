"""Local compatibility patches for the UniK3D source checkout."""
from __future__ import annotations

import re
from pathlib import Path

from .dependency_policy import DISABLE_XFORMERS_ENV

PATCH_MARKER = "# Patched by LichtFeld UniSHARP: keep optional training/logging imports lazy."
EVALUATION_PATCH_MARKER = "# Patched by LichtFeld UniSHARP: keep evaluation imports lazy."
ROBUST_LOSS_PATCH_MARKER = "# Patched by LichtFeld UniSHARP: use importlib.resources instead of pkg_resources."
XFORMERS_PATCH_MARKER = "# Patched by LichtFeld UniSHARP: disable optional xFormers native imports."
DISTRIBUTED_CV2_PATCH_MARKER = "# Patched by LichtFeld UniSHARP: remove unused cv2 import for inference."

_EAGER_OPTIONAL_IMPORTS_RE = re.compile(
    r"^from \.validation import validate\r?\n"
    r"from \.visualization import colorize, image_grid, log_train_artifacts\r?\n",
    re.MULTILINE,
)

_EAGER_EVALUATION_IMPORTS_RE = re.compile(
    r"^from \.evaluation_depth import \(DICT_METRICS, DICT_METRICS_3D, eval_3d,\r?\n"
    r"\s+eval_depth\)\r?\n",
    re.MULTILINE,
)

_PKG_RESOURCES_IMPORT_RE = re.compile(r"^from pkg_resources import resource_stream\r?\n", re.MULTILINE)
_DISTRIBUTED_CV2_IMPORT_RE = re.compile(r"^import cv2\r?\n", re.MULTILINE)

_XFORMERS_IMPORTS = {
    Path("unik3d/models/metadinov2/attention.py"): "from xformers.ops import fmha, memory_efficient_attention, unbind",
    Path("unik3d/models/metadinov2/block.py"): "from xformers.ops import fmha, index_select_cat, scaled_index_add",
    Path("unik3d/models/metadinov2/swiglu_ffn.py"): "from xformers.ops import SwiGLU",
}

_LAZY_OPTIONAL_EXPORTS = f"""{PATCH_MARKER}


def validate(*args, **kwargs):
    from .validation import validate as _validate

    return _validate(*args, **kwargs)


def colorize(*args, **kwargs):
    from .visualization import colorize as _colorize

    return _colorize(*args, **kwargs)


def image_grid(*args, **kwargs):
    from .visualization import image_grid as _image_grid

    return _image_grid(*args, **kwargs)


def log_train_artifacts(*args, **kwargs):
    from .visualization import log_train_artifacts as _log_train_artifacts

    return _log_train_artifacts(*args, **kwargs)

"""

_LAZY_EVALUATION_EXPORTS = f"""{EVALUATION_PATCH_MARKER}


class _LazyEvaluationMetrics:
    def __init__(self, name):
        self._name = name

    def _target(self):
        from . import evaluation_depth as _evaluation_depth

        return getattr(_evaluation_depth, self._name)

    def __contains__(self, key):
        return key in self._target()

    def __getitem__(self, key):
        return self._target()[key]

    def __iter__(self):
        return iter(self._target())

    def __len__(self):
        return len(self._target())

    def get(self, *args, **kwargs):
        return self._target().get(*args, **kwargs)

    def items(self):
        return self._target().items()

    def keys(self):
        return self._target().keys()

    def values(self):
        return self._target().values()


def eval_depth(*args, **kwargs):
    from .evaluation_depth import eval_depth as _eval_depth

    return _eval_depth(*args, **kwargs)


def eval_3d(*args, **kwargs):
    from .evaluation_depth import eval_3d as _eval_3d

    return _eval_3d(*args, **kwargs)


DICT_METRICS = _LazyEvaluationMetrics("DICT_METRICS")
DICT_METRICS_3D = _LazyEvaluationMetrics("DICT_METRICS_3D")

"""

_IMPORTLIB_RESOURCES_STREAM = f"""{ROBUST_LOSS_PATCH_MARKER}
from contextlib import contextmanager
from importlib import resources


@contextmanager
def resource_stream(package, resource_name):
    package_name = package.rsplit(".", 1)[0]
    resource = resources.files(package_name)
    for part in resource_name.split("/"):
        resource = resource.joinpath(part)
    with resource.open("rb") as stream:
        yield stream
"""


def patch_unik3d_for_inference(unik3d_dir: Path) -> bool:
    """Apply UniK3D source patches needed for plugin inference."""
    changed = patch_unik3d_utils_init(unik3d_dir)
    changed = patch_unik3d_robust_loss(unik3d_dir) or changed
    changed = patch_unik3d_xformers_imports(unik3d_dir) or changed
    changed = patch_unik3d_distributed_cv2_import(unik3d_dir) or changed
    return changed


def patch_unik3d_utils_init(unik3d_dir: Path) -> bool:
    """Avoid importing optional UniK3D training/evaluation modules during inference."""
    init_path = Path(unik3d_dir) / "unik3d" / "utils" / "__init__.py"
    if not init_path.is_file():
        return False

    text = init_path.read_text(encoding="utf-8")
    patched = text
    changed = False

    if PATCH_MARKER not in patched:
        patched, count = _EAGER_OPTIONAL_IMPORTS_RE.subn(_LAZY_OPTIONAL_EXPORTS, patched, count=1)
        changed = changed or count > 0

    if EVALUATION_PATCH_MARKER not in patched:
        patched, count = _EAGER_EVALUATION_IMPORTS_RE.subn(_LAZY_EVALUATION_EXPORTS, patched, count=1)
        changed = changed or count > 0

    if not changed:
        return False

    init_path.write_text(patched, encoding="utf-8")
    return True


def patch_unik3d_robust_loss(unik3d_dir: Path) -> bool:
    """Avoid UniK3D's deprecated pkg_resources import when losses are imported."""
    robust_loss_path = Path(unik3d_dir) / "unik3d" / "ops" / "losses" / "robust_loss.py"
    if not robust_loss_path.is_file():
        return False

    text = robust_loss_path.read_text(encoding="utf-8")
    if ROBUST_LOSS_PATCH_MARKER in text:
        return False

    patched, count = _PKG_RESOURCES_IMPORT_RE.subn(_IMPORTLIB_RESOURCES_STREAM, text, count=1)
    if count == 0:
        return False

    robust_loss_path.write_text(patched, encoding="utf-8")
    return True


def patch_unik3d_distributed_cv2_import(unik3d_dir: Path) -> bool:
    """Avoid requiring OpenCV for UniK3D distributed helpers during inference."""
    distributed_path = Path(unik3d_dir) / "unik3d" / "utils" / "distributed.py"
    if not distributed_path.is_file():
        return False

    text = distributed_path.read_text(encoding="utf-8")
    if DISTRIBUTED_CV2_PATCH_MARKER in text:
        return False

    patched, count = _DISTRIBUTED_CV2_IMPORT_RE.subn(f"{DISTRIBUTED_CV2_PATCH_MARKER}\n", text, count=1)
    if count == 0:
        return False

    distributed_path.write_text(patched, encoding="utf-8")
    return True


def patch_unik3d_xformers_imports(unik3d_dir: Path) -> bool:
    """Keep optional xFormers imports from loading broken native extensions by default."""
    changed = False
    for relative_path, import_stmt in _XFORMERS_IMPORTS.items():
        path = Path(unik3d_dir) / relative_path
        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8")
        if XFORMERS_PATCH_MARKER in text:
            continue

        pattern = re.compile(
            r"try:\r?\n"
            rf"    {re.escape(import_stmt)}\r?\n"
            r"\r?\n"
            r"    XFORMERS_AVAILABLE = True\r?\n"
            r"except ImportError:",
            re.MULTILINE,
        )
        replacement = (
            "try:\n"
            f"    {XFORMERS_PATCH_MARKER}\n"
            "    if (\n"
            f"        __import__(\"os\").environ.get(\"{DISABLE_XFORMERS_ENV}\", \"1\").strip().lower()\n"
            "        not in {\"0\", \"false\", \"no\", \"off\"}\n"
            "    ):\n"
            "        raise ImportError(\"xFormers disabled by UniSHARP plugin\")\n"
            f"    {import_stmt}\n"
            "\n"
            "    XFORMERS_AVAILABLE = True\n"
            "except Exception:"
        )
        patched, count = pattern.subn(replacement, text, count=1)
        if count == 0:
            continue

        path.write_text(patched, encoding="utf-8")
        changed = True

    return changed
