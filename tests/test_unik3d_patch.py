from core.unik3d_patch import (
    DISTRIBUTED_CV2_PATCH_MARKER,
    EVALUATION_PATCH_MARKER,
    PATCH_MARKER,
    ROBUST_LOSS_PATCH_MARKER,
    XFORMERS_PATCH_MARKER,
    patch_unik3d_distributed_cv2_import,
    patch_unik3d_for_inference,
    patch_unik3d_robust_loss,
    patch_unik3d_utils_init,
    patch_unik3d_xformers_imports,
)

ORIGINAL_UTILS_INIT = """from .camera import invert_pinhole, project_pinhole, unproject_pinhole
from .distributed import barrier
from .validation import validate
from .visualization import colorize, image_grid, log_train_artifacts

__all__ = [
    "validate",
    "colorize",
    "image_grid",
    "log_train_artifacts",
]
"""


def test_patch_unik3d_utils_init_lazies_training_only_imports(tmp_path):
    utils_dir = tmp_path / "UniK3D" / "unik3d" / "utils"
    utils_dir.mkdir(parents=True)
    init_path = utils_dir / "__init__.py"
    init_path.write_text(ORIGINAL_UTILS_INIT, encoding="utf-8")

    assert patch_unik3d_utils_init(tmp_path / "UniK3D") is True

    patched = init_path.read_text(encoding="utf-8")
    assert PATCH_MARKER in patched
    assert "from .validation import validate\n" not in patched
    assert "from .visualization import colorize, image_grid, log_train_artifacts\n" not in patched
    assert "def validate(*args, **kwargs):" in patched
    assert "def log_train_artifacts(*args, **kwargs):" in patched
    assert '"validate"' in patched
    assert '"log_train_artifacts"' in patched

    assert patch_unik3d_utils_init(tmp_path / "UniK3D") is False
    assert init_path.read_text(encoding="utf-8") == patched


def test_patch_unik3d_utils_init_ignores_missing_checkout(tmp_path):
    assert patch_unik3d_utils_init(tmp_path / "missing") is False


def test_patch_unik3d_utils_init_lazies_evaluation_imports(tmp_path):
    utils_dir = tmp_path / "UniK3D" / "unik3d" / "utils"
    utils_dir.mkdir(parents=True)
    init_path = utils_dir / "__init__.py"
    init_path.write_text(
        "from .evaluation_depth import (DICT_METRICS, DICT_METRICS_3D, eval_3d,\n"
        "                               eval_depth)\n",
        encoding="utf-8",
    )

    assert patch_unik3d_utils_init(tmp_path / "UniK3D") is True

    patched = init_path.read_text(encoding="utf-8")
    assert EVALUATION_PATCH_MARKER in patched
    assert "from .evaluation_depth import (DICT_METRICS" not in patched
    assert "def eval_depth(*args, **kwargs):" in patched
    assert 'DICT_METRICS = _LazyEvaluationMetrics("DICT_METRICS")' in patched


def test_patch_unik3d_robust_loss_replaces_pkg_resources(tmp_path):
    loss_dir = tmp_path / "UniK3D" / "unik3d" / "ops" / "losses"
    loss_dir.mkdir(parents=True)
    robust_loss_path = loss_dir / "robust_loss.py"
    robust_loss_path.write_text(
        "from pkg_resources import resource_stream\n\n"
        "def load():\n"
        "    return resource_stream(__name__, 'resources/partition_spline.npz')\n",
        encoding="utf-8",
    )

    assert patch_unik3d_robust_loss(tmp_path / "UniK3D") is True

    patched = robust_loss_path.read_text(encoding="utf-8")
    assert ROBUST_LOSS_PATCH_MARKER in patched
    assert "from pkg_resources import resource_stream" not in patched
    assert "from importlib import resources" in patched


def test_patch_unik3d_distributed_cv2_import_removes_unused_runtime_dep(tmp_path):
    utils_dir = tmp_path / "UniK3D" / "unik3d" / "utils"
    utils_dir.mkdir(parents=True)
    distributed_path = utils_dir / "distributed.py"
    distributed_path.write_text(
        "import os\n"
        "import cv2\n"
        "import torch\n"
        "\n"
        "def setup_multi_processes(cfg):\n"
        "    # cv2.setNumThreads(opencv_num_threads)\n"
        "    return None\n",
        encoding="utf-8",
    )

    assert patch_unik3d_distributed_cv2_import(tmp_path / "UniK3D") is True

    patched = distributed_path.read_text(encoding="utf-8")
    assert DISTRIBUTED_CV2_PATCH_MARKER in patched
    assert "import cv2\n" not in patched
    assert "# cv2.setNumThreads(opencv_num_threads)" in patched

    assert patch_unik3d_distributed_cv2_import(tmp_path / "UniK3D") is False


def test_patch_unik3d_xformers_imports_disables_native_import_by_default(tmp_path):
    model_dir = tmp_path / "UniK3D" / "unik3d" / "models" / "metadinov2"
    model_dir.mkdir(parents=True)
    attention_path = model_dir / "attention.py"
    attention_path.write_text(
        "try:\n"
        "    from xformers.ops import fmha, memory_efficient_attention, unbind\n"
        "\n"
        "    XFORMERS_AVAILABLE = True\n"
        "except ImportError:\n"
        "    XFORMERS_AVAILABLE = False\n",
        encoding="utf-8",
    )

    assert patch_unik3d_xformers_imports(tmp_path / "UniK3D") is True

    patched = attention_path.read_text(encoding="utf-8")
    assert XFORMERS_PATCH_MARKER in patched
    assert "UNISHARP_DISABLE_XFORMERS" in patched
    assert "except Exception:" in patched
    assert "except ImportError:" not in patched

    assert patch_unik3d_xformers_imports(tmp_path / "UniK3D") is False


def test_patch_unik3d_for_inference_patches_multiple_files(tmp_path):
    utils_dir = tmp_path / "UniK3D" / "unik3d" / "utils"
    loss_dir = tmp_path / "UniK3D" / "unik3d" / "ops" / "losses"
    utils_dir.mkdir(parents=True)
    loss_dir.mkdir(parents=True)
    (utils_dir / "__init__.py").write_text(ORIGINAL_UTILS_INIT, encoding="utf-8")
    (utils_dir / "distributed.py").write_text("import cv2\n", encoding="utf-8")
    (loss_dir / "robust_loss.py").write_text("from pkg_resources import resource_stream\n", encoding="utf-8")

    assert patch_unik3d_for_inference(tmp_path / "UniK3D") is True
    assert DISTRIBUTED_CV2_PATCH_MARKER in (utils_dir / "distributed.py").read_text(encoding="utf-8")
    assert patch_unik3d_for_inference(tmp_path / "UniK3D") is False
