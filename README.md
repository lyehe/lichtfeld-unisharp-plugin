# UniSHARP Plugin for LichtFeld Studio

Single image -> 3D Gaussian splat, inserted into the current LichtFeld scene.
This plugin vendors the inference path from
[Insta360-Research-Team/UniSHARP](https://github.com/Insta360-Research-Team/UniSHARP)
Python source and wraps its single-image inference flow in a LichtFeld panel.

## Features

- **Universal camera input** - auto-detects perspective, panorama, and fisheye images, with manual overrides.
- **Optional calibration** - pass a camera JSON, pinhole intrinsics, or Fisheye624 params when available.
- **Scene insertion** - inserts via LFS `scene.add_splat` when the host tensor bridge is available, while still saving `gaussians.ply` for compatibility and fallback.
- **Placement controls** - transform gizmo plus typed translate / rotate / scale fields.
- **Plugin-local assets** - checkpoints, cloned source dependencies, and CUDA caches stay inside this plugin directory.
- **VRAM release** - the model is unloaded when LFS training starts or when you click **Free VRAM**.
- **Inference-only runtime** - upstream training, dataset, validation, and metric code is intentionally omitted.

## Requirements

- Python `>=3.12,<3.13` in the LFS plugin environment.
- CUDA-capable NVIDIA GPU.
- Canonical plugin runtime stack:
  - `torch==2.8.0` from the PyTorch `cu128` index.
  - `torchvision==0.23.0` from the PyTorch `cu128` index.
  - `gsplat==1.5.3`.
- Git available on `PATH` for first-run cloning of UniK3D and 3DGEER source.
- Disk space for:
  - PyTorch / gsplat environment.
  - UniSHARP checkpoint `pretained_model.pt` from `Insta360-Research/Unisharp`.
  - UniK3D and 3DGEER source clones.

Fisheye mode also requires the 3DGEER CUDA rasterizer to be built after its source is cloned:

```powershell
cd .\3dgeer\submodules\geer-rasterizer
python setup.py build_ext --inplace
```

```bash
cd ./3dgeer/submodules/geer-rasterizer
python setup.py build_ext --inplace
```

Perspective and panorama modes do not require the 3DGEER rasterizer.
On Windows, building the fisheye rasterizer requires a CUDA-compatible MSVC
toolchain on `PATH`; `where cl` should find `cl.exe` before running the build.

The plugin does not use LFS as an inference rasterizer on current LFS `master`.
LFS provides scene insertion and viewport rendering, but not a public offscreen
`render_splat_data` API for arbitrary in-memory splats.

### Torch and xFormers policy

LFS plugins run in one Python process, so native Torch DLLs are process-global.
This plugin owns and validates its runtime stack from this repository:
`torch==2.8.0+cu128`, `torchvision==0.23.0+cu128`, and `gsplat==1.5.3`.

Do not install xFormers into this plugin. UniK3D treats xFormers as optional,
and the plugin disables it by default with `UNISHARP_DISABLE_XFORMERS=1`.
The reason is native ABI safety: if another plugin has already loaded a
different Torch stack, for example `torch 2.11 cu130`, xFormers built for
`torch 2.8 cu128` can fail at DLL load time with missing `c10.dll` entry
points. Reinstalling UniSHARP does not fix that; the proper fix is to keep
xFormers disabled or align all loaded ML plugins to one Torch/CUDA stack.

## Installation

### Manual dev install

```powershell
cd Lichtfeld-Unisharp-Plugin
.\install.ps1
```

```bash
cd Lichtfeld-Unisharp-Plugin
./install.sh
```

The installer creates:

```text
~/.lichtfeld/plugins/unisharp_plugin -> this plugin directory
```

On first LFS launch, the plugin environment is synced from the locked project
environment in `pyproject.toml` and `uv.lock`. Runtime assets are prepared after
you select an image or click **Retry** in the panel:

```text
models/ckpts/pretained_model.pt
UniK3D/
3dgeer/
cache/
```

The checkpoint filename is intentionally `pretained_model.pt`; that spelling
matches the upstream Hugging Face asset.

## Usage

1. Open the **UniSHARP** panel.
2. Choose or paste an image.
3. Leave camera mode as **Auto**, or choose **Perspective**, **Panorama**, or **Fisheye**.
4. Optional: open **Calibration** and provide a camera JSON, intrinsics, or Fisheye624 params.
5. Click **Generate**.
6. Place the inserted splat with the viewport gizmo or typed transform fields.

The generated output for each run is kept under `cache/jobs/<run>/...`, including
`gaussians.ply`, upstream metadata, and UniSHARP's preview renders.

By default, the generated Gaussians are inserted directly through
`scene.add_splat` when LFS exposes a compatible tensor bridge. Set
`UNISHARP_USE_DIRECT_LFS_INSERT=0` to force the PLY load path, or
`UNISHARP_USE_DIRECT_LFS_INSERT=1` to force the direct path with PLY fallback on
failure.

## Camera Calibration

A camera JSON can match UniSHARP's official inference format:

```json
{
  "camera": "perspective",
  "intrinsics": {
    "fx": 820.0,
    "fy": 820.0,
    "cx": 512.0,
    "cy": 384.0
  }
}
```

For fisheye:

```json
{
  "camera": "fisheye",
  "camera_params": [820.0, 820.0, 512.0, 384.0, 0.01, -0.001, 0.0, 0.0]
}
```

The panel's text fields also accept `fx fy cx cy`, 9 row-major K values, 8
Fisheye624 values, or all 16 Fisheye624 values.

## Notes

- **Plugin code:** MIT - see [LICENSE](LICENSE).
- **UniSHARP source:** vendored from the upstream UniSHARP repository.
- **Model checkpoint:** downloaded at runtime from
  [Insta360-Research/Unisharp](https://huggingface.co/Insta360-Research/Unisharp);
  it is not redistributed with this plugin.
- **External source dependencies:** UniK3D and 3DGEER are cloned at runtime into
  plugin-local directories and are not committed.

## Uninstall

```powershell
.\uninstall.ps1
```

```bash
./uninstall.sh
```

This removes the LFS plugin link. Delete `models/`, `UniK3D/`, `3dgeer/`, and
`cache/` manually if you want to reclaim downloaded assets.
