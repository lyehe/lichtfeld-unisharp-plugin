#!/usr/bin/env bash
set -euo pipefail

# One-shot installer for the UniSHARP LichtFeld Studio plugin on Linux / macOS.
# Creates:
#   <LFS plugins>/unisharp_plugin   -> this plugin directory

plugin_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lfs_plugins_dir="${LFS_PLUGINS_DIR:-$HOME/.lichtfeld/plugins}"
plugin_link="$lfs_plugins_dir/unisharp_plugin"

if [[ ! -f "$plugin_root/unisharp/__init__.py" ]]; then
    echo "Vendored UniSHARP package not found at $plugin_root/unisharp/." >&2
    echo "Re-clone the plugin repository." >&2
    exit 1
fi

echo
echo "== UniSHARP plugin installer =="
echo "Plugin root:     $plugin_root"
echo "LFS plugins dir: $lfs_plugins_dir"

mkdir -p "$lfs_plugins_dir"
if [[ -L "$plugin_link" ]]; then
    rm "$plugin_link"
elif [[ -e "$plugin_link" ]]; then
    echo "Path exists and is not a symlink: $plugin_link" >&2
    echo "Remove it manually if you want this installer to replace it." >&2
    exit 1
fi
ln -s "$plugin_root" "$plugin_link"

echo
echo "Install complete."
echo "  $plugin_link -> $plugin_root"
echo
echo "Next steps:"
echo "  1. Launch LichtFeld Studio."
echo "  2. Open the 'UniSHARP' panel."
echo "  3. Select an image or click Retry to sync dependencies and prepare assets."
echo "  4. Generate from the selected image."
echo
echo "Fisheye note: after ./3dgeer/ is cloned, build its rasterizer if needed:"
echo "  cd ./3dgeer/submodules/geer-rasterizer && python setup.py build_ext --inplace"
