#!/usr/bin/env bash
set -euo pipefail

lfs_plugins_dir="${LFS_PLUGINS_DIR:-$HOME/.lichtfeld/plugins}"
plugin_link="$lfs_plugins_dir/unisharp_plugin"

echo "== UniSHARP plugin uninstaller =="
if [[ -L "$plugin_link" ]]; then
    rm "$plugin_link"
    echo "Removed link: $plugin_link"
elif [[ -e "$plugin_link" ]]; then
    echo "Path exists and is not a symlink: $plugin_link" >&2
    echo "Remove it manually if you want to uninstall it." >&2
    exit 1
else
    echo "Not installed: $plugin_link"
fi
echo "Downloaded assets are left in this plugin directory: models/, UniK3D/, 3dgeer/, cache/."
