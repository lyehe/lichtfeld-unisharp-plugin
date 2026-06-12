#requires -Version 5.1
<#
.SYNOPSIS
  One-shot installer for the UniSHARP LichtFeld Studio plugin.

.DESCRIPTION
  Creates a directory junction:

    <LFS plugins>/unisharp_plugin  -> this plugin directory

  The plugin vendors the upstream UniSHARP Python source at ./unisharp/.
  Runtime assets are prepared on first load:
    - UniK3D source cloned into ./UniK3D/
    - 3DGEER source cloned into ./3dgeer/ for fisheye paths
    - UniSHARP checkpoint downloaded into ./models/

.PARAMETER LFSPluginsDir
  LichtFeld Studio plugins dir. Default: $env:USERPROFILE\.lichtfeld\plugins.

.PARAMETER Force
  Replace an existing target at <LFS plugins>/unisharp_plugin even if it is not a link.
#>

[CmdletBinding()]
param(
    [string]$LFSPluginsDir,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$PluginRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $LFSPluginsDir) {
    $LFSPluginsDir = Join-Path $env:USERPROFILE ".lichtfeld\plugins"
}

function Remove-LinkLike([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return }
    $item = Get-Item -LiteralPath $path -Force
    $isLink = ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq [IO.FileAttributes]::ReparsePoint
    if ($isLink) {
        & cmd /c rmdir (Get-Item -LiteralPath $path).FullName | Out-Null
        return
    }
    if (-not $Force) {
        throw "Path exists and is not a link: $path  (use -Force to remove)"
    }
    Remove-Item -LiteralPath $path -Recurse -Force
}

function New-DirectoryJunction([string]$link, [string]$target) {
    $linkParent = Split-Path -Parent $link
    if (-not (Test-Path $linkParent)) {
        New-Item -ItemType Directory -Path $linkParent -Force | Out-Null
    }
    Remove-LinkLike $link
    try {
        New-Item -ItemType Junction -Path $link -Target $target -ErrorAction Stop | Out-Null
    } catch {
        $result = & cmd /c mklink /J "$link" "$target" 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create junction '$link' -> '$target': $result"
        }
    }
}

Write-Host ""
Write-Host "== UniSHARP plugin installer ==" -ForegroundColor Cyan
Write-Host "Plugin root:     $PluginRoot"
Write-Host "LFS plugins dir: $LFSPluginsDir"

$vendored = Join-Path $PluginRoot "unisharp\__init__.py"
if (-not (Test-Path $vendored -PathType Leaf)) {
    throw "Vendored UniSHARP package not found at $vendored. Re-clone the plugin repository."
}

$pluginLink = Join-Path $LFSPluginsDir "unisharp_plugin"

Write-Host ""
Write-Host "Creating junction..."
New-DirectoryJunction -link $pluginLink -target $PluginRoot
Write-Host "  [OK] $pluginLink" -ForegroundColor Green
Write-Host "       -> $PluginRoot" -ForegroundColor DarkGray

Write-Host ""
Write-Host "Install complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Launch LichtFeld Studio."
Write-Host "  2. Wait for first-run uv sync and UniSHARP asset preparation."
Write-Host "  3. Open the 'UniSHARP' panel and generate from an image."
Write-Host ""
Write-Host "Fisheye note: after ./3dgeer/ is cloned, build its rasterizer if needed:"
Write-Host "  cd .\3dgeer\submodules\geer-rasterizer; python setup.py build_ext --inplace"
Write-Host ""
Write-Host "To uninstall: .\uninstall.ps1" -ForegroundColor DarkGray
