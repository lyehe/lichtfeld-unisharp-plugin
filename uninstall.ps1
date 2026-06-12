#requires -Version 5.1
[CmdletBinding()]
param(
    [string]$LFSPluginsDir
)

$ErrorActionPreference = "Stop"
if (-not $LFSPluginsDir) {
    $LFSPluginsDir = Join-Path $env:USERPROFILE ".lichtfeld\plugins"
}

function Remove-LinkIfPresent([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Host "Not installed: $path" -ForegroundColor DarkGray
        return
    }
    $item = Get-Item -LiteralPath $path -Force
    $isLink = ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq [IO.FileAttributes]::ReparsePoint
    if ($isLink) {
        & cmd /c rmdir (Get-Item -LiteralPath $path).FullName | Out-Null
        Write-Host "Removed link: $path" -ForegroundColor Green
        return
    }
    throw "Path exists but is not a junction/symlink: $path"
}

Write-Host "== UniSHARP plugin uninstaller ==" -ForegroundColor Cyan
Remove-LinkIfPresent (Join-Path $LFSPluginsDir "unisharp_plugin")
Write-Host "Downloaded assets are left in this plugin directory: models/, UniK3D/, 3dgeer/, cache/." -ForegroundColor DarkGray
