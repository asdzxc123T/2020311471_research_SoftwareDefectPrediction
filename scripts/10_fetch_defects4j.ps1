param(
  [string]$Dest = "data\\raw\\defects4j"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destAbs = Join-Path $root $Dest
$repoAbs = Join-Path $destAbs "defects4j"

New-Item -ItemType Directory -Force -Path $destAbs | Out-Null

if (Test-Path (Join-Path $repoAbs ".git")) {
  Write-Host "Defects4J repo already exists at: $repoAbs"
  exit 0
}

Write-Host "Cloning Defects4J into: $repoAbs"
git clone --depth 1 https://github.com/rjust/defects4j.git $repoAbs

Write-Host "NOTE: Defects4J requires perl/Java/build tools. See: https://github.com/rjust/defects4j"

