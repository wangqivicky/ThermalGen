$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

python "$ProjectRoot\scripts\export_stgl_pairs.py" `
  --split all `
  --patch-size 512 `
  --stride 256 `
  --output-size 512

