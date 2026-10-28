param([string]$Python = '3.12')
$ErrorActionPreference = 'Stop'
$gridRoot = Split-Path -Parent $PSScriptRoot
Push-Location $gridRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'Install uv (https://docs.astral.sh/uv/getting-started/installation/) first.'
    }
    & git -c core.autocrlf=false submodule update --init --recursive
    if ($LASTEXITCODE) { throw 'Submodule checkout failed.' }
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        & uv venv --python $Python .venv
        if ($LASTEXITCODE) { throw 'Virtual environment creation failed.' }
    }
    . "$PSScriptRoot/activate_windows.ps1"
    & uv pip install --python .venv/Scripts/python.exe -e .
    if ($LASTEXITCODE) { throw 'GRiD build failed.' }
} finally {
    Pop-Location
}
