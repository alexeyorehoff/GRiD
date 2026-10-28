param([switch]$SkipTorch, [switch]$SkipOracle)
$ErrorActionPreference = 'Stop'
$gridRoot = Split-Path -Parent $PSScriptRoot
Push-Location $gridRoot
try {
    & "$PSScriptRoot/base_install.ps1"
    . "$PSScriptRoot/activate_windows.ps1"
    & uv pip install --python .venv/Scripts/python.exe -r install/requirements-windows-dev.txt
    if ($LASTEXITCODE) { throw 'Test dependency installation failed.' }
    if (-not $SkipTorch) {
        & uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE) { throw 'PyTorch installation failed.' }
    }
    if (-not $SkipOracle) {
        $gridMamba = Join-Path $gridRoot '.cache/tools/Library/bin/micromamba.exe'
        if (-not (Test-Path -LiteralPath $gridMamba)) {
            New-Item -ItemType Directory -Path .cache/tools -Force | Out-Null
            Invoke-WebRequest -UseBasicParsing -Uri 'https://micro.mamba.pm/api/micromamba/win-64/latest' -OutFile .cache/tools/micromamba.tar.bz2
            & tar -xf .cache/tools/micromamba.tar.bz2 -C .cache/tools Library/bin/micromamba.exe
            if ($LASTEXITCODE) { throw 'micromamba extraction failed.' }
        }
        if (-not (Test-Path -LiteralPath '.cache/pinocchio/conda-meta/history')) {
            & $gridMamba create -y -p .cache/pinocchio -c conda-forge 'python=3.12' 'pinocchio=3.9' 'numpy=2.3.5'
            if ($LASTEXITCODE) { throw 'Pinocchio installation failed.' }
        }
        & uv pip install --python .venv/Scripts/python.exe 'numpy==2.3.5'
        if ($LASTEXITCODE) { throw 'NumPy installation failed.' }
        # Expose the native conda oracle to the project venv, and retain its DLL
        # directory handle for Python 3.8+'s explicit Windows loader policy.
        @'
from pathlib import Path
import sys
prefix = Path('.cache/pinocchio').resolve()
site = Path('.venv/Lib/site-packages')
if sys.version_info[:2] != (3, 12):
    raise RuntimeError('The Windows oracle recipe requires Python 3.12')
text = str(prefix / 'Lib/site-packages') + '\n'
text += 'import os; os._grid_oracle_dlls = os.add_dll_directory(' + repr(str(prefix / 'Library/bin')) + ')\n'
(site / 'grid_windows_oracle.pth').write_text(text, encoding='utf-8')
'@ | & .venv/Scripts/python.exe -
        if ($LASTEXITCODE) { throw 'Oracle configuration failed.' }
        & .venv/Scripts/python.exe -c "import pinocchio; print('Pinocchio', pinocchio.__version__)"
        if ($LASTEXITCODE) { throw 'Pinocchio import failed.' }
    }
} finally {
    Pop-Location
}
