# Dot-source from PowerShell: . ./install/activate_windows.ps1
# Configure the native x64 MSVC host compiler, CUDA Toolkit and project venv.
$gridPreviousErrorPreference = $ErrorActionPreference
$ErrorActionPreference = 'Stop'
try {
$gridRoot = Split-Path -Parent $PSScriptRoot
$gridVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    if (-not (Test-Path -LiteralPath $gridVswhere)) {
        throw 'Install Visual Studio 2022 with Desktop development with C++.'
    }
    $gridVs = & $gridVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $gridVs) { throw 'No Visual Studio C++ toolchain found.' }
    Import-Module (Join-Path $gridVs 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $gridVs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
}
$gridNvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
if (-not $gridNvcc) { throw 'Install the CUDA Toolkit and add its bin directory to PATH.' }
$env:CUDA_PATH = Split-Path -Parent (Split-Path -Parent $gridNvcc.Source)
$env:PYTHONUTF8 = '1'
if (-not $env:MPLBACKEND) { $env:MPLBACKEND = 'Agg' }
$gridGit = Get-Command git.exe -ErrorAction SilentlyContinue
if ($gridGit) {
    $gridGitBin = Join-Path (Split-Path -Parent (Split-Path -Parent $gridGit.Source)) 'bin'
    if (Test-Path -LiteralPath (Join-Path $gridGitBin 'bash.exe')) {
        # Workflow unit tests use Git for Windows' shell, not the WSL alias in
        # System32. CUDA and Python are still native Windows processes.
        $env:PATH = "$gridGitBin;$env:PATH"
    }
}
$gridActivate = Join-Path $gridRoot '.venv/Scripts/Activate.ps1'
if (Test-Path -LiteralPath $gridActivate) { . $gridActivate }
} finally {
    $ErrorActionPreference = $gridPreviousErrorPreference
}
