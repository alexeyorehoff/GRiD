param(
    [ValidateSet('all', 'cpu', 'cuda', 'wrappers')][string]$Lane = 'all',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$PytestArgs
)
$gridRoot = Split-Path -Parent $PSScriptRoot
Push-Location $gridRoot
try {
    . "$gridRoot/install/activate_windows.ps1"
    $gridTests = @()
    if ($Lane -eq 'cpu') {
        $gridTests += @(Get-ChildItem -LiteralPath test -Filter 'test_*.py' -File | Sort-Object Name | ForEach-Object FullName)
        $gridTests += @(Get-ChildItem -LiteralPath test/benchmarks -Filter 'test_*.py' -File | Sort-Object Name | ForEach-Object FullName)
        $gridTests += @('test/cuda_equivalents/test_pin_only_default.py',
            'test/cuda_equivalents/test_cuda_matmul_blockwrap_regression.py',
            'bindings/tests/test_mujoco_view.py', 'bindings/tests/test_mujoco_transforms.py',
            'bindings/tests/test_register_forwarding.py', 'docs/test_doc_examples.py')
    } elseif ($Lane -eq 'cuda') {
        $gridTests = @('test/cuda_equivalents')
    } elseif ($Lane -eq 'wrappers') {
        $gridTests = @('test/python_wrappers')
    }
    & .venv/Scripts/python.exe -m pytest -q @gridTests @PytestArgs
    $gridTestExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $gridTestExit
