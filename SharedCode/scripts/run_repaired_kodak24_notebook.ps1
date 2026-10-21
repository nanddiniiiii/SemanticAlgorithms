param(
    [string]$ExperimentRoot = "",
    [string]$Contributor = "Nandini.P",
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($ExperimentRoot)) {
    $ExperimentRoot = $projectRoot
}
$python = Join-Path $ExperimentRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Required repaired-notebook environment not found: $python"
}
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $RunId = "repaired-notebook-kodak24-$stamp"
}

$env:SEMCOM_EXPERIMENT_ROOT = $ExperimentRoot
$env:KODAK_BENCHMARK_CONTRIBUTOR = $Contributor
$env:KODAK_BENCHMARK_RUN_ID = $RunId
$env:KODAK_TIMING_REPETITIONS = "1"
$env:KODAK_WARMUP_REPETITIONS = "0"
$env:KODAK_COMPUTE_LPIPS = "1"
$env:KODAK_BENCHMARK_APPEND = "1"

Push-Location $projectRoot
try {
    & $python -m jupyter nbconvert `
        --to notebook `
        --execute `
        --inplace `
        --ExecutePreprocessor.timeout=-1 `
        "Notebooks\01_Kodak24_All_Model_Benchmark.ipynb"
    if ($LASTEXITCODE -ne 0) {
        throw "Notebook execution failed with exit code $LASTEXITCODE"
    }
    Write-Output "RUN_ID=$RunId"
}
finally {
    Pop-Location
}
