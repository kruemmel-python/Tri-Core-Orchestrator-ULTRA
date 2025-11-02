# tests\run_tests.ps1
param(
  [string]$DllPath = "$PSScriptRoot\..\build\CipherCore_OpenCl.dll",
  [int]$GpuIndex = 0,
  [switch]$Quiet
)

$ErrorActionPreference = "Stop"

# Projektwurzel
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Env setzen
$env:CIPHERCORE_DLL = (Resolve-Path $DllPath)
$env:CIPHERCORE_GPU = "$GpuIndex"

# Streamlit still schalten
$env:STREAMLIT_HEADLESS = "true"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:STREAMLIT_LOG_LEVEL = "error"

# Python/pytest-Warnings minimieren
$env:PYTHONWARNINGS = "ignore"

Write-Host "Using CIPHERCORE_DLL=$($env:CIPHERCORE_DLL)"
Write-Host "Using CIPHERCORE_GPU=$($env:CIPHERCORE_GPU)"

# Pytest-Args
$pytestArgs = @("tests", ".","-ra")
if ($Quiet) { $pytestArgs += "-q" }

# Run
python -m pytest @pytestArgs
