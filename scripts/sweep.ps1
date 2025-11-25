# Quick sweep helper for matmul_test.exe + plotting
# Uses paths relative to repo root.

param(
    [switch]$ProfileShared,    # run ncu on shared_mem_matmul_kernel
    [switch]$ProfileTensorCore # run ncu on tensor_core_kernel
)

$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "build\\bin\\matmul_test.exe"
$plot = Join-Path $root "scripts\\plot.py"

if (-Not (Test-Path $exe)) {
    Write-Host "Executable not found at $exe. Build first." -ForegroundColor Red
    exit 1
}

Push-Location $root

Write-Host "Running matmul_test.exe..." -ForegroundColor Cyan
& $exe

if ($LASTEXITCODE -ne 0) {
    Write-Host "matmul_test.exe failed with code $LASTEXITCODE" -ForegroundColor Red
    Pop-Location
    exit $LASTEXITCODE
}

# Generate plot if Python is available
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "Generating plot from results.csv..." -ForegroundColor Cyan
    & python $plot
}
else {
    Write-Host "Python not found in PATH; skip plotting." -ForegroundColor Yellow
}

# Optional profiling with Nsight Compute (adjust paths if needed)
if ($ProfileShared) {
    Write-Host "Profiling shared_mem_matmul_kernel with ncu..." -ForegroundColor Cyan
    ncu --set full --kernel-name "shared_mem_matmul_kernel" $exe
}

if ($ProfileTensorCore) {
    Write-Host "Profiling tensor_core_kernel with ncu..." -ForegroundColor Cyan
    ncu --set full --kernel-name "tensor_core_kernel" $exe
}

Pop-Location
