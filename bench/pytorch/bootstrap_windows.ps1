# Signed CPython 3.12 plus the CUDA 12.6 torch wheel.
# The uv-managed interpreter is unsigned. Windows Application Control blocks
# its _ctypes.pyd, so this script uses the python.org build.
$ErrorActionPreference = "Stop"
$env:UV_HTTP_TIMEOUT = "600"
$uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
if (-not (Test-Path $uv)) {
    irm https://astral.sh/uv/install.ps1 | iex
}
if (-not (Test-Path $uv)) {
    throw "uv was not installed at $uv"
}

function Find-SignedPython {
    $root = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (-not (Test-Path $root)) {
        return $null
    }
    $found = Get-ChildItem -Path $root -Recurse -Filter python.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq "python.exe" -and $_.Directory.Name -eq "Python312" } |
        Select-Object -First 1
    if ($found) {
        return $found.FullName
    }
    return $null
}

$py = Find-SignedPython
if (-not $py) {
    $winget = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\winget.exe"
    if (-not (Test-Path $winget)) {
        throw "winget is missing and no signed CPython 3.12 is installed"
    }
    & $winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
    $py = Find-SignedPython
}
if (-not $py) {
    throw "signed CPython 3.12 was not found. The uv-managed build is blocked by Application Control."
}
& $py -c "import ctypes"

$root = Join-Path $env:USERPROFILE "ppdl-w-bench"
New-Item -ItemType Directory -Force -Path $root | Out-Null
$venv = Join-Path $root "venv"
$vpy = Join-Path $venv "Scripts\python.exe"
$ready = $false
if (Test-Path $vpy) {
    & $vpy -c "import torch; raise SystemExit(0 if torch.__version__.split('+')[0] == '2.14.1' and torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -eq 0) {
        $ready = $true
    }
}
if (-not $ready) {
    if (Test-Path $venv) {
        Remove-Item -Recurse -Force $venv
    }
    & $uv venv --python $py $venv
    & $uv pip install --python $vpy --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.1"
}
& $vpy -c "import torch; print(torch.__version__); print('cuda', torch.version.cuda)"
