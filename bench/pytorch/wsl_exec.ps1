# Run one bash command inside the WSL distro named in wsl-distro.txt.
# PPDL_CMD is the UTF-8 command, base64-encoded, so the Windows shell
# does not expand it.
$ErrorActionPreference = "Stop"
if (-not $env:PPDL_CMD) {
    throw "PPDL_CMD is empty"
}
$cmd = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:PPDL_CMD))
$distroFile = Join-Path $env:USERPROFILE "ppdl-w-bench\wsl-distro.txt"
if (-not (Test-Path $distroFile)) {
    throw "missing $distroFile"
}
$distro = (Get-Content -Raw $distroFile).Trim()
& wsl.exe -d $distro -e bash -lc $cmd
exit $LASTEXITCODE
