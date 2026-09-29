param(
    [string]$CarlaRoot = $env:CARLA_ROOT,
    [int]$Port = 2000
)

if (-not $CarlaRoot) {
    throw 'Set CARLA_ROOT or pass -CarlaRoot to the CARLA 0.9.16 package directory.'
}
$launcher = Join-Path $CarlaRoot 'CarlaUE4.exe'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "CARLA launcher not found: $launcher"
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "Port $Port already has a listener. Stop the previous CARLA server before changing quality."
}

$process = Start-Process -FilePath $launcher -WorkingDirectory $CarlaRoot `
    -ArgumentList '-RenderOffScreen', '-quality-level=Epic', '-nosound', "-carla-rpc-port=$Port" `
    -WindowStyle Hidden -PassThru
Write-Output "Started CARLA Epic off-screen launcher PID=$($process.Id) port=$Port"
