$ErrorActionPreference = 'Stop'
$labRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $labRoot
$labPython = Join-Path $labRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $labPython)) {
    $labBootstrap = Get-Command python -ErrorAction SilentlyContinue
    if ($labBootstrap) { & $labBootstrap.Source -m venv (Join-Path $labRoot '.venv') }
    else {
        $labBundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (-not (Test-Path -LiteralPath $labBundled)) { throw 'Install Python 3.12 or later, then start again.' }
        & $labBundled -m venv (Join-Path $labRoot '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $labPython -c 'import fastapi, uvicorn, sklearn, mlflow, multipart' 2>$null
if ($LASTEXITCODE -ne 0) {
    & $labPython -m pip install -r (Join-Path $labRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
try {
    $labExisting = Invoke-RestMethod -Uri 'http://127.0.0.1:8770/api/status' -TimeoutSec 2
    if ($labExisting.name -eq 'Multimodal AI Lab') { Start-Process 'http://127.0.0.1:8770'; exit 0 }
    throw 'Port 8770 belongs to another service.'
} catch {
    if ($_.Exception.Message -eq 'Port 8770 belongs to another service.') { throw }
}
$env:MLFLOW_DISABLE_AGENT_HINT = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$labLogs = Join-Path $labRoot 'data\logs'
New-Item -ItemType Directory -Path $labLogs -Force | Out-Null
$labProcess = Start-Process -FilePath $labPython -ArgumentList @('-m','uvicorn','app:app','--host','127.0.0.1','--port','8770','--no-access-log') -WorkingDirectory $labRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $labLogs 'server-out.log') -RedirectStandardError (Join-Path $labLogs 'server-error.log') -PassThru
$labProcess.Id | Set-Content -LiteralPath (Join-Path $labLogs 'server.pid')
for ($labAttempt=0; $labAttempt -lt 30; $labAttempt++) {
    Start-Sleep -Milliseconds 500
    try { $labReady = Invoke-RestMethod -Uri 'http://127.0.0.1:8770/health/ready' -TimeoutSec 1; if ($labReady.status -eq 'ready') { Start-Process 'http://127.0.0.1:8770'; exit 0 } } catch {}
    if ($labProcess.HasExited) { throw ('Server failed. Inspect ' + (Join-Path $labLogs 'server-error.log')) }
}
throw 'Server did not become ready. Inspect data/logs/server-error.log.'
