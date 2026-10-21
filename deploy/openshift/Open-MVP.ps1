# Foreground loopback connection; Ctrl+C closes only this port-forward.
param([ValidateRange(1,65535)][int]$Port=18770)
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot 'Start-CrcMvp.ps1')
$taskEnv=& crc oc-env --shell powershell
if($LASTEXITCODE -ne 0){throw 'Could not locate the CRC client.'}
$taskEnv|Invoke-Expression
& oc whoami|Out-Null
if($LASTEXITCODE -ne 0){throw 'Authenticate the local operator oc session before opening the MVP.'}
$taskServer=& oc whoami --show-server
if(([uri]$taskServer).Host -ne 'api.crc.testing'){throw 'Only the local CRC cluster is permitted.'}
$taskAvailable=& oc get deployment/lab-app -n multimodal-ai-lab -o 'jsonpath={.status.availableReplicas}'
if($LASTEXITCODE -ne 0 -or [int]$taskAvailable -lt 1){throw 'The installed application is not ready.'}
Write-Host "Open http://127.0.0.1:$Port and sign in with your private local lab account."
Write-Host 'Keep this window open. Ctrl+C closes the connection without stopping OKD or deleting data.'
& oc port-forward -n multimodal-ai-lab service/lab-app "${Port}:8770" --address=127.0.0.1
if($LASTEXITCODE -ne 0){throw 'The local MVP connection closed or could not bind its port.'}
