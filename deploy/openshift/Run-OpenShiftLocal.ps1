# Starts OpenShift Local (CRC), logs oc in, and runs the application acceptance script.
# Prerequisites done by the user: CRC installed with the MSI (administrator), Windows session reopened,
# Red Hat pull secret saved outside every repository. This script never reads or prints the secret.
#
#   .\deploy\openshift\Run-OpenShiftLocal.ps1                       # full run
#   .\deploy\openshift\Run-OpenShiftLocal.ps1 -SkipStart            # cluster already running
#   .\deploy\openshift\Run-OpenShiftLocal.ps1 -Command verify -Phase after
param(
    # 'okd' = OKD community distribution bundled with CRC: no Red Hat account and no pull secret needed.
    # 'openshift' = Red Hat OpenShift Container Platform bundle: needs the pull secret from a Red Hat account.
    [ValidateSet('okd', 'openshift')]
    [string]$Preset = 'okd',
    [string]$PullSecret = 'E:\_SoftEng\yves-schillings\openshift-local\pull-secret.json',
    [int]$Cpus = 6,
    [int]$MemoryMiB = 16384,
    [int]$DiskGiB = 60,
    [switch]$SkipStart,
    [switch]$UseKubeadmin,
    [ValidateSet('all', 'preflight', 'build', 'deploy', 'verify', 'restart', 'report')]
    [string]$Command = 'all',
    [ValidateSet('before', 'after')]
    [string]$Phase = 'before'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo

if (-not (Get-Command crc -ErrorAction SilentlyContinue)) {
    throw 'crc is not on PATH. Install OpenShift Local with the MSI as administrator, then reopen the Windows session.'
}
if (-not $SkipStart) {
    if ($Preset -eq 'openshift' -and -not (Test-Path $PullSecret)) { throw "Pull secret not found at $PullSecret (path only; content is never read here)." }
    crc config set preset $Preset | Out-Null
    crc config set cpus $Cpus | Out-Null
    crc config set memory $MemoryMiB | Out-Null
    crc config set disk-size $DiskGiB | Out-Null
    crc config set consent-telemetry no | Out-Null
    crc setup
    if ($LASTEXITCODE -ne 0) { throw 'crc setup failed (group membership or Hyper-V); see the guide.' }
    if ($Preset -eq 'openshift') { crc start -p $PullSecret } else { crc start }
    if ($LASTEXITCODE -ne 0) { throw 'crc start failed.' }
}

# oc on PATH for this process only.
& crc oc-env | Invoke-Expression

if ($UseKubeadmin) {
    # The kubeadmin password is parsed from crc output and passed to oc login without being echoed.
    $line = (crc console --credentials | Select-String 'kubeadmin').Line
    if ($line -match "-p\s+(\S+)\s+(https://\S+)") {
        & oc login -u kubeadmin -p $Matches[1] $Matches[2] --insecure-skip-tls-verify=true | Out-Null
        Remove-Variable line, Matches -ErrorAction SilentlyContinue
    } else { throw 'Could not obtain kubeadmin credentials from crc.' }
} else {
    & oc login -u developer -p developer https://api.crc.testing:6443 --insecure-skip-tls-verify=true | Out-Null
}
Write-Host ("Logged in as " + (oc whoami))

$python = if (Test-Path "$repo\.venv\Scripts\python.exe") { "$repo\.venv\Scripts\python.exe" } else { 'python' }
if ($Command -eq 'verify') {
    & $python scripts\openshift_local_acceptance.py verify --phase $Phase
} else {
    & $python scripts\openshift_local_acceptance.py $Command
}
exit $LASTEXITCODE
