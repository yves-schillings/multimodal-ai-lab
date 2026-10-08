# Read-only readiness check. Does not install software, change Hyper-V or start a VM.
$ErrorActionPreference = 'Stop'
$taskOs = Get-CimInstance Win32_OperatingSystem
$taskComputer = Get-CimInstance Win32_ComputerSystem
$taskCpu = Get-CimInstance Win32_Processor
$taskCrc = Get-Command crc -ErrorAction SilentlyContinue
$taskOc = Get-Command oc -ErrorAction SilentlyContinue
. (Join-Path $PSScriptRoot 'Get-CrcSessionReadiness.ps1')
$taskSession = Get-CrcSessionReadiness
[pscustomobject]@{
    Windows = $taskOs.Caption
    LogicalCPUs = $taskCpu.NumberOfLogicalProcessors
    MemoryGiB = [math]::Round($taskComputer.TotalPhysicalMemory / 1GB, 1)
    FreeMemoryGiB = [math]::Round($taskOs.FreePhysicalMemory / 1MB, 1)
    CFreeGiB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
    HypervisorPresent = $taskComputer.HypervisorPresent
    CRCInstalled = [bool]$taskCrc
    OCInstalled = [bool]$taskOc
    CrcGroupInCurrentToken = $taskSession.CrcGroupInCurrentToken
    HyperVGroupInCurrentToken = $taskSession.HyperVGroupInCurrentToken
    SessionReady = $taskSession.Ready
    SessionNextAction = $taskSession.NextAction
    Purpose = 'OpenShift Local application testing; not full OpenShift AI certification'
} | ConvertTo-Json
if (-not $taskCrc) {
    Write-Host 'Next prerequisite: install CRC. Its OKD preset needs no Red Hat pull secret; the openshift preset requires one.'
}
