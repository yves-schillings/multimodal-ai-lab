# Read-only readiness check. Does not install software, change Hyper-V or start a VM.
$ErrorActionPreference = 'Stop'
$taskOs = Get-CimInstance Win32_OperatingSystem
$taskComputer = Get-CimInstance Win32_ComputerSystem
$taskCpu = Get-CimInstance Win32_Processor
$taskCrc = Get-Command crc -ErrorAction SilentlyContinue
$taskOc = Get-Command oc -ErrorAction SilentlyContinue
[pscustomobject]@{
    Windows = $taskOs.Caption
    LogicalCPUs = $taskCpu.NumberOfLogicalProcessors
    MemoryGiB = [math]::Round($taskComputer.TotalPhysicalMemory / 1GB, 1)
    FreeMemoryGiB = [math]::Round($taskOs.FreePhysicalMemory / 1MB, 1)
    CFreeGiB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
    HypervisorPresent = $taskComputer.HypervisorPresent
    CRCInstalled = [bool]$taskCrc
    OCInstalled = [bool]$taskOc
    Purpose = 'OpenShift Local application testing; not full OpenShift AI certification'
} | ConvertTo-Json
if (-not $taskCrc) {
    Write-Host 'Next prerequisite: install OpenShift Local from the official Red Hat download page and obtain a pull secret through your Red Hat account.'
}
