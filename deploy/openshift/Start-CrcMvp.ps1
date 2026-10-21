# Reuses the installed local OKD VM. No setup, rebuild, memory change or service stop.
param([switch]$CheckOnly)
$ErrorActionPreference='Stop'
$taskCrc=(Get-Command crc -ErrorAction Stop).Source
$taskStateText=& $taskCrc status -o json
if($LASTEXITCODE -ne 0){throw 'CRC status unavailable. Complete the documented installation before launching the MVP.'}
$taskState=$taskStateText|ConvertFrom-Json
if($taskState.preset -ne 'okd'){throw 'The MVP launcher only accepts the installed OKD preset.'}
if($taskState.crcStatus -eq 'Running' -and -not $CheckOnly){
    if($taskState.openshiftStatus -ne 'Running'){throw 'CRC is running but the cluster is not ready. Wait and retry.'}
    Write-Output 'Existing OKD cluster ready. Configuration and data retained.'
    return
}
. (Join-Path $PSScriptRoot 'Get-CrcSessionReadiness.ps1')
$taskSession=Get-CrcSessionReadiness
if(-not $taskSession.Ready){throw $taskSession.NextAction}
$taskFallback=$false
try {Get-ScheduledTask -TaskName crcDaemon -ErrorAction Stop|Out-Null}
catch {
    if($_.FullyQualifiedErrorId -notmatch '^HRESULT 0x80041318,Get-ScheduledTask$'){throw} # Known CIM schema failure only.
    $taskFallback=$true
}
if($taskFallback){
    # Native Task Scheduler inspection replaces only the broken CIM task query.
    $taskScheduler=New-Object -ComObject Schedule.Service
    $taskScheduler.Connect()
    $taskRegistered=$taskScheduler.GetFolder('\').GetTask('crcDaemon')
    $taskDefinition=$taskRegistered.Definition
    $taskExpectedLauncher=Join-Path (Split-Path $taskCrc) 'crc-background-launcher.exe'
    $taskVersionJson=& $taskCrc version -o json
    if($LASTEXITCODE -ne 0){throw 'CRC executable version could not be verified.'}
    $taskVersion=($taskVersionJson|ConvertFrom-Json).version -replace '^v',''
    $taskSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $taskPrincipal=$taskDefinition.Principal.UserId
    if($taskPrincipal -notmatch '^S-1-'){
        $taskPrincipal=([Security.Principal.NTAccount]::new($taskPrincipal)).Translate([Security.Principal.SecurityIdentifier]).Value
    }
    if(-not $taskRegistered.Enabled -or $taskRegistered.State -ne 4 -or
       $taskDefinition.Actions.Count -ne 1 -or
       $taskDefinition.Actions.Item(1).Path -ne $taskExpectedLauncher -or
       $taskDefinition.Actions.Item(1).Arguments -ne ('"'+$taskCrc+'" daemon') -or
       $taskPrincipal -ne $taskSid -or
       $taskDefinition.RegistrationInfo.Version -ne $taskVersion){
        throw 'Native CRC daemon identity/state check failed. No preflight bypass applied.'
    }
    Write-Output 'CRC daemon checked through native Task Scheduler: owned executable, account, version and running state match.'
}
if($CheckOnly){Write-Output 'Startup prerequisite checks passed. VM was not started.';return}
$taskSettingsPath=Join-Path $env:USERPROFILE '.crc/crc.json'
$taskSettings=Get-Content -LiteralPath $taskSettingsPath -Raw|ConvertFrom-Json
$taskKeys=@('skip-check-daemon-task-install','skip-check-daemon-task-running')
try{
    if($taskFallback){foreach($taskKey in $taskKeys){& $taskCrc config set $taskKey true|Out-Null;if($LASTEXITCODE -ne 0){throw 'Could not enable the scoped task-query compatibility fallback.'}}}
    & $taskCrc start
    if($LASTEXITCODE -ne 0){throw 'CRC start failed. Runtime acceptance was not established.'}
}finally{
    if($taskFallback){
        $taskRestoreFailures=@()
        foreach($taskKey in $taskKeys){
            try{
                $taskOld=$taskSettings.PSObject.Properties[$taskKey]
                if($null -ne $taskOld){& $taskCrc config set $taskKey $taskOld.Value|Out-Null}
                else{& $taskCrc config unset $taskKey|Out-Null}
                if($LASTEXITCODE -ne 0){$taskRestoreFailures+=$taskKey}
            }catch{$taskRestoreFailures+=$taskKey}
        }
        if($taskRestoreFailures.Count){throw ('Restoring original CRC check configuration failed: '+($taskRestoreFailures -join ', '))}
    }
}
