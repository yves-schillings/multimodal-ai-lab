# Read-only check of the current process token, not just account membership.
# A running desktop app can retain its old token after the CRC installer adds groups.
function Get-CrcSessionReadiness {
    $taskIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $taskTokenSids = @($taskIdentity.Groups | ForEach-Object { $_.Value })
    $taskCrcGroup = Get-LocalGroup -Name 'crc-users' -ErrorAction SilentlyContinue
    $taskCrcMember = [bool]($taskCrcGroup -and $taskTokenSids -contains $taskCrcGroup.SID.Value)
    $taskHyperVMember = $taskTokenSids -contains 'S-1-5-32-578'
    [pscustomobject]@{
        CrcGroupExists = [bool]$taskCrcGroup
        CrcGroupInCurrentToken = $taskCrcMember
        HyperVGroupInCurrentToken = $taskHyperVMember
        Ready = $taskCrcMember -and $taskHyperVMember
        NextAction = if ($taskCrcMember -and $taskHyperVMember) {
            'Current process has the required CRC and Hyper-V group memberships.'
        } elseif ($taskCrcGroup) {
            'Verify account membership in crc-users and Hyper-V Administrators, then sign out of Windows and sign back in. Reopen the terminal/app before retrying. Do not bypass helper-service access checks.'
        } else {
            'Complete CRC installation/setup, then sign out of Windows and sign back in after the required groups are assigned.'
        }
    }
}
