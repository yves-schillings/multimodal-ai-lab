# Local MVP operation and completion

The minimum viable product (MVP) is a fictional-data demonstrator on the existing local OKD cluster. OKD is the community Kubernetes distribution upstream of OpenShift Container Platform. Its name has no official letter-by-letter expansion.

The accepted scope includes document extraction and scanned-PDF optical character recognition, local speech transcription, human corrections and approval, classifier training and release, local authenticated accounts, case permissions, vector retrieval, cited generation, two isolated projects and recurring classifier evaluation. BGE-M3 produces text vectors for finding relevant passages. Qwen3:4b generates answers from those passages. Citation and revision checks and unsupported-question abstention accompany generation. These controls do not establish production answer quality.

## Open the installed MVP

Double-click `Start-MVP.cmd` in the repository root. Open the loopback URL printed in its window, then sign in with a private local account. Keep the window open while using the application. Ctrl+C closes that connection, without stopping the cluster or removing data.

For an alternative port:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File deploy\openshift\Open-MVP.ps1 -Port 18770
```

The launcher reuses the installed cluster and existing authenticated OpenShift client session. It does not run installation, rebuild images, reduce memory or redeploy the application. If the installed OKD VM is stopped, it invokes CRC start with the existing settings. Windows installation, group membership and operator authentication are prerequisites, not actions silently performed by this launcher.

The installed configuration is six virtual CPUs and 24 GiB of VM memory. A connection-only check is also available:

```powershell
deploy\openshift\Run-OpenShiftLocal.ps1
deploy\openshift\Start-CrcMvp.ps1 -CheckOnly
```

On this Windows host the ScheduledTasks CIM query fails with HRESULT 0x80041318. For this exact failure only, the startup helper checks the native Task Scheduler daemon's executable, arguments, account, version and running state. After those checks, a stopped-VM start can temporarily bypass the two broken daemon-task queries. It restores their prior configuration in a `finally` block. Other prerequisite errors remain blocking. The native prerequisite check and running-cluster launch were tested with Windows PowerShell 5.1. A new cold start was not performed during this closure exercise, because the accepted cluster and other work were retained.

`Run-OpenShiftLocal.ps1 -Command all` remains an explicit deployment operation. It is not needed to open the existing MVP. `Start.cmd` starts the separate desktop demonstration with simulated actors, rather than the authenticated OKD deployment.

## Renew an isolated project explicitly

A Controlled Project Environment (CPE) approval remains limited to 24 hours. It is not renewed automatically. The operator records a new approval reference and renews only the existing scope. Projects `a` and `b` have separate credentials and data.

Use the existing private operator kubeconfig and records directory outside the checkout. Substitute a genuine local owner approval reference:

```powershell
.\.venv\Scripts\python.exe scripts\local_cpe.py renew --project a --admin-kubeconfig <private-kubeconfig> --private-dir <existing-private-records> --approval-reference <new-owner-approval-reference> --hours 24
.\.venv\Scripts\python.exe scripts\cluster_cpe_acceptance.py --project a --admin-kubeconfig <private-kubeconfig> --private-dir <existing-private-records>
.\.venv\Scripts\python.exe scripts\local_cpe.py activate --project a --admin-kubeconfig <private-kubeconfig> --private-dir <existing-private-records>
```

Renewal revokes the gate, records the previous profile privately, changes only approval and expiry, and leaves the project provisioned but inactive. The update uses Kubernetes optimistic concurrency, so a concurrent profile edit cannot be overwritten. A conflict leaves the project revoked. Existing verification cannot activate the renewed profile. Fresh passing evidence must match the new profile and unchanged controls before activation.

Hours must be an integer from 1 to 24. Reusing the previous approval reference is rejected. Renewal does not change models, image, network policy, quotas, credentials or storage. It does not approve transcription, training or other additional CPE services.

## Rehearse a usable recovery without overwriting live data

The existing backup exercise restores application files and database copies privately. Run the additional serving rehearsal against its trusted `restored-application` directory:

```powershell
.\.venv\Scripts\python.exe scripts\mvp_restore_acceptance.py --backup-dir <private-restored-application> --credential-config <private-hashed-account-config> --credentials <private-acceptance-passwords> --private-dir <private-recovery-evidence>
```

The script verifies the backup manifest, creates a fresh private copy, checks SQLite integrity and absence of pending jobs, and verifies the active classifier checksum against its recorded metadata. It starts the actual application on an unused loopback port, signs in, reads an existing case, checks another owner's denial and requests a real classifier prediction. It terminates only its own test server and retains the private report and copy.

Supply only trusted local model artifacts. The rehearsal never promotes a model, rewrites live volumes or launches training. It verifies one saved case and the existing classifier, with local authentication. It does not establish restored RAG serving, restored MLflow service, off-site backup, complete cluster reconstruction or disaster recovery.

## Completion boundary

The small fictional acceptance exercises demonstrate the MVP. They do not require a larger representative corpus, distributed workers, GPU passthrough, full OpenShift AI, organizational identity integration or migration of the case store to PostgreSQL/object storage. Those are separate extensions.

The experimental small language-model adapter remains unpromoted: its independent abstention and malicious-source checks are insufficient. The MVP continues using the accepted pretrained local generation model. The original data volumes and deployment image remain in place.

Private execution records are retained outside Git. Office and website revisions require their separate editorial approval. Local implementation changes do not imply a new GitHub release or passed remote CI.
