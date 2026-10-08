# Local OpenShift preparation

## Three different runtime levels

1. **Docker Compose:** executable local application, OCR, MLflow and PostgreSQL. This does not install OpenShift AI.
2. **OpenShift Local (CRC):** a local single-node OpenShift development cluster. Use it to verify images, arbitrary user IDs, deployments, storage and policies.
3. **Red Hat OpenShift AI:** an Operator and platform components installed on an appropriately sized, entitled OpenShift cluster. It is not a standalone Docker image. Model serving and pipelines need their own configuration and acceptance.

The development host inspected on 7 October 2026 has Windows 11 Pro, an Intel i9-9900K,
16 logical CPUs and 64 GiB RAM. Around 27 GiB RAM and 245 GiB on C: were available.
The hypervisor is active. CRC and `oc` were not installed.

This host is a candidate for OpenShift Local application tests. It does **not** meet the
OpenShift AI 3.5 published single-node requirement of 32 CPUs and 128 GiB RAM. Do not
describe a smaller experimental installation as a supported OpenShift AI deployment.

## Preparation sequence

### Current host check: 9 October 2026

CRC 2.64.0 is now installed and configured with the OKD community preset. The
OKD preset does not require a Red Hat account or pull secret. The `openshift`
preset is a separate option and requires the vendor's pull secret.

After the user restarted Windows, the desktop process token includes both
`crc-users` and Hyper-V Administrators. The CRC admin helper is accessible.
The launcher checks these token memberships before attempting setup.

The owned `crcDaemon` scheduled task is present and running, but its inspection
through PowerShell/CIM still fails. Its executable, version, user and running
state were independently checked with `schtasks` and process inspection.
The two daemon-task checks were temporarily skipped for startup, with a
`finally` block restoring the configuration. Other preflight checks remain enabled.

Windows reserves TCP port 80. CRC uses its supported `ingress-http-port` setting
on available port 8080; HTTP application routes must include `:8080`.
HTTPS retains port 443. The CRC VM has started with 6 CPUs, 16 GiB RAM and a
60 GiB disk. The cluster completed initialization and all operators became stable.
The application deployment passed 27 workflow checks before pod replacement,
8 persistence checks afterwards, and all 9 network paths in both phases.
Containers use the arbitrary restricted UID; persistent volumes are unchanged.
See the [successful local run](../../docs/evidence/2026-10-09-openshift-local/attempt-2/README.md).
This accepts the local application deployment, not OpenShift AI or shared hosting.
Earlier host checks below are historical.

Run the read-only check from the repository root:

```powershell
.\deploy\openshift\Check-LocalHost.ps1
```

Install the Windows OpenShift Local distribution on C: from the official Red Hat page.
Sign in to your Red Hat account and obtain its pull secret. Keep that secret outside this
repository. The guided installer requires approval for host changes. Run `crc setup` and
`crc start` from the normal user account, not an administrator shell; allow CRC's required
elevation prompts. The account must be able to elevate.
Avoid simultaneously allocating all remaining RAM to CRC and Docker Desktop.

A read-only host check on 8 October 2026 still found no `crc`, `oc` or user `.crc`
directory. No running OpenShift Local cluster was verified. Manifests and this guide
are preparation, not deployment acceptance.

For application testing, start with 8 vCPUs and 16 GiB assigned to CRC. These are lab
planning values, not OpenShift AI sizing or a guarantee of sufficient workload capacity.
After installation, use the exact commands documented for the installed CRC version:

```powershell
crc setup
crc config set cpus 8
crc config set memory 16384
crc start --pull-secret '<path-outside-repository>'
crc oc-env --shell powershell | Invoke-Expression
crc console --credentials
```

Authenticate to the local cluster using its displayed credentials; never commit them.
Build/push the application and MLflow images into that cluster's image registry, set their
references in the manifests and create the database secret outside Git. Follow the acceptance
steps in [README.md](README.md). Keep application access through localhost port-forward while
identities are simulated. Record the tested cluster version and image digests.

## OpenShift AI evaluation after local application acceptance

Choose a compatible platform/version and an appropriate Red Hat entitlement or trial,
then follow the vendor's Operator installation procedure. Confirm compute, storage,
identity and model-serving prerequisites first. The Developer Sandbox provides another
evaluation route when local capacity is inadequate; only fictional lab data may be used.
No platform subscription or remote resource has been provisioned. Only the local
CRC/OKD application cluster described above has been provisioned and checked.

## References

- [CRC installation and Windows requirements](https://crc.dev/docs/installing/)
- [CRC commands and pull-secret preparation](https://crc.dev/docs/using/)
- [OpenShift Local download](https://console.redhat.com/openshift/create/local)
- [OpenShift AI 3.5 installation and sizing](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/installing_and_uninstalling_openshift_ai_self-managed/installing-and-deploying-openshift-ai_install)
- [OpenShift AI evaluation options](https://developers.redhat.com/products/red-hat-openshift-ai/download)
