# Container finding disposition — 7 October 2026

This review corrects the distinction between a vulnerability report and the automation gate. It does not authorise deployment or accept residual security risk.

The retained [Trivy report](../2026-10-07-closing/app-audit.json), generated on 7 October 2026 at 16:58 UTC, contains **76 HIGH and 1 CRITICAL package findings**. All 77 are Debian-package findings. That report supplies **no fixed version** for any of them. Its SHA-256 is `756c7ce0fa992b2164684303ef9b8526795d83f525e1cfbd7ddd5301d0d1e195`. Finding counts are package-level records, not necessarily distinct vulnerabilities.

The [derived summary](finding-summary.json) retains each identifier, package, installed version, severity, vendor-reference URL and fix state. Every record remains **not approved** for risk acceptance. The critical record is `CVE-2026-6653` in `libxml2`; no reachability assessment or mitigating-control claim is inferred from a missing fixed version.

## Automation policy

The original broad HIGH/CRITICAL scan failed. The current workflow retains the full report, then blocks on HIGH/CRITICAL findings **with an available fixed version**. Under that policy, this retained report contains zero blocking findings. This is a calculation against an existing report, **not a new image scan, a green GitHub run or security acceptance**. A fresh scan after publication can report different findings and available fixes.

The workflow change avoids an impossible remediation gate when the scanner provides no fix. It also leaves a separate residual-risk decision; passing the automation threshold cannot establish that an image is safe for sensitive data or shared exposure.

## Required follow-up before shared exposure

| Owner role | Required work | Acceptance evidence |
| --- | --- | --- |
| Platform engineer | Rebuild with the approved base-image digest and current package updates; rescan that exact image | Image digest, dated complete report and fixable-only gate result |
| Application and security engineers | Assess the critical `libxml2` record and document the reachable parsing paths; do not assume that OCR bounds remove package vulnerabilities | Reachability analysis with test evidence and explicit limits |
| Security owner | Review residual HIGH/CRITICAL records, compensating controls, exposure, expiry and retest conditions | Named approval or remediation decision; none is granted in this package |
| Platform owner | Keep the lab restricted to synthetic data and approved local access pending acceptance | Binding, identity, network and data-scope checks |

Reproduce the summary from the repository root:

```bash
python scripts/review_container_findings.py \
  docs/evidence/2026-10-07-closing/app-audit.json \
  --output docs/evidence/2026-10-07-security-review/finding-summary.json
```

The original evidence package and manifest are preserved unchanged. This disposition supplements them; it does not rewrite a historic result.
