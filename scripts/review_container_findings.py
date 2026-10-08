"""Summarise a retained Trivy report without suppressing findings or accepting risk.

Usage: python scripts/review_container_findings.py REPORT --output SUMMARY
The fixable-only gate is an automated remediation threshold, not deployment approval.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarise(report):
    findings = []
    for result in report.get("Results", []):
        for finding in result.get("Vulnerabilities", []):
            if finding.get("Severity") not in ("HIGH", "CRITICAL"):
                continue
            fixed = finding.get("FixedVersion", "")
            findings.append({
                "target": result.get("Target"),
                "package_type": result.get("Type"),
                "id": finding.get("VulnerabilityID"),
                "package": finding.get("PkgName"),
                "installed_version": finding.get("InstalledVersion"),
                "severity": finding.get("Severity"),
                "fixed_version": fixed,
                "remediation_state": "vendor fix reported" if fixed else "no fix reported by this scan",
                "risk_acceptance": "not approved",
                "reference": finding.get("PrimaryURL"),
            })
    fixable = sum(bool(v["fixed_version"]) for v in findings)
    return {
        "image_architecture": report.get("Metadata", {}).get("ImageConfig", {}).get("architecture"),
        "source_scan_time": report.get("CreatedAt"),
        "artifact_name": report.get("ArtifactName"),
        "artifact_type": report.get("ArtifactType"),
        "severity_counts": dict(Counter(v["severity"] for v in findings)),
        "package_type_counts": dict(Counter(v["package_type"] for v in findings)),
        "high_critical_findings": len(findings),
        "fixable_high_critical_findings": fixable,
        "fixable_only_gate": "FAIL" if fixable else "no fixable finding in this retained report",
        "deployment_approval": "not granted by this calculation",
        "findings": findings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = summarise(json.loads(raw))
    result["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}, indent=2))


if __name__ == "__main__":
    main()
