"""End-to-end synthetic demonstration against a running Multimodal AI Lab instance.

Phase "before": create a synthetic dossier, upload and review a document, draft, approve,
ask a question, prove that another actor is denied, train, promote and roll back a model.
Phase "after": re-read the saved identifiers and verify that everything survived a restart.

Uses only the standard library. Default target: http://127.0.0.1:8770 (loopback publish).
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Api:
    def __init__(self, base):
        self.base = base.rstrip("/")

    def call(self, method, path, actor=None, body=None, files=None, expect=200):
        url = self.base + path + (("&" if "?" in path else "?") + "actor=" + actor if actor else "")
        data, headers = None, {}
        if files:
            boundary = "----lab" + uuid.uuid4().hex
            parts = []
            for name, value in files.get("fields", {}).items():
                parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
            fname, content = files["file"]
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode() + content + b"\r\n")
            parts.append(f"--{boundary}--\r\n".encode())
            data = b"".join(parts)
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status, payload = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, payload = error.code, error.read()
        try:
            parsed = json.loads(payload.decode() or "null")
        except ValueError:
            parsed = payload.decode(errors="replace")
        if expect is not None and status != expect:
            raise SystemExit(f"{method} {path} expected {expect}, got {status}: {parsed}")
        return status, parsed

    def wait_job(self, job_id, actor):
        for _ in range(600):
            _, job = self.call("GET", f"/api/jobs/{job_id}", actor)
            if job["status"] in {"completed", "failed"}:
                return job
            time.sleep(0.25)
        raise SystemExit(f"Job {job_id} did not finish.")


def check(report, name, condition, detail=""):
    report["checks"].append({"check": name, "passed": bool(condition), "detail": detail})
    print(("PASS " if condition else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not condition:
        report["failed"] += 1


def phase_before(api, report, state_file):
    _, status = api.call("GET", "/api/status")
    check(report, "status reachable", status.get("name") == "Multimodal AI Lab", f"network_mode={status.get('network_mode')} mlflow={status.get('mlflow_tracking')}")
    check(report, "identity is simulated (no authentication claim)", status.get("simulated_identity") is True)
    # 1. synthetic dossier
    _, case = api.call("POST", "/api/cases", "officer-a", {"title": "Synthetic dossier: bicycle evidence inventory"})
    case_id = case["id"]
    check(report, "case created by officer-a", bool(case_id), case_id)
    sample = ROOT / "samples" / "evidence-inventory.txt"
    _, job = api.call("POST", f"/api/cases/{case_id}/upload", "officer-a", files={"fields": {"kind": "document", "language": "en"}, "file": ("evidence-inventory.txt", sample.read_bytes())})
    job = api.wait_job(job["job_id"], "officer-a")
    check(report, "document job completed", job["status"] == "completed", job.get("error") or job["status"])
    _, case = api.call("GET", f"/api/cases/{case_id}", "officer-a")
    document = case["documents"][0]
    check(report, "document extracted with explicit classification status", bool(document["text"]) and document["classification"].get("status") in {"classified", "unavailable"}, f"classification={document['classification'].get('status')} label={document['classification'].get('label')}")
    # 2. review, draft, approval by a separate reviewer
    _, case = api.call("POST", f"/api/cases/{case_id}/documents/{document['id']}/review", "officer-a", {"text": None, "expected_revision": case["revision"]})
    check(report, "source reviewed by officer-a", case["documents"][0]["reviewed"])
    _, case = api.call("POST", f"/api/cases/{case_id}/draft", "officer-a")
    check(report, "draft prepared from reviewed sources only", bool(case["draft"]))
    status_code, _ = api.call("POST", f"/api/cases/{case_id}/approve", "officer-a", {"text": None, "expected_revision": case["revision"]}, expect=None)
    check(report, "officer cannot approve own draft", status_code == 403, f"http {status_code}")
    _, case = api.call("POST", f"/api/cases/{case_id}/approve", "reviewer", {"text": None, "expected_revision": case["revision"]})
    check(report, "reviewer approved the draft", case.get("approved_revision") is not None and case.get("approved_by") == "reviewer", f"approved_revision={case.get('approved_revision')} status={case.get('status')}")
    _, answer = api.call("POST", f"/api/cases/{case_id}/question", "officer-a", {"question": "Which items are in the evidence inventory?"})
    check(report, "question answered from reviewed sources", not answer["abstained"] and answer["sources"], f"mode={answer.get('mode')} sources={len(answer['sources'])}")
    _, answer = api.call("POST", f"/api/cases/{case_id}/question", "officer-a", {"question": "What colour was the getaway car?"})
    check(report, "unsupported question abstains", answer["abstained"])
    # 3. denied access to a forbidden dossier
    status_code, _ = api.call("GET", f"/api/cases/{case_id}", "officer-b", expect=None)
    check(report, "officer-b denied on officer-a dossier", status_code == 403, f"http {status_code}")
    status_code, _ = api.call("POST", f"/api/cases/{case_id}/question", "officer-b", {"question": "Which items?"}, expect=None)
    check(report, "officer-b denied on question", status_code == 403, f"http {status_code}")
    status_code, _ = api.call("GET", "/api/cases/demo-case-b", "officer-a", expect=None)
    check(report, "officer-a denied on demo-case-b", status_code == 403, f"http {status_code}")
    status_code, _ = api.call("GET", "/api/cases", "unknown-user", expect=None)
    check(report, "unknown actor denied", status_code == 403, f"http {status_code}")
    # 4. model gate: train twice, promotion rules, rollback
    versions = []
    for _ in range(2):
        _, job = api.call("POST", "/api/models/train", "officer-a")
        job = api.wait_job(job["job_id"], "officer-a")
        check(report, "training job completed", job["status"] == "completed", job.get("error") or "")
        meta = job["result"]
        versions.append(meta["version"])
        check(report, "metrics recorded", meta["mlflow_status"] in {"recorded_on_server", "recorded_locally"}, f"{meta['mlflow_status']} accuracy={meta['accuracy']} macro_f1={meta['macro_f1']} gate_passed={meta['gate_passed']}")
    status_code, _ = api.call("POST", "/api/models/promote", "officer-a", {"version": versions[0]}, expect=None)
    check(report, "officer cannot promote (reviewer only)", status_code == 403, f"http {status_code}")
    status_code, _ = api.call("POST", "/api/models/promote", "reviewer", {"version": "classifier-does-not-exist"}, expect=None)
    check(report, "unknown candidate rejected", status_code == 422, f"http {status_code}")
    _, state = api.call("POST", "/api/models/promote", "reviewer", {"version": versions[0]})
    check(report, "reviewer promoted candidate 1", state["active_version"] == versions[0])
    _, state = api.call("POST", "/api/models/promote", "reviewer", {"version": versions[1]})
    check(report, "reviewer promoted candidate 2", state["active_version"] == versions[1] and state["previous_version"] == versions[0])
    _, state = api.call("POST", "/api/models/rollback", "reviewer")
    check(report, "rollback restored candidate 1", state["active_version"] == versions[0])
    _, prediction = api.call("POST", "/api/models/predict", "officer-a", {"text": "Evidence inventory: item one photograph, item two recording."})
    check(report, "active model classifies and requires human validation", prediction.get("requires_human_validation") is True, f"label={prediction.get('label')} version={prediction.get('model_version')}")
    state_file.write_text(json.dumps({"case_id": case_id, "document_id": document["id"], "versions": versions, "active_version": state["active_version"], "case_revision": case["revision"]}, indent=2))
    report["state"] = json.loads(state_file.read_text())


def phase_after(api, report, state_file):
    saved = json.loads(state_file.read_text())
    _, status = api.call("GET", "/api/status")
    check(report, "status reachable after restart", status.get("name") == "Multimodal AI Lab", f"network_mode={status.get('network_mode')}")
    _, case = api.call("GET", f"/api/cases/{saved['case_id']}", "officer-a")
    check(report, "case survived restart", case["id"] == saved["case_id"], f"revision={case['revision']}")
    check(report, "document and review survived restart", any(d["id"] == saved["document_id"] and d["reviewed"] for d in case["documents"]))
    check(report, "draft and approval survived restart", bool(case["draft"]) and case.get("approved_revision") is not None, f"status={case.get('status')}")
    status_code, _ = api.call("GET", f"/api/cases/{saved['case_id']}", "officer-b", expect=None)
    check(report, "denial still enforced after restart", status_code == 403, f"http {status_code}")
    _, state = api.call("GET", "/api/models", "reviewer")
    known = {v["version"] for v in state["versions"]}
    check(report, "model versions survived restart", set(saved["versions"]) <= known, f"{len(known)} versions")
    check(report, "active model survived restart", state["active_version"] == saved["active_version"], state["active_version"])
    _, prediction = api.call("POST", "/api/models/predict", "officer-a", {"text": "Incident report: officers attended a theft scene."})
    check(report, "active model still classifies", prediction.get("status") == "classified", f"label={prediction.get('label')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8770")
    parser.add_argument("--phase", choices=["before", "after"], default="before")
    parser.add_argument("--state", default=str(ROOT / "data" / "demo-e2e-state.json"))
    parser.add_argument("--report", default=None, help="Write the JSON report to this path.")
    args = parser.parse_args()
    api = Api(args.base)
    state_file = Path(args.state)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    report = {"phase": args.phase, "base": args.base, "checks": [], "failed": 0}
    (phase_before if args.phase == "before" else phase_after)(api, report, state_file)
    print(f"\n{args.phase}: {len(report['checks']) - report['failed']} passed, {report['failed']} failed")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    sys.exit(1 if report["failed"] else 0)


if __name__ == "__main__":
    main()
