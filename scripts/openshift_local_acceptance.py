"""OpenShift Local acceptance of the application deployment (prepared manifests, synthetic data).

Runs against an OpenShift Local (CRC) cluster that `oc` is already logged in to. Builds the two
images in the cluster registry, deploys deploy/openshift/overlays/openshift-local, then verifies:
pods ready, arbitrary non-root UID, bound persistent volumes, readiness endpoint, allowed and denied
network paths, the synthetic end-to-end scenario, and the same checks after all pods are deleted.

Every result is written to an evidence directory. The README written by `report` states
"application accepted on OpenShift Local" only when every check passed. OpenShift AI and KServe are
neither installed nor tested here; their acceptance is a separate scope.

Secrets: the PostgreSQL password is generated and passed to `oc create secret`; it is never printed
or written. The Red Hat pull secret is never read by this script.

Usage (from the repository root, after `oc login`):
    python scripts/openshift_local_acceptance.py all
    python scripts/openshift_local_acceptance.py preflight|build|deploy|verify --phase before|restart|verify --phase after|report
"""
import argparse
import datetime as dt
import json
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "deploy" / "openshift" / "overlays" / "openshift-local"
LOCAL_PORT = 18770
LABEL = "app.kubernetes.io/part-of=multimodal-ai-lab"


# --------------------------------------------------------------------------- helpers
def oc(*args, check=True, capture=True, quiet=False, timeout=1800):
    """Run an oc command. Commands containing a secret must be invoked with quiet=True."""
    cmd = ["oc", *args]
    if not quiet:
        print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise SystemExit(f"oc failed ({result.returncode}): {' '.join(args[:3])}\n{result.stderr if capture else ''}")
    return result


def oc_json(*args):
    return json.loads(oc(*args, "-o", "json").stdout)


def write(evidence, name, payload):
    evidence.mkdir(parents=True, exist_ok=True)
    path = evidence / name
    path.write_text(json.dumps(payload, indent=2) if not isinstance(payload, str) else payload, encoding="utf-8")
    try:
        display_path = path.resolve().relative_to(ROOT)
    except ValueError:
        display_path = path.resolve()
    print(f"  -> {display_path}")


def check(results, name, passed, detail=""):
    results.append({"check": name, "passed": bool(passed), "detail": detail})
    print(("PASS " if passed else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def pod_exec(ns, selector, *command, timeout=60):
    """Execute a command in the first pod matching a label selector; returns (rc, stdout+stderr)."""
    pods = oc_json("get", "pods", "-n", ns, "-l", selector, "--field-selector=status.phase=Running")["items"]
    if not pods:
        return 127, f"no running pod for {selector}"
    name = pods[0]["metadata"]["name"]
    container = pods[0]["spec"]["containers"][0]["name"]
    result = subprocess.run(["oc", "exec", "-n", ns, name, "-c", container, "--", *command], capture_output=True, text=True, timeout=timeout)
    return result.returncode, (result.stdout + result.stderr).strip()


def wait_rollout(ns, kind, name, timeout="600s"):
    oc("rollout", "status", f"{kind}/{name}", "-n", ns, f"--timeout={timeout}", capture=False)


def wait_pods_ready(ns, expected=3, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        pods = oc_json("get", "pods", "-n", ns, "-l", LABEL)["items"]
        ready = [p for p in pods if p["status"].get("phase") == "Running" and all(c.get("ready") for c in p["status"].get("containerStatuses", []))]
        if len(ready) >= expected:
            return pods
        time.sleep(5)
    raise SystemExit(f"{expected} ready pods not reached within {timeout}s")


def http_get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode() or "null")
    except urllib.error.HTTPError as error:
        return error.code, None
    except Exception as error:  # connection refused while the port-forward starts
        return None, str(error)


class PortForward:
    def __init__(self, ns, target="svc/lab-app", local=LOCAL_PORT, remote=8770):
        self.args = ["oc", "port-forward", "-n", ns, target, f"{local}:{remote}"]
        self.base = f"http://127.0.0.1:{local}"
        self.process = None

    def __enter__(self):
        self.process = subprocess.Popen(self.args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            status, _ = http_get(self.base + "/health/ready")
            if status == 200:
                return self
            time.sleep(1)
        self.__exit__(None, None, None)
        raise SystemExit("port-forward to lab-app did not become ready")

    def __exit__(self, *_):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()


# --------------------------------------------------------------------------- commands
def preflight(ns, evidence):
    info = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), "namespace": ns}
    info["whoami"] = oc("whoami").stdout.strip()
    info["oc_version"] = oc("version", "-o", "json", check=False).stdout.strip() or oc("version", check=False).stdout
    for key, args in {"clusterversion": ("get", "clusterversion", "version"), "nodes": ("get", "nodes"), "storageclasses": ("get", "storageclass")}.items():
        result = oc(*args, "-o", "json", check=False)
        info[key] = json.loads(result.stdout) if result.returncode == 0 and result.stdout.strip() else f"not readable by {info['whoami']}: {result.stderr.strip()[:200]}"
    info["crc_version"] = subprocess.run(["crc", "version"], capture_output=True, text=True).stdout.strip() if shutil.which("crc") else "crc not on PATH"
    # 'okd' = community distribution bundled with CRC (no pull secret); 'openshift' = Red Hat OpenShift Container Platform bundle.
    info["crc_preset"] = subprocess.run(["crc", "config", "get", "preset"], capture_output=True, text=True).stdout.strip() if shutil.which("crc") else "unknown"
    write(evidence, "preflight.json", info)
    return info


def build(ns, evidence):
    if oc("get", "project", ns, check=False).returncode != 0:
        oc("new-project", ns, capture=False)
    oc("project", ns, capture=False)
    digests = {}
    for name, dockerfile, context_files in (
        ("lab-app", ROOT / "Dockerfile", ["requirements.txt", "app.py", "lab", "static"]),
        ("lab-mlflow", ROOT / "deploy" / "docker" / "mlflow.Dockerfile", []),
    ):
        with tempfile.TemporaryDirectory(prefix=f"{name}-build-") as tmp:
            tmp = Path(tmp)
            shutil.copy(dockerfile, tmp / "Dockerfile")
            for item in context_files:
                source = ROOT / item
                if source.is_dir():
                    shutil.copytree(source, tmp / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                else:
                    shutil.copy(source, tmp / item)
            if oc("get", "bc", name, "-n", ns, check=False).returncode != 0:
                oc("new-build", "--binary", "--strategy=docker", f"--name={name}", "-n", ns, capture=False)
            oc("start-build", name, "-n", ns, f"--from-dir={tmp}", "--follow", "--wait", capture=False, timeout=3600)
        tag = oc_json("get", "istag", f"{name}:latest", "-n", ns)
        digests[name] = {"image": tag["image"]["dockerImageReference"], "digest": tag["image"]["metadata"]["name"]}
    write(evidence, "images.json", digests)
    return digests


def deploy(ns, evidence):
    if oc("get", "secret", "lab-postgres", "-n", ns, check=False).returncode != 0:
        password = secrets.token_urlsafe(24)
        result = subprocess.run(["oc", "create", "secret", "generic", "lab-postgres", f"--from-literal=POSTGRES_PASSWORD={password}", "-n", ns], capture_output=True, text=True)
        del password
        if result.returncode != 0:
            raise SystemExit("creating the PostgreSQL secret failed")
        print("$ oc create secret generic lab-postgres --from-literal=POSTGRES_PASSWORD=<generated, not recorded>")
    oc("apply", "-k", str(OVERLAY), "-n", ns, capture=False)
    wait_rollout(ns, "statefulset", "lab-postgres")
    wait_rollout(ns, "deployment", "lab-mlflow")
    wait_rollout(ns, "deployment", "lab-app")
    state = {"pods": oc_json("get", "pods", "-n", ns, "-l", LABEL), "pvcs": oc_json("get", "pvc", "-n", ns), "networkpolicies": oc_json("get", "networkpolicy", "-n", ns)}
    write(evidence, "deploy.json", state)
    return state


def probe_verdict(rc, output, expected_open):
    """An execution error is not evidence that a network policy denied traffic."""
    value = output.strip()
    observed = "open" if value == "OPEN" else "closed" if value == "CLOSED" else "error"
    return observed, rc == 0 and observed != "error" and (observed == "open") == expected_open


def network_matrix(ns):
    """Allowed and denied paths from the NetworkPolicy set. Each probe connects with a 5 s timeout."""
    py_connect = "import socket;s=socket.socket();s.settimeout(5);print('OPEN' if s.connect_ex(('{h}',{p}))==0 else 'CLOSED');s.close()"
    py_http = "import urllib.request;r=urllib.request.urlopen('http://{h}:{p}{path}',timeout=5);print('OPEN' if r.status==200 else 'ERROR')"
    sh_connect = "timeout 5 bash -c '</dev/tcp/{h}/{p}' 2>/dev/null; rc=$?; if [ $rc -eq 0 ]; then echo OPEN; elif [ $rc -eq 1 ] || [ $rc -eq 124 ]; then echo CLOSED; else exit $rc; fi"
    probe_name = "lab-probe-" + secrets.token_hex(4)
    app, mlflow, postgres, probe = "app.kubernetes.io/name=lab-app", "app.kubernetes.io/name=lab-mlflow", "app.kubernetes.io/name=lab-postgres", f"run={probe_name}"
    cases = [
        ("app -> mlflow:5000 /health", app, ["python", "-c", py_http.format(h="lab-mlflow", p=5000, path="/health")], True),
        ("app -> postgres:5432", app, ["python", "-c", py_connect.format(h="lab-postgres", p=5432)], False),
        ("app -> internet 1.1.1.1:443", app, ["python", "-c", py_connect.format(h="1.1.1.1", p=443)], False),
        ("mlflow -> postgres:5432", mlflow, ["python", "-c", py_connect.format(h="lab-postgres", p=5432)], True),
        ("mlflow -> app:8770", mlflow, ["python", "-c", py_connect.format(h="lab-app", p=8770)], False),
        ("postgres -> mlflow:5000", postgres, ["bash", "-c", sh_connect.format(h="lab-mlflow", p=5000)], False),
        ("probe pod -> app:8770 (ingress default deny)", probe, ["python", "-c", py_connect.format(h="lab-app", p=8770)], False),
        ("probe pod -> mlflow:5000 (ingress only from app)", probe, ["python", "-c", py_connect.format(h="lab-mlflow", p=5000)], False),
        ("probe pod -> postgres:5432 (ingress only from mlflow)", probe, ["python", "-c", py_connect.format(h="lab-postgres", p=5432)], False),
    ]
    image = f"image-registry.openshift-image-registry.svc:5000/{ns}/lab-app:latest"
    oc("run", probe_name, "-n", ns, f"--image={image}", "--restart=Never", f"--labels={probe}", "--command", "--", "sleep", "900")
    for _ in range(60):
        pods = oc_json("get", "pods", "-n", ns, "-l", probe)["items"]
        if pods and pods[0]["status"].get("phase") == "Running":
            break
        time.sleep(2)
    rows = []
    try:
        for name, selector, command, expect_open in cases:
            rc, output = pod_exec(ns, selector, *command, timeout=40)
            observed, passed = probe_verdict(rc, output, expect_open)
            rows.append({"path": name, "expected": "allowed" if expect_open else "denied", "observed": observed, "passed": passed, "output": output[:200]})
    finally:
        oc("delete", "pod", probe_name, "-n", ns, "--ignore-not-found", "--wait=false", check=False)
    return rows


def verify(ns, evidence, phase):
    results = []
    pods = wait_pods_ready(ns)
    names = {p["metadata"]["labels"].get("app.kubernetes.io/name"): p["metadata"]["name"] for p in pods}
    check(results, "three pods Running and Ready", len(names) >= 3, ", ".join(sorted(names)))
    pvcs = oc_json("get", "pvc", "-n", ns)["items"]
    bound = {p["metadata"]["name"]: p["status"].get("phase") for p in pvcs}
    check(results, "persistent volume claims Bound", all(v == "Bound" for v in bound.values()) and len(bound) >= 4, json.dumps(bound))
    uids = {}
    for comp, selector in (("app", "app.kubernetes.io/name=lab-app"), ("mlflow", "app.kubernetes.io/name=lab-mlflow"), ("postgres", "app.kubernetes.io/name=lab-postgres")):
        rc, out = pod_exec(ns, selector, "id", "-u")
        uids[comp] = out if rc == 0 else f"error: {out}"
    check(results, "containers run as non-root UID", all(u.isdigit() and int(u) != 0 for u in uids.values()), json.dumps(uids))
    check(results, "app UID is the arbitrary OpenShift UID, not the image default 10001", uids.get("app", "").isdigit() and int(uids["app"]) != 10001, uids.get("app"))
    rc, out = pod_exec(ns, "app.kubernetes.io/name=lab-app", "sh", "-c", "touch /app/data/.write-test && rm /app/data/.write-test && echo writable")
    check(results, "data volume writable by the arbitrary UID", "writable" in out, out[:80])
    matrix = network_matrix(ns)
    check(results, "network matrix: every allowed path open and every denied path closed", all(r["passed"] for r in matrix), f"{sum(r['passed'] for r in matrix)}/{len(matrix)}")
    with PortForward(ns) as pf:
        status, ready = http_get(pf.base + "/health/ready")
        check(results, "/health/ready through port-forward", status == 200 and ready and ready.get("network_mode") == "container", json.dumps(ready))
        report_path = evidence / f"demo-e2e-{phase}.json"
        cmd = [sys.executable, str(ROOT / "scripts" / "demo_e2e.py"), "--phase", phase, "--base", pf.base, "--state", str(evidence / "demo-e2e-state.json"), "--report", str(report_path)]
        print("$ " + " ".join(cmd))
        demo = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        (evidence / f"demo-e2e-{phase}.log").write_text(demo.stdout + demo.stderr, encoding="utf-8")
        demo_report = json.loads(report_path.read_text()) if report_path.exists() else {"checks": [], "failed": 1}
        check(results, f"synthetic end-to-end scenario ({phase})", demo.returncode == 0, f"{len(demo_report['checks']) - demo_report['failed']} passed, {demo_report['failed']} failed")
    payload = {"phase": phase, "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "pods": names, "pvcs": bound, "uids": uids, "network_matrix": matrix, "checks": results, "failed": sum(not r["passed"] for r in results)}
    write(evidence, f"verify-{phase}.json", payload)
    return payload


def restart(ns, evidence):
    before = {p["metadata"]["labels"].get("app.kubernetes.io/name"): p["metadata"]["name"] for p in oc_json("get", "pods", "-n", ns, "-l", LABEL)["items"]}
    pvcs_before = {p["metadata"]["name"]: p["spec"].get("volumeName") for p in oc_json("get", "pvc", "-n", ns)["items"]}
    oc("delete", "pods", "-n", ns, "-l", LABEL, "--wait=true", capture=False)
    wait_rollout(ns, "statefulset", "lab-postgres")
    wait_rollout(ns, "deployment", "lab-mlflow")
    wait_rollout(ns, "deployment", "lab-app")
    after = {p["metadata"]["labels"].get("app.kubernetes.io/name"): p["metadata"]["name"] for p in wait_pods_ready(ns)}
    pvcs_after = {p["metadata"]["name"]: p["spec"].get("volumeName") for p in oc_json("get", "pvc", "-n", ns)["items"]}
    payload = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), "pods_before": before, "pods_after": after, "all_pods_replaced": all(before.get(k) != v for k, v in after.items() if k != "lab-postgres") , "volumes_unchanged": pvcs_before == pvcs_after, "pvc_volumes": pvcs_after}
    write(evidence, "restart.json", payload)
    return payload


def report(ns, evidence):
    def load(name):
        path = evidence / name
        return json.loads(path.read_text()) if path.exists() else None

    pre, images, before, rst, after = (load(n) for n in ("preflight.json", "images.json", "verify-before.json", "restart.json", "verify-after.json"))
    complete = all(x is not None for x in (pre, images, before, rst, after))
    passed = complete and before["failed"] == 0 and after["failed"] == 0 and rst["volumes_unchanged"]
    preset_text = (pre or {}).get("crc_preset", "")
    distribution = "OKD preset, the community distribution of OpenShift bundled with CRC; not the Red Hat OpenShift Container Platform bundle" if "okd" in preset_text else "Red Hat OpenShift Container Platform bundle"
    verdict = f"application accepted on OpenShift Local ({distribution})" if passed else "NOT accepted: see failed checks"
    lines = [f"# OpenShift Local acceptance run - {dt.date.today().isoformat()}", "",
             f"Result: **{verdict}**.", "",
             "Scope: application, MLflow tracking server and PostgreSQL from `deploy/openshift/overlays/openshift-local` on a single-node",
             "OpenShift Local cluster with synthetic data only. Access through `oc port-forward`; no Route; simulated identities.",
             "**OpenShift AI and KServe are not installed and not tested by this run. Their acceptance is a separate scope.**", ""]
    if pre:
        server = pre.get("oc_version", "")
        lines += ["## Platform", "", f"- Login: `{pre.get('whoami')}`", f"- CRC: `{pre.get('crc_version', '').splitlines()[0] if pre.get('crc_version') else 'n/a'}`", f"- CRC preset: `{pre.get('crc_preset', 'unknown')}` ({distribution})",
                  f"- oc/server version: see `preflight.json`", f"- Storage classes: {'recorded' if isinstance(pre.get('storageclasses'), dict) else pre.get('storageclasses')}", ""]
    if images:
        lines += ["## Images built in the cluster registry", ""] + [f"- `{k}`: `{v['digest']}`" for k, v in images.items()] + [""]
    for title, data in (("Checks before restart", before), ("Checks after restart", after)):
        if data:
            lines += [f"## {title}", "", "| Check | Result | Detail |", "| --- | --- | --- |"]
            lines += [f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail'].replace('|', '/')[:120]} |" for c in data["checks"]]
            lines += ["", "Network matrix:", "", "| Path | Expected | Observed | Result |", "| --- | --- | --- | --- |"]
            lines += [f"| {r['path']} | {r['expected']} | {r['observed']} | {'PASS' if r['passed'] else 'FAIL'} |" for r in data["network_matrix"]]
            lines += [""]
    if rst:
        lines += ["## Restart", "", f"- All three pods deleted at {rst['timestamp']}; replacements Ready: {rst['pods_after']}", f"- Persistent volumes unchanged: {rst['volumes_unchanged']}", ""]
    lines += ["## Limits", "", "- Single-node developer cluster on one workstation; not a sizing or performance result.",
              "- Speech weights were not copied to the `lab-models` volume unless stated above; speech recognition reports unavailable in that case.",
              "- Port-forward bypasses NetworkPolicy by design (kubelet path); the ingress default-deny was verified from an in-namespace probe pod.",
              "- No Route, no trusted identity, no GPU, no model serving: unchanged target scope.", ""]
    write(evidence, "README.md", "\n".join(lines))
    print(f"\n{verdict}")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["preflight", "build", "deploy", "verify", "restart", "report", "all"])
    parser.add_argument("--phase", choices=["before", "after"], default="before")
    parser.add_argument("--namespace", default="multimodal-ai-lab")
    parser.add_argument("--skip-build", action="store_true", help="Reuse existing cluster image streams and record their current digests")
    parser.add_argument("--evidence", default=str(ROOT / "docs" / "evidence" / f"{dt.date.today().isoformat()}-openshift-local"))
    args = parser.parse_args()
    evidence = Path(args.evidence).resolve()
    ns = args.namespace
    if shutil.which("oc") is None:
        raise SystemExit("oc is not on PATH: run `crc oc-env | Invoke-Expression` first")
    if args.command == "all":
        preflight(ns, evidence)
        if args.skip_build:
            images = {}
            for name in ("lab-app", "lab-mlflow"):
                tag = oc_json("get", "istag", f"{name}:latest", "-n", ns)
                images[name] = {"image": tag["image"]["dockerImageReference"], "digest": tag["image"]["metadata"]["name"]}
            write(evidence, "images.json", images)
        else:
            build(ns, evidence)
        deploy(ns, evidence)
        before = verify(ns, evidence, "before")
        if before["failed"]:
            report(ns, evidence)
            sys.exit(1)
        restart(ns, evidence); verify(ns, evidence, "after")
        sys.exit(0 if report(ns, evidence) else 1)
    {"preflight": lambda: preflight(ns, evidence), "build": lambda: build(ns, evidence), "deploy": lambda: deploy(ns, evidence),
     "verify": lambda: verify(ns, evidence, args.phase), "restart": lambda: restart(ns, evidence), "report": lambda: sys.exit(0 if report(ns, evidence) else 1)}[args.command]()


if __name__ == "__main__":
    main()
