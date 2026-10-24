"""Shared helpers for the planned laboratory measurements (M1-M5); fictional data only.

The helpers authenticate with the private local credential file, run bounded Python code inside the
deployed application pod, compute order statistics and write evidence with SHA-256 digests.
No password, session token or CSRF token is ever printed or written to evidence.
"""
import base64
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.openshift_local_acceptance import PortForward

NS = "multimodal-ai-lab"

# The in-pod bootstrap stays short; the measured code and its input files travel on stdin.
BOOTSTRAP = ("import base64,json,os,sys,tempfile;p=json.loads(sys.stdin.read());d=tempfile.mkdtemp(prefix='measure-');"
             "[open(os.path.join(d,n),'wb').write(base64.b64decode(v)) for n,v in p['files'].items()];"
             "sys.path.insert(0,'/app');g={'WORK':d,'PARAMS':p['params']};exec(compile(p['code'],'measurement','exec'),g)")


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def summary(values):
    """Median and nearest-rank 95th percentile: the smallest value with at least 95% of samples at or below it."""
    data = sorted(values)
    if not data:
        return {"n": 0}
    n = len(data)
    middle = n // 2
    median = data[middle] if n % 2 else (data[middle - 1] + data[middle]) / 2
    p95 = data[max(0, math.ceil(0.95 * n) - 1)]
    return {"n": n, "min": round(data[0], 4), "median": round(median, 4), "p95": round(p95, 4),
            "max": round(data[-1], 4), "mean": round(sum(data) / n, 4)}


def oc(*args, timeout=600, input_bytes=None):
    result = subprocess.run(["oc", *args], capture_output=True, timeout=timeout, input=input_bytes)
    if result.returncode:
        raise RuntimeError("oc " + " ".join(args[:2]) + " failed: " + result.stderr.decode(errors="replace")[-400:])
    return result.stdout.decode()


def running_pod(label):
    pods = json.loads(oc("get", "pods", "-n", NS, "-l", "app.kubernetes.io/name=" + label, "-o", "json"))["items"]
    ready = [p for p in pods if p["status"].get("phase") == "Running" and not p["metadata"].get("deletionTimestamp")]
    if not ready:
        raise RuntimeError("No running pod for " + label)
    return ready[0]


def pod_python(code, params=None, files=None, label="lab-app", container="app", timeout=7200):
    """Run measurement code in the deployed pod; the code prints one JSON document as its last stdout line."""
    pod = running_pod(label)["metadata"]["name"]
    payload = {"code": code, "params": params or {},
               "files": {name: base64.b64encode(data).decode() for name, data in (files or {}).items()}}
    process = subprocess.Popen(["oc", "exec", "-i", "-n", NS, pod, "-c", container, "--", "python", "-c", BOOTSTRAP],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
    out, _ = process.communicate(json.dumps(payload).encode(), timeout=timeout)
    if process.returncode:
        raise RuntimeError("In-pod measurement failed with exit code " + str(process.returncode))
    return json.loads(out.decode().strip().splitlines()[-1])


def runtime_context():
    """Cluster, image and model identities that make a measurement reproducible."""
    context = {"recorded_at": now(), "host": {"platform": platform.platform(), "processor": platform.processor()}}
    for label in ("lab-app", "lab-inference", "lab-vector"):
        pod = running_pod(label)
        container = pod["spec"]["containers"][0]
        status = pod["status"]["containerStatuses"][0]
        context[label] = {"image": container["image"], "image_id": status.get("imageID"),
                          "resources": container.get("resources", {}), "node": pod["spec"].get("nodeName")}
    deployment = json.loads(oc("get", "deployment/lab-inference", "-n", NS, "-o", "json"))
    context["lab-inference"]["env"] = {e["name"]: e.get("value") for e in deployment["spec"]["template"]["spec"]["containers"][0].get("env", [])
                                       if e["name"].startswith("OLLAMA_")}
    node = json.loads(oc("get", "nodes", "-o", "json"))["items"][0]
    context["node"] = {"capacity": node["status"]["capacity"], "allocatable": node["status"]["allocatable"],
                       "os_image": node["status"]["nodeInfo"]["osImage"], "kubelet": node["status"]["nodeInfo"]["kubeletVersion"]}
    code = ("import hashlib,json,os,pathlib;r={'lab/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pathlib.Path('/app/lab').glob('*.py'))};"
            "r['env']={k:os.environ.get(k) for k in ('LAB_RETRIEVAL_BACKEND','LAB_EMBEDDING_MODEL','LAB_EMBEDDING_DIGEST','LAB_GENERATION_MODEL','LAB_GENERATION_DIGEST','LAB_WHISPER_MODEL_DIR','LAB_AUTH_MODE')};print(json.dumps(r))")
    context["app_source_sha256"] = pod_python(code)
    return context


class LabApi:
    """Authenticated API client over a loopback port-forward; renews an expired session once per request."""

    def __init__(self, base, credentials, account):
        self.base, self.account = base, account
        self.password = json.loads(Path(credentials).read_text(encoding="utf-8"))["passwords"][account]
        self.client = httpx.Client(base_url=base, timeout=900, trust_env=False)
        self.login()

    def login(self):
        self.client.cookies.clear()
        response = self.client.post("/api/auth/login", json={"username": self.account, "password": self.password})
        response.raise_for_status()
        self.client.headers["X-Lab-CSRF"] = response.json()["csrf"]

    def request(self, method, path, **kwargs):
        response = self.client.request(method, path, **kwargs)
        if response.status_code == 401:
            self.login()
            response = self.client.request(method, path, **kwargs)
        return response

    def close(self):
        self.client.close()


def upload_reviewed_document(api, case_id, name, text):
    """Upload one UTF-8 text document, wait for extraction and confirm it unchanged as reviewed."""
    response = api.request("POST", f"/api/cases/{case_id}/upload", data={"kind": "document"},
                           files={"file": (name, text.encode("utf-8"), "text/plain")})
    response.raise_for_status()
    job_id = response.json()["job_id"]
    for _ in range(240):
        job = api.request("GET", "/api/jobs/" + job_id).json()
        if job["status"] == "completed":
            break
        if job["status"] == "failed":
            raise RuntimeError("Fictional document extraction failed: " + name)
        time.sleep(0.5)
    else:
        raise RuntimeError("Fictional document extraction timed out: " + name)
    case = api.request("GET", f"/api/cases/{case_id}").json()
    document = next(d for d in case["documents"] if d["name"] == name)
    if not document["reviewed"]:
        response = api.request("POST", f"/api/cases/{case_id}/documents/{document['id']}/review",
                               json={"text": document["text"], "expected_revision": case["revision"]})
        response.raise_for_status()
    return document["id"]


def write_json(folder, name, payload):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_digests(folder):
    """SHA256SUMS lists every evidence file except itself, sorted by relative path."""
    lines = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(folder).as_posix())
    (folder / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
