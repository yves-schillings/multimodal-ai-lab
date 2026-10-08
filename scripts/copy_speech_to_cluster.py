"""Copy prepared local Whisper base weights into the OKD models PVC without downloading.

Requires oc login and an existing lab-models PVC. A temporary restricted pod mounts
the PVC writable; the application keeps its read-only mount. Existing model files
are verified, never overwritten. Only file names, sizes and hashes enter evidence.
"""
import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path


FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    return subprocess.run(["oc", *args], check=True, timeout=300, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="multimodal-ai-lab")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "whisper-base")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = {}
    for name in FILES:
        path = args.model_dir / name
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"Required local model file unavailable: {name}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        manifest[name] = {"bytes": path.stat().st_size, "sha256": digest}
    name = "lab-model-copy-" + uuid.uuid4().hex[:10]
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": name, "namespace": args.namespace},
           "spec": {"restartPolicy": "Never", "securityContext": {"runAsNonRoot": True},
                    "containers": [{"name": "copy", "image": f"image-registry.openshift-image-registry.svc:5000/{args.namespace}/lab-app:latest",
                                    "command": ["sleep", "900"],
                                    "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}, "seccompProfile": {"type": "RuntimeDefault"}},
                                    "resources": {"requests": {"cpu": "10m", "memory": "64Mi"}, "limits": {"cpu": "500m", "memory": "256Mi"}},
                                    "volumeMounts": [{"name": "models", "mountPath": "/app/models"}]}],
                    "volumes": [{"name": "models", "persistentVolumeClaim": {"claimName": "lab-models"}}]}}
    created = False
    try:
        run("create", "-f", "-", input=json.dumps(pod), text=True, capture_output=True)
        created = True
        run("wait", "-n", args.namespace, "--for=condition=Ready", f"pod/{name}", "--timeout=180s")
        remote = """
import hashlib,json,os,pathlib,sys,tarfile,shutil
manifest=json.loads(sys.argv[1]); root=pathlib.Path('/app/models'); dest=root/'whisper-base'
def verify(folder):
    for name,item in manifest.items():
        p=folder/name
        if not p.is_file() or p.is_symlink() or p.stat().st_size!=item['bytes']:
            raise RuntimeError('Model file missing or unexpected size: '+name)
        with p.open('rb') as source: actual=hashlib.file_digest(source,'sha256').hexdigest()
        if actual!=item['sha256']: raise RuntimeError('Model digest mismatch: '+name)
if dest.exists():
    verify(dest)
    # Drain the incoming stream so oc does not report a broken pipe on repeat runs.
    while sys.stdin.buffer.read(1024*1024): pass
    print('Existing local weights match; no files changed')
else:
    stage=root/('.copy-'+sys.argv[2]); stage.mkdir()
    try:
        with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
            for entry in archive:
                if entry.name not in manifest or not entry.isfile() or entry.size!=manifest[entry.name]['bytes']:
                    raise RuntimeError('Unexpected archive entry')
                with archive.extractfile(entry) as src, (stage/entry.name).open('xb') as dst:
                    shutil.copyfileobj(src,dst)
        verify(stage); stage.rename(dest)
        print('Local weights copied and digests verified')
    finally:
        if stage.exists(): shutil.rmtree(stage)
"""
        with tempfile.TemporaryFile() as archive:
            with tarfile.open(fileobj=archive, mode="w") as tar:
                for filename in FILES:
                    tar.add(args.model_dir / filename, arcname=filename, recursive=False)
            archive.seek(0)
            run("exec", "-i", "-n", args.namespace, name, "--", "python", "-c", remote,
                json.dumps(manifest), name, stdin=archive)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"namespace": args.namespace, "model": "Whisper base", "local_copy_verified": True, "files": manifest}, indent=2), encoding="utf-8")
        print("PASS local model weights verified on the models PVC")
    finally:
        if created:
            run("delete", "pod", name, "-n", args.namespace, "--wait=false", "--ignore-not-found", capture_output=True)


if __name__ == "__main__":
    main()
