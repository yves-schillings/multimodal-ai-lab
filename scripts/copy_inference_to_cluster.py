"""Copy only two pinned, locally downloaded Ollama models to the local inference PVC."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path


MODELS = {"bge-m3/latest": "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab",
          "qwen3/4b": "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"}


def oc(*args, **kwargs):
    return subprocess.run(["oc", *args], check=True, timeout=600, capture_output=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path(os.environ.get("OLLAMA_MODELS", Path.home()/".ollama/models")))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.model_root.resolve()
    files, manifest_files = {}, {}
    for name, digest in MODELS.items():
        relative = "manifests/registry.ollama.ai/library/"+name
        data = (root/relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("Local manifest does not match the pinned model: "+name)
        manifest_files[relative] = len(data)
        manifest = json.loads(data)
        for layer in [manifest["config"], *manifest["layers"]]:
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", layer["digest"]):
                raise RuntimeError("Invalid model layer digest")
            relative = "blobs/"+layer["digest"].replace(":", "-")
            path = root/relative
            if not path.is_file() or path.is_symlink() or path.stat().st_size != layer["size"]:
                raise RuntimeError("Missing or unexpected local layer")
            files[relative] = layer["size"]
    files.update(manifest_files)  # Copy model manifests only after all their blobs.
    expected = {}
    for relative, size in files.items():
        with (root/relative).open("rb") as source:
            expected[relative] = {"bytes": size, "sha256": hashlib.file_digest(source, "sha256").hexdigest()}
    name = "lab-inference-copy-"+uuid.uuid4().hex[:8]
    pod = {"apiVersion":"v1", "kind":"Pod", "metadata":{"name":name, "namespace":"multimodal-ai-lab"},
           "spec":{"restartPolicy":"Never", "automountServiceAccountToken":False,
                   "securityContext":{"runAsNonRoot":True,"seccompProfile":{"type":"RuntimeDefault"}},
                   "containers":[{"name":"copy", "image":"image-registry.openshift-image-registry.svc:5000/multimodal-ai-lab/lab-app:latest",
                                  "command":["sleep","900"], "securityContext":{"allowPrivilegeEscalation":False,"capabilities":{"drop":["ALL"]}},
                                  "resources":{"requests":{"cpu":"100m","memory":"128Mi"},"limits":{"cpu":"1","memory":"256Mi"}},
                                  "volumeMounts":[{"name":"models","mountPath":"/models"}]}],
                   "volumes":[{"name":"models","persistentVolumeClaim":{"claimName":"lab-inference-models"}}]}}
    remote = """
import hashlib,json,pathlib,sys,tarfile,shutil,os
expected=json.loads(sys.argv[1]); root=pathlib.Path('/models'); count=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
 for entry in archive:
  if entry.name not in expected or not entry.isfile() or entry.size!=expected[entry.name]['bytes']: raise RuntimeError('Unexpected archive entry')
  item=expected[entry.name]; destination=root/entry.name; destination.parent.mkdir(parents=True,exist_ok=True)
  if destination.exists():
   if destination.is_symlink(): raise RuntimeError('Existing symbolic link refused')
   with destination.open('rb') as file: digest=hashlib.file_digest(file,'sha256').hexdigest()
   if digest!=item['sha256']: raise RuntimeError('Existing model file differs; not overwritten')
   with archive.extractfile(entry) as incoming:
    while incoming.read(1024*1024): pass
  else:
   stage=destination.with_name(destination.name+'.stage-'+sys.argv[2])
   try:
    with archive.extractfile(entry) as source,stage.open('xb') as target: shutil.copyfileobj(source,target)
    with stage.open('rb') as file: digest=hashlib.file_digest(file,'sha256').hexdigest()
    if digest!=item['sha256']: raise RuntimeError('Copied model file digest mismatch')
    stage.rename(destination)
   finally: stage.unlink(missing_ok=True)
  count+=1
if count!=len(expected): raise RuntimeError('Missing archive entries')
print('All pinned local inference files verified')
"""
    created = False
    try:
        oc("create","-f","-",input=json.dumps(pod),text=True); created=True
        oc("wait","-n","multimodal-ai-lab","--for=condition=Ready","pod/"+name,"--timeout=180s",text=True)
        with tempfile.TemporaryFile() as archive:
            with tarfile.open(fileobj=archive,mode="w") as tar:
                for relative in files: tar.add(root/relative,arcname=relative,recursive=False)
            archive.seek(0)
            result=oc("exec","-i","-n","multimodal-ai-lab",name,"--","python","-c",remote,
                      json.dumps(expected),name,stdin=archive)
            print(result.stdout.decode().strip())
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps({"passed":True,"models":MODELS,"files":expected},indent=2),encoding="utf-8")
    finally:
        if created: oc("delete","pod",name,"-n","multimodal-ai-lab","--wait=false","--ignore-not-found",text=True)


if __name__ == "__main__": main()
