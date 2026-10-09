"""Provision local OKD password accounts, keeping reusable passwords in an owner-only private file."""
import argparse
import base64
import csv
import io
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.auth import password_hash


def oc(*args, **kwargs):
    return subprocess.run(["oc",*args],check=True,capture_output=True,timeout=240,**kwargs)


def protect_file(path):
    if os.name=="nt":
        identity=list(csv.reader(io.StringIO(subprocess.check_output(["whoami","/user","/fo","csv","/nh"],text=True))))[0][1]
        # icacls requires an asterisk before a numeric SID.
        subprocess.run(["icacls",str(path),"/inheritance:r","/grant:r","*"+identity+":(F)"],check=True,capture_output=True)
    else:
        path.chmod(0o600)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-output",type=Path,required=True)
    args=parser.parse_args()
    output=args.private_output.resolve()
    if output==ROOT or ROOT in output.parents:
        raise ValueError("Keep plaintext credentials outside the code repository.")
    exists=oc("get","secret","lab-local-credentials","-n","multimodal-ai-lab","--ignore-not-found","-o","name").stdout.strip()
    if output.exists():
        protect_file(output)
        data=json.loads(output.read_text(encoding="utf-8"))
    elif exists:
        raise RuntimeError("Existing credentials have no matching private file; do not replace accounts.")
    else:
        passwords={actor:secrets.token_urlsafe(24) for actor in ("officer-a","officer-b","reviewer")}
        data={"passwords":passwords,"accounts":{actor:{"enabled":True,"password_hash":password_hash(password)} for actor,password in passwords.items()}}
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("x",encoding="utf-8") as target:
            json.dump(data,target,indent=2)
        protect_file(output)
    # Keep existing Secret content: rerunning must not undo administrator account revocations.
    if not exists:
        secret={"apiVersion":"v1","kind":"Secret","metadata":{"name":"lab-local-credentials","namespace":"multimodal-ai-lab"},
                "type":"Opaque","data":{"credentials.json":base64.b64encode(json.dumps(data["accounts"]).encode()).decode()}}
        oc("create","-f","-",input=json.dumps(secret).encode())
    oc("patch","deployment","lab-app","-n","multimodal-ai-lab","--type","strategic","--patch-file",str(ROOT/"deploy/openshift/local-rag/app-auth-patch.yaml"))
    oc("rollout","status","deployment/lab-app","-n","multimodal-ai-lab","--timeout=180s")
    print("Local credentials activated. Passwords remain in the specified private file; none printed.")


if __name__=="__main__": main()
