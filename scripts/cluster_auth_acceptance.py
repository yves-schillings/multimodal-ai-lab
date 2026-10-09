"""Exercise real local OKD password sessions without printing credentials or session tokens."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.openshift_local_acceptance import PortForward


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    passwords=json.loads(args.credentials.read_text(encoding="utf-8"))["passwords"]
    report={"scope":"Local OKD password sessions; fictional accounts; no institutional identity claim","checks":[],"passed":False}
    def check(name,passed):
        report["checks"].append({"name":name,"passed":bool(passed)})
        print(("PASS " if passed else "FAIL ")+name,flush=True)
        if not passed: raise RuntimeError("Authentication acceptance failed: "+name)
    try:
        with PortForward("multimodal-ai-lab") as forward,httpx.Client(base_url=forward.base,timeout=300,trust_env=False) as client:
            check("real local identity mode",client.get("/api/status").json()["identity_mode"]=="local_password_session")
            check("actor selector cannot authenticate",client.get("/api/cases?actor=reviewer").status_code==401)
            check("wrong password refused",client.post("/api/auth/login",json={"username":"officer-a","password":"invalid"}).status_code==401)
            response=client.post("/api/auth/login",json={"username":"officer-a","password":passwords["officer-a"]});response.raise_for_status()
            csrf=response.json()["csrf"]
            check("session cookie is HttpOnly and SameSite Strict","HttpOnly" in response.headers["set-cookie"] and "SameSite=strict" in response.headers["set-cookie"])
            check("case permissions bound to authenticated account",client.get("/api/cases/demo-case-a").status_code==200 and client.get("/api/cases/demo-case-b").status_code==403)
            check("reviewer impersonation denied",client.get("/api/cases?actor=reviewer").status_code==403)
            check("CSRF-less write denied",client.post("/api/cases",json={"title":"Must not be created"}).status_code==401)
            client.headers["X-Lab-CSRF"]=csrf
            check("officer cannot train classifier",client.post("/api/models/train").status_code==403)
            created=client.post("/api/cases",json={"title":"Synthetic authenticated acceptance"});created.raise_for_status()
            check("authenticated mutation accepted",created.json()["owner_actor"]=="officer-a")
            # Persistent server sessions remain valid after replacing only the application pod.
            token=client.cookies.get("lab_session")
        subprocess.run(["oc","delete","pod","-n","multimodal-ai-lab","-l","app.kubernetes.io/name=lab-app"],check=True,capture_output=True)
        subprocess.run(["oc","rollout","status","deployment/lab-app","-n","multimodal-ai-lab","--timeout=180s"],check=True,capture_output=True)
        with PortForward("multimodal-ai-lab") as forward,httpx.Client(base_url=forward.base,timeout=300,trust_env=False) as client:
            client.cookies.set("lab_session",token);client.headers["X-Lab-CSRF"]=csrf
            check("local session persists after app replacement",client.get("/api/auth/me").status_code==200)
            check("logout accepted",client.post("/api/auth/logout").status_code==200)
            client.cookies.set("lab_session",token)
            check("logged-out token refused",client.get("/api/cases").status_code==401)
            client.cookies.clear()
            response=client.post("/api/auth/login",json={"username":"reviewer","password":passwords["reviewer"]});response.raise_for_status()
            check("reviewer uses own authenticated identity",client.get("/api/auth/me").json()["principal"]["id"]=="reviewer")
        report["passed"]=True
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__": main()
