"""Actual local OKD application RAG acceptance; synthetic data and port-forward only."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.openshift_local_acceptance import PortForward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    args = parser.parse_args()
    report = {"scope":"local OKD app + offline CPU Ollama + pgvector; synthetic data; simulated identities",
              "checks":[], "passed":False}
    def login(api):
        status=api.get('/api/status').json()
        report['identity_mode']=status['identity_mode']
        if not status['simulated_identity']:
            if not args.credentials:raise RuntimeError('Authenticated acceptance requires the private credential file.')
            passwords=json.loads(args.credentials.read_text(encoding='utf-8'))['passwords']
            response=api.post('/api/auth/login',json={'username':'officer-a','password':passwords['officer-a']})
            response.raise_for_status();api.headers['X-Lab-CSRF']=response.json()['csrf']
            report['scope']='local OKD app + offline CPU Ollama + pgvector; synthetic data; authenticated local account'
    def check(name, passed):
        report["checks"].append({"name":name,"passed":bool(passed)})
        print(("PASS " if passed else "FAIL ")+name,flush=True)
        if not passed: raise RuntimeError("Acceptance failed: "+name)
    try:
        with PortForward("multimodal-ai-lab") as forward, httpx.Client(base_url=forward.base,timeout=300,trust_env=False) as api:
            login(api)
            status=api.get("/api/status").json()
            check("app configured for local pgvector and generation", status["capabilities"]["generative_llm"] and "pgvector" in status["capabilities"]["retrieval"])
            response=api.post("/api/cases",params={"actor":"officer-a"},json={"title":"Synthetic offline RAG cluster acceptance"})
            response.raise_for_status(); case_id=response.json()["id"]
            report["case_id"]=case_id
            response=api.post(f"/api/cases/{case_id}/upload",params={"actor":"officer-a"},data={"kind":"document"},
                              files={"file":("synthetic-rag.txt",b"I left my blue bicycle beside the library at nine in the morning.","text/plain")})
            response.raise_for_status(); job_id=response.json()["job_id"]
            for attempt in range(60):
                job=api.get("/api/jobs/"+job_id,params={"actor":"officer-a"}).json()
                if job["status"]=="completed": break
                if job["status"]=="failed": raise RuntimeError("Synthetic document job failed")
                time.sleep(1)
            else: raise RuntimeError("Synthetic document job did not complete")
            def ask(actor="officer-a",question="Where was the bicycle left?"):
                return api.post(f"/api/cases/{case_id}/question",params={"actor":actor},json={"question":question})
            result=ask(); result.raise_for_status()
            check("unreviewed documents excluded",result.json()["abstained"])
            case=api.get(f"/api/cases/{case_id}",params={"actor":"officer-a"}).json()
            document=case["documents"][0]
            response=api.post(f"/api/cases/{case_id}/documents/{document['id']}/review",params={"actor":"officer-a"},
                              json={"text":document["text"],"expected_revision":case["revision"]})
            response.raise_for_status()
            response=ask(); response.raise_for_status(); answer=response.json()
            check("pod generates a cited answer from reviewed evidence",not answer["abstained"] and "library" in answer["answer"].lower() and bool(answer.get("claims")))
            check("supporting quotes match cited source",all(c["quote"] in answer["sources"][c["source"]-1]["text"] for c in answer["claims"]))
            check("other actor denied",ask("officer-b").status_code==403)
            response=ask(question="What is the chemical composition of Neptune's atmosphere?"); response.raise_for_status()
            check("unsupported question abstains",response.json()["abstained"])
            report["sample"]=answer
        volumes=json.loads(subprocess.check_output(["oc","get","pvc","lab-vector-data","lab-inference-models","-n","multimodal-ai-lab","-o","json"]))
        before={p["metadata"]["name"]:p["spec"]["volumeName"] for p in volumes["items"]}
        subprocess.run(["oc","delete","pod","-n","multimodal-ai-lab","-l","app.kubernetes.io/name in (lab-vector,lab-inference)"],check=True,stdout=subprocess.DEVNULL)
        for workload in ("statefulset/lab-vector","deployment/lab-inference"):
            subprocess.run(["oc","rollout","status",workload,"-n","multimodal-ai-lab","--timeout=180s"],check=True,stdout=subprocess.DEVNULL)
        volumes=json.loads(subprocess.check_output(["oc","get","pvc","lab-vector-data","lab-inference-models","-n","multimodal-ai-lab","-o","json"]))
        after={p["metadata"]["name"]:p["spec"]["volumeName"] for p in volumes["items"]}
        check("vector and inference PVCs unchanged after pod replacement",before==after)
        with PortForward("multimodal-ai-lab") as forward, httpx.Client(base_url=forward.base,timeout=300,trust_env=False) as api:
            login(api)
            response=api.post(f"/api/cases/{case_id}/question",params={"actor":"officer-a"},json={"question":"Where was the bicycle left?"})
            response.raise_for_status(); result=response.json()
            check("RAG remains available after service replacement",not result["abstained"] and "library" in result["answer"].lower())
        report["passed"]=True
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
