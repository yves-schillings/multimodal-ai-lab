"""Extend the actual original local OKD network matrix for offline RAG services."""
import argparse
import json
import secrets
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.openshift_local_acceptance import network_matrix, oc, oc_json, pod_exec, probe_verdict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    ns="multimodal-ai-lab"
    rows=network_matrix(ns)
    name="lab-rag-probe-"+secrets.token_hex(4)
    selector="run="+name
    image=f"image-registry.openshift-image-registry.svc:5000/{ns}/lab-app:latest"
    oc("run",name,"-n",ns,"--image="+image,"--restart=Never","--labels="+selector,"--command","--","sleep","600")
    py="import socket;s=socket.socket();s.settimeout(5);print('OPEN' if s.connect_ex(('{h}',{p}))==0 else 'CLOSED');s.close()"
    sh="timeout 5 bash -c '</dev/tcp/1.1.1.1/443' 2>/dev/null; rc=$?; if [ $rc -eq 0 ]; then echo OPEN; elif [ $rc -eq 1 ] || [ $rc -eq 124 ]; then echo CLOSED; else exit $rc; fi"
    cases=[
        ("app -> vector:5432","app.kubernetes.io/name=lab-app",["python","-c",py.format(h="lab-vector",p=5432)],True),
        ("app -> inference:11434","app.kubernetes.io/name=lab-app",["python","-c",py.format(h="lab-inference",p=11434)],True),
        ("unrelated pod -> vector:5432",selector,["python","-c",py.format(h="lab-vector",p=5432)],False),
        ("unrelated pod -> inference:11434",selector,["python","-c",py.format(h="lab-inference",p=11434)],False),
        ("vector -> internet 1.1.1.1:443","app.kubernetes.io/name=lab-vector",["bash","-c",sh],False),
        ("inference -> internet 1.1.1.1:443","app.kubernetes.io/name=lab-inference",["bash","-c",sh],False),
    ]
    try:
        for _ in range(60):
            pods=oc_json("get","pods","-n",ns,"-l",selector)["items"]
            if pods and pods[0]["status"].get("phase")=="Running": break
            time.sleep(2)
        for label,source,command,expected in cases:
            rc,output=pod_exec(ns,source,*command,timeout=40)
            observed,passed=probe_verdict(rc,output,expected)
            rows.append({"path":label,"expected":"allowed" if expected else "denied","observed":observed,"passed":passed})
            print(("PASS " if passed else "FAIL ")+label,flush=True)
    finally:
        oc("delete","pod",name,"-n",ns,"--wait=false","--ignore-not-found",check=False)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps({"scope":"Local OKD network connectivity; original matrix plus offline RAG paths","checks":rows,"passed":all(row['passed'] for row in rows)},indent=2),encoding="utf-8")
    if not all(row["passed"] for row in rows):
        raise RuntimeError("Network acceptance failed; execution errors are not treated as denials.")


if __name__=="__main__": main()
