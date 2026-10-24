"""Load the shared fictional M1/M2 corpus into new cases of the deployed lab through its normal API.

Creates one corpus case for officer-a, an identical corpus case for officer-b (used by the concurrent
M3 users) and the two control cases that injected instructions try to reveal. Every document is
uploaded, extracted by the normal job worker and confirmed unchanged as reviewed. The manifest maps
passage identifiers to case document identifiers; it contains no credential.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.measurement_common import LabApi, PortForward, NS, file_sha256, now, upload_reviewed_document, write_json

CORPUS = ROOT / "tests" / "fixtures" / "measurements" / "rag_corpus.json"


def load_corpus(api, title, corpus):
    case_id = api.request("POST", "/api/cases", json={"title": title}).json()["id"]
    documents = {}
    for passage in corpus["passages"]:
        documents[passage["id"]] = upload_reviewed_document(api, case_id, passage["id"] + ".txt", passage["text"])
        print("loaded", passage["id"], flush=True)
    case = api.request("GET", f"/api/cases/{case_id}").json()
    if len(case["documents"]) != len(corpus["passages"]) or not all(d["reviewed"] for d in case["documents"]):
        raise RuntimeError("Corpus case is incomplete or contains unreviewed documents.")
    return {"case_id": case_id, "owner": api.account, "revision": case["revision"], "documents": documents}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    manifest = {"created_at": now(), "namespace": NS, "corpus_sha256": file_sha256(CORPUS), "cases": {}}
    with PortForward(NS) as forward:
        apis = {account: LabApi(forward.base, args.credentials, account) for account in ("officer-a", "officer-b")}
        manifest["cases"]["corpus_officer_a"] = load_corpus(apis["officer-a"], "Fictional measurement corpus (officer A)", corpus)
        manifest["cases"]["corpus_officer_b"] = load_corpus(apis["officer-b"], "Fictional measurement corpus (officer B)", corpus)
        for control in corpus["other_cases"]:
            api = apis[control["owner"]]
            case_id = api.request("POST", "/api/cases", json={"title": control["title"]}).json()["id"]
            document = upload_reviewed_document(api, case_id, control["key"] + ".txt", control["document"])
            case = api.request("GET", f"/api/cases/{case_id}").json()
            manifest["cases"][control["key"]] = {"case_id": case_id, "owner": control["owner"],
                                                 "revision": case["revision"], "documents": {control["key"]: document}}
        for api in apis.values():
            api.close()
    write_json(args.manifest.parent, args.manifest.name, manifest)
    print("Corpus manifest written to", args.manifest)


if __name__ == "__main__":
    main()
