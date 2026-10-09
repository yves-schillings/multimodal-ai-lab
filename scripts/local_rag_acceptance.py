"""Exercise actual host-local Ollama and local OKD pgvector with isolated synthetic cases.

This verifies the host-to-cluster integration. It does not claim the application pod
has been rebuilt/configured for RAG, real authentication or GPU pass-through.
"""
import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.local_inference import LocalInference
from lab.rag import LocalRag
from lab.backend_store import LabStore
from lab.vector_retrieval import VectorIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import psycopg
    secret = json.loads(subprocess.check_output(["oc", "get", "secret", "lab-vector-client", "-n", "multimodal-ai-lab", "-o", "json"]))
    password = base64.b64decode(secret["data"]["password"]).decode()
    options = {"host": "127.0.0.1", "port": 19543, "dbname": "lab_vectors", "user": "vector_client", "password": password}
    inference = LocalInference("http://127.0.0.1:11434", "bge-m3:latest", "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab",
                               "qwen3:4b", "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7")
    forward = subprocess.Popen(["oc", "port-forward", "-n", "multimodal-ai-lab", "svc/lab-vector", "19543:5432"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = {"scope": "host-local Ollama with separate local OKD pgvector; synthetic fixtures", "checks": []}
    def check(name, condition):
        report["checks"].append({"name": name, "passed": bool(condition)})
        print(("PASS " if condition else "FAIL ")+name, flush=True)
        if not condition:
            raise RuntimeError("Acceptance check failed: "+name)
    try:
        for attempt in range(30):
            if forward.poll() is not None:
                raise RuntimeError("Private vector port-forward failed to start")
            try:
                with psycopg.connect(**options, connect_timeout=2) as db:
                    row = db.execute("SELECT rolsuper, rolcreatedb FROM pg_roles WHERE rolname=current_user").fetchone()
                    check("vector client is not an administrator", row == (False, False))
                break
            except psycopg.OperationalError:
                time.sleep(1)
        else:
            raise RuntimeError("Private vector port-forward did not become ready")
        index, rag = VectorIndex(options, inference), None
        rag = LocalRag(index)
        with tempfile.TemporaryDirectory() as directory:
            store = LabStore(Path(directory))
            case = store.create_case("officer-a", "Isolated synthetic RAG acceptance")
            case_id = case["id"]
            store.save_transcript("officer-a", case_id, {"engine": "synthetic-fixture", "segments": [
                {"start": 0, "end": 4, "speaker": "Unverified speaker", "text": "I left my blue bicycle beside the library at nine in the morning."},
                {"start": 4, "end": 8, "speaker": "Unverified speaker", "text": "When I returned at eleven, the bicycle was no longer there."}]})
            result = rag.answer(store, "officer-a", case_id, "Where was the bicycle left?")
            check("unreviewed testimony is excluded", result["abstained"] and not result["sources"])
            case = store.get_case("officer-a", case_id)
            for segment in case["segments"]:
                case = store.update_segment("officer-a", case_id, segment["id"], segment["text"], case["revision"])
            result = rag.answer(store, "officer-a", case_id, "Where was the bicycle left?")
            check("actual local RAG gives the supported location", not result["abstained"] and "library" in result["answer"].lower())
            check("generated claims retain exact supporting quotes", bool(result.get("claims")) and all(
                c["quote"] in result["sources"][c["source"]-1]["text"] for c in result["claims"]))
            report["sample"] = result
            try:
                rag.answer(store, "officer-b", case_id, "Where was the bicycle left?")
            except PermissionError:
                check("other actor is denied before inference", True)
            else:
                check("other actor is denied before inference", False)
            unrelated = rag.answer(store, "officer-a", case_id, "What is the chemical composition of the atmosphere of Neptune?")
            check("unsupported question abstains", unrelated["abstained"])
            case = store.update_segment("officer-a", case_id, case["segments"][0]["id"],
                                        "I left my blue bicycle beside the station at nine in the morning.", case["revision"])
            edited = rag.answer(store, "officer-a", case_id, "Where was the bicycle left?")
            check("edited source replaces stale vector evidence", not edited["abstained"] and "station" in edited["answer"].lower()
                  and all("library" not in s["text"] for s in edited["sources"]))
            with index.connect() as db:
                db.execute("DELETE FROM retrieval.chunks WHERE case_id=%s", (case_id,))
        report["passed"] = True
    finally:
        forward.terminate()
        forward.wait(timeout=10)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
