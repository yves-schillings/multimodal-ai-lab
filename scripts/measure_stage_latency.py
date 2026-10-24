"""M3: latency per stage on the laboratory CPU and the cited-answer path with 1, 2 and 4 simultaneous users.

Stages run inside the deployed lab-app pod with the service code and configuration: faster-whisper
transcription, Tesseract OCR, BGE-M3 question embedding, pgvector retrieval and Qwen3:4b generation.
The cited-answer path runs through the authenticated HTTP API over a loopback port-forward.
Ollama timing fields are recorded beside each wall-clock time so model loading can be separated from
computation. Fictional inputs only. Usage:

    python scripts/measure_stage_latency.py stages --credentials <private.json> --manifest <manifest.json> --out <evidence-dir>
    python scripts/measure_stage_latency.py concurrency --credentials <private.json> --manifest <manifest.json> --out <evidence-dir>
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.measurement_common import LabApi, PortForward, NS, now, pod_python, runtime_context, summary, write_json

CORPUS = ROOT / "tests" / "fixtures" / "measurements" / "rag_corpus.json"
SPEECH = ROOT / "docs" / "evidence" / "2026-10-07-closing" / "speech-audio"
REPETITIONS = 30

STAGE_CODE = r'''
import json, os, sys, time
from pathlib import Path
P = PARAMS
def log(*a):
    print(*a, file=sys.stderr, flush=True)
out = {"stage": P["stage"], "samples": []}
stage = P["stage"]
if stage == "transcription":
    from lab.speech import SpeechEngine
    engine = SpeechEngine(model_dir=Path(os.environ.get("LAB_WHISPER_MODEL_DIR", "/app/models/whisper-base")))
    t = time.perf_counter(); engine._load_model(); out["model_load_seconds"] = round(time.perf_counter() - t, 4)
    for i, item in enumerate(P["clips"]):
        t = time.perf_counter()
        r = engine.transcribe(Path(WORK) / item["name"], item["language"])
        wall = time.perf_counter() - t
        out["samples"].append({"rep": i + 1, "clip": item["name"], "language": item["language"],
                               "audio_seconds": r["duration_seconds"], "processing_seconds": r["processing_seconds"],
                               "wall_seconds": round(wall, 4), "text": " ".join(s["text"] for s in r["segments"])})
        log(stage, i + 1, round(wall, 3))
elif stage.startswith("ocr"):
    from lab.documents import extract_document
    for i, name in enumerate(P["items"]):
        t = time.perf_counter()
        r = extract_document(Path(WORK) / name)
        wall = time.perf_counter() - t
        out["samples"].append({"rep": i + 1, "file": name, "engine": r["engine"], "characters": len(r["text"]),
                               "wall_seconds": round(wall, 4)})
        log(stage, i + 1, round(wall, 3))
else:
    from lab.rag import configured_rag
    from lab.backend_store import LabStore
    from lab.vector_retrieval import vector_literal
    rag = configured_rag()
    index, inference = rag.index, rag.index.inference
    store = LabStore(Path(os.environ.get("LAB_DATA_DIR", "/app/data")))
    timing, generated = {}, {}
    original = inference._request
    def recorded(method, path, payload=None):
        response = original(method, path, payload)
        if path in ("/api/embed", "/api/generate"):
            timing.clear()
            timing.update({k: response.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count",
                           "prompt_eval_duration", "eval_count", "eval_duration")})
        if path == "/api/generate":
            generated["response"] = response.get("response", "")
        return response
    def failure_kind(text):
        # Diagnostic only: why the service withheld the answer.
        try:
            parsed = json.loads(text)
        except ValueError:
            return "invalid_json"
        if isinstance(parsed, dict) and "evidence" in parsed and "abstained" not in parsed:
            return "prompt_echoed"
        if not isinstance(parsed, dict) or not isinstance(parsed.get("abstained"), bool):
            return "missing_abstention_decision"
        return "unsupported_citation_or_quote"
    inference._request = recorded
    case = store.get_case(P["actor"], P["case_id"])
    for i, q in enumerate(P["questions"]):
        sample = {"rep": i + 1, "question_id": q["id"], "language": q["language"]}
        if stage == "embedding":
            t = time.perf_counter(); inference.embed([q["text"]]); sample["wall_seconds"] = round(time.perf_counter() - t, 4)
            sample["ollama"] = dict(timing)
        elif stage == "vector_search":
            vector = vector_literal(inference.embed([q["text"]])[0])
            t = time.perf_counter()
            with index.connect() as db:
                rows = db.execute("""SELECT source_id, content_hash, 1-(embedding <=> %s::vector) AS score
                    FROM retrieval.chunks WHERE case_id=%s AND revision=%s AND model_digest=%s
                    ORDER BY embedding <=> %s::vector, source_id LIMIT 3""",
                    (vector, P["case_id"], case["revision"], inference.embedding_digest, vector)).fetchall()
            sample["wall_seconds"] = round(time.perf_counter() - t, 4)
            sample["rows"] = len(rows)
        elif stage == "retrieval_full":
            t = time.perf_counter(); r = index.retrieve(store, P["actor"], P["case_id"], q["text"])
            sample["wall_seconds"] = round(time.perf_counter() - t, 4)
            sample["sources"] = len(r["sources"]); sample["ollama_embed"] = dict(timing)
        elif stage == "generation":
            sources = index.retrieve(store, P["actor"], P["case_id"], q["text"])["sources"]
            sample["sources"] = len(sources)
            if not sources:
                sample["skipped"] = "retrieval abstained"
            else:
                t = time.perf_counter()
                try:
                    claims = inference.generate(q["text"], sources)
                    sample["outcome"] = "claims" if claims else "abstained"
                    sample["claims"] = len(claims)
                except Exception as exc:
                    sample["outcome"] = type(exc).__name__
                    sample["failure_kind"] = failure_kind(generated.get("response", ""))
                    sample["response_head"] = generated.get("response", "")[:160]
                sample["wall_seconds"] = round(time.perf_counter() - t, 4)
                sample["ollama"] = dict(timing)
        out["samples"].append(sample)
        log(stage, i + 1, sample.get("wall_seconds"))
print(json.dumps(out))
'''


def stage_questions(corpus):
    """Ten questions per language, in corpus order: the same 30 inputs for every model stage."""
    selected = []
    for language in ("en", "fr", "nl"):
        selected += [q for q in corpus["questions"] if q["language"] == language][:10]
    return selected


def run_stages(args):
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    case = manifest["cases"]["corpus_officer_a"]
    questions = [{"id": q["id"], "language": q["language"], "text": q["text"]} for q in stage_questions(corpus)]
    questions = [questions[i % len(questions)] for i in range(REPETITIONS)]
    wanted = set(args.stages or ["transcription", "ocr", "embedding", "vector_search", "retrieval_full", "generation"])
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out, "runtime-context.json", runtime_context())
    raw_path = args.out / "stages-raw.json"
    results = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
    clips = sorted(SPEECH.glob("*.wav"))
    for language in ("fr", "nl") if "transcription" in wanted else ():
        chosen = [c for c in clips if c.name.startswith(language)]
        order = [{"name": c.name, "language": language} for _ in range(3) for c in chosen][:REPETITIONS]
        results["transcription_" + language] = pod_python(STAGE_CODE, {"stage": "transcription", "clips": order},
                                                          {c.name: c.read_bytes() for c in chosen})
        write_json(args.out, "stages-raw.json", results)
    ocr = json.loads((args.ocr_dir / "ocr-manifest.json").read_text(encoding="utf-8"))
    for stage, name in (("ocr_scanned_pdf", ocr["latency_items"]["scanned_pdf"]), ("ocr_png", ocr["latency_items"]["png"])) if "ocr" in wanted else ():
        results[stage] = pod_python(STAGE_CODE, {"stage": stage, "items": [name] * REPETITIONS},
                                    {name: (args.ocr_dir / "inputs" / name).read_bytes()})
        write_json(args.out, "stages-raw.json", results)
    for stage in [s for s in ("embedding", "vector_search", "retrieval_full", "generation") if s in wanted]:
        results[stage] = pod_python(STAGE_CODE, {"stage": stage, "questions": questions, "case_id": case["case_id"],
                                                 "actor": case["owner"]})
        write_json(args.out, "stages-raw.json", results)
    write_json(args.out, "stages-summary.json", summarize_stages(results))


def summarize_stages(results):
    table = {}
    for stage, data in results.items():
        timed = [s for s in data["samples"] if "wall_seconds" in s]
        row = {"wall_seconds": summary([s["wall_seconds"] for s in timed])}
        if stage.startswith("transcription"):
            audio = sum(s["audio_seconds"] for s in timed)
            processing = sum(s["processing_seconds"] for s in timed)
            row["model_load_seconds"] = data.get("model_load_seconds")
            row["processing_seconds"] = summary([s["processing_seconds"] for s in timed])
            row["real_time_factor_per_clip"] = summary([s["processing_seconds"] / s["audio_seconds"] for s in timed])
            row["real_time_factor_aggregate"] = round(processing / audio, 4)
            row["audio_seconds_total"] = round(audio, 3)
        if stage in ("embedding", "generation"):
            ollama = [s["ollama"] for s in timed if s.get("ollama")]
            row["ollama_load_seconds"] = summary([o["load_duration"] / 1e9 for o in ollama if o.get("load_duration")])
            row["ollama_total_seconds"] = summary([o["total_duration"] / 1e9 for o in ollama if o.get("total_duration")])
            if stage == "generation":
                row["output_tokens_per_second"] = summary([o["eval_count"] / (o["eval_duration"] / 1e9) for o in ollama
                                                           if o.get("eval_count") and o.get("eval_duration")])
                row["prompt_tokens"] = summary([o["prompt_eval_count"] for o in ollama if o.get("prompt_eval_count")])
                row["outcomes"] = {k: sum(1 for s in timed if s.get("outcome") == k) for k in sorted({s.get("outcome") for s in timed})}
                row["withheld_reasons"] = {k: sum(1 for s in timed if s.get("failure_kind") == k)
                                           for k in sorted({s["failure_kind"] for s in timed if s.get("failure_kind")})}
                row["skipped"] = sum(1 for s in data["samples"] if s.get("skipped"))
        table[stage] = row
    return table


def run_concurrency(args):
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    questions = stage_questions(corpus)
    cases = {"officer-a": manifest["cases"]["corpus_officer_a"]["case_id"], "officer-b": manifest["cases"]["corpus_officer_b"]["case_id"]}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"recorded_at": now(), "levels": {}, "method": "Each simulated user holds its own authenticated session and sends its share of the 30 questions one after another; users 1 and 3 are officer-a, users 2 and 4 officer-b on an identical corpus case."}
    with PortForward(NS) as forward:
        # Warm-up: the first question per case indexes the 60 reviewed passages; it is recorded, not counted.
        warmup = {}
        for account, case_id in cases.items():
            api = LabApi(forward.base, args.credentials, account)
            started = time.perf_counter()
            response = api.request("POST", f"/api/cases/{case_id}/question", json={"question": questions[0]["text"]})
            warmup[account] = {"status": response.status_code, "seconds": round(time.perf_counter() - started, 3)}
            api.close()
        report["warmup_first_question_with_indexing"] = warmup
        for users in args.levels:
            per_user = -(-REPETITIONS // users)
            samples, lock = [], threading.Lock()
            apis = [LabApi(forward.base, args.credentials, "officer-a" if u % 2 == 0 else "officer-b") for u in range(users)]

            def work(u):
                api = apis[u]
                for k in range(per_user):
                    q = questions[(u * per_user + k) % len(questions)]
                    started = time.perf_counter()
                    try:
                        response = api.request("POST", f"/api/cases/{cases[api.account]}/question", json={"question": q["text"]})
                        status = response.status_code
                        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    except Exception as exc:
                        status, body = type(exc).__name__, {}
                    ended = time.perf_counter()
                    with lock:
                        samples.append({"user": u + 1, "account": api.account, "question_id": q["id"], "status": status,
                                        "abstained": body.get("abstained"), "claims": len(body.get("claims") or []),
                                        "started": round(started, 3), "seconds": round(ended - started, 3)})
                    print(f"users={users} user={u + 1} {q['id']} {status} {ended - started:.1f}s", flush=True)

            begin = time.perf_counter()
            threads = [threading.Thread(target=work, args=(u,)) for u in range(users)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.perf_counter() - begin
            for api in apis:
                api.close()
            ok = [s for s in samples if s["status"] == 200]
            report["levels"][str(users)] = {"requests": len(samples), "succeeded": len(ok), "elapsed_seconds": round(elapsed, 3),
                                            "throughput_answers_per_minute": round(60 * len(ok) / elapsed, 3),
                                            "latency_seconds_successful": summary([s["seconds"] for s in ok]),
                                            "latency_seconds_all": summary([s["seconds"] for s in samples]),
                                            "statuses": {str(k): sum(1 for s in samples if s["status"] == k) for k in {s["status"] for s in samples}},
                                            "samples": sorted(samples, key=lambda s: s["started"])}
            write_json(args.out, "concurrency-raw.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stages", "concurrency"])
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ocr-dir", type=Path, help="OCR evidence folder with ocr-manifest.json and inputs/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repetitions", type=int, default=REPETITIONS, help="30 for the measurement; fewer only for a dry run")
    parser.add_argument("--stages", nargs="+", help="Run only these stage groups: transcription ocr embedding vector_search retrieval_full generation")
    args = parser.parse_args()
    globals()["REPETITIONS"] = args.repetitions
    run_stages(args) if args.command == "stages" else run_concurrency(args)


if __name__ == "__main__":
    main()
