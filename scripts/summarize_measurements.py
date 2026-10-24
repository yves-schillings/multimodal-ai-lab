"""Print Markdown result tables from the measurement result files, so evidence READMEs never retype numbers.

    python scripts/summarize_measurements.py m3|m1|m2|ocr --dir <evidence-dir>
"""
import argparse
import json
from pathlib import Path


def load(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def seconds(s):
    return f"{s['median']:.2f} s | {s['p95']:.2f} s"


def m3(folder):
    stages = load(folder, "stages-summary.json")
    print("| Stage | n | Median | 95th percentile | Detail |\n| --- | --- | --- | --- | --- |")
    labels = {"transcription_fr": "Transcription, French clip (Whisper base)", "transcription_nl": "Transcription, Dutch clip (Whisper base)",
              "ocr_scanned_pdf": "OCR, scanned PDF page", "ocr_png": "OCR, PNG page", "embedding": "Question embedding (BGE-M3)",
              "vector_search": "Vector search, SQL only (pgvector)", "retrieval_full": "Retrieval call (embedding, search, checks)",
              "generation": "Answer generation (Qwen3:4b)"}
    for stage, row in stages.items():
        detail = ""
        if stage.startswith("transcription"):
            rtf = row["real_time_factor_per_clip"]
            detail = f"real-time factor median {rtf['median']:.2f}, p95 {rtf['p95']:.2f}, aggregate {row['real_time_factor_aggregate']:.2f}; model load {row['model_load_seconds']:.2f} s"
        if stage == "embedding":
            detail = f"model load median {row['ollama_load_seconds']['median']:.2f} s"
        if stage == "generation":
            detail = (f"model load median {row['ollama_load_seconds']['median']:.2f} s; {row['output_tokens_per_second']['median']:.1f} output tokens/s; "
                      f"outcomes {row['outcomes']}; withheld {row.get('withheld_reasons')}")
        print(f"| {labels.get(stage, stage)} | {row['wall_seconds']['n']} | {seconds(row['wall_seconds'])} | {detail} |")
    path = folder / "concurrency-raw.json"
    if path.exists():
        report = load(folder, "concurrency-raw.json")
        print("\n| Simultaneous users | Requests | HTTP 200 | Other statuses | Median (200) | p95 (200) | Answers per minute |\n| --- | --- | --- | --- | --- | --- | --- |")
        for users, level in report["levels"].items():
            ok = level["latency_seconds_successful"]
            others = {k: v for k, v in level["statuses"].items() if k != "200"}
            med = f"{ok['median']:.1f} s" if ok.get("n") else "-"
            p95 = f"{ok['p95']:.1f} s" if ok.get("n") else "-"
            answered = sum(1 for s in level["samples"] if s["status"] == 200 and s["abstained"] is False)
            print(f"| {users} | {level['requests']} | {level['succeeded']} ({answered} answered) | {others or '-'} | {med} | {p95} | {level['throughput_answers_per_minute']:.2f} |")


def m1(folder):
    r = load(folder, "injection-results.json")
    for name, criterion in r["criteria"].items():
        print(f"- {name}: {'PASS' if criterion['passed'] else 'FAIL'}")
    print(json.dumps({k: v for k, v in r["criteria"]["unsupported_refused_at_least_18_of_20"].items()}, indent=1))
    print(json.dumps(r["answerable_context"], indent=1))
    print("findings:", r["criteria"]["no_injection_succeeds"]["findings"])
    print("denials:", r["criteria"]["no_injection_succeeds"]["denials"])


def m2(folder):
    r = load(folder, "relevance-results.json")
    print("| Language | Questions | Method | Recall@3 (service view) | Recall@3 (raw) | Recall@10 | MRR@10 |\n| --- | --- | --- | --- | --- | --- | --- |")
    for language, g in r["by_language"].items():
        for method in ("vector", "lexical"):
            m = g[method]
            print(f"| {language} | {g['questions']} | {method} | {m['recall_at_3_service']:.3f} | {m['recall_at_3']:.3f} | {m['recall_at_10']:.3f} | {m['rr_at_10']:.3f} |")
    print("decision:", r["decision_bge_m3_stays_default"], "relevant-passage similarity:", r["vector_similarity_of_relevant_passages"])


def ocr(folder):
    r = load(folder, "ocr-results.json")
    d = r["diagnostic_three_languages"]["conditions"]
    print("| Condition | Category | Threshold CER | Median CER | Max CER | Documents passed | Diagnostic eng+fra+nld median CER |\n| --- | --- | --- | --- | --- | --- | --- |")
    for cid, c in r["conditions"].items():
        print(f"| {cid} | {c['category']} | {c['threshold_cer']:.2f} | {c['cer']['median']:.4f} | {c['cer']['max']:.4f} | {c['documents_passed']}/{c['documents']} | {d[cid]['median']:.4f} |")
    print("categories:", r["categories"], "diagnostic:", r["diagnostic_three_languages"]["categories"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", choices=["m3", "m1", "m2", "ocr"])
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.measurement](args.dir)


if __name__ == "__main__":
    main()
