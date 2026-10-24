"""OCR quality on fictional typed documents, from clean scans to difficult ones, through the deployed service.

`generate` renders every fictional document of tests/fixtures/measurements/ocr_documents.json under each
scan condition with a fixed random seed. `run` sends the files to the deployed lab-app pod, extracts them
with the service function lab.documents.extract_document (pypdf, PDFium rendering, Tesseract) and scores
character and word error rates against the typed ground truth. Usage:

    python scripts/measure_ocr_quality.py generate --out <evidence-dir>
    python scripts/measure_ocr_quality.py run --out <evidence-dir>
"""
import argparse
import hashlib
import json
import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.measurement_common import now, pod_python, runtime_context, summary, write_json

FIXTURE = ROOT / "tests" / "fixtures" / "measurements" / "ocr_documents.json"
FONTS = Path("C:/Windows/Fonts")
BASE_DPI = 300

RUN_CODE = r'''
import json, subprocess, sys, time
from pathlib import Path
from lab.documents import extract_document
version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
langs = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
out = {"tesseract_version": (version.stdout or version.stderr).splitlines()[0],
       "tesseract_languages": [l for l in (langs.stdout or langs.stderr).splitlines()[1:] if l.strip()], "results": []}
for name in PARAMS["items"]:
    started = time.perf_counter()
    try:
        r = extract_document(Path(WORK) / name)
        out["results"].append({"file": name, "engine": r["engine"], "text": r["text"], "seconds": round(time.perf_counter() - started, 4)})
    except Exception as exc:
        out["results"].append({"file": name, "error": str(exc)[:200], "seconds": round(time.perf_counter() - started, 4)})
    print(name, file=sys.stderr, flush=True)
# Diagnostic only, not the service configuration: the same function with the installed French and Dutch data.
import pytesseract
original = pytesseract.image_to_string
pytesseract.image_to_string = lambda image, **kw: original(image, lang=PARAMS["diagnostic_lang"], **kw)
out["diagnostic"] = {"lang": PARAMS["diagnostic_lang"], "results": []}
for name in PARAMS["items"]:
    started = time.perf_counter()
    try:
        r = extract_document(Path(WORK) / name)
        out["diagnostic"]["results"].append({"file": name, "text": r["text"], "seconds": round(time.perf_counter() - started, 4)})
    except Exception as exc:
        out["diagnostic"]["results"].append({"file": name, "error": str(exc)[:200], "seconds": round(time.perf_counter() - started, 4)})
print(json.dumps(out, ensure_ascii=False))
'''


def normalise(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i]
        for j, h in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def error_rates(reference, hypothesis):
    ref, hyp = normalise(reference), normalise(hypothesis)
    return {"cer": round(edit_distance(ref, hyp) / len(ref), 4),
            "wer": round(edit_distance(ref.split(), hyp.split()) / len(ref.split()), 4),
            "reference_characters": len(ref), "reference_words": len(ref.split())}


def render(document, condition, seed):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    font = ImageFont.truetype(str(FONTS / document["font"]), round(document["points"] / 72 * BASE_DPI))
    paper, ink = condition.get("paper", 255), condition.get("ink", 0)
    line_height = round(font.size * 1.45)
    margin = round(0.6 * BASE_DPI)
    width = max(round(font.getlength(line)) for line in document["lines"]) + 2 * margin
    height = line_height * len(document["lines"]) + 2 * margin
    image = Image.new("L", (width, height), paper)
    draw = ImageDraw.Draw(image)
    for k, line in enumerate(document["lines"]):
        draw.text((margin, margin + k * line_height), line, font=font, fill=ink)
    if condition.get("rotate"):
        image = image.rotate(condition["rotate"], resample=Image.BICUBIC, expand=True, fillcolor=paper)
    dpi = condition["dpi"]
    if dpi != BASE_DPI:
        image = image.resize((round(image.width * dpi / BASE_DPI), round(image.height * dpi / BASE_DPI)), Image.LANCZOS)
    if condition.get("blur"):
        image = image.filter(ImageFilter.GaussianBlur(condition["blur"]))
    pixels = np.asarray(image, dtype=np.float32)
    if condition.get("shade"):
        ys, xs = np.mgrid[0:pixels.shape[0], 0:pixels.shape[1]]
        gradient = 1 - (1 - condition["shade"] / 255) * ((xs / pixels.shape[1] + ys / pixels.shape[0]) / 2)
        pixels = pixels * gradient
    if condition.get("binarize"):
        pixels = np.where(pixels > condition["binarize"], 255, 0).astype(np.float32)
    if condition.get("noise"):
        rng = np.random.default_rng(seed)
        mask = rng.random(pixels.shape) < condition["noise"]
        pixels[mask] = rng.choice([0, 255], size=int(mask.sum()))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "L")


def generate(args):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    inputs = args.out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    items = []
    for document in fixture["documents"]:
        for condition in fixture["conditions"]:
            seed = int(hashlib.sha256((document["id"] + condition["id"]).encode()).hexdigest()[:8], 16)
            image = render(document, condition, seed)
            name = f"{document['id']}__{condition['id']}.{condition['format']}"
            path = inputs / name
            if condition["format"] == "pdf":
                image.save(path, "PDF", resolution=condition["dpi"])
            elif condition["format"] == "jpg":
                image.save(path, "JPEG", quality=condition["jpeg_quality"], dpi=(condition["dpi"],) * 2)
            else:
                image.save(path, "PNG", dpi=(condition["dpi"],) * 2, optimize=True)
            items.append({"file": name, "document": document["id"], "language": document["language"], "condition": condition["id"],
                          "category": condition["category"], "pixels": [image.width, image.height], "seed": seed,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            print("generated", name, flush=True)
    import platform
    import numpy
    import PIL
    generator = {"python": platform.python_version(), "pillow": PIL.__version__, "numpy": numpy.__version__,
                 "fonts_sha256": {d["font"]: hashlib.sha256((FONTS / d["font"]).read_bytes()).hexdigest() for d in fixture["documents"]}}
    write_json(args.out, "ocr-manifest.json", {"generated_at": now(), "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                                               "generator": generator, "items": items, "latency_items": {"scanned_pdf": "D-EN-statement__scanned-pdf-300dpi.pdf",
                                                                                 "png": "D-EN-statement__clean-200dpi.png"}})


def run(args):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    manifest = json.loads((args.out / "ocr-manifest.json").read_text(encoding="utf-8"))
    truth = {d["id"]: "\n".join(d["lines"]) for d in fixture["documents"]}
    files = {item["file"]: (args.out / "inputs" / item["file"]).read_bytes() for item in manifest["items"]}
    for item in manifest["items"]:
        if hashlib.sha256(files[item["file"]]).hexdigest() != item["sha256"]:
            raise RuntimeError("Generated OCR input changed after its manifest was written: " + item["file"])
    write_json(args.out, "runtime-context.json", runtime_context())
    raw = pod_python(RUN_CODE, {"items": [item["file"] for item in manifest["items"]], "diagnostic_lang": "eng+fra+nld"}, files)
    write_json(args.out, "ocr-raw.json", raw)
    by_file = {r["file"]: r for r in raw["results"]}
    rows = []
    for item in manifest["items"]:
        result = by_file[item["file"]]
        row = {k: item[k] for k in ("file", "document", "language", "condition", "category")}
        row["seconds"] = result["seconds"]
        if "error" in result:
            row.update({"error": result["error"], "cer": 1.0, "wer": 1.0})
        else:
            row.update(error_rates(truth[item["document"]], result["text"]))
            row["engine"] = result["engine"]
        limit = fixture["thresholds"][item["category"]]
        row["threshold_cer"] = limit
        row["passed"] = row["cer"] <= limit
        rows.append(row)
    conditions = {}
    for condition in fixture["conditions"]:
        selected = [r for r in rows if r["condition"] == condition["id"]]
        conditions[condition["id"]] = {"category": condition["category"], "threshold_cer": fixture["thresholds"][condition["category"]],
                                       "cer": summary([r["cer"] for r in selected]), "wer": summary([r["wer"] for r in selected]),
                                       "documents_passed": sum(r["passed"] for r in selected), "documents": len(selected)}
    categories = {c: {"documents_passed": sum(r["passed"] for r in rows if r["category"] == c),
                      "documents": sum(1 for r in rows if r["category"] == c),
                      "cer": summary([r["cer"] for r in rows if r["category"] == c])} for c in ("clean", "difficult")}
    diagnostic = {}
    for result in raw["diagnostic"]["results"]:
        item = next(i for i in manifest["items"] if i["file"] == result["file"])
        rates = {"cer": 1.0, "wer": 1.0} if "error" in result else error_rates(truth[item["document"]], result["text"])
        diagnostic[result["file"]] = {"cer": rates["cer"], "wer": rates["wer"], "category": item["category"], "condition": item["condition"],
                                      "passed": rates["cer"] <= fixture["thresholds"][item["category"]]}
    diagnostic_summary = {"lang": raw["diagnostic"]["lang"], "scope": "Diagnostic only: the service calls Tesseract without a language, so English data only.",
                          "categories": {c: {"documents_passed": sum(d["passed"] for d in diagnostic.values() if d["category"] == c),
                                             "documents": sum(1 for d in diagnostic.values() if d["category"] == c),
                                             "cer": summary([d["cer"] for d in diagnostic.values() if d["category"] == c])} for c in ("clean", "difficult")},
                          "conditions": {c["id"]: summary([d["cer"] for d in diagnostic.values() if d["condition"] == c["id"]]) for c in fixture["conditions"]},
                          "documents": diagnostic}
    write_json(args.out, "ocr-results.json", {"recorded_at": now(), "tesseract_version": raw["tesseract_version"],
                                              "tesseract_languages": raw["tesseract_languages"], "thresholds": fixture["thresholds"],
                                              "categories": categories, "conditions": conditions, "documents": rows,
                                              "passed": all(r["passed"] for r in rows), "diagnostic_three_languages": diagnostic_summary})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "run"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args) if args.command == "generate" else run(args)


if __name__ == "__main__":
    main()
