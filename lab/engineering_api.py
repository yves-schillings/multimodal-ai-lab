"""Small optional AI Engineering routes over the existing lab controls.

The portal runs synthetic examples. A retained GPU benchmark is not a live GPU
deployment. Existing session, CSRF, case and CPE checks remain authoritative.
"""
import hashlib
import json
import os
import threading
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .auth import COOKIE
from .local_inference import LocalInference


class EngineeringQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)


class InvoiceExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invoice: dict[str, float] | None = Field(default=None, max_length=6)


MANUAL = (
    ("Fictional MX-7 printer consumables", "Fictional maintenance manual. The MX-7 label printer uses cartridge C42. The recommended replacement interval is 5,000 labels. This text is a synthetic engineering fixture."),
    ("Fictional MX-7 support codes", "Fictional support note. On the MX-7 label printer, code P17 means the paper tray is not latched. The operator closes the paper tray and presses Reset. This text is a synthetic engineering fixture."),
)


def _agentic_configuration(rag):
    if os.environ.get("LAB_ENGINEERING_AGENTIC", "disabled") != "enabled":
        return None
    from .agentic_search import AgenticSearch, ReviewedLexicalRetriever
    if rag is not None:
        retriever, inference = rag.index, rag.index.inference
    else:
        # Explicit operator pins are required. Requests cannot choose models,
        # endpoints, identities or tools, and no model is downloaded here.
        inference = LocalInference(
            os.environ["LAB_AGENTIC_INFERENCE_URL"],
            os.environ["LAB_EMBEDDING_MODEL"], os.environ["LAB_EMBEDDING_DIGEST"],
            os.environ["LAB_GENERATION_MODEL"], os.environ["LAB_GENERATION_DIGEST"],
        )
        retriever = ReviewedLexicalRetriever()
    return AgenticSearch(retriever, inference, max_attempts=3, max_seconds=90,
                         cpu_only=os.environ.get("LAB_AGENTIC_CPU_ONLY", "true").lower() == "true")


def _vision_report(store):
    report_path = Path(os.environ.get("LAB_ENGINEERING_VISION_REPORT", store.root / "engineering" / "vision-report.json"))
    if not report_path.is_file():
        raise HTTPException(503, "No retained vision run is configured for this profile.")
    if report_path.stat().st_size > 2 * 1024 * 1024:
        raise HTTPException(503, "The retained vision report exceeds its bounded size.")
    raw = report_path.read_bytes()
    try:
        report = json.loads(raw)
    except (ValueError, UnicodeError):
        raise HTTPException(503, "The retained vision report is invalid.") from None
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise HTTPException(503, "The retained vision report has an unsupported schema.")
    allowed = ("scope", "runtime", "dataset", "training", "model", "quality", "benchmarks",
               "http_benchmarks", "cache_demonstration", "gpu_memory", "cpu_cuda_comparison",
               "timing_protocol", "synthetic_acceptance", "limitations", "status", "load_benchmarks",
               "performance_probe", "load_protocol")
    result = {**{key: report[key] for key in allowed if key in report},
              "report_sha256": hashlib.sha256(raw).hexdigest(),
              "view": "retained_measured_run", "live_gpu_inference": False}
    load_path = os.environ.get("LAB_ENGINEERING_VISION_LOAD_REPORT")
    if load_path:
        path = Path(load_path)
        if not path.is_file() or path.stat().st_size > 400_000:
            raise HTTPException(503, "The retained concurrent vision report is unavailable.")
        probe = json.loads(path.read_text(encoding="utf-8"))
        if probe.get("schema_version") != "vision-open-loop-v1":
            raise HTTPException(503, "The concurrent vision report has an unsupported schema.")
        result["performance_probe"] = probe
    return result


def install_engineering_routes(app, store, auth, cpe, rag):
    """Attach one view and bounded routes without replacing the base application."""
    agentic = _agentic_configuration(rag)
    if cpe and agentic is not None and rag is None:
        raise RuntimeError("A CPE agent must use its approved configured RAG and inference gateway.")
    model_lock = threading.Lock()
    app.state.engineering_agentic = agentic

    def active(actor):
        return store._actor(actor)

    def reauthorize(request, actor, case_id=None):
        def check():
            if auth:
                principal = auth.authenticate(request.cookies.get(COOKIE), request.headers.get("X-Lab-CSRF"), mutation=True)
                if principal["id"] != actor:
                    raise PermissionError("The current account differs from the authorized actor.")
            active(actor)
            if cpe:
                cpe.authorize(actor, request.url.path)
            if case_id is not None:
                store.get_case(actor, case_id)
        return check

    @app.get("/engineering")
    def engineering_page():
        return FileResponse(Path(__file__).resolve().parents[1] / "static" / "engineering.html")

    @app.get("/api/engineering/status")
    def engineering_status(actor: str):
        active(actor)
        from .engineering_explainability import explainability_status
        xai = explainability_status()
        try:
            measured = _vision_report(store)
            vision = {"available": True, "view": measured["view"], "report_sha256": measured["report_sha256"]}
        except HTTPException:
            vision = {"available": False, "view": "no_retained_run"}
        return {"synthetic_only": True, "production_approved": False,
                "identity_mode": "local_password_session" if auth else "simulated_loopback",
                "agentic_search": {"configured": agentic is not None, "max_attempts": 3,
                                   "maximum_seconds": 90, "model_validated_on_request": True,
                                   "source_scope": "reviewed sources in one authorized case"},
                "explainability": xai, "can_run_explainability": actor == "reviewer" and cpe is None,
                "vision": vision}

    @app.post("/api/engineering/example")
    def engineering_example(actor: str):
        active(actor)
        case = store.create_case(actor, "Fictional MX-7 maintenance example")
        for title, text in MANUAL:
            item = store.add_document(actor, case["id"], title,
                                      {"text": text, "engine": "synthetic_engineering_fixture"},
                                      {"label": "synthetic_only"})
            case = store.get_case(actor, case["id"])
            case = store.review_document(actor, case["id"], item["document_id"], text, case["revision"])
        return {"case": case, "notice": "The selected fixed fictional fixture was loaded and marked reviewed by your explicit request. This is not a real document review."}

    @app.post("/api/cases/{case_id}/agent-question")
    def agent_question(case_id: str, body: EngineeringQuestion, request: Request, actor: str):
        check = reauthorize(request, actor, case_id)
        check()
        if agentic is None:
            raise HTTPException(503, "The bounded agentic model profile is not configured.")
        if not model_lock.acquire(blocking=False):
            raise HTTPException(429, "Another local engineering model request is running.")
        try:
            result = agentic.answer(store, actor, case_id, body.question, reauthorize=check)
            check()
            return result
        finally:
            model_lock.release()

    @app.post("/api/engineering/explain")
    def explain_example(body: InvoiceExample, request: Request, actor: str):
        check = reauthorize(request, actor)
        check()
        if actor != "reviewer":
            raise PermissionError("The reviewer account runs this synthetic model study.")
        if cpe:
            raise PermissionError("The current CPE profile does not approve this model-fitting study.")
        from .engineering_explainability import explainability_status, run_explainability_comparison
        if not explainability_status()["available"]:
            raise HTTPException(503, "This profile lacks the optional explanation dependencies.")
        if not model_lock.acquire(blocking=False):
            raise HTTPException(429, "Another local engineering model request is running.")
        try:
            result = run_explainability_comparison(body.invoice, seed=1729, n_samples=800, lime_samples=4000)
            check()
            return result
        finally:
            model_lock.release()

    @app.get("/api/engineering/vision")
    def retained_vision(actor: str):
        active(actor)
        return _vision_report(store)

    @app.get("/api/engineering/acceptance")
    def candidate_acceptance(actor: str):
        active(actor)
        from .engineering_acceptance import evaluate_candidate_evidence
        evidence = Path(os.environ.get("LAB_ENGINEERING_EVIDENCE_DIR", store.root / "engineering"))
        manifest = {}
        for name, filename in (("rag", "agentic-search-v3.json"), ("vision", "vision-resnet-open-loop.json"),
                               ("explainability", "explainability-comparison.json")):
            path = evidence / filename
            if not path.is_file() or path.stat().st_size > 400_000:
                raise HTTPException(503, "The bounded candidate evidence set is unavailable.")
            manifest[name] = json.loads(path.read_text(encoding="utf-8"))
        return evaluate_candidate_evidence(manifest)
