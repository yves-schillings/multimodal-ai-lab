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
    """Validate the bounded agent-question request body.

    Notes:
        question is a non-empty string of at most 2,000 characters.
        Additional request keys are rejected by Pydantic before the route runs.
        Identity, model names, endpoints and permissions cannot be supplied here.
    """
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)


class InvoiceExample(BaseModel):
    """Validate the optional six-field synthetic invoice request.

    Notes:
        invoice may be None to select the teaching example, or a mapping of at most
        six numeric values. The explanation module validates field names, ranges
        and binary flags. This schema does not accept real invoice documents.
    """
    model_config = ConfigDict(extra="forbid")
    invoice: dict[str, float] | None = Field(default=None, max_length=6)


MANUAL = (
    ("Fictional MX-7 printer consumables", "Fictional maintenance manual. The MX-7 label printer uses cartridge C42. The recommended replacement interval is 5,000 labels. This text is a synthetic engineering fixture."),
    ("Fictional MX-7 support codes", "Fictional support note. On the MX-7 label printer, code P17 means the paper tray is not latched. The operator closes the paper tray and presses Reset. This text is a synthetic engineering fixture."),
)


def _agentic_configuration(rag):
    """Construct the optional bounded agent from operator-owned configuration.

    Args:
        rag: Configured RAG service with index and pinned inference client, or None.

    Returns:
        AgenticSearch using the configured vector index or the explicit lexical
        profile; None when LAB_ENGINEERING_AGENTIC is not exactly 'enabled'.

    Raises:
        KeyError: A required LAB_AGENTIC_* or model-pin environment variable is absent.
        ValueError: LocalInference rejects an endpoint or model digest.

    Notes:
        Reads environment variables; constructs clients without pulling models.
        LAB_AGENTIC_CPU_ONLY defaults to true. The portal uses three searches and
        90 seconds; this construction is not evidence that an inference call works.
    """
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
    """Read bounded retained vision evidence and expose approved top-level fields.

    Args:
        store: StatementStore whose root supplies the default engineering report folder.

    Returns:
        Dictionary with selected report fields, SHA-256 of the original report,
        view='retained_measured_run' and live_gpu_inference=False. An optional
        vision-open-loop-v1 report is attached as performance_probe.

    Raises:
        HTTPException: 503 for an absent, oversized, invalid or unsupported main report, or an absent/oversized load probe or an object probe with an unsupported schema.
        OSError: A configured file cannot be read.
        ValueError: The optional load probe contains invalid JSON.
        AttributeError: The optional load probe is valid JSON but is not an object;
            its current parser calls get() without first checking the type.

    Notes:
        Reads LAB_ENGINEERING_VISION_REPORT (2 MiB maximum) and optionally
        LAB_ENGINEERING_VISION_LOAD_REPORT (400,000 bytes maximum). The allowlist
        is top-level only. Reading evidence never starts a model or GPU service.
    """
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
    """Register the optional study page and API routes on an existing FastAPI app.

    Args:
        app: FastAPI application with the base middleware and error mapping already installed.
        store: Authoritative StatementStore for actor, case, revision and reviewed-source checks.
        auth: LocalSessions instance, or None for the simulated loopback profile.
        cpe: Controlled-project policy object, or None when that profile is not active.
        rag: Approved configured RAG service, or None for the explicit lexical profile.

    Returns:
        None. Routes and app.state.engineering_agentic are installed in place.

    Raises:
        RuntimeError: An enabled CPE agent has no approved configured RAG/inference route.
        KeyError: Required enabled-agent configuration is missing.

    Notes:
        Creates one non-blocking lock shared by explanation fitting and agent
        inference. Base middleware remains responsible for HTTP authentication
        and policy enforcement; caller-selected actors alone are not authentication.
    """
    agentic = _agentic_configuration(rag)
    if cpe and agentic is not None and rag is None:
        raise RuntimeError("A CPE agent must use its approved configured RAG and inference gateway.")
    model_lock = threading.Lock()
    app.state.engineering_agentic = agentic

    def active(actor):
        """Validate an actor through the store's installed actor-checking profile.

        Args:
            actor: Server-resolved actor identifier, or simulated actor in loopback mode.

        Returns:
            Validated actor identifier string returned by StatementStore._actor.

        Raises:
            PermissionError: The actor is unknown, or an authenticated wrapper
                rejects its disabled account.

        Notes:
            The bare store checks registered simulated identifiers. Enabled-account
            checks belong to the authenticated wrapper. This helper is not a
            substitute for the base application's session check.
        """
        return store._actor(actor)

    def reauthorize(request, actor, case_id=None):
        """Build a callable that repeats the current request's access checks.

        Args:
            request: FastAPI Request supplying the session cookie, CSRF header and route.
            actor: Fixed authorized actor identifier that the callback must preserve.
            case_id: Optional fixed case identifier; None omits the case ownership check.

        Returns:
            Zero-argument check callback used before and after long-running work.

        Notes:
            The returned callback reads current session, credential and policy state
            on every call; building it does not authenticate a request by itself.
        """
        def check():
            """Reauthenticate and reauthorize the fixed request, actor and optional case.

            Returns:
                None when all enabled checks pass.

            Raises:
                PermissionError: Session, CSRF, actor or case access is no longer valid.
                CpeInactive: The controlled-project profile is inactive or expired;
                    the base application maps this separate runtime error to 503.
                KeyError: The selected case no longer exists.

            Notes:
                Closes over request, actor and case_id. In local-session mode it checks
                mutation=True, including CSRF. It then checks the actor, CPE route and
                case in order. No model is called and no authorization state is changed.
            """
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
        """Serve the static synthetic-study HTML page for GET /engineering.

        Returns:
            FileResponse for static/engineering.html relative to the repository.

        Notes:
            The page shell can be served before login; protected API calls still
            pass through the existing application middleware. No model work runs here.
        """
        return FileResponse(Path(__file__).resolve().parents[1] / "static" / "engineering.html")

    @app.get("/api/engineering/status")
    def engineering_status(actor: str):
        """Report optional study capabilities for an enabled actor.

        Args:
            actor: Actor identifier checked against the authoritative store.

        Returns:
            Capability dictionary describing identity mode, agent bounds, optional
            explanation libraries and retained vision availability; production_approved
            is always False.

        Raises:
            PermissionError: The actor is not enabled.

        Notes:
            Imports the explanation capability probe and may read a retained vision
            report. A 503 report condition becomes vision.available=False. Agent
            digests are validated on actual requests, not certified by this status call.
        """
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
        """Create and mark reviewed the fixed fictional MX-7 maintenance fixture.

        Args:
            actor: Authorized actor that will own the new synthetic case.

        Returns:
            Dictionary containing the completed case and a notice about its synthetic
            review status.

        Raises:
            PermissionError: The actor is not enabled or case access is denied.

        Notes:
            Writes one new case and two fixed documents to the local store, updating
            revisions through the regular review workflow. Each invocation creates a
            new example. No uploaded or real document is automatically approved.
        """
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
        """Run one bounded question against the fixed authorized case.

        Args:
            case_id: Path-selected case identifier, checked before and after inference.
            body: Validated EngineeringQuestion containing the original question.
            request: Request whose session, CSRF token and CPE route can be rechecked.
            actor: Authorized actor selected by the base application controls.

        Returns:
            Agent result with cited claims or an explicit abstention and stop reason.

        Raises:
            HTTPException: 503 when no agent is configured, or 429 while the shared model lock is held.
            PermissionError: Authorization is invalid or is revoked during work.
            ConflictError: The case or reviewed sources change during processing.

        Notes:
            May call the configured local model and retriever. Always releases the
            shared model lock. Exact-quote checks do not establish that a claim concerns
            the correct object; the known two-step defect remains uncorrected.
        """
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
        """Fit and explain the two synthetic invoice models for the reviewer.

        Args:
            body: InvoiceExample with optional numeric teaching inputs; None selects defaults.
            request: Request whose current authorization is checked around fitting.
            actor: Must be the authorized reviewer account.

        Returns:
            Comparison report from 800 synthetic rows, seed 1729 and 4,000 LIME
            samples, including raw explanation diagnostics and limitations.

        Raises:
            PermissionError: Actor is not reviewer, current access fails, or a CPE profile is active.
            HTTPException: 503 for missing optional dependencies or 429 for an occupied model lock.
            ValueError: The explanation module rejects invoice fields or values.

        Notes:
            Performs CPU fitting/explanation under the shared lock and rechecks
            access before returning. A numerical explanation check is not a business
            acceptance or permission to use the model on real data.
        """
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
        """Return the configured retained vision report to an enabled actor.

        Args:
            actor: Authorized actor identifier checked by active().

        Returns:
            Bounded selected evidence fields from _vision_report.

        Raises:
            PermissionError: The actor is disabled or unknown.
            HTTPException: 503 when the retained measurement profile is unavailable.

        Notes:
            Reads evidence only; this GET endpoint does not start live image inference.
        """
        active(actor)
        return _vision_report(store)

    @app.get("/api/engineering/acceptance")
    def candidate_acceptance(actor: str):
        """Evaluate the three configured retained evidence files without approving release.

        Args:
            actor: Enabled actor identifier authorized by the base application.

        Returns:
            Technical evidence-gate report with pass/fail/review_pending checks,
            content hashes and separate human-approval requirements.

        Raises:
            PermissionError: The actor is unknown or disabled.
            HTTPException: 503 if a required file is absent or exceeds 400,000 bytes.
            ValueError: A report contains invalid JSON or an invalid evidence manifest.
            OSError: A configured file cannot be read.

        Notes:
            Reads LAB_ENGINEERING_EVIDENCE_DIR or store.root/engineering. Filenames
            remain agentic-search-v3.json, vision-resnet-open-loop.json and
            explainability-comparison.json. This route does not automatically load the
            28 October comparison: use the separate gate CLI for that recorded profile.
            No registry state, human decision or production release is written.
        """
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
