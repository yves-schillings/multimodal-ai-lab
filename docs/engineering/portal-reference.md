# Engineering portal reference

The optional `/engineering` view exposes synthetic study actions through the existing application. It does not add an independent identity system, a production model endpoint or release authority. Source contracts are in [`engineering_api.py`](../../lab/engineering_api.py) and [`engineering.js`](../../static/engineering.js); the complete Python list is in the [function index](FUNCTION_INDEX.md).

## API models and configuration

`EngineeringQuestion` accepts only `question`, a non-empty string of at most 2,000 characters. Extra fields such as a model endpoint or another actor are rejected. `InvoiceExample` accepts `invoice=null` or a numeric mapping of at most six keys; the explanation module validates their names and ranges.

`_agentic_configuration(rag)` returns None unless `LAB_ENGINEERING_AGENTIC=enabled`. With an approved RAG service it uses that service's vector index and pinned inference client. Otherwise the operator must supply `LAB_AGENTIC_INFERENCE_URL`, `LAB_EMBEDDING_MODEL`, `LAB_EMBEDDING_DIGEST`, `LAB_GENERATION_MODEL` and `LAB_GENERATION_DIGEST`; that explicit profile uses the reviewed-only lexical adapter. `LAB_AGENTIC_CPU_ONLY` defaults to true. No model is pulled. An enabled CPE agent without an approved RAG route is refused at route installation.

`_vision_report(store)` reads a main report (up to 2 MiB, schema_version=1) from `LAB_ENGINEERING_VISION_REPORT` or `store.root/engineering/vision-report.json`. It selects listed top-level fields and adds the original file's SHA-256. An optional `LAB_ENGINEERING_VISION_LOAD_REPORT` must be at most 400,000 bytes with schema `vision-open-loop-v1`. These files are operator-selected; user requests cannot select arbitrary report paths. This helper does not recursively sanitize every nested field. Its optional-probe parser currently assumes a JSON object: valid JSON such as `[]` or `null` raises AttributeError rather than the structured 503 returned for an object with the wrong schema.

`install_engineering_routes(app, store, auth, cpe, rag)` installs the view/routes and one model lock. Its nested `active(actor)` returns the validated actor ID string; the bare store recognizes simulated actor IDs, while an authenticated wrapper can also reject disabled accounts. `reauthorize(request, actor, case_id)` builds a `check()` callback that repeats session/CSRF, fixed actor, CPE route and optional case checks. PermissionError maps to 403; CpeInactive is a separate runtime error mapped to 503. Those helpers work with the base middleware; store actor selection by itself is not authentication.

## Every HTTP route

| Handler | Method and route | Inputs and work | Result and failures |
| --- | --- | --- | --- |
| `engineering_page` | GET `/engineering` | Serves the fixed static HTML shell | FileResponse; displaying the shell does not authorize protected API calls |
| `engineering_status` | GET `/api/engineering/status` | Enabled actor; checks optional libraries and retained vision | Identity/capability flags, 3 searches/90 seconds, production_approved=false. Missing report becomes vision.available=false |
| `engineering_example` | POST `/api/engineering/example` | Authorized actor, ordinary session/CSRF controls | Creates a new fictional MX-7 case with two fixed reviewed documents; changes SQLite state. Returns case and synthetic-review notice |
| `agent_question` | POST `/api/cases/{case_id}/agent-question` | Fixed case, EngineeringQuestion, current request and actor | Cited result or abstention; repeated authorization before/after work. HTTP 503 for no profile, 429 for the occupied lock; permission/source conflicts propagate to base error handling |
| `explain_example` | POST `/api/engineering/explain` | Reviewer only, no current CPE; InvoiceExample | Synthetic model comparison, seed 1729, 800 rows, 4,000 LIME samples. HTTP 503 for missing libraries, 429 for busy lock; invalid inputs and revoked access are rejected |
| `retained_vision` | GET `/api/engineering/vision` | Enabled actor | Approved retained report fields; view=retained_measured_run and live_gpu_inference=false. HTTP 503 for unavailable evidence |
| `candidate_acceptance` | GET `/api/engineering/acceptance` | Enabled actor, operator-selected evidence directory | In-memory evidence gate. Reads `agentic-search-v3.json`, `vision-resnet-open-loop.json` and `explainability-comparison.json`, each at most 400,000 bytes. Missing/oversized file gives 503 |

The acceptance route does not automatically discover the 28 October comparison records. Use `scripts/gate_agentic_retrieval.py` for that exact profile. Its fourth case is a distinct blocking check. A route result does not update a registry, grant access or record a human release decision.

The model lock serializes agent and explanation work within one application process. It is not a distributed scheduler or project quota. It is released in `finally` even when work fails. Long-running actions recheck the session before returning. The current agent still checks exact quotations without proving object correspondence; the known wrong-bicycle result remains possible.

## Every browser procedure

All displayed model/source text is inserted with `textContent`; it is never treated as executable HTML. Server-side controls remain authoritative.

| Procedure | Parameters and result | Effects and error behavior |
| --- | --- | --- |
| `el(id)` | DOM ID string -> element or null | Reads the page. A missing required control can cause a later null-property error |
| `fmt(value, digits=3)` | Numeric/coercible scalar -> fixed decimal string, or Unavailable | No scientific calculation. Invalid digits can raise RangeError |
| `node(tag, text, className)` | Element name, optional text and CSS class -> new unattached element | Escapes text through textContent; caller inserts it |
| `message(text)` | Notice text -> void | Replaces/clears the message area's text |
| `api(path, method, body)` | Fixed route, method, optional JSON -> Promise of parsed response | Adds actor and tab CSRF header. Fetch, malformed JSON and HTTP errors propagate |
| `details(target, label, data)` | Parent, summary, serializable data -> void | Appends a details/pre panel; JSON serialization can fail |
| `table(target, headers, rows)` | Parent, labels and scalar rows -> void | Builds escaped TH/TD cells; no HTML supplied by the data is evaluated |
| `guarded(button, action)` | Optional button and zero-argument async action -> Promise<void> | Clears messages, disables/re-enables button, catches and displays errors |
| `refresh()` | No arguments -> Promise<void> | Fetches status/cases, retains a still-authorized selection, initializes numeric fields and controls |
| `shapBars(out, shap)` | Parent and computed SHAP report -> void | Displays signed log-odds shares; widths use absolute magnitudes, not percentage points |
| `renderExplanation(comparison)` | Comparison report -> void | Renders two model results, tree SHAP and the linear-model LIME diagnostic. Does not treat the methods as a same-model comparison |
| `explain(useInputs, button)` | Boolean edited/default choice and button -> Promise<void> | Reads numeric fields or sends invoice=null, calls the fixed explanation route, renders or shows error |
| `init()` | No arguments -> Promise<void> | Discovers identity mode. Shows login when a session or per-tab token is missing, otherwise refreshes capabilities |

`renderExplanation` uses the small UI diagnostic of an in-range LIME case output and an absolute case error at most 0.05. This is not the complete model acceptance gate or a guarantee that the explanation is useful. Failed raw outputs remain visible as study evidence.

## Every handler and local callback

| Handler/callback location | Argument | Responsibility |
| --- | --- | --- |
| `actor.onchange` and its guarded action | No argument is consumed | Set the simulated actor, refresh permitted cases and clear the previous answer |
| `login.onsubmit` | SubmitEvent | Cancel browser submission and launch the local login action |
| Login action | No arguments | Post username/password, clear password control, store returned tab CSRF, select server principal and refresh |
| `logout.onclick` and its guarded action | No argument is consumed | Revoke session, clear tab token and reload only after success |
| `sample.onclick` and its guarded action | No argument is consumed | Create/select the fixed fictional example and show its explicit notice |
| `search.onsubmit` | SubmitEvent | Cancel submission and launch the guarded case-question action |
| Question action | No arguments | Require a selected case, post the question, display answer/abstention, sources, stop reason and action trace |
| `vision.onclick` and its guarded action | No argument is consumed | Read retained evidence and display timing/precision/worker/count tables |
| Explanation action inside `explain` | No arguments | Collect optional inputs, call the explanation route and render the report |
| `explain-default.onclick` | No argument is consumed | Request the fixed default invoice |
| `explain.onsubmit` | SubmitEvent | Cancel submission and request the edited invoice |
| `acceptance.onclick` and its guarded action | No argument is consumed | Read the configured evidence gate and display decision, checks and hashes |
| Final `init().catch` | Error | Show a startup failure in the message area |
| `table`: header/row/cell iteration | Label, row array or scalar respectively | Create escaped table elements in their original order |
| `refresh`: case iterator / selection predicate | Case dictionary | Create options / decide whether the previous case remains selectable |
| Vision benchmark/probe row maps | One benchmark or configuration | Select display fields without recomputing percentiles |
| Vision percentile map | Existing metric key | Format one recorded latency in milliseconds |
| `shapBars`: magnitude map | SHAP contribution | Compute a display normalization magnitude |
| LIME contribution row map | Surrogate condition/weight | Render the condition and fitted local weight |
| Invoice input iterator | Numeric DOM input | Copy its name/value into the bounded request object |
| Gate check row map | Gate check dictionary | Render ID, status and recorded reason |

## Portal behavior tests

[`test_engineering_api.py`](../../tests/test_engineering_api.py) documents every fixture, helper and fake-agent method. The module uses temporary credentials and an in-process TestClient, not the user's accounts or a live service.

- `encoded` hashes a test-only passphrase once; `secured` writes isolated enabled accounts and restores patched environment settings; `login` obtains a test cookie and CSRF header.
- Session/CSRF/case tests reject spoofed actors and access to another actor's case.
- A scripted `RevokingAgent.answer` disables an account during work; the route must withhold its sentinel answer.
- Extra endpoint/model/identity body fields are rejected. Non-reviewers and unapproved CPE profiles cannot start explanation fitting.
- Retained vision tests verify the top-level allowlist and explicit non-live label.
- Missing agent configuration returns 503 rather than silently changing the requested service.

See the [runbook](RUNBOOK.md) for command lines, prerequisites, evidence filenames, stop/cleanup procedures and the separation between unit tests and real measurements.
