# Search, local inference and comparison reference

This reference documents the implemented prototype. It does not introduce a
product change or claim that its generated answers are factually verified.
For operational prerequisites, environment preparation, exact commands and
safe output locations, use the [engineering procedures and runbook](README.md).
The source links below point to the corresponding definitions and full contracts.

## What runs, and where the boundaries are

| Function | Implementation | What it controls |
| --- | --- | --- |
| Retrieve by weighted words | `ReviewedLexicalRetriever` | Fits TF-IDF on reviewed case passages at every search, keeps up to three lexical matches |
| Encode passage/query text | `LocalInference.embed` and pinned BGE-M3 | Returns one finite, nonzero vector with 1.024 components per text |
| Persist and rank vectors | `VectorIndex` and PostgreSQL/pgvector | Stores IDs, hashes, revisions, model digests and vectors; text is reloaded from the case store |
| Choose the next action | `LocalInference.plan_search` and Qwen3:4b | Proposes `retrieve`, `answer` or `abstain`, within a constrained JSON schema |
| Write claims | `LocalInference.generate` and the same Qwen3:4b | Returns one to three claims with numbered source and verbatim quote |
| Enforce the search loop | `AgenticSearch` | Owns actor/case/revision, reauthorisation, bounded evidence and stop rules |
| Run classic RAG once | `LocalRag` | One retrieval and one generation, with a final authoritative source check |
| Evaluate recorded runs | `gate_agentic_retrieval.evaluate` | Adapts the recorded trial verdicts and evaluates the evidence gate without rerunning models |

One language model in planner and generator roles remains one agent. The model
cannot replace the caller's actor, case or retriever. Ollama is the local model
runtime. BGE-M3 is an embedding model; pgvector is the vector storage/search
extension, not the answer-writing model.

### Preparation and request paths

1. The caller authorises a case and obtains its human-reviewed text.
2. `reviewed_sources` bounds and labels the passages; `VectorIndex.synchronize`
   embeds and persists them when their revision/model/hash manifest differs.
3. Each question/reformulation is embedded and compared with vectors of that
   case, revision and model. Returned text is reloaded from the authoritative store.
4. The planner proposes a step; the controller checks permissions, budgets and
   source integrity before continuing. A generator call uses the original question.
5. Exact quotes and source numbers are checked before a result is returned.
   These checks do **not** establish semantic support or a correct entity join.

For the explicit lexical path, steps 2 and 3 instead fit and query a local TF-IDF
representation; no embedding or PostgreSQL call occurs. The adapter reports
count-cosine explicitly when TF-IDF is unavailable; pgvector failures never
silently select lexical retrieval.

## Runtime and evidence contracts

- **Reviewed source bounds.** At most 128 source chunks per case. Transcript
  text is truncated to 1.500 characters per segment; reviewed documents are
  split into non-overlapping chunks of that size. Truncation can omit evidence.
- **Query bounds.** At most 2.000 characters. The vector query keeps up to three
  rows with similarity `1 - cosine distance >= 0.45`. Lexical retrieval needs a
  shared token and score at least `0.10`, also capped at three.
- **Selected evidence.** At most three chunks and 4.500 combined characters.
  New hits take priority over old hits, so a previously found identifier anchor
  can leave the window. There is no guaranteed multi-hop evidence preservation.
- **Agent limits.** One to three searches and a configured elapsed-time bound.
  The recorded scripts use 90 seconds/question. Budget-aware generation calls
  are capped at 4.096 context tokens and 384 generated tokens per call, with
  `num_gpu=0` in the CPU experiments and `keep_alive=0`.
- **Deadline meaning.** Late output is withheld and HTTP phase timeouts are
  bounded by remaining time. This does not cancel computation already executing
  on the model server or an in-flight database statement.
- **Classic RAG difference.** `LocalRag.answer` has no iterative search loop or
  independently created end-to-end agent budget; an outer request context may
  supply a budget. It is distinct from `AgenticSearch.answer`.
- **Authorisation.** The case store checks access before/after retrieval. The
  agent also accepts an existing session-revocation callback. Synthetic actors
  used in recorded comparisons are not production authentication or OpenFGA.
- **Stored data.** The vector table contains vectors and provenance, not a copy
  of passage text. Embeddings are data and are not anonymisation. Database
  permissions, network security and retention need their own controls.
- **Synchronisation limits.** Direct `synchronize` calls trust an authorised
  snapshot from their caller. A matching empty manifest can leave older-revision
  rows stored; retrieval still excludes those rows by its current-revision filter.
- **Secrets.** `vector_options` and `configured_rag` read a password file. Keep
  it outside the repository and never print the returned connection dictionary.
  The named inference gateway additionally reads its bearer-token file.

## Procedures: which command does what

The [runbook](README.md) owns safe executable procedures and prerequisite checks.
The following CLI entry points expose their arguments without running a trial:

```powershell
python -B scripts/measure_agentic_engineering.py --help
python -B scripts/compare_agentic_retrieval.py --help
python -B scripts/gate_agentic_retrieval.py --help
```

| Procedure | Entry point and required inputs | Side effects and evidence |
| --- | --- | --- |
| Historical lexical demonstration | `measure_agentic_engineering.py --out <new-file> --max-seconds 90` | Real local CPU Qwen calls; temporary synthetic store; incremental JSON and exit code 0 only if all three historical rules pass |
| Compare one backend | `compare_agentic_retrieval.py --backend lexical|pgvector --protocol <protocol.json> --out <new-file>` | Real CPU model calls; pgvector mode additionally embeds, writes rows and runs synthetic guards; output parent must already exist |
| Retain fixtures for persistence | Add `--fixture-dir <private-store>` to the pgvector comparison | Keeps the synthetic case store needed to compare row/source hashes later |
| Inspect persistence | `compare_agentic_retrieval.py --persistence <run.json> --fixture-dir <same-store> --out <new-file>` | Does not restart anything; checks rows and reuse after an externally managed restart, then deletes the run's named synthetic index rows |
| Gate recorded evidence | `gate_agentic_retrieval.py --run <run.json> --vision <vision.json> --explainability <comparison.json> --out <new-file>` | Reads three existing records, computes a new gate and input hashes; no model, cluster or performance rerun |

The newer comparison gate is a **CLI procedure**. It does not automatically
replace the fixed historical input files read by `/api/engineering/acceptance`.
Use distinct new output names and retain failed trials; never rewrite historical
records to make them look like results from a different retriever or model.
The persistence command is destructive only to the explicitly listed synthetic
rows, so verify the run's case IDs and use the fixture index/database scope.

## Known limits: exact quotes and answer correctness

The implemented claim schema has one source/quote pair per claim. It verifies
that the quoted text exists; it does not verify that the text entails the claim,
that the claim identifies the correct bicycle, or that two passages join on the
same frame number. The recorded fourth comparison case demonstrates this defect.
A statement about another frame number can pass the citation check.

`compare_agentic_retrieval.judge` is deliberately narrow: the expected word must
occur in the answer and its quote/revision/attempt checks must pass. A negated
answer or one containing an extra false location can satisfy that substring
rule. Expected passages are separately used for retrieval-only diagnostics;
they are not independently required by this answer oracle.

The fourth case has its own `multi_step` gate requirement. Both recorded
backends remain blocked. Adding that gate does not repair the controller or
create an independent semantic verifier. Pure gate evaluation trusts stored
trial verdicts, and human/domain review remains separate from technical status.
Four English synthetic questions and single CPU timings are not general answer
quality, multilingual acceptance, throughput capacity or production readiness.

## Per-definition API index

Every class, method, nested helper and module has an English source docstring.
The links open the complete parameter/return/error/side-effect contracts.
Lambdas are documented with inline comments at their call sites.

### `lab/agentic_search.py`

| Definition | Purpose |
| --- | --- |
| [`_manifest`](../../lab/agentic_search.py#L26) | Return the reviewed-source identity/hash mapping for one case snapshot. |
| [`ReviewedLexicalRetriever`](../../lab/agentic_search.py#L40) | Explicit reviewed-only lexical retrieval adapter for bounded case searches. |
| [`ReviewedLexicalRetriever.retrieve`](../../lab/agentic_search.py#L50) | Rank reviewed passages for an authorised case and recheck its snapshot. |
| [`AgenticSearch`](../../lab/agentic_search.py#L101) | Orchestrate one agent while retaining control outside the language model. |
| [`AgenticSearch.__init__`](../../lab/agentic_search.py#L110) | Store dependencies and validate the search/time bounds; perform no I/O. |
| [`AgenticSearch.answer`](../../lab/agentic_search.py#L134) | Search a case, ask for bounded actions and return a cited answer or stop. |
| [`AgenticSearch.answer.authorization`](../../lab/agentic_search.py#L168) | Invoke the optional session guard and reject a revoked/changed actor. |
| [`AgenticSearch.answer.checkpoint`](../../lab/agentic_search.py#L185) | Reauthorise and compare current case evidence with the initial snapshot. |
| [`AgenticSearch.answer.validate_sources`](../../lab/agentic_search.py#L221) | Validate a retriever payload against the initial authoritative manifest. |
| [`AgenticSearch.answer.stop`](../../lab/agentic_search.py#L255) | Finalise a stop result after checking access and evidence once more. |

### `lab/local_inference.py`

| Definition | Purpose |
| --- | --- |
| [`InferenceUnavailable`](../../lab/local_inference.py#L29) | Signal missing/invalid local inference or output that fails its contract. |
| [`InferenceDeadlineExceeded`](../../lab/local_inference.py#L39) | Report an expired request budget so callers withhold late model output. |
| [`InvalidSearchPlan`](../../lab/local_inference.py#L47) | Report that a model action fails the small permitted search-plan contract. |
| [`validate_search_plan`](../../lab/local_inference.py#L65) | Validate and normalise the sole model-controlled search action object. |
| [`LocalInference`](../../lab/local_inference.py#L95) | Client for pinned local embedding, planning and cited-answer generation. |
| [`LocalInference.__init__`](../../lab/local_inference.py#L103) | Validate local endpoint/model pins and retain client configuration. |
| [`LocalInference.request_budget`](../../lab/local_inference.py#L145) | Temporarily bind deadline, guard and per-call options to this context. |
| [`LocalInference._request`](../../lab/local_inference.py#L186) | Perform one guarded local HTTP request and parse its JSON response. |
| [`LocalInference._request.checkpoint`](../../lab/local_inference.py#L207) | Run current guards and return the allowed HTTP phase timeout in seconds. |
| [`LocalInference.verify`](../../lab/local_inference.py#L249) | Check that a local Ollama tag still has its configured digest. |
| [`LocalInference.embed`](../../lab/local_inference.py#L268) | Encode a bounded list of texts with the pinned embedding model. |
| [`LocalInference.generate`](../../lab/local_inference.py#L295) | Generate and structurally validate up to three cited factual claims. |
| [`LocalInference.plan_search`](../../lab/local_inference.py#L356) | Ask the pinned model for the next permitted action after retrieval. |

### `lab/vector_retrieval.py`

| Definition | Purpose |
| --- | --- |
| [`reviewed_sources`](../../lab/vector_retrieval.py#L16) | Build bounded source dictionaries from reviewed case segments/documents. |
| [`source_hash`](../../lab/vector_retrieval.py#L47) | Hash one source object's full JSON representation for integrity comparison. |
| [`vector_literal`](../../lab/vector_retrieval.py#L62) | Serialise one finite 1,024-dimensional vector for a bound SQL parameter. |
| [`VectorIndex`](../../lab/vector_retrieval.py#L78) | Persist/reuse reviewed-source embeddings and retrieve authorised passages. |
| [`VectorIndex.__init__`](../../lab/vector_retrieval.py#L87) | Retain database connection options and the pinned embedding client. |
| [`VectorIndex.connect`](../../lab/vector_retrieval.py#L99) | Open a psycopg connection with bounded connection/query timeouts. |
| [`VectorIndex.synchronize`](../../lab/vector_retrieval.py#L114) | Reuse an unchanged manifest or atomically replace this model's case rows. |
| [`VectorIndex.retrieve`](../../lab/vector_retrieval.py#L154) | Search one authorised case and return sources reloaded from its store. |

### `lab/rag.py`

| Definition | Purpose |
| --- | --- |
| [`configured_rag`](../../lab/rag.py#L21) | Build the explicitly selected local RAG dependencies from environment settings. |
| [`LocalRag`](../../lab/rag.py#L54) | One-search generative RAG facade over a trusted VectorIndex dependency. |
| [`LocalRag.__init__`](../../lab/rag.py#L63) | Retain a vector index and its inference client without calling either. |
| [`LocalRag.answer`](../../lab/rag.py#L73) | Retrieve once, optionally generate claims, then recheck the case snapshot. |

### `scripts/measure_agentic_engineering.py`

| Definition | Purpose |
| --- | --- |
| [`measure`](../../scripts/measure_agentic_engineering.py#L40) | Execute and record the three historical reviewed-only lexical trials. |

### `scripts/compare_agentic_retrieval.py`

| Definition | Purpose |
| --- | --- |
| [`TimedInference`](../../scripts/compare_agentic_retrieval.py#L75) | Transparent model-client wrapper that records elapsed milliseconds per call. |
| [`TimedInference.__init__`](../../scripts/compare_agentic_retrieval.py#L82) | Retain the client and initialise an empty in-memory call log. |
| [`TimedInference.__getattr__`](../../scripts/compare_agentic_retrieval.py#L90) | Delegate an otherwise missing attribute to the original inference client. |
| [`TimedInference._timed`](../../scripts/compare_agentic_retrieval.py#L101) | Call one client method and always append its elapsed milliseconds. |
| [`TimedInference.embed`](../../scripts/compare_agentic_retrieval.py#L122) | Time the wrapped embedding call for a text batch, preserving its contract. |
| [`TimedInference.plan_search`](../../scripts/compare_agentic_retrieval.py#L133) | Time and delegate planning arguments without changing model decisions. |
| [`TimedInference.generate`](../../scripts/compare_agentic_retrieval.py#L144) | Time and delegate cited-answer generation without validating it again. |
| [`TimedRetriever`](../../scripts/compare_agentic_retrieval.py#L156) | Transparent retriever wrapper recording ranked returned passages and timing. |
| [`TimedRetriever.__init__`](../../scripts/compare_agentic_retrieval.py#L159) | Retain a retriever, optional inference client and after-search callback. |
| [`TimedRetriever.retrieve`](../../scripts/compare_agentic_retrieval.py#L172) | Delegate one search, log its returned passages and invoke the callback. |
| [`sha256`](../../scripts/compare_agentic_retrieval.py#L197) | Return SHA-256 of the exact bytes of one local file. |
| [`environment`](../../scripts/compare_agentic_retrieval.py#L202) | Collect local reproducibility metadata without exposing runtime secrets. |
| [`environment.run`](../../scripts/compare_agentic_retrieval.py#L213) | Run one metadata command and return stripped standard output. |
| [`vector_options`](../../scripts/compare_agentic_retrieval.py#L233) | Read private PostgreSQL connection settings for the local vector run. |
| [`cpu_budget`](../../scripts/compare_agentic_retrieval.py#L248) | Return a context manager requesting CPU-only bounded inference. |
| [`review_all`](../../scripts/compare_agentic_retrieval.py#L261) | Mark every current transcript segment reviewed through the real store API. |
| [`new_case`](../../scripts/compare_agentic_retrieval.py#L276) | Create a synthetic transcript case and optionally mark its segments reviewed. |
| [`lexical_ranking`](../../scripts/compare_agentic_retrieval.py#L295) | Rank all reviewed sources using the historical lexical scoring rules. |
| [`vector_ranking`](../../scripts/compare_agentic_retrieval.py#L324) | Rank all current indexed case rows with BGE-M3 cosine similarity. |
| [`expected_summary`](../../scripts/compare_agentic_retrieval.py#L348) | Look up each expected passage text in a diagnostic ranking. |
| [`quotes_checked`](../../scripts/compare_agentic_retrieval.py#L365) | Check claim source numbers and verbatim quote membership in a result. |
| [`stop_kind`](../../scripts/compare_agentic_retrieval.py#L381) | Map one stop-reason string to its actor/category for reporting. |
| [`judge`](../../scripts/compare_agentic_retrieval.py#L391) | Apply the protocol's narrow fixture rules to one recorded outcome. |
| [`run_trial`](../../scripts/compare_agentic_retrieval.py#L427) | Execute one bounded real-agent question and record steps/searches/verdicts. |
| [`guard_trials`](../../scripts/compare_agentic_retrieval.py#L459) | Run synthetic access, source-change, revocation and reuse checks. |
| [`guard_trials.record`](../../scripts/compare_agentic_retrieval.py#L482) | Append and print one boolean guard observation with structured details. |
| [`indexed_rows`](../../scripts/compare_agentic_retrieval.py#L549) | Read vector-row provenance metadata for selected synthetic cases. |
| [`measure`](../../scripts/compare_agentic_retrieval.py#L567) | Run one protocol/backend comparison and preserve all results in fresh JSON. |
| [`persistence`](../../scripts/compare_agentic_retrieval.py#L655) | Inspect a retained vector run after an externally managed PostgreSQL restart. |

### `scripts/gate_agentic_retrieval.py`

| Definition | Purpose |
| --- | --- |
| [`gate_trial`](../../scripts/gate_agentic_retrieval.py#L35) | Adapt a recorded comparison trial to the evidence gate's trial contract. |
| [`sha256`](../../scripts/gate_agentic_retrieval.py#L58) | Hash the exact bytes of an input evidence file for gate provenance. |
| [`evaluate`](../../scripts/gate_agentic_retrieval.py#L63) | Evaluate three recorded evidence inputs and save a new gate report. |

## Source and verification boundary

This documentation changes no executable behaviour. Definitions and source
contracts describe the retained implementation, including its known failures.
The root verification compares syntax trees after removing docstrings and runs
regression checks separately from any live inference experiment.
