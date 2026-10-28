# AI engineering laboratory runbook

This guide documents preparation, execution, interpretation and clean-up for the code used by the AI engineering article. Use synthetic cases and a separate evidence directory. The retained measurements describe particular runs; these procedures do not certify a new machine or authorize production.

Start with the [function index](FUNCTION_INDEX.md) when studying an individual function. Its links open the complete contract in the source: inputs, outputs, errors, side effects and limits. The [search reference](search-reference.md), [vision reference](vision-reference.md), [explanation and gate reference](explainability-acceptance-reference.md) and [portal reference](portal-reference.md) explain each component in context.

## 1. Choose the execution environment

| Work | Environment | Required capabilities | What it does not require |
| --- | --- | --- | --- |
| Documentation checks | Any supported repository Python | Python standard library | Models, database, network, GPU |
| Controller, API and gate behavior tests | Base application virtual environment | Installed `requirements.txt` | Live models or an OKD cluster; tests use controlled clients |
| TF-IDF agent trial | Base environment and local Ollama | Pinned Qwen3:4b | pgvector or embedding calls |
| BGE-M3 comparison | Base environment, local Ollama, isolated pgvector database | Pinned BGE-M3 and Qwen3:4b, PostgreSQL with pgvector, client credentials | GPU; the comparison explicitly forces CPU |
| SHAP/LIME study | Separate Python 3.12 environment | `requirements-explainability.txt` | GPU, external data or inference service |
| Vision measurement | Existing Linux/WSL PyTorch environment | Matching PyTorch/CUDA for `--require-cuda`; CPU is also supported | YOLO/TensorRT/Triton downloads; the probes use synthetic inputs |

Run commands from the repository root. Creating environments or downloading dependencies is an operator setup step, not something the experiment scripts silently perform. Keep credentials, case stores, reports and checkpoints outside the Git checkout.

For a new base environment, use the Python version supported by the main application, create `.venv`, and install `requirements.txt`. Reuse an already working environment when available. The separate explanation environment avoids mixing scientific dependencies into that base environment.

## 2. Check documentation before studying or changing code

```powershell
.\.venv\Scripts\python.exe -B scripts\check_engineering_documentation.py
```

Expected result: `passed` is true and `errors` is empty. This checks every named function, method, nested helper and class in the scope manifest, including the associated tests. It also checks that parameter names occur in their contracts. Python lambdas and browser callbacks are documented at their call sites; their descriptions require human review.

After documentation moves source lines, regenerate the source-linked index:

```powershell
.\.venv\Scripts\python.exe -B scripts\check_engineering_documentation.py --write-index
```

`--json-out <path>` writes the detailed coverage report. This option replaces that selected report file. A successful coverage result does not prove that prose is accurate; compare each contract with the implementation and review the specific limitations below.

## 3. Prepare an isolated evidence directory

Choose a new directory for each run. The date and sequence below are examples to replace before execution.

```powershell
$evidenceDir = Join-Path $env:USERPROFILE 'ai-engineering-evidence\run-YYYYMMDD-01'
New-Item -ItemType Directory -Path $evidenceDir -ErrorAction Stop | Out-Null
$protocol = (Resolve-Path 'docs\evidence\agentic-retrieval-20261028\protocol.json').Path
```

Review the protocol before running: questions, expected passages, answer rules, loop rule and limits must be known in advance. New experiment designs get a new protocol and report; do not rewrite historical results to match an improved method.

## 4. Check the local model service and pins

Use the existing local Ollama service. If none is running, start it through its normal installed launcher or `ollama serve` in a dedicated terminal. Do not start a second service on the same port.

```powershell
Invoke-RestMethod 'http://127.0.0.1:11434/api/version'
(Invoke-RestMethod 'http://127.0.0.1:11434/api/tags').models |
    Select-Object name, digest
```

The comparison code pins these exact artifacts:

| Role | Model | Required digest |
| --- | --- | --- |
| Question/passage embeddings | `bge-m3:latest` | `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab` |
| Planner and answer generator | `qwen3:4b` | `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7` |

A name alone is insufficient. Tags can move; a fresh pull may produce a different digest. The scripts do not pull models or substitute a cloud endpoint. A missing or mismatched pin must be resolved deliberately or recorded as unavailable. The historical lexical run does not exercise an embedding call despite constructing a client with an embedding-model configuration.

Avoid concurrent heavy inference while comparing durations. The retained comparison forced CPU and `keep_alive=0`; its model reload time is part of the observations, not a production latency target.

## 5. Prepare pgvector access

Use an existing, isolated laboratory PostgreSQL database with the `vector` extension and a provisioned client role. The role needs schema/table operations required by `VectorIndex.synchronize` in the `retrieval` schema; it must not be a database administrator. This runbook does not grant roles or create an administrator account.

Set the connection variables for that test database:

```powershell
$env:LAB_VECTOR_HOST = '127.0.0.1'
$env:LAB_VECTOR_PORT = '19543'
$env:LAB_VECTOR_DATABASE = 'lab_vectors'
$env:LAB_VECTOR_USER = 'vector_client'
$env:LAB_VECTOR_PASSWORD_FILE = 'C:\Private\lab-vector-password.txt'
```

The password file path is an example. Use an existing operator-managed private file containing only the password; do not paste its contents into shell history, source files, reports or the article. No credential value is reproduced in the public evidence.

For the existing local OKD layout, the database service is `lab-vector` in namespace `multimodal-ai-lab`. In a dedicated terminal with an already approved cluster context, a loopback-only forward is:

```powershell
oc -n multimodal-ai-lab port-forward service/lab-vector 19543:5432 --address 127.0.0.1
```

Verify the selected cluster/project and service before running this command. The terminal must stay open while the comparison uses the database. This is a local access path, not a public database listener. Use the [existing deployment documentation](../../deploy/) for installation and cluster prerequisites rather than recreating a cluster for a documentation check.

The scripts create/update/delete rows for their synthetic case IDs. Use a test database without important existing data under those IDs, especially `demo-case-a`. A fresh evidence path alone does not isolate shared database rows.

## 6. Reproduce the historical lexical trial

```powershell
.\.venv\Scripts\python.exe -B scripts\measure_agentic_engineering.py --out "$evidenceDir\historical-lexical-rerun.json" --max-seconds 90
```

The output filename must not already exist. This script uses `ReviewedLexicalRetriever`, three fixed questions, simulated actors and pinned Qwen3:4b on CPU. It writes incremental results so failed or interrupted trials remain visible. Exit 0 means all historical trial criteria passed; exit 1 means at least one failed. A protective controller stop can still fail the stricter expected model-abstention rule. Do not interpret an experiment failure as a failed test runner or discard its report.

## 7. Compare TF-IDF and BGE-M3 with the same agent

Run the backends one after the other under the same hardware and model conditions:

```powershell
.\.venv\Scripts\python.exe -B scripts\compare_agentic_retrieval.py --backend lexical --protocol "$protocol" --fixture-dir "$evidenceDir\tfidf-fixture" --out "$evidenceDir\tfidf.json"
.\.venv\Scripts\python.exe -B scripts\compare_agentic_retrieval.py --backend pgvector --protocol "$protocol" --fixture-dir "$evidenceDir\vector-fixture" --out "$evidenceDir\bge-pgvector.json"
```

Keep `vector-fixture` until the restart check completes. Each report filename must be new. The comparison writes trial observations, ranked passages, model calls, durations, options, source/protocol hashes and environment information. It does not contain a general semantic answer judge. Its process can complete normally while trial verdicts fail; inspect the JSON rather than treating process exit 0 as model acceptance.

The pgvector branch has no lexical fallback. BGE-M3 produces 1,024-dimensional vectors; PostgreSQL/pgvector stores and searches them. The controller keeps at most three chunks and three searches. Similarity thresholds are backend-specific and do not represent calibrated confidence that a passage answers the question.

Distinguish these verdicts:

- **Historical rule:** preserves the original requirement that the paraphrase needs a repeated search.
- **Answer rule:** a correct answer on the first search can succeed. The unsupported case must meet the specified abstention rule.
- **Loop rule:** the separate two-step fixture must demonstrate repeated search and the expected answer.
- **Semantic review:** inspect the claims and their cited passages. A word such as `library` occurring in a response, or an exact quote copied from a different bicycle entry, is insufficient proof of a correct answer.

The retained BGE run passed the three basic questions and failed the separate wrong-object/two-step case. This documentation does not implement the planned identifier-chain correction. Preserve both the failure and the stronger independent review requirement when repeating the experiment.

## 8. Check index reuse and persistence

The vector comparison already records forbidden-case, unreviewed-source, edited-source, revocation and unchanged-index checks. Read their individual booleans and call counts.

For restart persistence, first retain the test store and confirm the database rows exist from that run. Restart only the intended laboratory database workload through the normal operator procedure. Record its new start time and wait for readiness. The persistence subcommand itself does not restart a pod:

```powershell
.\.venv\Scripts\python.exe -B scripts\compare_agentic_retrieval.py --persistence "$evidenceDir\bge-pgvector.json" --fixture-dir "$evidenceDir\vector-fixture" --out "$evidenceDir\persistence.json"
```

It verifies source hashes/revisions against the retained store, resynchronizes unchanged cases and records zero new embedding calls when reuse succeeds. **It then deletes the synthetic comparison rows** named in the run record and records the remaining count. Keep the report; a restart result is a local storage test, not disaster recovery or production isolation acceptance. Missing fixture files, a mismatched run/store or an unavailable database invalidate this procedure.

## 9. Run the explanation study

Prepare the separate environment once if it is not already available:

```powershell
py -3.12 -m venv ..\multimodal-xai-venv
..\multimodal-xai-venv\Scripts\python.exe -m pip install -r requirements-explainability.txt
```

Run a new same-split comparison:

```powershell
..\multimodal-xai-venv\Scripts\python.exe -B scripts\measure_explainability_engineering.py --compare --output "$evidenceDir\explainability.json"
```

The command prints the report and optionally writes identical JSON. **Unlike the search comparison, this output path is overwritten if it already exists**; select a fresh path. Exit 0 means the top-level numerical checks passed, 1 means one failed, and 2 covers caught dependency/input/JSON errors. A LIME fidelity diagnostic can fail within a normally executed report. Always read the raw diagnostics and limitations.

The six optional invoice inputs are documented in the [explanation reference](explainability-acceptance-reference.md). Only bounded numeric synthetic features are accepted. The linear baseline and tree model use the same synthetic split; Tree SHAP and LIME in the retained illustration explain different models. SHAP sums in its selected output scale, here log-odds, and its reconstruction check does not prove causal or useful explanations.

The separate review compares SHAP and LIME on **both** models, the same 20 stratified test cases and the same output, on log-odds and on probability, with one check per quality (performance and calibration, additivity, local fidelity, range, stability, reference sensitivity, agreement) plus coefficients, interactions and bounded counterfactuals:

```powershell
..\multimodal-xai-venv\Scripts\python.exe -B scripts\review_explainability_engineering.py --output "$evidenceDir\explainability-review.json"
```

It reads its parameters and thresholds from the [review protocol](../evidence/ai-engineering-review-20261028/explainability/protocol.json) and stops with exit 2 if they differ from the code. **It refuses an existing output file.** It runs on the CPU with one thread in about 75 seconds. Exit 0 means every check passed and 1 means the report was written with at least one failed check; failures are findings. The [review README](../evidence/ai-engineering-review-20261028/explainability/README.md) explains the retained results and their scales.

## 10. Run the vision probes

Select an already prepared Linux/WSL interpreter by setting `VISION_PYTHON`, and an evidence path outside the checkout in `EVIDENCE_DIR`. Run from the repository as visible to that Linux environment. Do not combine Windows drive syntax with Linux CLI paths.

The earlier small trained-classifier procedure is:

```bash
"$VISION_PYTHON" scripts/measure_vision_engineering.py --output "$EVIDENCE_DIR/tiny-trained-01" --http
```

The separate retained-style load probe is:

```bash
"$VISION_PYTHON" scripts/measure_vision_engineering.py --load-probe --output "$EVIDENCE_DIR/resnet-load-01" --require-cuda --gpu-memory-mib 512 --duration 10 --target-rps 100 --small-batch 8 --batch-wait-ms 5 --queue-size 64 --warmup 10
```

Use a new/empty output directory. The CLI refuses evidence inside the checkout. The [vision reference](vision-reference.md) documents every limit, return code, report and handler. The probe starts temporary loopback HTTP services, workers and queues, then closes them. `--serve` is a separate opt-in small-demo mode and is incompatible with `--load-probe`; stop that persistent demo normally with Ctrl+C.

Read offered, completed, rejected and successful counts together. Successful-request percentiles do not include rejected requests. Five milliseconds is the batch-filling wait, not a bound on total queueing. FP16 or another worker is an experimental setting, not a promised improvement. The ResNet-shaped probe has random weights: its latency result cannot qualify image-detection quality. Allocator reservations do not include all GPU driver/context memory. That probe also fixes **one CPU thread per call** (`cpu_threads_per_call: 1`), so its CPU row is not a fair CPU baseline.

### 10.1 Trained packaging model and load ramp

The protocol for this procedure is [`docs/evidence/ai-engineering-review-20261028/vision/protocol.json`](../evidence/ai-engineering-review-20261028/vision/protocol.json); read it before running. First train and test the ResNet-18 shape on procedural packaging images (train/validation/test from disjoint seeds, hash-checked):

```bash
"$VISION_PYTHON" -B scripts/measure_vision_engineering.py --train-packaging --output "$EVIDENCE_DIR/training" --require-cuda --epochs 15 --training-batch 64 --cpu-threads 4 --train-gpu-memory-mib 4096
```

The weights file and its `.json` sidecar (file SHA-256, test-split hash, input size) land in that directory. Then run one configuration per command, each in a new output directory:

```bash
"$VISION_PYTHON" -B scripts/measure_vision_engineering.py --ramp --ramp-config gpu_fp32_dynamic --trained-weights "$EVIDENCE_DIR/training/packaging_resnet18_224.pt" --output "$EVIDENCE_DIR/ramp/gpu_fp32_dynamic" --duration 20 --ramp-warmup-seconds 5 --key-point-repeats 3 --gpu-memory-mib 512 --queue-size 64
```

Configurations: `cpu_fp32_dynamic` (PyTorch default threads, all physical cores), `cpu_fp32_dynamic_1thread` (historical one-thread comparison), `gpu_fp32_batch1`, `gpu_fp32_dynamic`, `gpu_fp16_dynamic`, `gpu_fp32_dynamic_2workers` and `gpu_fp32_dynamic_cache50`. Thread settings belong to each configuration and are recorded in the report; `--cpu-threads` applies to training only. The ramp verifies the weights hash, regenerates the test split and refuses a mismatch, then climbs 10 to 3,200 req/s from separate client processes and stops after a step with more than 20 % failures or a generator-limited step. It reruns the key rates to three runs. Each step takes about 30 seconds; a GPU configuration takes 4 to 7 minutes.

Read `capacity_at_slo` together with each step's `failure_share`, `within_slo_share_of_offered`, `generator_limited`, `p99_observations` and `quality_under_load`. Latency is measured from the scheduled arrival, and late, rejected and unsent requests count as failures. If another job shares the GPU, serialize access to it; the scripts do not take a lock themselves.

## 11. Recompute the candidate evidence gate

The gate can be recomputed from retained reports with **no models or cluster**. Choose one search comparison record and the relevant vision and explanation reports:

```powershell
.\.venv\Scripts\python.exe -B scripts\gate_agentic_retrieval.py --run "$evidenceDir\bge-pgvector.json" --vision 'docs\evidence\ai-engineering-20261027\vision-resnet-open-loop.json' --explainability 'docs\evidence\ai-engineering-20261027\explainability-comparison.json' --out "$evidenceDir\candidate-gate.json"
```

The output path must be new. This command's process exit status is not the acceptance decision: read `gate.decision`, the per-check statuses and input hashes. A failed/missing two-step case blocks; LIME and random-weight vision failures also remain independent checks. The best successful technical result is `technical_candidate`, with human review required and production approval false. Trusted input hashes identify content, not its authenticity or completeness.

## 12. Open the optional browser study

Use the existing base application's approved startup profile. A local command for an already configured base environment is:

```powershell
$env:LAB_DATA_DIR = Join-Path $evidenceDir 'portal-case-store'
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8770
```

Open `/engineering` on that local service. The [portal reference](portal-reference.md) documents all routes, request bodies, roles and errors. Configure identity, CPE and inference through the base application; do not weaken them to make a study run pass.

Important profile differences:

- The agent route uses three searches and **90 seconds**. It serializes local agent/explanation work with a non-blocking shared lock.
- The explanation action is reviewer-only and disabled for the current CPE profile.
- The vision action reads retained evidence; it does not perform live GPU inference.
- The portal acceptance route reads its three fixed evidence filenames. It does **not automatically load** the new BGE comparison; use procedure 11 for that profile. The older search file lacks the new multi-step evidence and therefore remains blocked.

## 13. Run behavior tests separately from measurements

Base-environment tests do not need a running model or database:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests\test_agentic_search.py tests\test_local_rag.py tests\test_engineering_acceptance.py tests\test_engineering_api.py tests\test_engineering_documentation.py -q -p no:cacheprovider
```

Run explanation tests in the separate Python 3.12 environment and vision tests in the selected PyTorch environment:

```powershell
..\multimodal-xai-venv\Scripts\python.exe -B -m pytest tests\test_engineering_explainability.py tests\test_explainability_review.py -q -p no:cacheprovider
```

```bash
"$VISION_PYTHON" -B -m pytest tests/test_engineering_vision.py -q -p no:cacheprovider
```

Read skipped tests and their reasons. Scripted clients prove specific control-flow behavior, not the frequency of mistakes made by Qwen. A passing unit suite can coexist with a failed real question. The documentation check is another independent gate and does not replace these tests.

## 14. Diagnose failures and close temporary resources

| Observation | Check next | Preserve |
| --- | --- | --- |
| Local model unavailable | Ollama endpoint, model name and exact digest; current deadline | Error type, model pin, failed trial and stop reason |
| Database refused or connection timeout | Forwarding terminal, selected test database, client-role grants, private password-file path | Failure report; never print credentials |
| No relevant passage | Review status, case rights, source revision, ranks and threshold | Ranked passages and abstention |
| Exact quote supports the wrong object | Object/identifier correspondence and the full evidence chain | Wrong claim and exact source as a failed case |
| Deadline or repeated query | Timed steps, remaining budget and action trace | Explicit stop; do not silently increase limits |
| Explanation unavailable/fails | Optional environment, raw LIME output, SHAP scale and diagnostics | Failed explanation and its numeric checks |
| Vision lacks CUDA or rejects load | Selected interpreter/runtime, device requirement, accounting and queue limits | Rejections and CPU/GPU execution scope |
| Gate blocked | Each failed and review-pending check, including missing evidence | Technical result separate from any human decision |

After the run, stop only the forwarding terminal and temporary demo service you started. Do not stop a shared Ollama service or cluster as part of report generation. Verify report files are complete, keep the protocol and hashes with them, and remove credentials/runtime stores only through the normal owner-managed process. Do not delete historical evidence.

## 15. Review a documentation-only change

Before integrating documentation, check that every scoped named definition has a substantive contract, callbacks have call-site explanations, source links resolve and procedure options match `--help`/implementation. Compare executable Python ASTs with docstrings removed and JavaScript tokens with comments removed. Run the relevant unit suite. Do not rerun models merely to update comments.

Documentation changes move source lines and alter source hashes. Historical reports must retain the hashes of the exact old code they measured. Do not edit evidence hashes to match newly documented files, or silently repoint article links at shifted lines. Update live source links to the new source revision only when that revision is actually published.
