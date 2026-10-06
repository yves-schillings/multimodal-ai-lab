# AI Lab: Multimodal Case Processing

A synthetic-data lab for document classification, speech transcription, reviewed-source search and human approval. The application connects a case workflow to measured model training and release decisions.

## Run locally

On Windows, double-click `Start.cmd`. The first launch creates `.venv` and installs the pinned dependencies. The application opens at **http://127.0.0.1:8770** and remains bound to loopback.

Alternatively, with Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8770
```

Choose a demo officer, review the synthetic transcript, search its reviewed sources and prepare a draft. Switch to the reviewer to approve it. Corrections invalidate the draft and its approval. The personas are explicitly simulated identities; they are not real authentication.

In **Model lab**, train a candidate. The local sklearn TF-IDF/logistic regression pipeline uses 24 synthetic training samples and 8 held-out samples. MLflow records metrics locally. The reviewer may promote a version passing accuracy and macro-F1 gates, then restore a previous version. The small synthetic evaluation does not establish production accuracy.

## Speech preparation

Speech recognition executes locally through faster-whisper. Model acquisition is a separate preparation step; application requests never download a model automatically:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_speech.py
```

This downloads the public `base` model weights to the ignored `models/whisper-base` directory. Afterwards, import a synthetic recording of at most five minutes and 25 MiB. English, French and Dutch are supported by the adapter; actual quality depends on the recording and requires review.

## Document input

Import UTF-8 TXT/MD/CSV or an unencrypted text PDF. Image OCR requires a separately installed local Tesseract executable. Scanned PDFs are not silently treated as extracted text: the first release asks for local OCR preparation. Uploaded source text, extraction and review remain separate.

## Public browser sandbox

Try the [deployed synthetic sandbox](https://secloudis-case-ai-lab.subllings.chatgpt.site). See the [code guide](docs/code-guide.md) for a walkthrough and the [deployment stages](docs/deployment.md) for the OpenShift AI target, controlled cloud bursting and future CPE lifecycle.

`public-demo/` contains a self-contained browser application. It uses synthetic fixtures only, reviewed-source lexical search, an extractive draft and a browser Naive Bayes classifier trained on its own synthetic corpus. Its weak baseline intentionally fails the release gate. Browser experiment state stays in the tab.

The public sandbox has no live speech or LLM inference, server-side authentication, case uploads or MLflow server. These limits are visible in the interface. The local Python implementation provides file processing and separate sklearn/MLflow evidence. Neither version claims to be an official institutional system.

## Architecture and implementation status

See `docs/architecture.md` for component responsibilities, boundaries and the target deployment path. The current implementation uses SQLite and one durable local worker. PostgreSQL/pgvector, RabbitMQ, trusted OpenID Connect identities, generative RAG and Kubernetes/OpenShift AI are enterprise extensions, not deployed components of this release.

All examples are fictional. Real recordings or case documents require an explicitly authorised environment. Azure is a later option; transfer of real data requires explicit authorisation before activation.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
node tests/browser_core.test.mjs
```

The Python tests cover case isolation, stale revisions, human review, approval separation, bounded input, durable-job behaviour and model gates. The browser checks cover classifier gates, reviewed-source eligibility and insufficient evidence.

Source files and technical documentation belong in this repository. Presentation decks, article drafts, private source material, recordings, model weights, runtime databases and credentials do not.
