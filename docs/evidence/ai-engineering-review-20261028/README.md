# AI engineering review experiments (28 October 2026)

Three experiments answer the review of the article "AI Engineering for Agentic RAG, Inference and Explainability":

| Folder | Question | Status |
|---|---|---|
| [`rag/`](rag/README.md) | Does fixed RAG, fixed RAG with a reranker, agentic RAG or an identifier-linked correction answer better, at what cost? | See its README |
| [`vision/`](vision/README.md) | How fast and how well does a **trained** vision model serve requests as the load rises? | See its README |
| [`explainability/`](explainability/README.md) | Do SHAP and LIME pass separate checks of additivity, fidelity and stability on the **same** model and cases? | See its README |

Every folder holds a protocol written before its runs, the raw run records, a summary and the exact commands.
All data is synthetic or fictional. No personal data, no secrets and no credentials are stored here.

## Machine

- Intel Core i9-9900K (8 cores, 16 logical processors), 64 GB RAM, one NVIDIA GeForce RTX 2080 Ti (11 GB), Windows 11 with WSL 2 (Ubuntu).
- Windows: CPython 3.12.14 in the repository `.venv`; Ollama 0.32.13.
- WSL: CPython 3.12.3 with PyTorch 2.8.0+cu126 (CUDA 12.6) for the reranker service and the vision runs.

## Deterministic and variable parts

- **Deterministic:** the evaluation sets, the splits, the judge rules, the retrieval ranking rule, the identifier rule, the synthetic data generators with their seeds, and the summaries computed from the raw records.
- **Variable:** language-model outputs (temperature 0 is nearly but not strictly deterministic on a GPU), all timings (other processes, model reloads, thermal state), and GPU kernels selected at run time.
