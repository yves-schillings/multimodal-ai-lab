"""Serve the pinned BAAI/bge-reranker-v2-m3 cross-encoder on the loopback interface for the RAG benchmark.

The weights must already be in the local Hugging Face cache: the service loads them with
``local_files_only=True`` and never downloads at run time. It binds to 127.0.0.1 only, has no
authentication and accepts synthetic benchmark text; it is a lab component, not a production endpoint.

Endpoints:

- ``GET /health``: model ID, cache revision, SHA-256 of the weights file, device and precision.
- ``POST /rerank`` with ``{"query": str, "passages": [str, ...]}``: one raw logit per passage, in order.

Usage (in the Linux/WSL PyTorch environment)::

    python scripts/serve_reranker.py --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --port 8790
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from time import perf_counter

MODEL_ID = "BAAI/bge-reranker-v2-m3"


def weights_sha256(snapshot):
    """Return the SHA-256 of ``model.safetensors`` in a local snapshot folder.

    Args:
        snapshot (Path): The Hugging Face snapshot directory of the pinned revision.
    Returns:
        str: Hexadecimal digest, read in 16 MiB blocks.
    Raises:
        OSError: The weights file cannot be read.
    """
    digest = hashlib.sha256()
    with open(Path(snapshot) / "model.safetensors", "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(revision, device, half):
    """Load the tokenizer and model of the pinned revision from the local cache only.

    Args:
        revision (str): The snapshot commit to load.
        device (str): ``cuda`` or ``cpu``.
        half (bool): Use FP16 weights (on CUDA only).
    Returns:
        tuple: (tokenizer, model in evaluation mode, identity dictionary for ``/health``).
    Raises:
        OSError: The revision is not in the local cache.
    """
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = snapshot_download(MODEL_ID, revision=revision, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True)
    dtype = torch.float16 if half and device == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    identity = {"model": MODEL_ID, "revision": revision, "weights_sha256": weights_sha256(snapshot),
                "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "precision": "fp16" if dtype == torch.float16 else "fp32", "max_length": 512,
                "torch": torch.__version__}
    return tokenizer, model, identity


def make_handler(tokenizer, model, identity):
    """Build the request handler class bound to one loaded model.

    Args:
        tokenizer: The loaded tokenizer.
        model: The loaded sequence-classification model.
        identity (dict): The ``/health`` answer.
    Returns:
        type: A ``BaseHTTPRequestHandler`` subclass.
    """
    import torch

    class Handler(BaseHTTPRequestHandler):
        """Answer ``/health`` and ``/rerank``; every other path returns 404."""

        def _send(self, status, payload):
            """Write one JSON response with its status code.

            Args:
                status (int): HTTP status code.
                payload (dict): JSON-serialisable body.
            """
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            """Return the model identity on ``/health``."""
            if self.path == "/health":
                self._send(200, identity)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            """Score a query against 1 to 32 passages on ``/rerank`` and return raw logits."""
            if self.path != "/rerank":
                self._send(404, {"error": "not found"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                query, passages = request["query"], request["passages"]
                if (not isinstance(query, str) or not query.strip() or len(query) > 2000
                        or not isinstance(passages, list) or not 1 <= len(passages) <= 32
                        or any(not isinstance(p, str) or len(p) > 2000 for p in passages)):
                    raise ValueError("bounds")
            except (KeyError, TypeError, ValueError):
                self._send(400, {"error": "Send a query and 1 to 32 passages of at most 2000 characters."})
                return
            started = perf_counter()
            with torch.inference_mode():
                batch = tokenizer([[query, p] for p in passages], padding=True, truncation=True,
                                  max_length=512, return_tensors="pt").to(model.device)
                scores = model(**batch).logits.view(-1).float().tolist()
            self._send(200, {"scores": scores, "elapsed_ms": round((perf_counter() - started) * 1000, 2)})

        def log_message(self, *args):
            """Ignore ``args``: request lines would only repeat the benchmark's own log."""

    return Handler


def main():
    """Parse the options, load the pinned model and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revision", required=True, help="Pinned snapshot commit of the reranker")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--cpu", action="store_true", help="Run on the CPU instead of CUDA")
    parser.add_argument("--fp32", action="store_true", help="Keep FP32 weights on CUDA")
    args = parser.parse_args()
    tokenizer, model, identity = load(args.revision, "cpu" if args.cpu else "cuda", not args.fp32)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(tokenizer, model, identity))
    print(json.dumps({"listening": f"http://127.0.0.1:{args.port}", **identity}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
