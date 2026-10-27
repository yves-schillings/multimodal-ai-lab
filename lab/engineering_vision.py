"""Small, synthetic vision engineering laboratory with optional PyTorch.

This module imports on the base application without torch. GPU execution belongs
to the explicitly selected host runtime, not to the CPU-only cluster baseline.
The fixtures and all model-quality conclusions are synthetic demonstrations.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import random
import queue
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any, Sequence

IMAGE_SIZE = 32
PREPROCESS_VERSION = "gray32-u8-div255-v1"
CLASS_NAMES = ("continuous_synthetic_seal", "interrupted_synthetic_seal")


class VisionDependencyUnavailable(RuntimeError):
    """The optional model runtime is absent from this interpreter."""


def optional_runtime_status() -> dict[str, Any]:
    """Report installation only. A real CUDA call is required to prove GPU use."""
    return {"torch_installed": importlib.util.find_spec("torch") is not None,
            "cuda_execution_verified": False,
            "note": "Use the executable benchmark to verify the selected runtime."}


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise VisionDependencyUnavailable(
            "This optional vision lab needs an existing PyTorch environment. "
            "The base application remains usable without it."
        ) from exc
    return torch


def validate_image(pixels: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    if not isinstance(pixels, (list, tuple)) or len(pixels) != IMAGE_SIZE:
        raise ValueError("pixels must contain exactly 32 rows")
    rows = []
    for row in pixels:
        if not isinstance(row, (list, tuple)) or len(row) != IMAGE_SIZE:
            raise ValueError("each image row must contain exactly 32 pixels")
        if any(type(value) is not int or not 0 <= value <= 255 for value in row):
            raise ValueError("pixels must be integer grayscale values from 0 to 255")
        rows.append(tuple(row))
    return tuple(rows)


def image_digest(pixels: Sequence[Sequence[int]]) -> str:
    valid = validate_image(pixels)
    return hashlib.sha256(bytes(value for row in valid for value in row)).hexdigest()


def make_cache_key(pixels, weights_version: str, preprocessing_version: str):
    if not weights_version or not preprocessing_version:
        raise ValueError("weight and preprocessing versions are required")
    return weights_version, preprocessing_version, image_digest(pixels)


class PredictionCache:
    """Bounded synthetic-result cache. It neither checks nor grants access.

    A caller handling real data must authorize the user before any cache lookup
    and include the permitted data scope in its own cache policy.
    """
    def __init__(self, max_entries: int = 64):
        if not 1 <= max_entries <= 4096:
            raise ValueError("max_entries must be between 1 and 4096")
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            if key not in self._entries:
                self.misses += 1
                return None
            self.hits += 1
            self._entries.move_to_end(key)
            return copy.deepcopy(self._entries[key])

    def put(self, key, value):
        with self._lock:
            self._entries[key] = copy.deepcopy(value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)


@dataclass(frozen=True)
class SyntheticImage:
    image_id: str
    label: int
    pixels: tuple[tuple[int, ...], ...]


def generate_synthetic_examples(count: int, seed: int, split: str = "fixture"):
    if not 2 <= count <= 4096:
        raise ValueError("synthetic example count must be between 2 and 4096")
    rng = random.Random(seed)
    examples = []
    for index in range(count):
        label = index % 2
        pixels = [[rng.randrange(0, 25) for _ in range(IMAGE_SIZE)]
                  for _ in range(IMAGE_SIZE)]
        x0 = rng.randrange(10, 19)
        width = rng.choice((3, 4))
        brightness = rng.randrange(180, 246)
        gap_start = rng.randrange(12, 17)
        for y in range(4, 29):
            for x in range(x0, min(x0 + width, IMAGE_SIZE)):
                if not (label == 1 and gap_start <= y < gap_start + 6):
                    pixels[y][x] = max(0, min(255, brightness + rng.randrange(-10, 11)))
        examples.append(SyntheticImage(f"{split}-{index:04d}", label,
                                       validate_image(pixels)))
    return examples


def build_tiny_cnn(seed: int = 47):
    torch = require_torch()
    torch.manual_seed(seed)
    return torch.nn.Sequential(OrderedDict([
        ("conv1", torch.nn.Conv2d(1, 4, 3, padding=1)),
        ("relu1", torch.nn.ReLU()),
        ("pool1", torch.nn.MaxPool2d(2)),
        ("conv2", torch.nn.Conv2d(4, 8, 3, padding=1)),
        ("relu2", torch.nn.ReLU()),
        ("pool2", torch.nn.MaxPool2d(2)),
        ("flatten", torch.nn.Flatten()),
        ("classifier", torch.nn.Linear(8 * 8 * 8, 2)),
    ]))


def preprocess_images(images, device="cpu"):
    torch = require_torch()
    if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 64:
        raise ValueError("a batch must contain between 1 and 64 images")
    valid = [validate_image(image) for image in images]
    return torch.tensor(valid, dtype=torch.float32).unsqueeze(1).div_(255.0).to(device)


def weights_fingerprint(model) -> str:
    """Hash the actual tensor values, names, shapes and dtypes, independently of device."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(cpu.dtype), list(cpu.shape)]).encode())
        for item in cpu.reshape(-1).tolist():
            digest.update(struct.pack("<f", float(item)))
    return digest.hexdigest()


class VisionPredictor:
    """Own an immutable model snapshot. Replace the predictor to change its weights."""
    def __init__(self, model, device="cpu", preprocessing_version=PREPROCESS_VERSION,
                 cache: PredictionCache | None = None):
        self.torch = require_torch()
        self.weights_version = weights_fingerprint(model)
        self.preprocessing_version = preprocessing_version
        self.device = str(device)
        self._model = copy.deepcopy(model).to(device).eval()
        self.cache = cache if cache is not None else PredictionCache()
        self.compute_calls = 0

    def predict_batch(self, images):
        tensor = preprocess_images(images, self.device)
        with self.torch.inference_mode():
            probabilities = self._model(tensor).softmax(dim=1).cpu().tolist()
        self.compute_calls += 1
        return [{"predicted_class": int(max(range(2), key=lambda i: row[i])),
                 "label": CLASS_NAMES[int(max(range(2), key=lambda i: row[i]))],
                 "probabilities": row,
                 "weights_version": self.weights_version,
                 "preprocessing_version": self.preprocessing_version,
                 "cache_hit": False} for row in probabilities]

    def predict(self, pixels, use_cache=True):
        key = make_cache_key(pixels, self.weights_version, self.preprocessing_version)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                cached["cache_hit"] = True
                return cached
        result = self.predict_batch([pixels])[0]
        if use_cache:
            self.cache.put(key, result)
        return result


def train_tiny_model(examples, seed=47, epochs=20, batch_size=32):
    torch = require_torch()
    if not 1 <= epochs <= 100:
        raise ValueError("epochs must be between 1 and 100")
    model = build_tiny_cnn(seed).cpu()
    inputs = preprocess_images([item.pixels for item in examples]) if len(examples) <= 64 else (
        torch.tensor([item.pixels for item in examples], dtype=torch.float32).unsqueeze(1).div_(255.0))
    labels = torch.tensor([item.label for item in examples], dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 1)
    losses = []
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(examples), generator=generator)
        loss_sum = 0.0
        for offset in range(0, len(examples), batch_size):
            indexes = permutation[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs[indexes]), labels[indexes])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indexes)
        losses.append(loss_sum / len(examples))
    model.eval()
    return model, {"device": "cpu", "seed": seed, "epochs": epochs,
                   "batch_size": batch_size, "optimizer": "Adam", "learning_rate": 0.01,
                   "loss_by_epoch": losses, "duration_seconds": time.perf_counter() - start}


def evaluate_model(engine, examples):
    predictions = []
    for offset in range(0, len(examples), 64):
        predictions.extend(engine.predict_batch([x.pixels for x in examples[offset:offset + 64]]))
    confusion = [[0, 0], [0, 0]]
    for item, prediction in zip(examples, predictions):
        confusion[item.label][prediction["predicted_class"]] += 1
    correct = sum(confusion[i][i] for i in range(2))
    return {"samples": len(examples), "correct": correct,
            "accuracy": correct / len(examples), "confusion_matrix": confusion}, predictions


def _quantile(values, q):
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    lower = math.floor(location)
    upper = math.ceil(location)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def summarize_timings(seconds, batch_size):
    return {"samples": len(seconds), "batch_size_images": batch_size,
            "p50_ms": _quantile(seconds, 0.50) * 1000,
            "p95_ms": _quantile(seconds, 0.95) * 1000,
            "p99_ms": _quantile(seconds, 0.99) * 1000,
            "mean_ms": sum(seconds) / len(seconds) * 1000,
            "throughput_images_per_second": len(seconds) * batch_size / sum(seconds)}


def measure_calls(call, warmup, repeats, batch_size, synchronize=lambda: None):
    for _ in range(warmup):
        call()
        synchronize()
    values = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter_ns()
        call()
        synchronize()
        values.append((time.perf_counter_ns() - start) / 1e9)
    return summarize_timings(values, batch_size)


def measure_engine(engine, examples, batch_size, warmup=10, repeats=40):
    torch = engine.torch
    images = [item.pixels for item in examples[:batch_size]]
    prepared = preprocess_images(images, engine.device)
    sync = (lambda: torch.cuda.synchronize(engine.device)) if engine.device.startswith("cuda") else (lambda: None)

    def model_only():
        with torch.inference_mode():
            return engine._model(prepared)

    return {"device": engine.device, "batch_size_images": batch_size,
            "model_call": measure_calls(model_only, warmup, repeats, batch_size, sync),
            "full_pipeline": measure_calls(lambda: engine.predict_batch(images), warmup, repeats, batch_size, sync),
            "model_scope": "Synchronized wall time for forward call on an already prepared device tensor",
            "pipeline_scope": "Input validation, tensor conversion/normalization, device transfer, forward call, softmax and CPU result conversion; cache disabled"}


def demonstrate_cache(engine, pixels):
    first = engine.predict(pixels)
    calls_after_first = engine.compute_calls
    second = engine.predict(pixels)
    base_key = make_cache_key(pixels, engine.weights_version, engine.preprocessing_version)
    changed = copy.deepcopy(engine._model).cpu()
    with engine.torch.no_grad():
        next(changed.parameters()).reshape(-1)[0] += 0.001
    changed_weights = weights_fingerprint(changed)
    modified = [list(row) for row in pixels]
    modified[0][0] = (modified[0][0] + 1) % 256
    checks = {
        "first_request_missed_cache": not first["cache_hit"],
        "repeat_request_hit_cache": second["cache_hit"],
        "repeat_did_not_recompute": engine.compute_calls == calls_after_first,
        "actual_weight_change_has_new_fingerprint": changed_weights != engine.weights_version,
        "weight_change_has_no_cached_result": engine.cache.get(make_cache_key(pixels, changed_weights, engine.preprocessing_version)) is None,
        "preprocessing_version_change_has_no_cached_result": engine.cache.get(make_cache_key(pixels, engine.weights_version, engine.preprocessing_version + "-candidate")) is None,
        "input_change_has_no_cached_result": engine.cache.get(make_cache_key(modified, engine.weights_version, engine.preprocessing_version)) is None,
    }
    return {"checks": checks, "passed": all(checks.values()),
            "key_components": ["actual weights fingerprint", "preprocessing version", "validated pixel digest"],
            "cache_does_not_grant_access": True,
            "scope": "Synthetic fixtures only. A changed preprocessor key demonstrates invalidation, not acceptance of a new preprocessing algorithm."}


def create_local_http_server(engine, port=0):
    """Optional synthetic demo, loopback only. No authentication claim is made."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, code, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self._reply(404, {"error": "not found"})
            self._reply(200, {"scope": "synthetic localhost demo", "device": engine.device,
                              "weights_version": engine.weights_version})

        def do_POST(self):
            if self.path != "/predict":
                return self._reply(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    return self._reply(413, {"error": "request body must be 1 to 65536 bytes"})
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or "pixels" not in payload:
                    raise ValueError("a JSON object with pixels is required")
                result = engine.predict(payload["pixels"], use_cache=False)
                self._reply(200, result)
            except (ValueError, TypeError, UnicodeDecodeError) as exc:
                self._reply(400, {"error": str(exc)})
    return HTTPServer(("127.0.0.1", port), Handler)


def measure_http(engine, pixels, warmup=5, repeats=20):
    from urllib.request import Request, urlopen
    server = create_local_http_server(engine)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/predict"
    payload = json.dumps({"pixels": pixels}).encode("utf-8")

    def request():
        req = Request(url, payload, {"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as response:
            result = json.loads(response.read())
        if result["weights_version"] != engine.weights_version:
            raise RuntimeError("HTTP response used a different model version")

    try:
        timing = measure_calls(request, warmup, repeats, 1)
        return {"device": engine.device, **timing,
                "scope": "Sequential localhost HTTP request, JSON parsing, uncached pipeline and response. Test server closes after measurement.",
                "bind_address": "127.0.0.1", "authentication_claimed": False}
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def build_resnet18_probe(seed=47, num_classes=1000):
    """Canonical ResNet-18-shaped, RANDOM weights: a performance probe only.

    No torchvision, downloaded weights, pretrained semantics or quality claim.
    """
    torch = require_torch()
    nn = torch.nn

    class BasicBlock(nn.Module):
        def __init__(self, incoming, outgoing, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(outgoing)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(outgoing)
            self.downsample = (nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride, bias=False),
                                            nn.BatchNorm2d(outgoing))
                               if stride != 1 or incoming != outgoing else nn.Identity())

        def forward(self, x):
            identity = self.downsample(x)
            x = self.relu(self.bn1(self.conv1(x)))
            return self.relu(self.bn2(self.conv2(x)) + identity)

    class ResNet18Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
            self.layers = nn.Sequential(
                BasicBlock(64, 64), BasicBlock(64, 64),
                BasicBlock(64, 128, 2), BasicBlock(128, 128),
                BasicBlock(128, 256, 2), BasicBlock(256, 256),
                BasicBlock(256, 512, 2), BasicBlock(512, 512))
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

        def forward(self, x):
            return self.fc(self.pool(self.layers(self.stem(x))).flatten(1))

    torch.manual_seed(seed)
    return ResNet18Probe().eval()


def fast_weights_fingerprint(model):
    """Exact state bytes with dtype and shape, efficient for larger probes."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(cpu.dtype), list(cpu.shape)]).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def latency_summary(milliseconds):
    if not milliseconds:
        return {"samples": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    return {"samples": len(milliseconds), "p50_ms": _quantile(milliseconds, .50),
            "p95_ms": _quantile(milliseconds, .95), "p99_ms": _quantile(milliseconds, .99),
            "max_ms": max(milliseconds)}


class ProbeQueueFull(RuntimeError):
    pass


class BoundedBatchVisionService:
    """Local performance-only queue and explicit model instances, no auth claim.

    Every request is synthetic RGB bytes, exactly size*size*3, cache disabled.
    The batch timer starts when a worker removes its first request. A full queue
    rejects immediately. Accepted futures finish or fail; close joins all workers.
    """
    def __init__(self, batch_call, *, image_size=224, workers=1, max_batch=8,
                 max_wait_ms=5, queue_size=64):
        from concurrent.futures import Future
        if workers not in (1, 2) or not 1 <= max_batch <= 8 or not 0 <= max_wait_ms <= 5:
            raise ValueError("Use one or two workers, batch 1..8 and wait 0..5 ms")
        if not 1 <= queue_size <= 256 or not 32 <= image_size <= 224:
            raise ValueError("Use queue size 1..256 and image size 32..224")
        self.Future = Future
        self.batch_call = batch_call
        self.image_size = image_size
        self.body_bytes = image_size * image_size * 3
        self.queue = queue.Queue(queue_size)
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.records = []
        self.rejected = 0
        self.max_queue_depth = 0
        self.worker_threads = [threading.Thread(target=self._run, args=(i,),
                                               name=f"vision-batch-{i}", daemon=True)
                               for i in range(workers)]
        for worker in self.worker_threads:
            worker.start()

    def submit(self, payload):
        if not isinstance(payload, bytes) or len(payload) != self.body_bytes:
            raise ValueError(f"Expected exactly {self.body_bytes} RGB uint8 bytes")
        if self.stop.is_set():
            raise RuntimeError("Service is closing")
        future = self.Future()
        item = (payload, time.perf_counter(), future)
        try:
            self.queue.put_nowait(item)
        except queue.Full as exc:
            with self.lock:
                self.rejected += 1
            raise ProbeQueueFull("Bounded inference queue is full") from exc
        with self.lock:
            self.max_queue_depth = max(self.max_queue_depth, self.queue.qsize())
        return future

    def _run(self, worker_index):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                first = self.queue.get(timeout=.05)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.perf_counter() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break
            started = time.perf_counter()
            try:
                results, timing = self.batch_call(worker_index, [x[0] for x in batch])
                if len(results) != len(batch):
                    raise RuntimeError("The model returned an incomplete batch")
                ended = time.perf_counter()
                record = {"worker": worker_index, "size": len(batch), **timing,
                          "worker_pipeline_ms": (ended - started) * 1000}
                with self.lock:
                    self.records.append(record)
                for (_, submitted, future), result in zip(batch, results):
                    future.set_result({**result, "queue_wait_ms": (started - submitted) * 1000,
                                       "service_latency_ms": (ended - submitted) * 1000,
                                       "batch_size": len(batch)})
            except Exception as exc:
                for _, _, future in batch:
                    future.set_exception(exc)
            finally:
                for _ in batch:
                    self.queue.task_done()

    def metrics(self):
        with self.lock:
            records = list(self.records)
            histogram = {}
            for item in records:
                size = str(item["size"])
                histogram[size] = histogram.get(size, 0) + 1
            total_images = sum(x["size"] for x in records)
            return {"batches": len(records), "images": total_images,
                    "batch_occupancy_histogram": histogram,
                    "mean_batch_size": total_images / len(records) if records else None,
                    "queue_rejections": self.rejected, "max_queue_depth": self.max_queue_depth,
                    "batch_model_latency": latency_summary([x["model_ms"] for x in records]),
                    "batch_pipeline_latency": latency_summary([x["worker_pipeline_ms"] for x in records])}

    def close(self):
        self.stop.set()
        for worker in self.worker_threads:
            worker.join(timeout=30)
        if any(worker.is_alive() for worker in self.worker_threads):
            raise RuntimeError("A batch worker did not stop within 30 seconds")


def make_resnet_batch_runtime(base_model, device="cpu", precision="fp32", workers=1, image_size=224):
    """Prepare same seed/weights snapshots, then CPU or CUDA pipeline callables."""
    torch = require_torch()
    if precision not in ("fp32", "fp16") or (precision == "fp16" and not device.startswith("cuda")):
        raise ValueError("FP16 is measured only on the explicitly selected CUDA route")
    dtype = torch.float16 if precision == "fp16" else torch.float32
    models = [copy.deepcopy(base_model).to(device=device, dtype=dtype).eval() for _ in range(workers)]
    streams = [torch.cuda.Stream(device=device) for _ in models] if device.startswith("cuda") else [None] * workers
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    def batch_call(worker_index, payloads):
        from contextlib import nullcontext
        stream = streams[worker_index]
        manager = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with manager, torch.inference_mode():
            images = [torch.frombuffer(bytearray(p), dtype=torch.uint8).reshape(image_size, image_size, 3)
                      .permute(2, 0, 1).float().div_(255) for p in payloads]
            tensor = torch.stack(images).to(device=device, dtype=dtype)
            if stream is not None:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(stream)
                logits = models[worker_index](tensor)
                end.record(stream)
            else:
                start_cpu = time.perf_counter()
                logits = models[worker_index](tensor)
                model_ms = (time.perf_counter() - start_cpu) * 1000
            probabilities = logits.float().softmax(dim=1)
            values, indexes = probabilities.max(dim=1)
            values, indexes = values.cpu().tolist(), indexes.cpu().tolist()
            if stream is not None:
                stream.synchronize()
                model_ms = start.elapsed_time(end)
            if not all(math.isfinite(x) for x in values):
                raise RuntimeError("Non-finite model output")
            return [{"top_class_index": int(i), "top_probability": float(v),
                     "random_weights": True, "quality_claimed": False} for i, v in zip(indexes, values)], {"model_ms": model_ms}

    return batch_call, models, streams


def create_batched_http_server(service, port=0):
    """Threaded, loopback-only binary RGB service for the open-loop experiment."""
    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 128

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, value):
            data = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if self.path != "/predict-rgb":
                return self.reply(404, {"error": "not found"})
            self.connection.settimeout(15)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length != service.body_bytes:
                    return self.reply(400, {"error": "Incorrect RGB uint8 body size"})
                payload = self.rfile.read(length)
                result = service.submit(payload).result(timeout=15)
                self.reply(200, result)
            except ProbeQueueFull as exc:
                self.reply(429, {"error": str(exc)})
            except ValueError as exc:
                self.reply(400, {"error": str(exc)})
            except Exception as exc:
                self.reply(503, {"error_type": type(exc).__name__, "error": str(exc)[:300]})

    return Server(("127.0.0.1", port), Handler)


def run_open_loop_http(server, payloads, target_rps=100, duration_seconds=10):
    """Schedule arrivals independently of responses; never silently close the loop."""
    import http.client
    from concurrent.futures import ThreadPoolExecutor
    if not 1 <= target_rps <= 200 or not 1 <= duration_seconds <= 30:
        raise ValueError("Use offered rate 1..200 req/s and duration 1..30 seconds")
    count = int(target_rps * duration_seconds)
    results, submitted, dropped, lateness = [], [], 0, []
    pending_slots = threading.BoundedSemaphore(256)

    def request(payload, due):
        started = time.perf_counter()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=16)
        try:
            connection.request("POST", "/predict-rgb", payload, {"Content-Type": "application/octet-stream"})
            response = connection.getresponse()
            body = json.loads(response.read())
            ended = time.perf_counter()
            return {"status": response.status, "ended": ended,
                    "http_ms": (ended - started) * 1000, "scheduled_arrival_ms": (ended - due) * 1000,
                    "queue_wait_ms": body.get("queue_wait_ms"), "service_ms": body.get("service_latency_ms"),
                    "error_type": body.get("error_type")}
        except Exception as exc:
            ended = time.perf_counter()
            return {"status": "client_error", "ended": ended, "http_ms": (ended - started) * 1000,
                    "scheduled_arrival_ms": (ended - due) * 1000, "error_type": type(exc).__name__}
        finally:
            connection.close()
            pending_slots.release()

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=128, thread_name_prefix="vision-load") as pool:
        for index in range(count):
            due = started + index / target_rps
            remaining = due - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            lateness.append(max(0, time.perf_counter() - due) * 1000)
            if pending_slots.acquire(blocking=False):
                submitted.append(pool.submit(request, payloads[index % len(payloads)], due))
            else:
                dropped += 1
        injection_ended = time.perf_counter()
        for future in submitted:
            results.append(future.result())
    ended = time.perf_counter()
    good = [x for x in results if x["status"] == 200]
    statuses = {}
    for result in results:
        key = str(result["status"])
        statuses[key] = statuses.get(key, 0) + 1
    return {"offered_requests": count, "submitted_requests": len(submitted),
            "completed_requests": len(results), "successful_requests": len(good),
            "error_requests": len(results) - len(good), "client_safety_drops": dropped,
            "http_status_counts": statuses, "offered_rate_per_second": target_rps,
            "requested_duration_seconds": duration_seconds,
            "injection_duration_seconds": injection_ended - started,
            "total_with_drain_seconds": ended - started,
            "achieved_success_rate_during_window_per_second": sum(x["ended"] <= started + duration_seconds for x in good) / duration_seconds,
            "successful_requests_per_second_including_drain": len(good) / (ended - started),
            "arrival_scheduler_lateness": latency_summary(lateness),
            "http_end_to_end_latency": latency_summary([x["http_ms"] for x in good]),
            "scheduled_arrival_to_response_latency": latency_summary([x["scheduled_arrival_ms"] for x in good]),
            "queue_wait_latency": latency_summary([x["queue_wait_ms"] for x in good]),
            "service_queue_and_pipeline_latency": latency_summary([x["service_ms"] for x in good]),
            "all_response_latency": latency_summary([x["http_ms"] for x in results]),
            "error_types": sorted({x["error_type"] for x in results if x.get("error_type")}),
            "accounting_complete": count == len(results) + dropped,
            "scope": "Open-loop localhost binary HTTP; end-to-end includes service queue, pipeline and response. No authentication or production SLA acceptance."}
