"""Provide synthetic vision exercises with explicit measurement boundaries.

The tiny grayscale CNN path trains on fictional stripe images and can compare CPU
and CUDA outputs. The random-weight RGB ResNet-shaped path measures a bounded,
batched localhost service; it does not measure detection quality. The packaging
path generates procedural bottle images, trains the same ResNet-18 shape from
scratch, and drives the batched service with a multi-process open-loop load
ramp whose summaries keep every failed request visible. PyTorch is imported
lazily, so the base application can import this module without it.
Callers own server shutdown, access control and runtime selection. No model
weights or packages are downloaded. CUDA allocator budgets are set by the CLI,
not by this module, and do not cover driver/context memory.
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
    """Signal that the selected interpreter cannot import optional PyTorch.

    This RuntimeError subclass is raised by require_torch; it carries a human-readable
    message and has no additional attributes or recovery side effects.
    """


def optional_runtime_status() -> dict[str, Any]:
    """Inspect whether the optional torch package can be located without importing it.

    Returns:
        dict[str, Any]: Installation availability, a false CUDA-verification flag and
        an explanation. Package presence never establishes that CUDA can execute.

    Side effects:
        Performs import-spec discovery only; allocates no tensor or GPU context.
        Import-system discovery errors other than an absent spec propagate.
    """
    return {"torch_installed": importlib.util.find_spec("torch") is not None,
            "cuda_execution_verified": False,
            "note": "Use the executable benchmark to verify the selected runtime."}


def require_torch():
    """Import and return PyTorch from the currently selected interpreter.

    Returns:
        module: The imported torch module; Python may reuse its import cache.

    Raises:
        VisionDependencyUnavailable: PyTorch raises ImportError during import.
        Other import/runtime failures propagate. No installation or download occurs.

    Side effects:
        Executes torch's normal import initialization. This is not a CUDA-use check.
    """
    try:
        import torch
    except ImportError as exc:
        raise VisionDependencyUnavailable(
            "This optional vision lab needs an existing PyTorch environment. "
            "The base application remains usable without it."
        ) from exc
    return torch


def validate_image(pixels: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Validate one synthetic 32-by-32 grayscale image and freeze its rows.

    Args:
        pixels (list[list[int]] | tuple[tuple[int, ...], ...]): Exactly 32 rows of
            32 built-in integers in the inclusive range 0..255. Booleans are rejected.

    Returns:
        tuple[tuple[int, ...], ...]: An immutable validated image with unchanged values.

    Raises:
        ValueError: Shape, container type, pixel type or range is invalid.

    Side effects:
        None; the input is not modified and no model is executed.
    """
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
    """Hash the validated grayscale pixel bytes in row-major order.

    Args:
        pixels (Sequence[Sequence[int]]): Image accepted by validate_image.

    Returns:
        str: A 64-character lowercase SHA-256 hexadecimal digest.

    Raises:
        ValueError: Image validation fails. Metadata and user access scope are not
            included in this digest; identical pixel bytes intentionally share a hash.
    """
    valid = validate_image(pixels)
    return hashlib.sha256(bytes(value for row in valid for value in row)).hexdigest()


def make_cache_key(pixels, weights_version: str, preprocessing_version: str):
    """Bind a cached prediction to weights, preprocessing label and image content.

    Args:
        pixels (Sequence[Sequence[int]]): One valid 32-by-32 grayscale image.
        weights_version (str): Nonempty model fingerprint or caller-owned version.
        preprocessing_version (str): Nonempty preprocessing-version label.

    Returns:
        tuple[str, str, str]: Weight version, preprocessing version and pixel digest.

    Raises:
        ValueError: Either version is empty or the image is invalid.

    Limitations:
        Version labels are trusted identifiers, not validation of an algorithm.
        No user, tenant or authorization scope is encoded here.
    """
    if not weights_version or not preprocessing_version:
        raise ValueError("weight and preprocessing versions are required")
    return weights_version, preprocessing_version, image_digest(pixels)


class PredictionCache:
    """Store a bounded least-recently-used cache of synthetic prediction values.

    get and put protect entries and hit/miss counters with one lock and deep-copy
    values to prevent external mutation. Eviction removes the least-recent entry.
    The cache provides neither authorization nor tenant isolation. A real service
    must authorize before lookup and define its permitted-data scope separately.
    """
    def __init__(self, max_entries: int = 64):
        """Create an empty cache and its entry/counter lock.

        Args:
            max_entries (int): Retained-key capacity, inclusive range 1..4096; default 64.

        Returns:
            None.

        Raises:
            ValueError: Capacity is outside the permitted interval.
            TypeError: A supplied value cannot be compared with integer bounds.

        Side effects:
            Initializes an OrderedDict, a lock, and zero hit/miss counters.
        """
        if not 1 <= max_entries <= 4096:
            raise ValueError("max_entries must be between 1 and 4096")
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """Read a deep copy and mark an existing key as most recently used.

        Args:
            key (Hashable): Normally the tuple returned by make_cache_key.

        Returns:
            Any | None: A copied cached value, or None on a miss. Stored None is therefore
            indistinguishable from a miss to this API's callers.

        Raises:
            TypeError: The key is unhashable. Copy failures propagate.

        Side effects:
            Under the cache lock, increments hits or misses and updates LRU order.
        """
        with self._lock:
            if key not in self._entries:
                self.misses += 1
                return None
            self.hits += 1
            self._entries.move_to_end(key)
            return copy.deepcopy(self._entries[key])

    def put(self, key, value):
        """Store an isolated copy, update recency and evict excess entries.

        Args:
            key (Hashable): Cache identity, including the applicable version scope.
            value (Any): Deep-copyable result to retain.

        Returns:
            None.

        Raises:
            TypeError: The key is unhashable. Deep-copy failures propagate.

        Side effects:
            Mutates the cache under its lock, evicting oldest keys until within capacity.
        """
        with self._lock:
            self._entries[key] = copy.deepcopy(value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)


@dataclass(frozen=True)
class SyntheticImage:
    """Describe one immutable fictional grayscale classification fixture.

    Attributes:
        image_id (str): Split-prefixed fixture identifier, not a real image identity.
        label (int): 0 for a continuous stripe or 1 for an interrupted stripe.
        pixels (tuple[tuple[int, ...], ...]): Validated 32-by-32 uint8-style grayscale
            values. Frozen dataclass fields prevent reassignment; pixel tuples are
            immutable. Construction itself does not independently validate fields.
    """
    image_id: str
    label: int
    pixels: tuple[tuple[int, ...], ...]


def generate_synthetic_examples(count: int, seed: int, split: str = "fixture"):
    """Generate deterministic alternating-class fictional stripe images.

    Args:
        count (int): Number of fixtures, inclusive range 2..4096.
        seed (int): Seed for a private random.Random instance.
        split (str): Prefix for image_id values; default fixture.

    Returns:
        list[SyntheticImage]: Alternating labels with noisy backgrounds and either
        continuous or six-pixel-interrupted bright vertical stripes.

    Raises:
        ValueError: Count is out of bounds or generated-image validation fails.

    Side effects:
        Allocates fixture arrays without changing global random state. The task is
        deliberately artificial; its accuracy cannot establish real inspection quality.
    """
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
    """Construct an untrained two-class CNN for the grayscale stripe exercise.

    Args:
        seed (int): Seed passed to torch.manual_seed; default 47.

    Returns:
        torch.nn.Sequential: CPU model with two convolution/ReLU/pooling stages and
        a 512-to-2 linear classifier, initially in training mode.

    Raises:
        VisionDependencyUnavailable: PyTorch cannot be imported.
        PyTorch initialization errors propagate.

    Side effects:
        Resets PyTorch's global RNG and allocates model parameters. No weights are
        downloaded; calling this function alone does not train the model.
    """
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
    """Validate, normalize and transfer a grayscale image batch.

    Args:
        images (list | tuple): Between 1 and 64 valid 32-by-32 grayscale images.
        device (str | torch.device): Target tensor device; defaults to CPU.

    Returns:
        torch.Tensor: float32 tensor shaped (batch, 1, 32, 32), divided by 255 so
        pixel intensities lie in 0..1, on the requested device.

    Raises:
        ValueError: Batch bounds or any image validation fails.
        VisionDependencyUnavailable: PyTorch is missing.
        PyTorch device/allocation errors propagate.

    Side effects:
        Allocates/transfers tensors; requesting CUDA can initialize a GPU context.
    """
    torch = require_torch()
    if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 64:
        raise ValueError("a batch must contain between 1 and 64 images")
    valid = [validate_image(image) for image in images]
    return torch.tensor(valid, dtype=torch.float32).unsqueeze(1).div_(255.0).to(device)


def weights_fingerprint(model) -> str:
    """Hash model state names, shapes, dtypes and float32-serialized values.

    Args:
        model (torch.nn.Module): Model exposing state_dict tensor values.

    Returns:
        str: SHA-256 fingerprint independent of each tensor's current device.

    Raises:
        Tensor conversion/iteration errors and float32 packing errors propagate.

    Side effects:
        Copies detached state tensors to contiguous CPU storage. Each scalar is
        serialized as a little-endian float32, so this is intended for the tiny
        float32 model, not arbitrary bit-exact float64/integer state. Use
        fast_weights_fingerprint for state-byte hashing of the larger probe.
    """
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(cpu.dtype), list(cpu.shape)]).encode())
        for item in cpu.reshape(-1).tolist():
            digest.update(struct.pack("<f", float(item)))
    return digest.hexdigest()


class VisionPredictor:
    """Own a copied model snapshot and version-aware synthetic result cache.

    The supplied model is copied to one device and placed in evaluation mode.
    Callers must treat the private snapshot as immutable and replace this object
    when changing weights. Cache operations are locked, but prediction counters and
    model execution are not a general multi-worker serving synchronization contract.
    A caller must authorize users independently; cached results grant no access.
    """
    def __init__(self, model, device="cpu", preprocessing_version=PREPROCESS_VERSION,
                 cache: PredictionCache | None = None):
        """Snapshot a model and select its device, version label and cache.

        Args:
            model (torch.nn.Module): Source model copied without mutating the caller's model.
            device (str | torch.device): Target execution device; default CPU.
            preprocessing_version (str): Label used in cache identity; no alternate
                preprocessing algorithm is installed by changing this string.
            cache (PredictionCache | None): Shared version-scoped cache, or a new cache.

        Returns:
            None.

        Raises:
            VisionDependencyUnavailable: PyTorch is absent. Copy, hash, device transfer
            and allocation failures propagate.

        Side effects:
            Allocates a model copy, possibly a CUDA context, and initializes compute_calls.
        """
        self.torch = require_torch()
        self.weights_version = weights_fingerprint(model)
        self.preprocessing_version = preprocessing_version
        self.device = str(device)
        self._model = copy.deepcopy(model).to(device).eval()
        self.cache = cache if cache is not None else PredictionCache()
        self.compute_calls = 0

    def predict_batch(self, images):
        """Execute an uncached two-class prediction for a grayscale batch.

        Args:
            images (list | tuple): Between 1 and 64 valid grayscale images.

        Returns:
            list[dict]: Predicted index/label, two probabilities, weight/preprocessing
            versions and cache_hit=False for each image, preserving input order.

        Raises:
            Image/preprocessing validation and model/device execution errors propagate.

        Side effects:
            Allocates device tensors and transfers probabilities to CPU. Uses
            inference_mode and increments compute_calls once per successfully processed
            batch; it does not consult or populate the cache.
        """
        tensor = preprocess_images(images, self.device)
        with self.torch.inference_mode():
            probabilities = self._model(tensor).softmax(dim=1).cpu().tolist()
        self.compute_calls += 1
        # The two argmax key lambdas read row[i]; ties keep the first class index.
        return [{"predicted_class": int(max(range(2), key=lambda i: row[i])),
                 "label": CLASS_NAMES[int(max(range(2), key=lambda i: row[i]))],
                 "probabilities": row,
                 "weights_version": self.weights_version,
                 "preprocessing_version": self.preprocessing_version,
                 "cache_hit": False} for row in probabilities]

    def predict(self, pixels, use_cache=True):
        """Predict one image using the versioned cache when requested.

        Args:
            pixels (Sequence[Sequence[int]]): Valid 32-by-32 grayscale image.
            use_cache (bool): Look up and populate the cache when true; default true.

        Returns:
            dict: One prediction; a cached copy is marked cache_hit=True.

        Raises:
            Image/version validation, deep-copy or model errors propagate.

        Side effects:
            Updates cache recency/counters and, on a miss, executes predict_batch.
            Authorization is the caller's responsibility before any cache lookup.
        """
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
    """Train the tiny CPU CNN on caller-provided synthetic labelled examples.

    Args:
        examples (Sequence[SyntheticImage]): Nonempty training fixtures with labels
            accepted by a two-class CrossEntropyLoss; callers supply valid data.
        seed (int): Model seed and minibatch generator seed base; default 47.
        epochs (int): Training epochs, inclusive range 1..100.
        batch_size (int): Positive minibatch size; default 32.

    Returns:
        tuple[torch.nn.Module, dict]: Trained CPU evaluation-mode model and metadata
        including Adam learning rate, loss per epoch and wall duration in seconds.

    Raises:
        ValueError: Epochs fail explicit validation; malformed/empty batches can fail
            preprocessing or range operations. PyTorch training errors propagate.

    Side effects:
        Resets model RNG, allocates tensors and mutates model/optimizer state.
        This function supplies no held-out quality check or real-data claim.
    """
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
    """Measure two-class accuracy on nonempty held-out fixtures.

    Args:
        engine (VisionPredictor): Predictor whose uncached predict_batch is called.
        examples (Sequence[SyntheticImage]): Nonempty labelled evaluation fixtures.

    Returns:
        tuple[dict, list[dict]]: Sample/correct counts, accuracy and 2-by-2 confusion
        matrix (true-label rows, predicted-label columns), plus ordered predictions.

    Raises:
        Predictor errors propagate; empty examples cause division by zero and invalid
        labels can cause indexing errors. The caller must provide the declared task.

    Side effects:
        Executes batches of at most 64 and increments engine compute counters.
        Accuracy describes this fixture split only.
    """
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
    """Linearly interpolate a quantile of a nonempty numeric sequence.

    Args:
        values (Sequence[float]): Observations in any consistent unit.
        q (float): Requested fraction in 0..1, as supplied by internal callers.

    Returns:
        float: Quantile in the input unit using position (n - 1) * q.

    Raises:
        Empty or incompatible input raises normal indexing/arithmetic errors;
        no public validation or NaN filtering is applied. Input order is unchanged.
    """
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    lower = math.floor(location)
    upper = math.ceil(location)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def summarize_timings(seconds, batch_size):
    """Summarize nonempty positive per-call durations measured in seconds.

    Args:
        seconds (Sequence[float]): Successful measurement durations, in seconds.
        batch_size (int): Images processed by each measured call.

    Returns:
        dict: Sample count, batch images, p50/p95/p99/mean in milliseconds and
        aggregate images per second across the measured call time.

    Raises:
        Empty or zero-total durations cause indexing/division errors. Callers must
        supply finite durations and a meaningful fixed batch size.

    Limitations:
        Throughput here covers sequential measured calls, not peak concurrent capacity.
    """
    return {"samples": len(seconds), "batch_size_images": batch_size,
            "p50_ms": _quantile(seconds, 0.50) * 1000,
            "p95_ms": _quantile(seconds, 0.95) * 1000,
            "p99_ms": _quantile(seconds, 0.99) * 1000,
            "mean_ms": sum(seconds) / len(seconds) * 1000,
            "throughput_images_per_second": len(seconds) * batch_size / sum(seconds)}


# The default synchronization lambda is deliberately a CPU no-op, not a GPU barrier.
def measure_calls(call, warmup, repeats, batch_size, synchronize=lambda: None):
    """Warm a callable, then time repeated synchronized wall-clock calls.

    Args:
        call (Callable[[], Any]): Work invoked once per warm-up and measurement.
        warmup (int): Number of unrecorded calls.
        repeats (int): Positive number of measured calls.
        batch_size (int): Images per call, used only for throughput reporting.
        synchronize (Callable[[], Any]): Device barrier; the default no-op lambda
            is appropriate for synchronous CPU work.

    Returns:
        dict: summarize_timings output, converting recorded seconds to milliseconds.

    Raises:
        Callable/barrier errors propagate; invalid or empty timing inputs are not
        independently validated here.

    Side effects:
        Executes caller work and barriers. Each measured call is bracketed by barriers;
        no concurrency, network request or cache policy is introduced by this helper.
    """
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
    """Compare prepared-tensor forward time with the uncached grayscale pipeline.

    Args:
        engine (VisionPredictor): Model snapshot, torch module and device to measure.
        examples (Sequence[SyntheticImage]): At least batch_size fixtures.
        batch_size (int): Images per call, within preprocess_images limits.
        warmup (int): Unrecorded calls per path; default 10.
        repeats (int): Measured calls per path; default 40.

    Returns:
        dict: Device, batch size, model_call and full_pipeline timing summaries with
        separate textual scopes.

    Raises:
        Preprocessing, inference and timing errors propagate.

    Side effects:
        Prepares a device tensor; synchronizes CUDA before/after timed boundaries.
        Full-pipeline calls increment engine counters, with cache disabled.
        Forward-only timing excludes validation, transfer, softmax and result conversion.
    """
    torch = engine.torch
    images = [item.pixels for item in examples[:batch_size]]
    prepared = preprocess_images(images, engine.device)
    # CUDA lambda waits for this device; the CPU lambda needs no asynchronous barrier.
    sync = (lambda: torch.cuda.synchronize(engine.device)) if engine.device.startswith("cuda") else (lambda: None)

    def model_only():
        """Run the captured model on the captured already prepared device tensor.

        Returns:
            torch.Tensor: Raw two-class logits for the fixed batch.

        Raises:
            Model/device errors propagate to measure_calls.

        Side effects:
            Uses inference_mode. The outer CUDA barrier supplies synchronization; this
            closure excludes preprocessing, cache lookup and probability conversion.
        """
        with torch.inference_mode():
            return engine._model(prepared)

    return {"device": engine.device, "batch_size_images": batch_size,
            "model_call": measure_calls(model_only, warmup, repeats, batch_size, sync),
            # Capture the same images for every uncached full-pipeline measurement.
            "full_pipeline": measure_calls(lambda: engine.predict_batch(images), warmup, repeats, batch_size, sync),
            "model_scope": "Synchronized wall time for forward call on an already prepared device tensor",
            "pipeline_scope": "Input validation, tensor conversion/normalization, device transfer, forward call, softmax and CPU result conversion; cache disabled"}


def demonstrate_cache(engine, pixels):
    """Exercise synthetic cache reuse and version/input invalidation.

    Args:
        engine (VisionPredictor): Prefer a fresh predictor/cache for a cold first call.
        pixels (Sequence[Sequence[int]]): One valid grayscale fixture.

    Returns:
        dict: Individual boolean checks, combined passed flag, key components and
        explicit authorization/preprocessing limitations.

    Raises:
        Prediction, copy, parameter mutation and hashing errors propagate.

    Side effects:
        Performs two predictions, increments cache counters, and changes a separate
        CPU copy of model weights plus a copied image to construct distinct keys.
        The owned engine weights and original pixels remain unchanged. Changing only
        a preprocessing label demonstrates invalidation, not acceptance of new code.
    """
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
    """Construct an unstarted sequential grayscale HTTP demo on IPv4 loopback.

    Args:
        engine (VisionPredictor): Snapshot used for uncached /predict requests.
        port (int): TCP port; 0 asks the OS to allocate an available local port.

    Returns:
        HTTPServer: Caller-owned server bound to 127.0.0.1. Call serve_forever in a
        thread or the foreground and explicitly shutdown/server_close when finished.

    Raises:
        OSError: Bind or socket creation fails. Handler-specific input errors become
            HTTP responses; model errors not caught by the handler can propagate.

    Side effects:
        Binds a local socket immediately, without starting a serving thread.
        This synthetic endpoint has no authentication, TLS or production access claim.
    """
    class Handler(BaseHTTPRequestHandler):
        """Handle the captured predictor's synthetic grayscale localhost requests.

        GET /health reports the model version; POST /predict accepts bounded JSON
        pixels and bypasses the result cache. This class captures engine from its
        factory. HTTPServer invokes one request at a time; no identity is authenticated.
        """
        def log_message(self, *args):
            """Suppress BaseHTTPRequestHandler's default stderr request logging.

            Args:
                *args (Any): Formatting string and substitution values supplied by the
                    standard-library handler; intentionally ignored.

            Returns:
                None. No request body, access record or audit evidence is retained here.
            """
            pass

        def _reply(self, code, payload):
            """Serialize a JSON payload and write one HTTP response.

            Args:
                code (int): HTTP status code.
                payload (JSON-serializable object): Response body.

            Returns:
                None.

            Raises:
                Serialization and socket/write errors propagate.

            Side effects:
                Writes UTF-8 JSON plus Content-Type and byte-accurate Content-Length headers
                on the current request connection; adds no authentication or CORS policy.
            """
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            """Serve /health, or return JSON HTTP 404 for another path.

            Returns:
                None: The response is written to the request socket.

            Side effects:
                Reveals only the synthetic scope, engine device and weight version.
                Executes no inference. JSON/socket errors from _reply propagate.
            """
            if self.path != "/health":
                return self._reply(404, {"error": "not found"})
            self._reply(200, {"scope": "synthetic localhost demo", "device": engine.device,
                              "weights_version": engine.weights_version})

        def do_POST(self):
            """Validate a bounded /predict JSON request and run uncached inference.

            Returns:
                None: Writes HTTP 200 prediction, 404 for another path, 413 for a declared
                body length outside 1..65536 bytes, or 400 for handled input errors.

            Raises:
                Model/runtime and socket errors outside ValueError, TypeError and
                UnicodeDecodeError propagate through the standard-library server.

            Side effects:
                Reads the declared body length and executes engine.predict(use_cache=False).
                Only the captured synthetic grayscale task is accepted; no identity is checked.
            """
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
    """Measure sequential end-to-end requests against an owned localhost server.

    Args:
        engine (VisionPredictor): Predictor whose cache is bypassed.
        pixels (Sequence[Sequence[int]]): JSON-serializable valid grayscale fixture.
        warmup (int): Unrecorded HTTP calls; default 5.
        repeats (int): Measured HTTP calls; default 20.

    Returns:
        dict: Device, latency/throughput summaries and scope/authentication metadata.

    Raises:
        HTTP, JSON, model-version mismatch or timing errors propagate after cleanup.

    Side effects:
        Binds an ephemeral loopback port and starts one daemon serving thread.
        Finally shuts down and closes its server, then joins for up to five seconds.
        Each client call has a ten-second timeout. This is sequential wall time, not
        a concurrent load or production-network measurement.
    """
    from urllib.request import Request, urlopen
    server = create_local_http_server(engine)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/predict"
    payload = json.dumps({"pixels": pixels}).encode("utf-8")

    def request():
        """Send one captured JSON fixture and verify its returned model version.

        Returns:
            None: The response is consumed and only its weights_version is checked.

        Raises:
            RuntimeError: Returned weights_version differs from the captured predictor.
            HTTP, timeout and JSON errors propagate to the outer measurement.

        Side effects:
            Opens a localhost HTTP request with a ten-second timeout and closes its
            response context. The enclosing factory owns serving-thread cleanup.
        """
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
    """Construct a random-weight ResNet-18-shaped RGB performance probe.

    Args:
        seed (int): Global PyTorch initialization seed; default 47.
        num_classes (int): Output logit width; default 1000. Callers should supply a
            meaningful positive dimension; no explicit dimension validation occurs.

    Returns:
        torch.nn.Module: CPU evaluation-mode residual network with [2,2,2,2] basic
        block stages and random weights.

    Raises:
        VisionDependencyUnavailable: PyTorch is absent.
        PyTorch construction errors propagate.

    Side effects:
        Resets torch's RNG and allocates convolution/batch-normalization/classifier
        parameters. No torchvision, pretrained weights, download, trained semantics
        or quality claim is introduced.
    """
    torch = require_torch()
    nn = torch.nn

    class BasicBlock(nn.Module):
        """Implement one random-weight residual block captured by the probe factory.

        The main path has two 3-by-3 convolutions with batch normalization. A projected
        1-by-1 identity path handles stride/channel changes; otherwise it is identity.
        Input/output tensors use the normal (batch, channels, height, width) layout.
        """
        def __init__(self, incoming, outgoing, stride=1):
            """Allocate main and residual projection paths.

            Args:
                incoming (int): Input channels.
                outgoing (int): Output channels.
                stride (int): First-convolution/projection spatial stride; default 1.

            Returns:
                None.

            Raises:
                PyTorch layer-construction/allocation errors propagate.

            Side effects:
                Registers convolution, batch-normalization and ReLU submodules.
                The outer probe later applies its own weight initialization.
            """
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
            """Apply the residual block to one prepared RGB-feature tensor.

            Args:
                x (torch.Tensor): Floating tensor shaped (batch, incoming, height, width)
                    on the same device/dtype as this block.

            Returns:
                torch.Tensor: ReLU-transformed sum, with outgoing channels and the configured
                downsampling stride.

            Raises:
                PyTorch shape/device errors propagate. ReLU is configured in-place;
                normal PyTorch training/evaluation and autograd rules still apply.
            """
            identity = self.downsample(x)
            x = self.relu(self.bn1(self.conv1(x)))
            return self.relu(self.bn2(self.conv2(x)) + identity)

    class ResNet18Probe(nn.Module):
        """Combine an RGB stem, eight residual blocks and an output classifier.

        The nested class captures num_classes and torch.nn from the factory.
        Its default weights carry no trained image-label meaning.
        """
        def __init__(self):
            """Build the residual network and initialize its learnable state.

            Returns:
                None.

            Raises:
                PyTorch layer-construction/allocation errors propagate.

            Side effects:
                Creates the 7-by-7 RGB stem, [2,2,2,2] block stages, adaptive pooling and
                num_classes-wide linear head. Applies Kaiming-normal convolution weights
                and unit/zero batch-normalization parameters. RNG seeding is owned by the
                enclosing factory immediately before this constructor is called.
            """
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
            """Return raw logits from a caller-prepared RGB tensor.

            Args:
                x (torch.Tensor): Floating (batch, 3, height, width) tensor, normally
                    image_size square and compatible with the model's device/dtype.

            Returns:
                torch.Tensor: Shape (batch, num_classes), with raw unnormalized logits.

            Raises:
                PyTorch shape/device errors propagate. This method performs no input-byte
                validation, normalization, softmax, caching or quality assessment.
            """
            return self.fc(self.pool(self.layers(self.stem(x))).flatten(1))

    torch.manual_seed(seed)
    return ResNet18Probe().eval()


def fast_weights_fingerprint(model):
    """Hash contiguous model state bytes with tensor name, dtype and shape.

    Args:
        model (torch.nn.Module): State tensors convertible to contiguous CPU NumPy arrays.

    Returns:
        str: SHA-256 fingerprint of sorted state entries, independent of device.

    Raises:
        Tensor transfer or NumPy conversion errors propagate; unsupported tensor
        dtypes are not silently cast.

    Side effects:
        Detaches and copies state to CPU as needed. Unlike weights_fingerprint, this
        serializes the tensor's actual NumPy bytes rather than scalar float32 values.
    """
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(cpu.dtype), list(cpu.shape)]).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def latency_summary(milliseconds):
    """Summarize already-millisecond latency observations.

    Args:
        milliseconds (Sequence[float]): Samples in milliseconds, or an empty sequence.

    Returns:
        dict: Sample count and interpolated p50/p95/p99/max milliseconds. Empty
        input yields zero samples and None for all percentiles/max.

    Raises:
        Sorting/arithmetic errors propagate for incompatible values. Finite,
        nonnegative samples are the caller's responsibility.
    """
    if not milliseconds:
        return {"samples": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    return {"samples": len(milliseconds), "p50_ms": _quantile(milliseconds, .50),
            "p95_ms": _quantile(milliseconds, .95), "p99_ms": _quantile(milliseconds, .99),
            "max_ms": max(milliseconds)}


class ProbeQueueFull(RuntimeError):
    """Signal immediate rejection when the bounded inference queue has no slot.

    This RuntimeError subclass is raised by submit and translated to HTTP 429 by
    the binary probe handler. It does not mean a model execution failed.
    """
    pass


class BoundedBatchVisionService:
    """Run a bounded synthetic RGB queue with one or two batching workers.

    Each accepted payload has exactly image_size * image_size * 3 uint8 bytes.
    Workers collect up to max_batch requests, with a collection timer starting only
    after the first request is removed. Queue wait before that moment is additional.
    No cache or authentication is provided. Metrics/counters use a lock; the model
    callback is responsible for safe independent worker instances.

    Callers must stop submissions before close and must not cancel returned futures:
    submission/closure is not one atomic transaction, and cancellation is not a
    supported path in the worker completion code. A hanging callback can outlive the
    30-second join and causes close to raise rather than prove complete shutdown.
    """
    def __init__(self, batch_call, *, image_size=224, workers=1, max_batch=8,
                 max_wait_ms=5, queue_size=64):
        """Validate bounds, create the shared queue and start daemon workers.

        Args:
            batch_call (Callable): Called as batch_call(worker_index, payloads), returning
                an equally sized list of result dicts and timing dict containing model_ms.
            image_size (int): RGB square width/height in pixels, 32..224.
            workers (int): One or two threads/model-instance indices.
            max_batch (int): Maximum requests collected per batch, 1..8.
            max_wait_ms (float): Maximum collection wait after the first dequeue, 0..5 ms.
            queue_size (int): Pending-request capacity, 1..256.

        Returns:
            None.

        Raises:
            ValueError: Configuration bounds fail. Thread creation errors propagate.

        Side effects:
            Immediately starts worker threads; owner must eventually call close.
            Queue capacity counts waiting requests, not currently executing batches.
        """
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
        """Validate and enqueue one RGB payload without waiting for capacity.

        Args:
            payload (bytes): Exactly image_size * image_size * 3 interleaved RGB bytes.

        Returns:
            concurrent.futures.Future: Resolves to a result dict augmented with queue
            wait, service latency in milliseconds and actual batch size.

        Raises:
            ValueError: Type/length is invalid.
            RuntimeError: Closure has been requested.
            ProbeQueueFull: Queue is currently full; rejection count is incremented.

        Side effects:
            Captures submission time, enqueues work and updates observed queue depth.
            Stop concurrent submissions before close; future cancellation is unsupported.
        """
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
        """Drain the shared queue through one worker's model callback.

        Args:
            worker_index (int): Index selecting the independent model snapshot/stream.

        Returns:
            None: Runs until stop is set and the waiting queue has drained.

        Raises:
            Ordinary callback exceptions are placed on every future in that batch.
            Unsupported future cancellation/completion errors can escape worker handling.

        Side effects:
            Dequeues and batches requests; collection wait is bounded separately from
            total queue latency. Records completed-batch metrics under the lock, resolves
            futures, and marks each queued item done. No per-request hard deadline
            cancels model computation; a blocked callback can delay shutdown.
        """
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
        """Snapshot completed-batch throughput and latency counters under the lock.

        Returns:
            dict: Completed batch/image totals, occupancy histogram, mean batch size,
            queue rejections/depth, and model/pipeline latency summaries in milliseconds.

        Raises:
            Missing or nonnumeric model_ms values from a malformed callback can cause
            aggregation errors.

        Side effects:
            None beyond a temporary locked read. Failed batches are absent from records;
            caller-level HTTP accounting is required to retain their requests/errors.
        """
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
        """Request queue draining and join each owned worker for up to 30 seconds.

        Returns:
            None: All checked worker threads have stopped.

        Raises:
            RuntimeError: Any worker is still alive after its join allowance.

        Side effects:
            Sets stop, prevents ordinary later submissions and waits for owned threads.
            It does not terminate a running callback, close an HTTP server, or manage
            other processes. Owners must stop new submissions before invoking this method.
        """
        self.stop.set()
        for worker in self.worker_threads:
            worker.join(timeout=30)
        if any(worker.is_alive() for worker in self.worker_threads):
            raise RuntimeError("A batch worker did not stop within 30 seconds")


def make_resnet_batch_runtime(base_model, device="cpu", precision="fp32", workers=1, image_size=224,
                              class_names=None):
    """Prepare per-worker model copies and a timed RGB batch callback.

    Args:
        base_model (torch.nn.Module): Unchanged source probe copied for each worker.
        device (str): CPU or a CUDA device string; default cpu.
        precision (str): fp32, or fp16 only for a CUDA device.
        workers (int): Number of model snapshots; normally one or two.
        image_size (int): Square RGB width/height in pixels; normally 32..224.
        class_names (Sequence[str] | None): None keeps the random-weight probe
            result format. A sequence whose length equals the model's logit width
            marks caller-supplied trained weights and adds predicted_class/label.

    Returns:
        tuple[Callable, list[torch.nn.Module], list]: batch_call, copied evaluation
        models and one CUDA stream per model (or None entries on CPU).

    Raises:
        ValueError: Precision is unsupported or FP16 is requested for CPU.
        PyTorch device/allocation errors propagate; caller validates other bounds.
        A class index outside class_names raises IndexError inside batch_call.

    Side effects:
        Allocates device model copies/streams and synchronizes their initialization.
        Sets no allocator budget and downloads nothing. Without class_names, outputs
        remain performance-only because the probe has random weights. With
        class_names, this function trusts the caller's claim that the weights are
        trained; quality comes only from a separate held-out evaluation.
    """
    torch = require_torch()
    if precision not in ("fp32", "fp16") or (precision == "fp16" and not device.startswith("cuda")):
        raise ValueError("FP16 is measured only on the explicitly selected CUDA route")
    dtype = torch.float16 if precision == "fp16" else torch.float32
    models = [copy.deepcopy(base_model).to(device=device, dtype=dtype).eval() for _ in range(workers)]
    streams = [torch.cuda.Stream(device=device) for _ in models] if device.startswith("cuda") else [None] * workers
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    def batch_call(worker_index, payloads):
        """Run one worker's captured model on raw synthetic RGB payloads.

        Args:
            worker_index (int): Valid index into captured models and CUDA streams.
            payloads (Sequence[bytes]): Nonempty bodies containing image_size-square
                interleaved RGB uint8 values; the service validates body lengths first.

        Returns:
            tuple[list[dict], dict]: Top-class indices/probabilities with explicit
            random_weights/quality_claimed flags, plus forward-only model_ms. When
            the factory received class_names, each result also carries
            predicted_class and label, and random_weights is False.

        Raises:
            RuntimeError: Any returned top probability is non-finite.
            Indexing, shape, allocation and PyTorch execution errors propagate.

        Side effects:
            Copies buffers to CPU tensors, converts HWC to CHW, divides by 255, stacks,
            transfers to device/precision, executes inference and copies results to CPU.
            CUDA events time forward execution on that worker's stream; its final
            synchronization completes asynchronous work before returning. CPU timing
            brackets forward only. No class semantics, cache or image decoder is supplied.
        """
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
            if class_names is not None:
                return [{"top_class_index": int(i), "predicted_class": int(i), "label": class_names[int(i)],
                         "top_probability": float(v), "random_weights": False, "quality_claimed": False}
                        for i, v in zip(indexes, values)], {"model_ms": model_ms}
            return [{"top_class_index": int(i), "top_probability": float(v),
                     "random_weights": True, "quality_claimed": False} for i, v in zip(indexes, values)], {"model_ms": model_ms}

    return batch_call, models, streams


def make_payload_cache_key(payload, weights_version, preprocessing_version):
    """Bind a cached RGB prediction to weights, preprocessing and exact body bytes.

    Args:
        payload (bytes): Raw RGB request body exactly as received.
        weights_version (str): Nonempty fingerprint of the serving weights.
        preprocessing_version (str): Nonempty preprocessing-version label.

    Returns:
        tuple[str, str, str]: Weight version, preprocessing version and the
        SHA-256 hexadecimal digest of the body bytes.

    Raises:
        ValueError: Either version label is empty.
        TypeError: payload is not a bytes-like object accepted by hashlib.

    Limitations:
        Only byte-identical bodies share a key. A re-captured camera frame of the
        same object normally differs in noise and therefore misses. No user,
        tenant or authorization scope is encoded.
    """
    if not weights_version or not preprocessing_version:
        raise ValueError("weight and preprocessing versions are required")
    return weights_version, preprocessing_version, hashlib.sha256(payload).hexdigest()


def create_batched_http_server(service, port=0, cache=None, cache_scope=None):
    """Bind a threaded synthetic RGB probe server to IPv4 loopback.

    Args:
        service (BoundedBatchVisionService): Existing started batching service.
        port (int): TCP port; 0 requests an OS-allocated local port.
        cache (PredictionCache | None): Optional exact-body result cache consulted
            before the bounded queue. None keeps the original uncached behaviour.
        cache_scope (tuple[str, str] | None): Weights version and preprocessing
            version bound into every cache key; required when cache is supplied.

    Returns:
        ThreadingHTTPServer: Bound but not serving until the caller invokes
        serve_forever. The caller owns server and service cleanup separately.

    Raises:
        ValueError: cache is supplied without a two-part cache_scope.
        OSError: Socket bind/allocation fails.

    Side effects:
        Binds 127.0.0.1; request threads are daemon threads with listen backlog 128.
        No authentication, TLS, sensitive-data processing or production SLA is claimed.
        With a cache, a hit is answered without queueing or model execution and is
        marked cache_hit=True with queue_wait_ms=0 and batch_size=0. Concurrent
        identical misses are not coalesced: each executes the model.
    """
    if cache is not None and (not isinstance(cache_scope, tuple) or len(cache_scope) != 2):
        raise ValueError("A cache requires a (weights_version, preprocessing_version) scope")
    class Server(ThreadingHTTPServer):
        """Configure the captured RGB server's daemon threads and listen backlog.

        ThreadingHTTPServer supplies lifecycle methods. daemon_threads=True prevents
        request threads from keeping the interpreter alive; request_queue_size=128 is
        the TCP listen backlog, distinct from the model's bounded inference queue.
        """
        daemon_threads = True
        request_queue_size = 128

    class Handler(BaseHTTPRequestHandler):
        """Translate binary localhost prediction requests into service futures.

        This factory-local handler captures service. Only POST /predict-rgb is handled;
        its connection and result waits are bounded, but timeouts do not cancel queued
        or executing inference. Access control is intentionally absent.
        """
        def log_message(self, *args):
            """Suppress the standard handler's stderr request logging.

            Args:
                *args (Any): Standard-library log format and substitutions, ignored.

            Returns:
                None. This provides neither request logging nor an audit trail.
            """
            pass

        def reply(self, code, value):
            """Write a JSON status response, tolerating a disconnected probe client.

            Args:
                code (int): HTTP response status.
                value (JSON-serializable object): Body value.

            Returns:
                None.

            Raises:
                Serialization/header errors propagate. BrokenPipeError and
                ConnectionResetError during body writing are intentionally swallowed.

            Side effects:
                Writes JSON, Content-Type and byte Content-Length on this connection.
            """
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
            """Accept a correctly sized RGB body and await its inference future.

            Returns:
                None: Writes HTTP 200 for a result, 404 for another path, 400 for malformed
                length/value input, 429 for queue saturation, or 503 for other caught errors.

            Side effects:
                Sets a 15-second socket timeout, reads the declared exact body and waits up
                to 15 seconds for the service future. This timeout does not cancel inference.
                Error messages are capped at 300 characters for generic exceptions.
                Request identity is not authenticated; the endpoint is a local synthetic probe.
                When the factory received a cache, the body digest is looked up first;
                a miss stores the model result without its per-request timing fields.
            """
            if self.path != "/predict-rgb":
                return self.reply(404, {"error": "not found"})
            self.connection.settimeout(15)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length != service.body_bytes:
                    return self.reply(400, {"error": "Incorrect RGB uint8 body size"})
                payload = self.rfile.read(length)
                if cache is None:
                    result = service.submit(payload).result(timeout=15)
                    return self.reply(200, result)
                lookup_started = time.perf_counter()
                key = make_payload_cache_key(payload, *cache_scope)
                cached = cache.get(key)
                if cached is not None:
                    return self.reply(200, {**cached, "cache_hit": True, "queue_wait_ms": 0.0,
                                            "service_latency_ms": (time.perf_counter() - lookup_started) * 1000,
                                            "batch_size": 0})
                result = service.submit(payload).result(timeout=15)
                cache.put(key, {k: v for k, v in result.items()
                                if k not in ("queue_wait_ms", "service_latency_ms", "batch_size")})
                self.reply(200, {**result, "cache_hit": False})
            except ProbeQueueFull as exc:
                self.reply(429, {"error": str(exc)})
            except ValueError as exc:
                self.reply(400, {"error": str(exc)})
            except Exception as exc:
                self.reply(503, {"error_type": type(exc).__name__, "error": str(exc)[:300]})

    return Server(("127.0.0.1", port), Handler)


def run_open_loop_http(server, payloads, target_rps=100, duration_seconds=10):
    """Offer scheduled localhost requests independently of response completion.

    Args:
        server (HTTPServer): Already-serving RGB server with a loopback port.
        payloads (Sequence[bytes]): Nonempty cyclic input fixture list; caller owns
            image validation and supplies the required body dimensions.
        target_rps (int | float): Offered arrival rate, 1..200 requests per second.
        duration_seconds (int | float): Injection window, 1..30 seconds.

    Returns:
        dict: Offered/submitted/completed/success/error/drop accounting, observed
        rates and p50/p95/p99/max latency summaries. Successful-request percentiles
        exclude HTTP failures; all_response_latency retains completed errors too.

    Raises:
        ValueError: Rate/duration bounds fail.
        Scheduling/submission or malformed caller-input errors can propagate.
        Ordinary request/network/JSON failures are returned as client_error records.

    Side effects:
        Opens one HTTP connection per request, using up to 128 client threads and
        256 pending slots. Arrivals use absolute due times; client saturation drops
        requests and records them instead of waiting for previous responses.
        Waits for submitted calls to drain, but does not close server or service.
        Network calls have 16-second timeouts; total run time may exceed the injection
        window. Reports characterize this localhost workload, not sustainable capacity.
    """
    import http.client
    from concurrent.futures import ThreadPoolExecutor
    if not 1 <= target_rps <= 200 or not 1 <= duration_seconds <= 30:
        raise ValueError("Use offered rate 1..200 req/s and duration 1..30 seconds")
    count = int(target_rps * duration_seconds)
    results, submitted, dropped, lateness = [], [], 0, []
    pending_slots = threading.BoundedSemaphore(256)

    def request(payload, due):
        """Send one captured RGB request and record wall-clock stage latencies.

        Args:
            payload (bytes): One synthetic image body.
            due (float): Scheduled perf_counter arrival timestamp in seconds.

        Returns:
            dict: HTTP status or client_error, completion timestamp, HTTP milliseconds,
            scheduled-arrival milliseconds, optional server queue/service milliseconds
            and error type.

        Side effects:
            Opens a 16-second-timeout localhost connection and always closes it and
            releases the captured pending-slot semaphore after the request block.
            Network/JSON exceptions become error records, without retrying or cancelling
            server work. The measured HTTP time starts when this client worker begins,
            while scheduled_arrival_ms also includes client executor/scheduler delay.
        """
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


# ---------------------------------------------------------------------------
# Trained synthetic packaging-line classifier and multi-process load ramp.
# ---------------------------------------------------------------------------

PACKAGING_CLASS_NAMES = ("good", "missing_cap", "scratch", "misaligned_label")
PACKAGING_PREPROCESS_VERSION = "rgb-u8-hwc-to-chw-float32-div255-v1"
# Bottle, cap, label and label-band base colours; each draw adds a small jitter.
_BOTTLE_COLOURS = ((150, 88, 34), (46, 112, 62), (168, 178, 184), (52, 82, 148))
_CAP_COLOURS = ((196, 42, 40), (228, 228, 222), (40, 70, 170), (34, 34, 38), (220, 186, 40), (150, 152, 158))
_LABEL_COLOURS = ((236, 234, 226), (240, 226, 186), (238, 238, 168), (200, 222, 240))
_BAND_COLOURS = ((180, 30, 40), (30, 90, 170), (30, 130, 70), (90, 40, 120))


@dataclass(frozen=True)
class PackagingImage:
    """Describe one immutable procedural packaging-line RGB image.

    Attributes:
        image_id (str): Split-prefixed fixture identifier, not a real image identity.
        label (int): Index into PACKAGING_CLASS_NAMES.
        payload (bytes): Raw interleaved HWC RGB uint8 bytes, exactly the HTTP body.
        sha256 (str): SHA-256 hexadecimal digest of payload, used for split checks.
    """
    image_id: str
    label: int
    payload: bytes
    sha256: str


def _jitter_colour(rng, colour, spread):
    """Return one RGB colour with independent uniform jitter on each channel.

    Args:
        rng (random.Random): Caller-owned generator; three draws are consumed.
        colour (Sequence[float]): Base red, green and blue values in 0..255.
        spread (float): Maximum absolute change per channel.

    Returns:
        list[float]: Jittered channels clipped to 0..255.

    Raises:
        Arithmetic errors propagate for non-numeric input. No validation occurs.
    """
    return [max(0.0, min(255.0, c + rng.uniform(-spread, spread))) for c in colour]


def _segment_distance(xx, yy, x0, y0, x1, y1):
    """Compute each pixel centre's Euclidean distance to one line segment.

    Args:
        xx (torch.Tensor): Pixel-centre x coordinates, shape (height, width).
        yy (torch.Tensor): Pixel-centre y coordinates with the same shape.
        x0 (float): Segment start x in pixels.
        y0 (float): Segment start y in pixels.
        x1 (float): Segment end x in pixels.
        y1 (float): Segment end y in pixels; the segment must have nonzero length.

    Returns:
        torch.Tensor: Float distance map with the shape of xx.

    Raises:
        ZeroDivisionError: Start and end coincide. Tensor errors propagate.
    """
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    t = (((xx - x0) * dx + (yy - y0) * dy) / length_sq).clamp(0, 1)
    return ((xx - (x0 + t * dx)) ** 2 + (yy - (y0 + t * dy)) ** 2).sqrt()


def draw_packaging_image(rng, generator, label, image_size=224, grid=None):
    """Render one procedural bottle on a conveyor as raw RGB bytes.

    Args:
        rng (random.Random): Geometry/colour generator; consumed in a fixed order.
        generator (torch.Generator): CPU generator for pixel noise.
        label (int): 0 good, 1 missing_cap, 2 scratch, 3 misaligned_label.
        image_size (int): Square side in pixels, 32..224; geometry scales with it.
        grid (tuple[torch.Tensor, torch.Tensor] | None): Optional precomputed
            (yy, xx) pixel-centre grids for image_size, reused across images.

    Returns:
        bytes: image_size * image_size * 3 interleaved HWC RGB uint8 values.

    Raises:
        ValueError: label or image_size is outside the supported range.
        VisionDependencyUnavailable: PyTorch is absent.

    Side effects:
        Consumes rng and generator state; allocates CPU tensors only.

    Scene:
        Every class shares nuisance variation: background tint and gradient, belt
        slats, bottle position/size/colour, cylindrical shading, a vertical glare
        band, thin dark label print lines, dust specks, global gain and Gaussian
        noise. Good labels sit within +/-2 degrees and +/-4 px (at 224 px).
        missing_cap removes the cap and shows a dark neck opening with threads.
        scratch adds one to three thin bright or dark line segments on the body,
        14..55 px long and 22..70 grey levels of contrast. misaligned_label
        rotates the label 6..15 degrees, shifts it 12..22 px, or combines 4..8
        degrees with 8..13 px. These are fictional rules, not a real line spec.
    """
    torch = require_torch()
    if label not in range(len(PACKAGING_CLASS_NAMES)):
        raise ValueError("label must index PACKAGING_CLASS_NAMES")
    if not 32 <= image_size <= 224:
        raise ValueError("image_size must be between 32 and 224")
    size = image_size
    s = size / 224.0
    if grid is None:
        axis = torch.arange(size, dtype=torch.float32) + 0.5
        grid = torch.meshgrid(axis, axis, indexing="ij")
    yy, xx = grid
    # Background wall: tinted grey with a vertical gradient.
    base = rng.uniform(70, 120)
    gradient = rng.uniform(-25, 25)
    tint = [rng.uniform(-8, 8) for _ in range(3)]
    img = torch.stack([base + tint[c] + gradient * (yy / size - 0.5) for c in range(3)])
    # Conveyor belt with periodic slats along the bottom edge.
    belt_top = size * (0.86 + rng.uniform(-0.02, 0.02))
    slat_offset = rng.uniform(0, 16) * s
    belt_value = img * 0.55 + 8
    slats = torch.remainder(xx + slat_offset, 16 * s) < 2 * s
    belt_value = torch.where(slats, belt_value * 0.8, belt_value)
    img = torch.where(yy >= belt_top, belt_value, img)
    # Bottle geometry: body, tapered shoulder and neck standing on the belt.
    cx = size / 2 + rng.uniform(-14, 14) * s
    body_w = rng.uniform(70, 88) * s
    bottom = belt_top + rng.uniform(-2, 2) * s
    body_top = bottom - rng.uniform(105, 120) * s
    neck_w = rng.uniform(24, 32) * s
    shoulder_h = rng.uniform(18, 26) * s
    neck_bottom = body_top - shoulder_h
    neck_top = neck_bottom - rng.uniform(20, 28) * s
    dx = (xx - cx).abs()
    taper = ((yy - neck_bottom) / shoulder_h).clamp(0, 1)
    half_width = torch.where(yy >= body_top, torch.full_like(yy, body_w / 2),
                             neck_w / 2 + (body_w / 2 - neck_w / 2) * taper)
    bottle = (yy >= neck_top) & (yy <= bottom) & (dx <= half_width)
    body = bottle & (yy >= body_top)
    shade = 0.62 + 0.38 * torch.cos((dx / half_width.clamp(min=1.0)).clamp(0, 1) * (math.pi / 2))
    colour = _jitter_colour(rng, rng.choice(_BOTTLE_COLOURS), 15)
    surface = torch.stack([colour[c] * shade for c in range(3)])
    # Label: a band wrapped on the body; misalignment rotates and/or shifts it.
    label_h = rng.uniform(42, 56) * s
    body_mid = (body_top + bottom) / 2
    angle, shift = rng.uniform(-2, 2), rng.uniform(-4, 4) * s
    if label == 3:
        mode = rng.choice(("rotate", "shift", "both"))
        sign_a, sign_s = rng.choice((-1, 1)), rng.choice((-1, 1))
        if mode == "rotate":
            angle = sign_a * rng.uniform(6, 15)
        elif mode == "shift":
            shift = sign_s * rng.uniform(12, 22) * s
        else:
            angle, shift = sign_a * rng.uniform(4, 8), sign_s * rng.uniform(8, 13) * s
    theta = math.radians(angle)
    ly = body_mid + shift
    u = (xx - cx) * math.cos(theta) + (yy - ly) * math.sin(theta)
    v = -(xx - cx) * math.sin(theta) + (yy - ly) * math.cos(theta)
    label_mask = body & (v.abs() <= label_h / 2)
    paper = _jitter_colour(rng, rng.choice(_LABEL_COLOURS), 10)
    band = _jitter_colour(rng, rng.choice(_BAND_COLOURS), 15)
    band_centre = -label_h / 2 + rng.uniform(8, 12) * s
    band_half = rng.uniform(3, 5) * s
    band_mask = label_mask & ((v - band_centre).abs() <= band_half)
    for c in range(3):
        surface[c] = torch.where(label_mask, paper[c] * shade, surface[c])
        surface[c] = torch.where(band_mask, band[c] * shade, surface[c])
    # Nuisance: thin dark print lines parallel to the label edges.
    for _ in range(rng.randint(2, 4)):
        line_v = rng.uniform(band_centre + band_half + 4 * s, label_h / 2 - 3 * s)
        start = rng.uniform(-0.6, 0.0) * body_w / 2
        end = start + rng.uniform(0.25, 0.6) * body_w
        print_mask = label_mask & ((v - line_v).abs() <= 0.8 * s) & (u >= start) & (u <= end)
        surface = torch.where(print_mask, surface * 0.35, surface)
    # Nuisance: soft vertical glare band on the glass.
    glare_x = cx - body_w / 2 * rng.uniform(0.3, 0.5)
    glare_sigma = rng.uniform(2.5, 5.0) * s
    surface = surface + rng.uniform(10, 40) * torch.exp(-((xx - glare_x) ** 2) / (2 * glare_sigma ** 2))
    if label == 2:
        for _ in range(rng.choice((1, 1, 2, 3))):
            length = rng.uniform(14, 55) * s
            direction = rng.uniform(0, math.pi)
            mx = cx + rng.uniform(-0.75, 0.75) * body_w / 2
            my = rng.uniform(body_top + 8 * s, bottom - 8 * s)
            hx, hy = math.cos(direction) * length / 2, math.sin(direction) * length / 2
            width = rng.uniform(0.9, 1.8) * s
            contrast = rng.uniform(22, 70) * (1 if rng.random() < 0.6 else -1)
            distance = _segment_distance(xx, yy, mx - hx, my - hy, mx + hx, my + hy)
            profile = (1 - distance / (width / 2 + 0.5)).clamp(0, 1) * body
            surface = surface + contrast * profile
    img = torch.where(bottle, surface, img)
    if label == 1:
        # Missing cap: dark neck opening and two visible thread lines.
        rim = ((xx - cx) / (neck_w / 2)) ** 2 + ((yy - neck_top) / (2.5 * s)) ** 2 <= 1
        img = torch.where(rim, img * 0.35, img)
        for k in (1, 2):
            thread = ((yy - (neck_top + (3 + 4 * k) * s)).abs() <= 0.7 * s) & (dx <= neck_w / 2)
            img = torch.where(thread, img * 0.75, img)
    else:
        cap_h = rng.uniform(13, 18) * s
        cap_w = neck_w + rng.uniform(5, 9) * s
        cap = (dx <= cap_w / 2) & (yy >= neck_top - cap_h) & (yy < neck_top + 2 * s)
        cap_colour = _jitter_colour(rng, rng.choice(_CAP_COLOURS), 12)
        cap_shade = 0.7 + 0.3 * torch.cos((dx / (cap_w / 2)).clamp(0, 1) * (math.pi / 2))
        ridges = torch.remainder(xx - cx + 100, 3 * s) < 1.2 * s
        cap_shade = torch.where(ridges, cap_shade * 0.82, cap_shade)
        cap_value = torch.stack([cap_colour[c] * cap_shade for c in range(3)])
        img = torch.where(cap, cap_value, img)
    # Nuisance: dust specks anywhere in the frame.
    for _ in range(rng.randint(0, 6)):
        px, py = rng.uniform(0, size), rng.uniform(0, size)
        radius = rng.uniform(0.8, 2.0) * s
        speck = (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2
        img = torch.where(speck, img + rng.uniform(-50, 50), img)
    img = img * rng.uniform(0.88, 1.12)
    img = img + torch.randn(3, size, size, generator=generator) * rng.uniform(2, 6)
    pixels = img.clamp(0, 255).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    return pixels.numpy().tobytes()


def generate_packaging_images(count, seed, split="fixture", image_size=224):
    """Generate deterministic, class-balanced procedural packaging images.

    Args:
        count (int): Number of images, inclusive range 4..20000.
        seed (int): Seed for both the geometry random.Random and the noise generator.
        split (str): Prefix for image_id values; default fixture.
        image_size (int): Square side in pixels, 32..224.

    Returns:
        list[PackagingImage]: Labels cycle 0,1,2,3 so each class has count/4 images
        when count is a multiple of four. Payload hashes are precomputed.

    Raises:
        ValueError: count or image_size is out of bounds.
        VisionDependencyUnavailable: PyTorch is absent.

    Side effects:
        Allocates CPU tensors without changing global random or torch RNG state.
        The same seed reproduces identical bytes with the same torch build; the
        aggregate hash recorded by packaging_split_report lets a rerun verify it.
    """
    torch = require_torch()
    if not 4 <= count <= 20000:
        raise ValueError("packaging image count must be between 4 and 20000")
    if not 32 <= image_size <= 224:
        raise ValueError("image_size must be between 32 and 224")
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    axis = torch.arange(image_size, dtype=torch.float32) + 0.5
    grid = torch.meshgrid(axis, axis, indexing="ij")
    images = []
    for index in range(count):
        label = index % len(PACKAGING_CLASS_NAMES)
        payload = draw_packaging_image(rng, generator, label, image_size, grid)
        images.append(PackagingImage(f"{split}-{index:05d}", label, payload,
                                     hashlib.sha256(payload).hexdigest()))
    return images


def packaging_split_report(splits):
    """Summarize split sizes, class balance and image-hash overlap.

    Args:
        splits (dict[str, Sequence[PackagingImage]]): Ordered split name to images.

    Returns:
        dict: Per split count, unique images, duplicates within the split, class
        counts and aggregate SHA-256 over the ordered image digests; pairwise
        cross-split overlap counts and their total.

    Raises:
        AttributeError/IndexError: Items lack sha256/label or a label is outside
        PACKAGING_CLASS_NAMES.

    Limitations:
        Hash disjointness proves no byte-identical image is shared. It does not
        prove that near-duplicates are absent or that the generator rules differ.
    """
    summary, hash_sets = {}, {}
    for name, images in splits.items():
        hashes = [item.sha256 for item in images]
        hash_sets[name] = set(hashes)
        counts = [0] * len(PACKAGING_CLASS_NAMES)
        for item in images:
            counts[item.label] += 1
        summary[name] = {"count": len(hashes), "unique_images": len(hash_sets[name]),
                         "duplicates_within_split": len(hashes) - len(hash_sets[name]),
                         "class_counts": dict(zip(PACKAGING_CLASS_NAMES, counts)),
                         "aggregate_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest()}
    names = list(splits)
    overlaps = {f"{a}|{b}": len(hash_sets[a] & hash_sets[b])
                for i, a in enumerate(names) for b in names[i + 1:]}
    return {"splits": summary, "pairwise_overlap": overlaps,
            "cross_split_overlap_total": sum(overlaps.values()),
            "hash_scope": "SHA-256 of raw HWC RGB uint8 bytes"}


def packaging_images_to_uint8(images, image_size=224):
    """Stack payloads into one CPU uint8 tensor in channel-first layout.

    Args:
        images (Sequence[PackagingImage | bytes]): Nonempty images or raw payloads.
        image_size (int): Square side used to reshape each payload.

    Returns:
        torch.Tensor: uint8 tensor shaped (count, 3, image_size, image_size).

    Raises:
        ValueError: The sequence is empty.
        RuntimeError: A payload length does not match image_size.

    Side effects:
        Copies all payload bytes once. Conversion to float and division by 255 is
        left to the caller so it matches make_resnet_batch_runtime exactly.
    """
    torch = require_torch()
    payloads = [item.payload if isinstance(item, PackagingImage) else item for item in images]
    if not payloads:
        raise ValueError("at least one image is required")
    flat = torch.frombuffer(bytearray(b"".join(payloads)), dtype=torch.uint8)
    return flat.reshape(len(payloads), image_size, image_size, 3).permute(0, 3, 1, 2).contiguous()


def predict_packaging_logits(model, uint8_images, device="cpu", precision="fp32", batch_size=64):
    """Run evaluation-mode inference over a uint8 image tensor in fixed batches.

    Args:
        model (torch.nn.Module): Trained classifier; it is not mutated.
        uint8_images (torch.Tensor): (count, 3, height, width) uint8 CPU tensor.
        device (str): CPU or CUDA device for execution.
        precision (str): fp32, or fp16 on CUDA only.
        batch_size (int): Images per forward call, at least 1.

    Returns:
        torch.Tensor: float32 CPU logits shaped (count, classes).

    Raises:
        ValueError: precision is unsupported, fp16 is requested on CPU or
            batch_size is below 1.
        PyTorch device/allocation errors propagate.

    Side effects:
        Copies the model to the requested device/dtype. Inputs follow the serving
        preprocessing: uint8 to float32, divide by 255, then cast for fp16.
    """
    torch = require_torch()
    if precision not in ("fp32", "fp16") or (precision == "fp16" and not str(device).startswith("cuda")):
        raise ValueError("FP16 evaluation is supported only on CUDA")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dtype = torch.float16 if precision == "fp16" else torch.float32
    runner = copy.deepcopy(model).to(device=device, dtype=dtype).eval()
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(uint8_images), batch_size):
            batch = uint8_images[offset:offset + batch_size].float().div_(255).to(device=device, dtype=dtype)
            outputs.append(runner(batch).float().cpu())
    return torch.cat(outputs)


def classification_metrics(labels, predictions, class_names=PACKAGING_CLASS_NAMES):
    """Compute accuracy, per-class precision/recall and a confusion matrix.

    Args:
        labels (Sequence[int]): Nonempty true class indexes.
        predictions (Sequence[int]): Predicted class indexes of the same length.
        class_names (Sequence[str]): Names in class-index order.

    Returns:
        dict: Sample/correct counts, accuracy, per-class support, predicted count,
        true positives, precision and recall (None when undefined), macro averages
        over defined values, and the confusion matrix with rows = true class and
        columns = predicted class.

    Raises:
        ValueError: Inputs are empty or have different lengths.
        IndexError: An index is outside class_names.
    """
    if not labels or len(labels) != len(predictions):
        raise ValueError("labels and predictions must be nonempty and equally long")
    n = len(class_names)
    confusion = [[0] * n for _ in range(n)]
    for truth, predicted in zip(labels, predictions):
        confusion[truth][predicted] += 1
    per_class = {}
    for k, name in enumerate(class_names):
        tp = confusion[k][k]
        predicted = sum(confusion[r][k] for r in range(n))
        support = sum(confusion[k])
        per_class[name] = {"support": support, "predicted": predicted, "true_positives": tp,
                           "precision": tp / predicted if predicted else None,
                           "recall": tp / support if support else None}
    correct = sum(confusion[k][k] for k in range(n))
    precisions = [x["precision"] for x in per_class.values() if x["precision"] is not None]
    recalls = [x["recall"] for x in per_class.values() if x["recall"] is not None]
    return {"samples": len(labels), "correct": correct, "accuracy": correct / len(labels),
            "per_class": per_class, "class_names": list(class_names),
            "macro_precision": sum(precisions) / len(precisions) if precisions else None,
            "macro_recall": sum(recalls) / len(recalls) if recalls else None,
            "confusion_matrix": confusion,
            "confusion_matrix_axes": "rows = true class, columns = predicted class, class_names order"}


def train_packaging_classifier(train, validation, *, image_size=224, epochs=15, batch_size=64, seed=47,
                               device="cpu", learning_rate=1e-3, weight_decay=1e-4,
                               class_names=PACKAGING_CLASS_NAMES, progress=None):
    """Train the ResNet-18 shape from scratch and keep the best validation epoch.

    Args:
        train (Sequence[PackagingImage]): Nonempty labelled training images.
        validation (Sequence[PackagingImage]): Nonempty labelled validation images,
            used only for epoch selection and never for gradient updates.
        image_size (int): Square side of every payload.
        epochs (int): Training epochs, inclusive range 1..100.
        batch_size (int): Minibatch size, inclusive range 2..256.
        seed (int): Model initialization seed; seed + 1 seeds shuffling and flips.
        device (str): CPU or CUDA device used for training.
        learning_rate (float): AdamW peak rate for a one-cycle schedule, > 0.
        weight_decay (float): AdamW decoupled weight decay, >= 0.
        class_names (Sequence[str]): Output classes; sets the classifier width.
        progress (Callable[[dict], Any] | None): Optional callback receiving each
            epoch record after validation.

    Returns:
        tuple[torch.nn.Module, dict]: Evaluation-mode model on device loaded with
        the selected epoch's weights, plus history (training loss/accuracy,
        validation loss/accuracy and duration per epoch), the selected epoch, the
        selection rule and optimizer settings.

    Raises:
        ValueError: A bound fails or a split is empty.
        PyTorch allocation/execution errors propagate.

    Side effects:
        Resets torch's global RNG through build_resnet18_probe, allocates device
        tensors and mutates model/optimizer state. Augmentation is a random
        horizontal flip only. cuDNN training kernels are not forced to be
        deterministic, so a rerun can produce different weights; the saved file
        hash identifies the weights that were actually evaluated.
    """
    torch = require_torch()
    if not 1 <= epochs <= 100 or not 2 <= batch_size <= 256:
        raise ValueError("Use 1..100 epochs and batch size 2..256")
    if not train or not validation or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Nonempty splits, positive learning rate and nonnegative decay are required")
    model = build_resnet18_probe(seed, num_classes=len(class_names)).to(device)
    x_train = packaging_images_to_uint8(train, image_size)
    y_train = torch.tensor([item.label for item in train], dtype=torch.long)
    x_val = packaging_images_to_uint8(validation, image_size)
    y_val = torch.tensor([item.label for item in validation], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train) / batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                    total_steps=epochs * steps_per_epoch, pct_start=0.15)
    loss_function = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 1)
    history, best, best_state = [], None, None
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(train), generator=generator)
        loss_sum, correct = 0.0, 0
        for offset in range(0, len(train), batch_size):
            indexes = permutation[offset:offset + batch_size]
            batch = x_train[indexes].to(device).float().div_(255)
            flips = (torch.rand(len(indexes), generator=generator) < 0.5).to(device)
            batch = torch.where(flips[:, None, None, None], batch.flip(-1), batch)
            targets = y_train[indexes].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(indexes)
            correct += int((logits.argmax(dim=1) == targets).sum())
        model.eval()
        val_logits = predict_packaging_logits(model, x_val, device, "fp32", batch_size)
        val_loss = float(loss_function(val_logits, y_val))
        val_correct = int((val_logits.argmax(dim=1) == y_val).sum())
        record = {"epoch": epoch, "training_loss": loss_sum / len(train),
                  "training_accuracy_augmented": correct / len(train),
                  "validation_loss": val_loss, "validation_accuracy": val_correct / len(validation),
                  "learning_rate_end": scheduler.get_last_lr()[0],
                  "duration_seconds": time.perf_counter() - started}
        history.append(record)
        if best is None or (record["validation_accuracy"], -record["validation_loss"]) > (
                best["validation_accuracy"], -best["validation_loss"]):
            best = record
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if progress is not None:
            progress(record)
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "selected_epoch": best["epoch"],
                   "selection_rule": "Highest validation accuracy; ties broken by lower validation loss. The test split is not consulted.",
                   "optimizer": "AdamW", "learning_rate_peak": learning_rate, "weight_decay": weight_decay,
                   "schedule": "OneCycleLR, 15 % warm-up", "batch_size": batch_size, "epochs": epochs,
                   "augmentation": "Random horizontal flip (p = 0.5)", "seed": seed, "device": str(device),
                   "initialization": "Kaiming-normal random initialization; no pretrained weights downloaded"}


def open_loop_request_plan(request_index, image_count, repeat_share=None, hot_images=(), plan_seed=0):
    """Choose the test image, request kind and uniqueness marker for one request.

    Args:
        request_index (int): Global zero-based scheduled request index.
        image_count (int): Number of images available in the payload file, >= 1.
        repeat_share (float | None): None cycles through all images in order. A
            value in 0..1 sends that expected share as byte-identical repeats of
            hot_images and makes every other request byte-unique.
        hot_images (Sequence[int]): Image indexes used for repeats; required when
            repeat_share is positive.
        plan_seed (int): Offset mixed into the deterministic request hash.

    Returns:
        tuple[int, str, int | None]: Image index, kind (cycle, repeat or unique)
        and a marker integer for unique requests (otherwise None).

    Raises:
        ZeroDivisionError/IndexError: image_count is zero or hot_images is empty
        while a repeat is selected.

    Limitations:
        A unique request is the selected test image with its first pixel's three
        bytes replaced by the 24-bit request index. This imitates a fresh camera
        frame for an exact-hash cache; the one-pixel change is not a new scene.
    """
    if repeat_share is None:
        return request_index % image_count, "cycle", None
    mixed = ((request_index + plan_seed) * 2654435761) % 4294967296
    if mixed / 4294967296 < repeat_share:
        return hot_images[mixed % len(hot_images)], "repeat", None
    return request_index % image_count, "unique", request_index % 16777216


async def _open_loop_client_main(settings):
    """Send one process's share of scheduled requests with raw HTTP/1.0 sockets.

    Args:
        settings (dict): host, port, path, payload_path, image_bytes, image_count,
            target_rps, total_requests, process_index, process_count, t0
            (monotonic start), timeout_seconds, max_inflight, repeat_share,
            hot_images and plan_seed, as built by run_open_loop_ramp_step.

    Returns:
        list[tuple]: One record per assigned request: index, image index, kind,
        scheduled offset, send offset (None if not sent), end offset, status,
        predicted class, queue wait ms, service ms, batch size, cache-hit flag and
        error type. Offsets are seconds relative to t0.

    Raises:
        OSError: The payload file cannot be read. Per-request network, timeout and
        parsing failures are captured as client_error records instead.

    Side effects:
        Opens one loopback TCP connection per sent request. Requests are launched
        at absolute due times without waiting for earlier responses. When
        max_inflight requests are already open, the request is recorded as
        not_sent instead of being delayed.
    """
    import asyncio
    from pathlib import Path
    blob = Path(settings["payload_path"]).read_bytes()
    size = settings["image_bytes"]
    header = (f"POST {settings['path']} HTTP/1.0\r\nHost: {settings['host']}\r\n"
              f"Content-Type: application/octet-stream\r\nContent-Length: {size}\r\n\r\n").encode()
    rate, t0 = settings["target_rps"], settings["t0"]
    records, tasks = [], set()
    inflight = 0

    async def one(index, image_index, kind, due, payload):
        """Run one HTTP exchange and append its timing/outcome record.

        Args:
            index (int): Global scheduled request index.
            image_index (int): Payload-file image index sent in this request.
            kind (str): cycle, repeat or unique, from open_loop_request_plan.
            due (float): Absolute monotonic scheduled arrival in seconds.
            payload (bytes | bytearray): Body sent after the HTTP header.

        Returns:
            None: The record is appended to the enclosing records list.

        Side effects:
            Opens, uses and closes one connection; decrements the enclosing
            in-flight counter. Exceptions become client_error records.
        """
        nonlocal inflight
        sent = time.monotonic()
        status, body, error = "client_error", {}, None

        async def exchange():
            """Connect, send the request and read the full HTTP/1.0 response.

            Returns:
                bytes: Raw response bytes until the server closes the connection.

            Raises:
                OSError and asyncio cancellation/timeouts propagate to one().

            Side effects:
                Closes the stream writer after reading, even after a read error.
            """
            reader, writer = await asyncio.open_connection(settings["host"], settings["port"])
            try:
                writer.write(header)
                writer.write(payload)
                await writer.drain()
                return await reader.read()
            finally:
                writer.close()

        try:
            raw = await asyncio.wait_for(exchange(), settings["timeout_seconds"])
            head, _, content = raw.partition(b"\r\n\r\n")
            status = int(head.split(b" ", 2)[1])
            body = json.loads(content) if content else {}
            error = body.get("error_type")
        except Exception as exc:
            status, body, error = "client_error", {}, type(exc).__name__
        ended = time.monotonic()
        inflight -= 1
        records.append((index, image_index, kind, due - t0, sent - t0, ended - t0, status,
                        body.get("predicted_class"), body.get("queue_wait_ms"),
                        body.get("service_latency_ms"), body.get("batch_size"),
                        body.get("cache_hit"), error))

    for index in range(settings["process_index"], settings["total_requests"], settings["process_count"]):
        due = t0 + index / rate
        delay = due - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        image_index, kind, marker = open_loop_request_plan(
            index, settings["image_count"], settings["repeat_share"], settings["hot_images"], settings["plan_seed"])
        if inflight >= settings["max_inflight"]:
            records.append((index, image_index, kind, due - t0, None, None, "not_sent",
                            None, None, None, None, None, "client_in_flight_cap"))
            continue
        payload = blob[image_index * size:(image_index + 1) * size]
        if marker is not None:
            payload = bytearray(payload)
            payload[0:3] = marker.to_bytes(3, "big")
        inflight += 1
        task = asyncio.ensure_future(one(index, image_index, kind, due, payload))
        tasks.add(task)
        # The callback removes finished tasks so the set holds only open requests.
        task.add_done_callback(tasks.discard)
    if tasks:
        await asyncio.gather(*list(tasks), return_exceptions=True)
    return records


def open_loop_client_process(result_queue, settings):
    """Run one spawned load-generator process and return its records.

    Args:
        result_queue (multiprocessing.Queue): Receives one result dictionary.
        settings (dict): Client settings described in _open_loop_client_main.

    Returns:
        None: Puts {process_index, records, cpu_seconds, error} on result_queue.

    Raises:
        Queue/pickling errors can propagate. Client exceptions are captured into
        the error field so the parent still receives an explicit result.

    Side effects:
        Raises this process's soft open-file limit up to min(hard, 65536) where
        permitted, then runs an asyncio event loop. Imports no torch.
    """
    import asyncio
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        wanted = min(hard, 65536)
        if soft < wanted:
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
    except (ImportError, ValueError, OSError):
        pass
    started = time.process_time()
    try:
        records = asyncio.run(_open_loop_client_main(settings))
        error = None
    except Exception as exc:
        records, error = [], f"{type(exc).__name__}: {exc}"[:500]
    result_queue.put({"process_index": settings["process_index"], "records": records,
                      "cpu_seconds": time.process_time() - started, "error": error})


def read_cpu_snapshot():
    """Read system-wide and per-process CPU counters at one instant.

    Returns:
        dict: Monotonic time; busy and total jiffies from /proc/stat; the sum of
        user + system CPU seconds over every /proc/<pid>/stat process; and this
        process's user + system CPU seconds from os.times(). Linux-only fields
        are None when unavailable.

    Raises:
        Nothing for missing /proc files or processes that exit during the scan;
        parsing errors of an unexpected /proc/stat format propagate.

    Limitations:
        Under WSL, /proc covers the Linux VM only; Windows desktop load is
        invisible although it shares the physical CPU. The /proc/stat aggregate
        is tick-sampled and was observed to undercount millisecond bursts in
        this WSL VM; the per-process sums use precise scheduler run time.
        Processes that exit between two snapshots drop out of the sum.
    """
    import os
    busy = total = all_processes = None
    try:
        with open("/proc/stat", encoding="ascii") as handle:
            fields = [int(x) for x in handle.readline().split()[1:9]]
        total = sum(fields)
        busy = total - fields[3] - fields[4]
    except OSError:
        pass
    try:
        ticks = os.sysconf("SC_CLK_TCK")
        all_processes = 0.0
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat", encoding="ascii", errors="replace") as handle:
                    # Fields after the parenthesised command name; utime/stime are 14 and 15.
                    rest = handle.read().rsplit(")", 1)[1].split()
                all_processes += (int(rest[11]) + int(rest[12])) / ticks
            except (OSError, IndexError, ValueError):
                continue
    except (OSError, ValueError, AttributeError):
        all_processes = None
    times = os.times()
    return {"t": time.monotonic(), "busy_jiffies": busy, "total_jiffies": total,
            "all_processes_cpu_seconds": all_processes,
            "process_cpu_seconds": times.user + times.system}


def cpu_usage_between(before, after):
    """Convert two read_cpu_snapshot results into utilization figures.

    Args:
        before (dict): Earlier snapshot.
        after (dict): Later snapshot from the same process.

    Returns:
        dict: Wall seconds, tick-sampled system busy share in 0..1 (None without
        /proc/stat), CPU cores used by all Linux processes alive at both
        snapshots (None when absent from either snapshot), and this process's CPU
        usage expressed in cores.

    Raises:
        KeyError/TypeError: A snapshot lacks t, busy/total jiffies or process
        CPU seconds. all_processes_cpu_seconds is optional.
    """
    wall = after["t"] - before["t"]
    system = None
    if before["total_jiffies"] is not None and after["total_jiffies"] is not None:
        delta_total = after["total_jiffies"] - before["total_jiffies"]
        if delta_total > 0:
            system = (after["busy_jiffies"] - before["busy_jiffies"]) / delta_total
    process_cores = all_cores = None
    if wall > 0:
        process_cores = (after["process_cpu_seconds"] - before["process_cpu_seconds"]) / wall
        if before.get("all_processes_cpu_seconds") is not None and after.get("all_processes_cpu_seconds") is not None:
            all_cores = (after["all_processes_cpu_seconds"] - before["all_processes_cpu_seconds"]) / wall
    return {"wall_seconds": wall, "system_busy_share_tick_sampled": system,
            "all_linux_processes_cores": all_cores, "server_process_cores": process_cores}


def parse_nvidia_smi_sample(line):
    """Parse one 'utilization.gpu, memory.used' CSV line without units.

    Args:
        line (str): Text such as '17, 4715'; extra leading columns are ignored.

    Returns:
        dict | None: utilization_percent and memory_used_mib floats, or None for
        a malformed or unsupported line such as '[N/A]'.

    Raises:
        Nothing for malformed text.
    """
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        return None
    try:
        return {"utilization_percent": float(parts[-2]), "memory_used_mib": float(parts[-1])}
    except ValueError:
        return None


class NvidiaSmiSampler:
    """Sample device-wide GPU utilization and memory from an nvidia-smi subprocess.

    Each parsed line is stamped with the reader thread's monotonic time. Values
    are device-wide and include other processes and desktop applications.
    """
    def __init__(self, interval_ms=1000, executable="nvidia-smi", gpu_index=0):
        """Store sampling settings without starting a subprocess.

        Args:
            interval_ms (int): Sampling period passed to nvidia-smi -lms.
            executable (str): nvidia-smi executable name or path.
            gpu_index (int): Device index passed to nvidia-smi -i.

        Returns:
            None.
        """
        self.command = [executable, "-i", str(gpu_index), "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits", "-lms", str(interval_ms)]
        self.samples = []
        self.process = None
        self.thread = None
        self.available = False

    def start(self):
        """Start the sampling subprocess and its line-reader thread.

        Returns:
            bool: True when the subprocess started, False when the executable is
            missing or cannot be started.

        Side effects:
            Spawns one subprocess and one daemon reader thread.
        """
        import subprocess
        try:
            self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return False
        self.available = True

        def read_lines():
            """Append each parsed nvidia-smi line with its arrival time.

            Returns:
                None: Runs until the subprocess closes its output.

            Side effects:
                Mutates the sampler's samples list from this daemon thread.
            """
            for line in self.process.stdout:
                sample = parse_nvidia_smi_sample(line)
                if sample is not None:
                    self.samples.append({"t": time.monotonic(), **sample})

        self.thread = threading.Thread(target=read_lines, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Terminate the subprocess and return the collected samples.

        Returns:
            list[dict]: Samples with monotonic t, utilization_percent and
            memory_used_mib; empty when sampling never started.

        Side effects:
            Terminates and reaps the subprocess and joins the reader thread for up
            to five seconds each. Safe to call more than once.
        """
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
        if self.thread is not None:
            self.thread.join(timeout=5)
        return list(self.samples)


def summarize_gpu_samples(samples, start, end):
    """Summarize nvidia-smi samples whose arrival time falls in a window.

    Args:
        samples (Sequence[dict]): NvidiaSmiSampler.stop output.
        start (float): Window start in monotonic seconds.
        end (float): Window end in monotonic seconds.

    Returns:
        dict: Sample count, mean/max utilization percent and mean/max memory
        used in MiB (None values when the window holds no sample), with scope.

    Raises:
        KeyError: A sample lacks the expected fields.
    """
    window = [x for x in samples if start <= x["t"] <= end]
    util = [x["utilization_percent"] for x in window]
    memory = [x["memory_used_mib"] for x in window]
    return {"samples": len(window),
            "utilization_percent_mean": sum(util) / len(util) if util else None,
            "utilization_percent_max": max(util) if util else None,
            "memory_used_mib_mean": sum(memory) / len(memory) if memory else None,
            "memory_used_mib_max": max(memory) if memory else None,
            "scope": "Device-wide nvidia-smi at 1 Hz: includes other processes and Windows desktop applications"}


def _sleep_until(deadline):
    """Sleep in short slices until a monotonic deadline has passed.

    Args:
        deadline (float): Target time.monotonic() value in seconds.

    Returns:
        None.
    """
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.25))


def run_open_loop_ramp_step(port, payload_path, *, image_bytes, image_count, target_rps, warmup_seconds,
                            measure_seconds, process_count=None, max_inflight_per_process=256,
                            request_timeout_seconds=5.0, start_delay_seconds=2.0, repeat_share=None,
                            hot_images=(), plan_seed=0, host="127.0.0.1", path="/predict-rgb"):
    """Offer one open-loop load step from spawned client processes.

    Args:
        port (int): Loopback port of an already serving RGB server.
        payload_path (str | Path): File holding image_count concatenated payloads.
        image_bytes (int): Bytes per payload.
        image_count (int): Number of payloads in the file, >= 1.
        target_rps (float): Offered arrival rate, 1..10000 requests per second.
        warmup_seconds (float): Leading seconds of the same stream excluded from
            the measured cohort, 0..60.
        measure_seconds (float): Measured window length, 1..120 seconds.
        process_count (int | None): Client processes; None picks
            min(4, ceil(target_rps / 500)).
        max_inflight_per_process (int): Open-request cap per process, 1..4096;
            requests beyond it are recorded as not_sent.
        request_timeout_seconds (float): Client limit for one complete exchange.
        start_delay_seconds (float): Delay before the first scheduled request,
            leaving time for process start-up.
        repeat_share (float | None): Passed to open_loop_request_plan.
        hot_images (Sequence[int]): Repeat image indexes for the cache workload.
        plan_seed (int): Request-plan hash offset.
        host (str): Server address, loopback by default.
        path (str): Request path.

    Returns:
        dict: records (dicts with offsets relative to t0), t0 (monotonic), the
        measured window offsets, process count, client CPU seconds and errors,
        and server-process/system CPU usage across the measured window.

    Raises:
        ValueError: A bound fails.
        queue.Empty: A client process returns nothing within its allowance.

    Side effects:
        Spawns, joins and if needed terminates client processes. The calling
        thread sleeps through the run; the server must run in other threads.
        Arrival times are absolute: a slow response never delays later sends.
    """
    import multiprocessing
    if not 1 <= target_rps <= 10000 or not 0 <= warmup_seconds <= 60 or not 1 <= measure_seconds <= 120:
        raise ValueError("Use 1..10000 req/s, warm-up 0..60 s and measurement 1..120 s")
    if image_count < 1 or not 1 <= max_inflight_per_process <= 4096:
        raise ValueError("Use at least one image and an in-flight cap of 1..4096")
    processes_wanted = process_count or min(4, max(1, math.ceil(target_rps / 500)))
    total = int(round(target_rps * (warmup_seconds + measure_seconds)))
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    t0 = time.monotonic() + start_delay_seconds
    base = {"host": host, "port": port, "path": path, "payload_path": str(payload_path),
            "image_bytes": image_bytes, "image_count": image_count, "target_rps": target_rps,
            "total_requests": total, "process_count": processes_wanted, "t0": t0,
            "timeout_seconds": request_timeout_seconds, "max_inflight": max_inflight_per_process,
            "repeat_share": repeat_share, "hot_images": list(hot_images), "plan_seed": plan_seed}
    workers = [context.Process(target=open_loop_client_process, args=(result_queue, {**base, "process_index": k}),
                               name=f"vision-client-{k}", daemon=True) for k in range(processes_wanted)]
    for worker in workers:
        worker.start()
    outputs = []
    try:
        _sleep_until(t0 + warmup_seconds)
        before = read_cpu_snapshot()
        _sleep_until(t0 + warmup_seconds + measure_seconds)
        after = read_cpu_snapshot()
        allowance = start_delay_seconds + request_timeout_seconds + 120
        for _ in workers:
            outputs.append(result_queue.get(timeout=allowance))
    finally:
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
    names = ("index", "image_index", "kind", "due", "sent", "ended", "status", "predicted_class",
             "queue_wait_ms", "service_ms", "batch_size", "cache_hit", "error_type")
    records = [dict(zip(names, row)) for output in outputs for row in output["records"]]
    return {"records": records, "t0": t0, "window_offsets_seconds": [warmup_seconds, warmup_seconds + measure_seconds],
            "process_count": processes_wanted, "scheduled_requests": total,
            "max_inflight_per_process": max_inflight_per_process,
            "client_cpu_seconds": [x["cpu_seconds"] for x in outputs],
            "client_errors": [x["error"] for x in outputs if x["error"]],
            "cpu_window": cpu_usage_between(before, after)}


def summarize_open_loop_step(records, *, target_rps, warmup_seconds, measure_seconds, slo_p99_ms=100.0,
                             slo_max_failure_share=0.01, deadline_ms=1000.0, labels=None,
                             offline_predictions=None, generator_min_ratio=0.95,
                             reliable_p99_observations=1000):
    """Summarize one load step without hiding any offered request.

    Args:
        records (Sequence[dict]): Records from run_open_loop_ramp_step.
        target_rps (float): Offered rate of the step.
        warmup_seconds (float): Start offset of the measured cohort.
        measure_seconds (float): Length of the measured window, > 0.
        slo_p99_ms (float): Latency objective applied to p99 of answered requests
            and to the within-SLO share.
        slo_max_failure_share (float): Maximum failure share for the SLO.
        deadline_ms (float): An HTTP 200 later than this after its scheduled
            arrival is a deadline miss and counts as a failure.
        labels (Sequence[int] | None): True class per payload-file image.
        offline_predictions (Sequence[int] | None): Offline class per image from
            the same runtime, for agreement under load.
        generator_min_ratio (float): Achieved send rate below this fraction of
            target_rps marks the point generator_limited.
        reliable_p99_observations (int): Observation count below which p99 is
            flagged unreliable.

    Returns:
        dict: Offered/sent counts, achieved send rate, mutually exclusive
        outcomes (ok, deadline_miss, rejected, error, not_sent), failure share,
        share of offered requests answered within slo_p99_ms, latency summaries
        from scheduled arrival and from send time over every HTTP 200 (late
        answers included), client lateness, queue and compute time, cache hits,
        accuracy/agreement under load, generator_limited with its reason, the
        p99 reliability flag and slo_met.

    Raises:
        KeyError: A record lacks a required field.
        ZeroDivisionError: measure_seconds is zero.

    Limitations:
        Latency is measured from the scheduled arrival, so client lag counts
        against the service. The cohort is defined by scheduled time; requests
        from the warm-up still occupy the service during the window.
    """
    start, end = warmup_seconds, warmup_seconds + measure_seconds
    cohort = [r for r in records if start <= r["due"] < end]
    offered = len(cohort)
    sent_in_window = sum(1 for r in records if r["sent"] is not None and start <= r["sent"] < end)
    achieved = sent_in_window / measure_seconds
    outcomes = {"ok": 0, "deadline_miss": 0, "rejected": 0, "error": 0, "not_sent": 0}
    answered, statuses, errors = [], {}, {}
    for r in cohort:
        statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
        if r["error_type"]:
            errors[r["error_type"]] = errors.get(r["error_type"], 0) + 1
        if r["sent"] is None:
            outcomes["not_sent"] += 1
        elif r["status"] == 200:
            answered.append(r)
            late = (r["ended"] - r["due"]) * 1000 > deadline_ms
            outcomes["deadline_miss" if late else "ok"] += 1
        elif r["status"] == 429:
            outcomes["rejected"] += 1
        else:
            outcomes["error"] += 1
    failures = offered - outcomes["ok"]
    from_arrival = [(r["ended"] - r["due"]) * 1000 for r in answered]
    from_send = [(r["ended"] - r["sent"]) * 1000 for r in answered]
    lateness = [(r["sent"] - r["due"]) * 1000 for r in cohort if r["sent"] is not None]
    computed = [r for r in answered if not r["cache_hit"] and r["queue_wait_ms"] is not None
                and r["service_ms"] is not None]
    latency = latency_summary(from_arrival)
    generator_limited = achieved < generator_min_ratio * target_rps
    reason = None
    if generator_limited:
        reason = "client_in_flight_cap_reached" if outcomes["not_sent"] else "client_send_scheduler_behind"
    failure_share = failures / offered if offered else None
    quality = None
    predicted = [r for r in answered if r["predicted_class"] is not None]
    if labels is not None and predicted:
        correct = sum(labels[r["image_index"]] == r["predicted_class"] for r in predicted)
        quality = {"predictions": len(predicted), "correct": correct, "accuracy_under_load": correct / len(predicted)}
        if offline_predictions is not None:
            offline_correct = sum(labels[r["image_index"]] == offline_predictions[r["image_index"]] for r in predicted)
            agree = sum(offline_predictions[r["image_index"]] == r["predicted_class"] for r in predicted)
            quality.update(offline_accuracy_same_requests=offline_correct / len(predicted),
                           agreement_with_offline=agree / len(predicted), disagreements=len(predicted) - agree)
    slo_met = (not generator_limited and failure_share is not None and failure_share <= slo_max_failure_share
               and latency["p99_ms"] is not None and latency["p99_ms"] <= slo_p99_ms)
    within_slo = (sum(x <= slo_p99_ms for x in from_arrival) / offered) if offered else None
    completions = sum(1 for r in records if r["status"] == 200 and start <= r["ended"] < end)
    return {"target_rps": target_rps, "offered_requests": offered, "sent_in_window": sent_in_window,
            "achieved_send_rate_rps": achieved, "generator_limited": generator_limited,
            "generator_limited_reason": reason, "outcomes": outcomes, "http_status_counts": statuses,
            "error_types": errors, "failure_share": failure_share,
            "within_slo_share_of_offered": within_slo,
            "successful_throughput_rps": outcomes["ok"] / measure_seconds,
            "completions_200_in_window_rps": completions / measure_seconds,
            "latency_from_scheduled_arrival_ms": latency,
            "latency_from_send_ms": latency_summary(from_send),
            "p99_observations": latency["samples"],
            "p99_reliable": latency["samples"] >= reliable_p99_observations,
            "client_send_lateness_ms": latency_summary(lateness),
            "server_queue_wait_ms": latency_summary([r["queue_wait_ms"] for r in computed]),
            "server_compute_and_response_ms": latency_summary([r["service_ms"] - r["queue_wait_ms"] for r in computed]),
            "cache_hits": sum(1 for r in answered if r["cache_hit"]),
            "repeat_requests": sum(1 for r in cohort if r["kind"] == "repeat"),
            "quality_under_load": quality, "slo_met": slo_met,
            "slo": {"p99_ms_max": slo_p99_ms, "failure_share_max": slo_max_failure_share, "deadline_ms": deadline_ms}}


def ramp_should_stop(step, stop_failure_share=0.20):
    """Apply the ramp stop rule to one ladder step summary.

    Args:
        step (dict): summarize_open_loop_step output, or a failed-step record with
            status failed.
        stop_failure_share (float): Climbing stops when failures exceed this share.

    Returns:
        tuple[bool, str | None]: Whether to stop and the reason.

    Raises:
        Nothing for a well-formed dictionary; missing keys are read with get().
    """
    if step.get("status") == "failed":
        return True, "step_execution_failed"
    if step.get("failure_share") is not None and step["failure_share"] > stop_failure_share:
        return True, f"failure_share_above_{stop_failure_share}"
    if step.get("generator_limited"):
        return True, "generator_limited"
    return False, None


def select_sustainable_capacity(steps):
    """Select the highest offered rate whose runs all met the SLO.

    Args:
        steps (Sequence[dict]): Step summaries with target_rps, slo_met,
            generator_limited, successful_throughput_rps and latency fields;
            repeats share their target_rps.

    Returns:
        dict: max_sustainable_offered_rps (None if the lowest rate failed), the
        next failing rate, per-rate verdicts, and at the selected rate the run
        count, mean/min/max successful throughput and worst-run p50/p95/p99,
        failure share and observations.

    Raises:
        KeyError: A step lacks target_rps.

    Rule:
        Rates are visited in ascending order and the search ends at the first rate
        with any failing run, so a later lucky pass cannot raise capacity. A run
        that failed to execute, or was generator-limited, counts as not passing.
    """
    rates = sorted({s["target_rps"] for s in steps})
    verdicts, highest, first_failing = {}, None, None
    for rate in rates:
        runs = [s for s in steps if s["target_rps"] == rate]
        passed = all(s.get("slo_met") for s in runs)
        verdicts[str(rate)] = {"runs": len(runs), "all_runs_met_slo": passed,
                               "any_generator_limited": any(s.get("generator_limited") for s in runs)}
        if first_failing is None:
            if passed:
                highest = rate
            else:
                first_failing = rate
    result = {"max_sustainable_offered_rps": highest, "first_failing_offered_rps": first_failing,
              "per_rate": verdicts,
              "rule": "Highest ascending offered rate before the first rate with any run outside the SLO; generator-limited or failed runs do not pass."}
    if highest is not None:
        runs = [s for s in steps if s["target_rps"] == highest]
        throughput = [s["successful_throughput_rps"] for s in runs]
        # The key lambda picks the run with the worst p99 tail at the selected rate.
        worst = max(runs, key=lambda s: s["latency_from_scheduled_arrival_ms"]["p99_ms"])
        result.update(runs_at_max=len(runs), successful_throughput_rps_mean=sum(throughput) / len(throughput),
                      successful_throughput_rps_min=min(throughput), successful_throughput_rps_max=max(throughput),
                      worst_run_latency_ms=worst["latency_from_scheduled_arrival_ms"],
                      worst_run_failure_share=max(s["failure_share"] for s in runs),
                      all_runs_p99_reliable=all(s["p99_reliable"] for s in runs))
    return result


def replicas_for_target(target_rps, capacity_rps, headroom=0.3, zones=3):
    """Compute a replica-count sizing scenario from one measured capacity.

    Args:
        target_rps (float): Required request rate, > 0.
        capacity_rps (float): Per-replica capacity that met the SLO, > 0.
        headroom (float): Fraction of capacity kept unused, 0 <= headroom < 1.
        zones (int): Failure zones for the zone-loss variant, >= 2.

    Returns:
        dict: Usable capacity per replica, base replicas
        ceil(target / (capacity * (1 - headroom))), the N+1 count, and the
        per-zone count that survives the loss of one whole zone, with formulas.

    Raises:
        ValueError: A bound fails.

    Limitations:
        A formula only. It assumes linear scaling and ignores every shared
        component (load balancer, network, decoding, storage, control plane).
    """
    if target_rps <= 0 or capacity_rps <= 0 or not 0 <= headroom < 1 or zones < 2:
        raise ValueError("Use positive target/capacity, headroom 0..<1 and at least two zones")
    usable = capacity_rps * (1 - headroom)
    base = math.ceil(target_rps / usable)
    per_zone = math.ceil(base / (zones - 1))
    return {"target_rps": target_rps, "capacity_rps_at_slo": capacity_rps, "headroom": headroom,
            "usable_rps_per_replica": usable, "base_replicas": base, "n_plus_1_replicas": base + 1,
            "zones": zones, "replicas_per_zone_surviving_one_zone_loss": per_zone,
            "zone_redundant_total_replicas": per_zone * zones,
            "formula": "base = ceil(target / (capacity_at_SLO * (1 - headroom))); N+1 = base + 1; per_zone = ceil(base / (zones - 1)); total = per_zone * zones",
            "validated_measurement": False}
