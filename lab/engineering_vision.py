"""Provide two synthetic vision exercises with explicit measurement boundaries.

The tiny grayscale CNN path trains on fictional stripe images and can compare CPU
and CUDA outputs. The random-weight RGB ResNet-shaped path measures a bounded,
batched localhost service; it does not measure detection quality. PyTorch is
imported lazily, so the base application can import this module without it.
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


def make_resnet_batch_runtime(base_model, device="cpu", precision="fp32", workers=1, image_size=224):
    """Prepare per-worker model copies and a timed RGB batch callback.

    Args:
        base_model (torch.nn.Module): Unchanged source probe copied for each worker.
        device (str): CPU or a CUDA device string; default cpu.
        precision (str): fp32, or fp16 only for a CUDA device.
        workers (int): Number of model snapshots; normally one or two.
        image_size (int): Square RGB width/height in pixels; normally 32..224.

    Returns:
        tuple[Callable, list[torch.nn.Module], list]: batch_call, copied evaluation
        models and one CUDA stream per model (or None entries on CPU).

    Raises:
        ValueError: Precision is unsupported or FP16 is requested for CPU.
        PyTorch device/allocation errors propagate; caller validates other bounds.

    Side effects:
        Allocates device model copies/streams and synchronizes their initialization.
        Sets no allocator budget and downloads nothing; all outputs remain
        performance-only because the probe has random weights.
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
            random_weights/quality_claimed flags, plus forward-only model_ms.

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
            return [{"top_class_index": int(i), "top_probability": float(v),
                     "random_weights": True, "quality_claimed": False} for i, v in zip(indexes, values)], {"model_ms": model_ms}

    return batch_call, models, streams


def create_batched_http_server(service, port=0):
    """Bind a threaded synthetic RGB probe server to IPv4 loopback.

    Args:
        service (BoundedBatchVisionService): Existing started batching service.
        port (int): TCP port; 0 requests an OS-allocated local port.

    Returns:
        ThreadingHTTPServer: Bound but not serving until the caller invokes
        serve_forever. The caller owns server and service cleanup separately.

    Raises:
        OSError: Socket bind/allocation fails.

    Side effects:
        Binds 127.0.0.1; request threads are daemon threads with listen backlog 128.
        No authentication, TLS, sensitive-data processing or production SLA is claimed.
    """
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
            """
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
