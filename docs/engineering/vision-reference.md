# Vision engineering function reference

This reference explains the implementation, measurement harness and verification cases for the synthetic vision laboratory. Read the short runbook first; use this file when tracing a specific function or checking what a measurement includes.

The grayscale path trains a tiny convolutional neural network (CNN) on fictional stripe fixtures. The RGB path uses an untrained ResNet-18-shaped network only to measure service behavior. Neither path measures YOLO, TensorRT, Triton or a production model deployment.

## Execution map

1. **Tiny model:** generate_synthetic_examples -> train_tiny_model -> VisionPredictor -> evaluate_model -> sequential model/pipeline/optional HTTP timing.

2. **Concurrent probe:** build_resnet18_probe -> make_resnet_batch_runtime -> BoundedBatchVisionService -> create_batched_http_server -> run_open_loop_http.

3. **Evidence:** measure_vision_engineering.main selects one path, validates bounds and writes reports outside the checkout. Its load-probe branch saves each configuration separately.

4. **Verification:** pure input/cache tests and scripted slow callbacks check boundaries and accounting; optional torch/CUDA tests check shape and snapshot consistency.

## Measurement and safety boundaries

- Times returned by the public summaries use **milliseconds**. measure_calls internally records seconds; the load probe also reports request rates and seconds for injection/drain windows.

- **Queue wait**, **batch-filling wait**, **model-forward time**, **worker pipeline time**, **HTTP time** and **scheduled-arrival-to-response time** are different measurements. A five-millisecond batch timer does not cap total latency.

- Successful-request percentiles exclude rejected/error responses. Offered, successful, failed and client-dropped counts must be read with the latency percentiles.

- The synthetic cache keys include weights, preprocessing version and pixels. They do not contain a tenant/user policy; authentication must precede any real-data cache lookup.

- The optional HTTP servers bind to **127.0.0.1** and provide **no authentication**. They are synthetic localhost exercises.

- CUDA memory caps concern the PyTorch allocator, not driver/context overhead or other processes. Thread/cuDNN/allocator settings are process-global and are not restored by this harness.

- Callers own server/service shutdown and must stop submissions before closing the batching service. Request timeouts do not cancel queued or running inference; cancelled service futures are not a supported path.

- Numerical consistency and fixture accuracy do not authorize production use. The random-weight probe always carries no-quality/no-release flags.

## Anonymous callables

All six lambdas in lab/engineering_vision.py are explained beside their source: two probability-index readers for argmax, a default no-op synchronization callback, a CUDA synchronization callback, its CPU no-op alternative and an uncached full-pipeline timing closure. They do not create separate AI agents.

## Definition index

- [lab/engineering_vision.py](#labengineering_visionpy) — 50 functions/methods and 11 classes.

- [scripts/measure_vision_engineering.py](#scriptsmeasure_vision_engineeringpy) — 4 functions/methods and 0 classes.

- [tests/test_engineering_vision.py](#teststest_engineering_visionpy) — 13 functions/methods and 3 classes.


## lab/engineering_vision.py

[Open module](../../lab/engineering_vision.py)


Provide two synthetic vision exercises with explicit measurement boundaries.

The tiny grayscale CNN path trains on fictional stripe images and can compare CPU
and CUDA outputs. The random-weight RGB ResNet-shaped path measures a bounded,
batched localhost service; it does not measure detection quality. PyTorch is
imported lazily, so the base application can import this module without it.
Callers own server shutdown, access control and runtime selection. No model
weights or packages are downloaded. CUDA allocator budgets are set by the CLI,
not by this module, and do not cover driver/context memory.


### VisionDependencyUnavailable

    VisionDependencyUnavailable

[Class source](../../lab/engineering_vision.py#L33)

Signal that the selected interpreter cannot import optional PyTorch.

This RuntimeError subclass is raised by require_torch; it carries a human-readable
message and has no additional attributes or recovery side effects.


### optional_runtime_status

    optional_runtime_status()

[Function source](../../lab/engineering_vision.py#L41)

Inspect whether the optional torch package can be located without importing it.


**Returns**

    dict[str, Any]: Installation availability, a false CUDA-verification flag and
    an explanation. Package presence never establishes that CUDA can execute.


**Side effects and ownership**

    Performs import-spec discovery only; allocates no tensor or GPU context.
    Import-system discovery errors other than an absent spec propagate.


### require_torch

    require_torch()

[Function source](../../lab/engineering_vision.py#L57)

Import and return PyTorch from the currently selected interpreter.


**Returns**

    module: The imported torch module; Python may reuse its import cache.


**Raises**

    VisionDependencyUnavailable: PyTorch raises ImportError during import.
    Other import/runtime failures propagate. No installation or download occurs.


**Side effects and ownership**

    Executes torch's normal import initialization. This is not a CUDA-use check.


### validate_image

    validate_image(pixels: Sequence[Sequence[int]])

[Function source](../../lab/engineering_vision.py#L80)

Validate one synthetic 32-by-32 grayscale image and freeze its rows.


**Parameters**

    pixels (list[list[int]] | tuple[tuple[int, ...], ...]): Exactly 32 rows of
        32 built-in integers in the inclusive range 0..255. Booleans are rejected.


**Returns**

    tuple[tuple[int, ...], ...]: An immutable validated image with unchanged values.


**Raises**

    ValueError: Shape, container type, pixel type or range is invalid.


**Side effects and ownership**

    None; the input is not modified and no model is executed.


### image_digest

    image_digest(pixels: Sequence[Sequence[int]])

[Function source](../../lab/engineering_vision.py#L108)

Hash the validated grayscale pixel bytes in row-major order.


**Parameters**

    pixels (Sequence[Sequence[int]]): Image accepted by validate_image.


**Returns**

    str: A 64-character lowercase SHA-256 hexadecimal digest.


**Raises**

    ValueError: Image validation fails. Metadata and user access scope are not
        included in this digest; identical pixel bytes intentionally share a hash.


### make_cache_key

    make_cache_key(pixels, weights_version: str, preprocessing_version: str)

[Function source](../../lab/engineering_vision.py#L125)

Bind a cached prediction to weights, preprocessing label and image content.


**Parameters**

    pixels (Sequence[Sequence[int]]): One valid 32-by-32 grayscale image.
    weights_version (str): Nonempty model fingerprint or caller-owned version.
    preprocessing_version (str): Nonempty preprocessing-version label.


**Returns**

    tuple[str, str, str]: Weight version, preprocessing version and pixel digest.


**Raises**

    ValueError: Either version is empty or the image is invalid.


**Limitations**

    Version labels are trusted identifiers, not validation of an algorithm.
    No user, tenant or authorization scope is encoded here.


### PredictionCache

    PredictionCache

[Class source](../../lab/engineering_vision.py#L148)

Store a bounded least-recently-used cache of synthetic prediction values.

get and put protect entries and hit/miss counters with one lock and deep-copy
values to prevent external mutation. Eviction removes the least-recent entry.
The cache provides neither authorization nor tenant isolation. A real service
must authorize before lookup and define its permitted-data scope separately.


### PredictionCache.__init__

    PredictionCache.__init__(self, max_entries: int=64)

[Function source](../../lab/engineering_vision.py#L156)

Create an empty cache and its entry/counter lock.


**Parameters**

    max_entries (int): Retained-key capacity, inclusive range 1..4096; default 64.


**Returns**

    None.


**Raises**

    ValueError: Capacity is outside the permitted interval.
    TypeError: A supplied value cannot be compared with integer bounds.


**Side effects and ownership**

    Initializes an OrderedDict, a lock, and zero hit/miss counters.


### PredictionCache.get

    PredictionCache.get(self, key)

[Function source](../../lab/engineering_vision.py#L180)

Read a deep copy and mark an existing key as most recently used.


**Parameters**

    key (Hashable): Normally the tuple returned by make_cache_key.


**Returns**

    Any | None: A copied cached value, or None on a miss. Stored None is therefore
    indistinguishable from a miss to this API's callers.


**Raises**

    TypeError: The key is unhashable. Copy failures propagate.


**Side effects and ownership**

    Under the cache lock, increments hits or misses and updates LRU order.


### PredictionCache.put

    PredictionCache.put(self, key, value)

[Function source](../../lab/engineering_vision.py#L204)

Store an isolated copy, update recency and evict excess entries.


**Parameters**

    key (Hashable): Cache identity, including the applicable version scope.
    value (Any): Deep-copyable result to retain.


**Returns**

    None.


**Raises**

    TypeError: The key is unhashable. Deep-copy failures propagate.


**Side effects and ownership**

    Mutates the cache under its lock, evicting oldest keys until within capacity.


### SyntheticImage

    SyntheticImage

[Class source](../../lab/engineering_vision.py#L228)

Describe one immutable fictional grayscale classification fixture.


**Attributes**

    image_id (str): Split-prefixed fixture identifier, not a real image identity.
    label (int): 0 for a continuous stripe or 1 for an interrupted stripe.
    pixels (tuple[tuple[int, ...], ...]): Validated 32-by-32 uint8-style grayscale
        values. Frozen dataclass fields prevent reassignment; pixel tuples are
        immutable. Construction itself does not independently validate fields.


### generate_synthetic_examples

    generate_synthetic_examples(count: int, seed: int, split: str='fixture')

[Function source](../../lab/engineering_vision.py#L243)

Generate deterministic alternating-class fictional stripe images.


**Parameters**

    count (int): Number of fixtures, inclusive range 2..4096.
    seed (int): Seed for a private random.Random instance.
    split (str): Prefix for image_id values; default fixture.


**Returns**

    list[SyntheticImage]: Alternating labels with noisy backgrounds and either
    continuous or six-pixel-interrupted bright vertical stripes.


**Raises**

    ValueError: Count is out of bounds or generated-image validation fails.


**Side effects and ownership**

    Allocates fixture arrays without changing global random state. The task is
    deliberately artificial; its accuracy cannot establish real inspection quality.


### build_tiny_cnn

    build_tiny_cnn(seed: int=47)

[Function source](../../lab/engineering_vision.py#L283)

Construct an untrained two-class CNN for the grayscale stripe exercise.


**Parameters**

    seed (int): Seed passed to torch.manual_seed; default 47.


**Returns**

    torch.nn.Sequential: CPU model with two convolution/ReLU/pooling stages and
    a 512-to-2 linear classifier, initially in training mode.


**Raises**

    VisionDependencyUnavailable: PyTorch cannot be imported.
    PyTorch initialization errors propagate.


**Side effects and ownership**

    Resets PyTorch's global RNG and allocates model parameters. No weights are
    downloaded; calling this function alone does not train the model.


### preprocess_images

    preprocess_images(images, device='cpu')

[Function source](../../lab/engineering_vision.py#L315)

Validate, normalize and transfer a grayscale image batch.


**Parameters**

    images (list | tuple): Between 1 and 64 valid 32-by-32 grayscale images.
    device (str | torch.device): Target tensor device; defaults to CPU.


**Returns**

    torch.Tensor: float32 tensor shaped (batch, 1, 32, 32), divided by 255 so
    pixel intensities lie in 0..1, on the requested device.


**Raises**

    ValueError: Batch bounds or any image validation fails.
    VisionDependencyUnavailable: PyTorch is missing.
    PyTorch device/allocation errors propagate.


**Side effects and ownership**

    Allocates/transfers tensors; requesting CUDA can initialize a GPU context.


### weights_fingerprint

    weights_fingerprint(model)

[Function source](../../lab/engineering_vision.py#L341)

Hash model state names, shapes, dtypes and float32-serialized values.


**Parameters**

    model (torch.nn.Module): Model exposing state_dict tensor values.


**Returns**

    str: SHA-256 fingerprint independent of each tensor's current device.


**Raises**

    Tensor conversion/iteration errors and float32 packing errors propagate.


**Side effects and ownership**

    Copies detached state tensors to contiguous CPU storage. Each scalar is
    serialized as a little-endian float32, so this is intended for the tiny
    float32 model, not arbitrary bit-exact float64/integer state. Use
    fast_weights_fingerprint for state-byte hashing of the larger probe.


### VisionPredictor

    VisionPredictor

[Class source](../../lab/engineering_vision.py#L368)

Own a copied model snapshot and version-aware synthetic result cache.

The supplied model is copied to one device and placed in evaluation mode.
Callers must treat the private snapshot as immutable and replace this object
when changing weights. Cache operations are locked, but prediction counters and
model execution are not a general multi-worker serving synchronization contract.
A caller must authorize users independently; cached results grant no access.


### VisionPredictor.__init__

    VisionPredictor.__init__(self, model, device='cpu', preprocessing_version=PREPROCESS_VERSION, cache: PredictionCache | None=None)

[Function source](../../lab/engineering_vision.py#L377)

Snapshot a model and select its device, version label and cache.


**Parameters**

    model (torch.nn.Module): Source model copied without mutating the caller's model.
    device (str | torch.device): Target execution device; default CPU.
    preprocessing_version (str): Label used in cache identity; no alternate
        preprocessing algorithm is installed by changing this string.
    cache (PredictionCache | None): Shared version-scoped cache, or a new cache.


**Returns**

    None.


**Raises**

    VisionDependencyUnavailable: PyTorch is absent. Copy, hash, device transfer
    and allocation failures propagate.


**Side effects and ownership**

    Allocates a model copy, possibly a CUDA context, and initializes compute_calls.


### VisionPredictor.predict_batch

    VisionPredictor.predict_batch(self, images)

[Function source](../../lab/engineering_vision.py#L406)

Execute an uncached two-class prediction for a grayscale batch.


**Parameters**

    images (list | tuple): Between 1 and 64 valid grayscale images.


**Returns**

    list[dict]: Predicted index/label, two probabilities, weight/preprocessing
    versions and cache_hit=False for each image, preserving input order.


**Raises**

    Image/preprocessing validation and model/device execution errors propagate.


**Side effects and ownership**

    Allocates device tensors and transfers probabilities to CPU. Uses
    inference_mode and increments compute_calls once per successfully processed
    batch; it does not consult or populate the cache.


### VisionPredictor.predict

    VisionPredictor.predict(self, pixels, use_cache=True)

[Function source](../../lab/engineering_vision.py#L436)

Predict one image using the versioned cache when requested.


**Parameters**

    pixels (Sequence[Sequence[int]]): Valid 32-by-32 grayscale image.
    use_cache (bool): Look up and populate the cache when true; default true.


**Returns**

    dict: One prediction; a cached copy is marked cache_hit=True.


**Raises**

    Image/version validation, deep-copy or model errors propagate.


**Side effects and ownership**

    Updates cache recency/counters and, on a miss, executes predict_batch.
    Authorization is the caller's responsibility before any cache lookup.


### train_tiny_model

    train_tiny_model(examples, seed=47, epochs=20, batch_size=32)

[Function source](../../lab/engineering_vision.py#L465)

Train the tiny CPU CNN on caller-provided synthetic labelled examples.


**Parameters**

    examples (Sequence[SyntheticImage]): Nonempty training fixtures with labels
        accepted by a two-class CrossEntropyLoss; callers supply valid data.
    seed (int): Model seed and minibatch generator seed base; default 47.
    epochs (int): Training epochs, inclusive range 1..100.
    batch_size (int): Positive minibatch size; default 32.


**Returns**

    tuple[torch.nn.Module, dict]: Trained CPU evaluation-mode model and metadata
    including Adam learning rate, loss per epoch and wall duration in seconds.


**Raises**

    ValueError: Epochs fail explicit validation; malformed/empty batches can fail
        preprocessing or range operations. PyTorch training errors propagate.


**Side effects and ownership**

    Resets model RNG, allocates tensors and mutates model/optimizer state.
    This function supplies no held-out quality check or real-data claim.


### evaluate_model

    evaluate_model(engine, examples)

[Function source](../../lab/engineering_vision.py#L517)

Measure two-class accuracy on nonempty held-out fixtures.


**Parameters**

    engine (VisionPredictor): Predictor whose uncached predict_batch is called.
    examples (Sequence[SyntheticImage]): Nonempty labelled evaluation fixtures.


**Returns**

    tuple[dict, list[dict]]: Sample/correct counts, accuracy and 2-by-2 confusion
    matrix (true-label rows, predicted-label columns), plus ordered predictions.


**Raises**

    Predictor errors propagate; empty examples cause division by zero and invalid
    labels can cause indexing errors. The caller must provide the declared task.


**Side effects and ownership**

    Executes batches of at most 64 and increments engine compute counters.
    Accuracy describes this fixture split only.


### _quantile

    _quantile(values, q)

[Function source](../../lab/engineering_vision.py#L547)

Linearly interpolate a quantile of a nonempty numeric sequence.


**Parameters**

    values (Sequence[float]): Observations in any consistent unit.
    q (float): Requested fraction in 0..1, as supplied by internal callers.


**Returns**

    float: Quantile in the input unit using position (n - 1) * q.


**Raises**

    Empty or incompatible input raises normal indexing/arithmetic errors;
    no public validation or NaN filtering is applied. Input order is unchanged.


### summarize_timings

    summarize_timings(seconds, batch_size)

[Function source](../../lab/engineering_vision.py#L568)

Summarize nonempty positive per-call durations measured in seconds.


**Parameters**

    seconds (Sequence[float]): Successful measurement durations, in seconds.
    batch_size (int): Images processed by each measured call.


**Returns**

    dict: Sample count, batch images, p50/p95/p99/mean in milliseconds and
    aggregate images per second across the measured call time.


**Raises**

    Empty or zero-total durations cause indexing/division errors. Callers must
    supply finite durations and a meaningful fixed batch size.


**Limitations**

    Throughput here covers sequential measured calls, not peak concurrent capacity.


### measure_calls

    measure_calls(call, warmup, repeats, batch_size, synchronize=lambda: None)

[Function source](../../lab/engineering_vision.py#L595)

Warm a callable, then time repeated synchronized wall-clock calls.


**Parameters**

    call (Callable[[], Any]): Work invoked once per warm-up and measurement.
    warmup (int): Number of unrecorded calls.
    repeats (int): Positive number of measured calls.
    batch_size (int): Images per call, used only for throughput reporting.
    synchronize (Callable[[], Any]): Device barrier; the default no-op lambda
        is appropriate for synchronous CPU work.


**Returns**

    dict: summarize_timings output, converting recorded seconds to milliseconds.


**Raises**

    Callable/barrier errors propagate; invalid or empty timing inputs are not
    independently validated here.


**Side effects and ownership**

    Executes caller work and barriers. Each measured call is bracketed by barriers;
    no concurrency, network request or cache policy is introduced by this helper.


### measure_engine

    measure_engine(engine, examples, batch_size, warmup=10, repeats=40)

[Function source](../../lab/engineering_vision.py#L630)

Compare prepared-tensor forward time with the uncached grayscale pipeline.


**Parameters**

    engine (VisionPredictor): Model snapshot, torch module and device to measure.
    examples (Sequence[SyntheticImage]): At least batch_size fixtures.
    batch_size (int): Images per call, within preprocess_images limits.
    warmup (int): Unrecorded calls per path; default 10.
    repeats (int): Measured calls per path; default 40.


**Returns**

    dict: Device, batch size, model_call and full_pipeline timing summaries with
    separate textual scopes.


**Raises**

    Preprocessing, inference and timing errors propagate.


**Side effects and ownership**

    Prepares a device tensor; synchronizes CUDA before/after timed boundaries.
    Full-pipeline calls increment engine counters, with cache disabled.
    Forward-only timing excludes validation, transfer, softmax and result conversion.


### measure_engine.model_only

    measure_engine.model_only()

[Function source](../../lab/engineering_vision.py#L658)

Run the captured model on the captured already prepared device tensor.


**Returns**

    torch.Tensor: Raw two-class logits for the fixed batch.


**Raises**

    Model/device errors propagate to measure_calls.


**Side effects and ownership**

    Uses inference_mode. The outer CUDA barrier supplies synchronization; this
    closure excludes preprocessing, cache lookup and probability conversion.


### demonstrate_cache

    demonstrate_cache(engine, pixels)

[Function source](../../lab/engineering_vision.py#L682)

Exercise synthetic cache reuse and version/input invalidation.


**Parameters**

    engine (VisionPredictor): Prefer a fresh predictor/cache for a cold first call.
    pixels (Sequence[Sequence[int]]): One valid grayscale fixture.


**Returns**

    dict: Individual boolean checks, combined passed flag, key components and
    explicit authorization/preprocessing limitations.


**Raises**

    Prediction, copy, parameter mutation and hashing errors propagate.


**Side effects and ownership**

    Performs two predictions, increments cache counters, and changes a separate
    CPU copy of model weights plus a copied image to construct distinct keys.
    The owned engine weights and original pixels remain unchanged. Changing only
    a preprocessing label demonstrates invalidation, not acceptance of new code.


### create_local_http_server

    create_local_http_server(engine, port=0)

[Function source](../../lab/engineering_vision.py#L727)

Construct an unstarted sequential grayscale HTTP demo on IPv4 loopback.


**Parameters**

    engine (VisionPredictor): Snapshot used for uncached /predict requests.
    port (int): TCP port; 0 asks the OS to allocate an available local port.


**Returns**

    HTTPServer: Caller-owned server bound to 127.0.0.1. Call serve_forever in a
    thread or the foreground and explicitly shutdown/server_close when finished.


**Raises**

    OSError: Bind or socket creation fails. Handler-specific input errors become
        HTTP responses; model errors not caught by the handler can propagate.


**Side effects and ownership**

    Binds a local socket immediately, without starting a serving thread.
    This synthetic endpoint has no authentication, TLS or production access claim.


### create_local_http_server.Handler

    create_local_http_server.Handler

[Class source](../../lab/engineering_vision.py#L746)

Handle the captured predictor's synthetic grayscale localhost requests.

GET /health reports the model version; POST /predict accepts bounded JSON
pixels and bypasses the result cache. This class captures engine from its
factory. HTTPServer invokes one request at a time; no identity is authenticated.


### create_local_http_server.Handler.log_message

    create_local_http_server.Handler.log_message(self, *args)

[Function source](../../lab/engineering_vision.py#L753)

Suppress BaseHTTPRequestHandler's default stderr request logging.


**Parameters**

    *args (Any): Formatting string and substitution values supplied by the
        standard-library handler; intentionally ignored.


**Returns**

    None. No request body, access record or audit evidence is retained here.


### create_local_http_server.Handler._reply

    create_local_http_server.Handler._reply(self, code, payload)

[Function source](../../lab/engineering_vision.py#L765)

Serialize a JSON payload and write one HTTP response.


**Parameters**

    code (int): HTTP status code.
    payload (JSON-serializable object): Response body.


**Returns**

    None.


**Raises**

    Serialization and socket/write errors propagate.


**Side effects and ownership**

    Writes UTF-8 JSON plus Content-Type and byte-accurate Content-Length headers
    on the current request connection; adds no authentication or CORS policy.


### create_local_http_server.Handler.do_GET

    create_local_http_server.Handler.do_GET(self)

[Function source](../../lab/engineering_vision.py#L789)

Serve /health, or return JSON HTTP 404 for another path.


**Returns**

    None: The response is written to the request socket.


**Side effects and ownership**

    Reveals only the synthetic scope, engine device and weight version.
    Executes no inference. JSON/socket errors from _reply propagate.


### create_local_http_server.Handler.do_POST

    create_local_http_server.Handler.do_POST(self)

[Function source](../../lab/engineering_vision.py#L804)

Validate a bounded /predict JSON request and run uncached inference.


**Returns**

    None: Writes HTTP 200 prediction, 404 for another path, 413 for a declared
    body length outside 1..65536 bytes, or 400 for handled input errors.


**Raises**

    Model/runtime and socket errors outside ValueError, TypeError and
    UnicodeDecodeError propagate through the standard-library server.


**Side effects and ownership**

    Reads the declared body length and executes engine.predict(use_cache=False).
    Only the captured synthetic grayscale task is accepted; no identity is checked.


### measure_http

    measure_http(engine, pixels, warmup=5, repeats=20)

[Function source](../../lab/engineering_vision.py#L835)

Measure sequential end-to-end requests against an owned localhost server.


**Parameters**

    engine (VisionPredictor): Predictor whose cache is bypassed.
    pixels (Sequence[Sequence[int]]): JSON-serializable valid grayscale fixture.
    warmup (int): Unrecorded HTTP calls; default 5.
    repeats (int): Measured HTTP calls; default 20.


**Returns**

    dict: Device, latency/throughput summaries and scope/authentication metadata.


**Raises**

    HTTP, JSON, model-version mismatch or timing errors propagate after cleanup.


**Side effects and ownership**

    Binds an ephemeral loopback port and starts one daemon serving thread.
    Finally shuts down and closes its server, then joins for up to five seconds.
    Each client call has a ten-second timeout. This is sequential wall time, not
    a concurrent load or production-network measurement.


### measure_http.request

    measure_http.request()

[Function source](../../lab/engineering_vision.py#L863)

Send one captured JSON fixture and verify its returned model version.


**Returns**

    None: The response is consumed and only its weights_version is checked.


**Raises**

    RuntimeError: Returned weights_version differs from the captured predictor.
    HTTP, timeout and JSON errors propagate to the outer measurement.


**Side effects and ownership**

    Opens a localhost HTTP request with a ten-second timeout and closes its
    response context. The enclosing factory owns serving-thread cleanup.


### build_resnet18_probe

    build_resnet18_probe(seed=47, num_classes=1000)

[Function source](../../lab/engineering_vision.py#L894)

Construct a random-weight ResNet-18-shaped RGB performance probe.


**Parameters**

    seed (int): Global PyTorch initialization seed; default 47.
    num_classes (int): Output logit width; default 1000. Callers should supply a
        meaningful positive dimension; no explicit dimension validation occurs.


**Returns**

    torch.nn.Module: CPU evaluation-mode residual network with [2,2,2,2] basic
    block stages and random weights.


**Raises**

    VisionDependencyUnavailable: PyTorch is absent.
    PyTorch construction errors propagate.


**Side effects and ownership**

    Resets torch's RNG and allocates convolution/batch-normalization/classifier
    parameters. No torchvision, pretrained weights, download, trained semantics
    or quality claim is introduced.


### build_resnet18_probe.BasicBlock

    build_resnet18_probe.BasicBlock

[Class source](../../lab/engineering_vision.py#L918)

Implement one random-weight residual block captured by the probe factory.

The main path has two 3-by-3 convolutions with batch normalization. A projected
1-by-1 identity path handles stride/channel changes; otherwise it is identity.
Input/output tensors use the normal (batch, channels, height, width) layout.


### build_resnet18_probe.BasicBlock.__init__

    build_resnet18_probe.BasicBlock.__init__(self, incoming, outgoing, stride=1)

[Function source](../../lab/engineering_vision.py#L925)

Allocate main and residual projection paths.


**Parameters**

    incoming (int): Input channels.
    outgoing (int): Output channels.
    stride (int): First-convolution/projection spatial stride; default 1.


**Returns**

    None.


**Raises**

    PyTorch layer-construction/allocation errors propagate.


**Side effects and ownership**

    Registers convolution, batch-normalization and ReLU submodules.
    The outer probe later applies its own weight initialization.


### build_resnet18_probe.BasicBlock.forward

    build_resnet18_probe.BasicBlock.forward(self, x)

[Function source](../../lab/engineering_vision.py#L953)

Apply the residual block to one prepared RGB-feature tensor.


**Parameters**

    x (torch.Tensor): Floating tensor shaped (batch, incoming, height, width)
        on the same device/dtype as this block.


**Returns**

    torch.Tensor: ReLU-transformed sum, with outgoing channels and the configured
    downsampling stride.


**Raises**

    PyTorch shape/device errors propagate. ReLU is configured in-place;
    normal PyTorch training/evaluation and autograd rules still apply.


### build_resnet18_probe.ResNet18Probe

    build_resnet18_probe.ResNet18Probe

[Class source](../../lab/engineering_vision.py#L972)

Combine an RGB stem, eight residual blocks and an output classifier.

The nested class captures num_classes and torch.nn from the factory.
Its default weights carry no trained image-label meaning.


### build_resnet18_probe.ResNet18Probe.__init__

    build_resnet18_probe.ResNet18Probe.__init__(self)

[Function source](../../lab/engineering_vision.py#L978)

Build the residual network and initialize its learnable state.


**Returns**

    None.


**Raises**

    PyTorch layer-construction/allocation errors propagate.


**Side effects and ownership**

    Creates the 7-by-7 RGB stem, [2,2,2,2] block stages, adaptive pooling and
    num_classes-wide linear head. Applies Kaiming-normal convolution weights
    and unit/zero batch-normalization parameters. RNG seeding is owned by the
    enclosing factory immediately before this constructor is called.


### build_resnet18_probe.ResNet18Probe.forward

    build_resnet18_probe.ResNet18Probe.forward(self, x)

[Function source](../../lab/engineering_vision.py#L1010)

Return raw logits from a caller-prepared RGB tensor.


**Parameters**

    x (torch.Tensor): Floating (batch, 3, height, width) tensor, normally
        image_size square and compatible with the model's device/dtype.


**Returns**

    torch.Tensor: Shape (batch, num_classes), with raw unnormalized logits.


**Raises**

    PyTorch shape/device errors propagate. This method performs no input-byte
    validation, normalization, softmax, caching or quality assessment.


### fast_weights_fingerprint

    fast_weights_fingerprint(model)

[Function source](../../lab/engineering_vision.py#L1030)

Hash contiguous model state bytes with tensor name, dtype and shape.


**Parameters**

    model (torch.nn.Module): State tensors convertible to contiguous CPU NumPy arrays.


**Returns**

    str: SHA-256 fingerprint of sorted state entries, independent of device.


**Raises**

    Tensor transfer or NumPy conversion errors propagate; unsupported tensor
    dtypes are not silently cast.


**Side effects and ownership**

    Detaches and copies state to CPU as needed. Unlike weights_fingerprint, this
    serializes the tensor's actual NumPy bytes rather than scalar float32 values.


### latency_summary

    latency_summary(milliseconds)

[Function source](../../lab/engineering_vision.py#L1055)

Summarize already-millisecond latency observations.


**Parameters**

    milliseconds (Sequence[float]): Samples in milliseconds, or an empty sequence.


**Returns**

    dict: Sample count and interpolated p50/p95/p99/max milliseconds. Empty
    input yields zero samples and None for all percentiles/max.


**Raises**

    Sorting/arithmetic errors propagate for incompatible values. Finite,
    nonnegative samples are the caller's responsibility.


### ProbeQueueFull

    ProbeQueueFull

[Class source](../../lab/engineering_vision.py#L1076)

Signal immediate rejection when the bounded inference queue has no slot.

This RuntimeError subclass is raised by submit and translated to HTTP 429 by
the binary probe handler. It does not mean a model execution failed.


### BoundedBatchVisionService

    BoundedBatchVisionService

[Class source](../../lab/engineering_vision.py#L1085)

Run a bounded synthetic RGB queue with one or two batching workers.

Each accepted payload has exactly image_size * image_size * 3 uint8 bytes.
Workers collect up to max_batch requests, with a collection timer starting only
after the first request is removed. Queue wait before that moment is additional.
No cache or authentication is provided. Metrics/counters use a lock; the model
callback is responsible for safe independent worker instances.

Callers must stop submissions before close and must not cancel returned futures:
submission/closure is not one atomic transaction, and cancellation is not a
supported path in the worker completion code. A hanging callback can outlive the
30-second join and causes close to raise rather than prove complete shutdown.


### BoundedBatchVisionService.__init__

    BoundedBatchVisionService.__init__(self, batch_call, *, image_size=224, workers=1, max_batch=8, max_wait_ms=5, queue_size=64)

[Function source](../../lab/engineering_vision.py#L1099)

Validate bounds, create the shared queue and start daemon workers.


**Parameters**

    batch_call (Callable): Called as batch_call(worker_index, payloads), returning
        an equally sized list of result dicts and timing dict containing model_ms.
    image_size (int): RGB square width/height in pixels, 32..224.
    workers (int): One or two threads/model-instance indices.
    max_batch (int): Maximum requests collected per batch, 1..8.
    max_wait_ms (float): Maximum collection wait after the first dequeue, 0..5 ms.
    queue_size (int): Pending-request capacity, 1..256.


**Returns**

    None.


**Raises**

    ValueError: Configuration bounds fail. Thread creation errors propagate.


**Side effects and ownership**

    Immediately starts worker threads; owner must eventually call close.
    Queue capacity counts waiting requests, not currently executing batches.


### BoundedBatchVisionService.submit

    BoundedBatchVisionService.submit(self, payload)

[Function source](../../lab/engineering_vision.py#L1145)

Validate and enqueue one RGB payload without waiting for capacity.


**Parameters**

    payload (bytes): Exactly image_size * image_size * 3 interleaved RGB bytes.


**Returns**

    concurrent.futures.Future: Resolves to a result dict augmented with queue
    wait, service latency in milliseconds and actual batch size.


**Raises**

    ValueError: Type/length is invalid.
    RuntimeError: Closure has been requested.
    ProbeQueueFull: Queue is currently full; rejection count is incremented.


**Side effects and ownership**

    Captures submission time, enqueues work and updates observed queue depth.
    Stop concurrent submissions before close; future cancellation is unsupported.


### BoundedBatchVisionService._run

    BoundedBatchVisionService._run(self, worker_index)

[Function source](../../lab/engineering_vision.py#L1180)

Drain the shared queue through one worker's model callback.


**Parameters**

    worker_index (int): Index selecting the independent model snapshot/stream.


**Returns**

    None: Runs until stop is set and the waiting queue has drained.


**Raises**

    Ordinary callback exceptions are placed on every future in that batch.
    Unsupported future cancellation/completion errors can escape worker handling.


**Side effects and ownership**

    Dequeues and batches requests; collection wait is bounded separately from
    total queue latency. Records completed-batch metrics under the lock, resolves
    futures, and marks each queued item done. No per-request hard deadline
    cancels model computation; a blocked callback can delay shutdown.


### BoundedBatchVisionService.metrics

    BoundedBatchVisionService.metrics(self)

[Function source](../../lab/engineering_vision.py#L1235)

Snapshot completed-batch throughput and latency counters under the lock.


**Returns**

    dict: Completed batch/image totals, occupancy histogram, mean batch size,
    queue rejections/depth, and model/pipeline latency summaries in milliseconds.


**Raises**

    Missing or nonnumeric model_ms values from a malformed callback can cause
    aggregation errors.


**Side effects and ownership**

    None beyond a temporary locked read. Failed batches are absent from records;
    caller-level HTTP accounting is required to retain their requests/errors.


### BoundedBatchVisionService.close

    BoundedBatchVisionService.close(self)

[Function source](../../lab/engineering_vision.py#L1264)

Request queue draining and join each owned worker for up to 30 seconds.


**Returns**

    None: All checked worker threads have stopped.


**Raises**

    RuntimeError: Any worker is still alive after its join allowance.


**Side effects and ownership**

    Sets stop, prevents ordinary later submissions and waits for owned threads.
    It does not terminate a running callback, close an HTTP server, or manage
    other processes. Owners must stop new submissions before invoking this method.


### make_resnet_batch_runtime

    make_resnet_batch_runtime(base_model, device='cpu', precision='fp32', workers=1, image_size=224)

[Function source](../../lab/engineering_vision.py#L1285)

Prepare per-worker model copies and a timed RGB batch callback.


**Parameters**

    base_model (torch.nn.Module): Unchanged source probe copied for each worker.
    device (str): CPU or a CUDA device string; default cpu.
    precision (str): fp32, or fp16 only for a CUDA device.
    workers (int): Number of model snapshots; normally one or two.
    image_size (int): Square RGB width/height in pixels; normally 32..224.


**Returns**

    tuple[Callable, list[torch.nn.Module], list]: batch_call, copied evaluation
    models and one CUDA stream per model (or None entries on CPU).


**Raises**

    ValueError: Precision is unsupported or FP16 is requested for CPU.
    PyTorch device/allocation errors propagate; caller validates other bounds.


**Side effects and ownership**

    Allocates device model copies/streams and synchronizes their initialization.
    Sets no allocator budget and downloads nothing; all outputs remain
    performance-only because the probe has random weights.


### make_resnet_batch_runtime.batch_call

    make_resnet_batch_runtime.batch_call(worker_index, payloads)

[Function source](../../lab/engineering_vision.py#L1317)

Run one worker's captured model on raw synthetic RGB payloads.


**Parameters**

    worker_index (int): Valid index into captured models and CUDA streams.
    payloads (Sequence[bytes]): Nonempty bodies containing image_size-square
        interleaved RGB uint8 values; the service validates body lengths first.


**Returns**

    tuple[list[dict], dict]: Top-class indices/probabilities with explicit
    random_weights/quality_claimed flags, plus forward-only model_ms.


**Raises**

    RuntimeError: Any returned top probability is non-finite.
    Indexing, shape, allocation and PyTorch execution errors propagate.


**Side effects and ownership**

    Copies buffers to CPU tensors, converts HWC to CHW, divides by 255, stacks,
    transfers to device/precision, executes inference and copies results to CPU.
    CUDA events time forward execution on that worker's stream; its final
    synchronization completes asynchronous work before returning. CPU timing
    brackets forward only. No class semantics, cache or image decoder is supplied.


### create_batched_http_server

    create_batched_http_server(service, port=0)

[Function source](../../lab/engineering_vision.py#L1370)

Bind a threaded synthetic RGB probe server to IPv4 loopback.


**Parameters**

    service (BoundedBatchVisionService): Existing started batching service.
    port (int): TCP port; 0 requests an OS-allocated local port.


**Returns**

    ThreadingHTTPServer: Bound but not serving until the caller invokes
    serve_forever. The caller owns server and service cleanup separately.


**Raises**

    OSError: Socket bind/allocation fails.


**Side effects and ownership**

    Binds 127.0.0.1; request threads are daemon threads with listen backlog 128.
    No authentication, TLS, sensitive-data processing or production SLA is claimed.


### create_batched_http_server.Server

    create_batched_http_server.Server

[Class source](../../lab/engineering_vision.py#L1388)

Configure the captured RGB server's daemon threads and listen backlog.

ThreadingHTTPServer supplies lifecycle methods. daemon_threads=True prevents
request threads from keeping the interpreter alive; request_queue_size=128 is
the TCP listen backlog, distinct from the model's bounded inference queue.


### create_batched_http_server.Handler

    create_batched_http_server.Handler

[Class source](../../lab/engineering_vision.py#L1398)

Translate binary localhost prediction requests into service futures.

This factory-local handler captures service. Only POST /predict-rgb is handled;
its connection and result waits are bounded, but timeouts do not cancel queued
or executing inference. Access control is intentionally absent.


### create_batched_http_server.Handler.log_message

    create_batched_http_server.Handler.log_message(self, *args)

[Function source](../../lab/engineering_vision.py#L1405)

Suppress the standard handler's stderr request logging.


**Parameters**

    *args (Any): Standard-library log format and substitutions, ignored.


**Returns**

    None. This provides neither request logging nor an audit trail.


### create_batched_http_server.Handler.reply

    create_batched_http_server.Handler.reply(self, code, value)

[Function source](../../lab/engineering_vision.py#L1416)

Write a JSON status response, tolerating a disconnected probe client.


**Parameters**

    code (int): HTTP response status.
    value (JSON-serializable object): Body value.


**Returns**

    None.


**Raises**

    Serialization/header errors propagate. BrokenPipeError and
    ConnectionResetError during body writing are intentionally swallowed.


**Side effects and ownership**

    Writes JSON, Content-Type and byte Content-Length on this connection.


### create_batched_http_server.Handler.do_POST

    create_batched_http_server.Handler.do_POST(self)

[Function source](../../lab/engineering_vision.py#L1443)

Accept a correctly sized RGB body and await its inference future.


**Returns**

    None: Writes HTTP 200 for a result, 404 for another path, 400 for malformed
    length/value input, 429 for queue saturation, or 503 for other caught errors.


**Side effects and ownership**

    Sets a 15-second socket timeout, reads the declared exact body and waits up
    to 15 seconds for the service future. This timeout does not cancel inference.
    Error messages are capped at 300 characters for generic exceptions.
    Request identity is not authenticated; the endpoint is a local synthetic probe.


### run_open_loop_http

    run_open_loop_http(server, payloads, target_rps=100, duration_seconds=10)

[Function source](../../lab/engineering_vision.py#L1476)

Offer scheduled localhost requests independently of response completion.


**Parameters**

    server (HTTPServer): Already-serving RGB server with a loopback port.
    payloads (Sequence[bytes]): Nonempty cyclic input fixture list; caller owns
        image validation and supplies the required body dimensions.
    target_rps (int | float): Offered arrival rate, 1..200 requests per second.
    duration_seconds (int | float): Injection window, 1..30 seconds.


**Returns**

    dict: Offered/submitted/completed/success/error/drop accounting, observed
    rates and p50/p95/p99/max latency summaries. Successful-request percentiles
    exclude HTTP failures; all_response_latency retains completed errors too.


**Raises**

    ValueError: Rate/duration bounds fail.
    Scheduling/submission or malformed caller-input errors can propagate.
    Ordinary request/network/JSON failures are returned as client_error records.


**Side effects and ownership**

    Opens one HTTP connection per request, using up to 128 client threads and
    256 pending slots. Arrivals use absolute due times; client saturation drops
    requests and records them instead of waiting for previous responses.
    Waits for submitted calls to drain, but does not close server or service.
    Network calls have 16-second timeouts; total run time may exceed the injection
    window. Reports characterize this localhost workload, not sustainable capacity.


### run_open_loop_http.request

    run_open_loop_http.request(payload, due)

[Function source](../../lab/engineering_vision.py#L1512)

Send one captured RGB request and record wall-clock stage latencies.


**Parameters**

    payload (bytes): One synthetic image body.
    due (float): Scheduled perf_counter arrival timestamp in seconds.


**Returns**

    dict: HTTP status or client_error, completion timestamp, HTTP milliseconds,
    scheduled-arrival milliseconds, optional server queue/service milliseconds
    and error type.


**Side effects and ownership**

    Opens a 16-second-timeout localhost connection and always closes it and
    releases the captured pending-slot semaphore after the request block.
    Network/JSON exceptions become error records, without retrying or cancelling
    server work. The measured HTTP time starts when this client worker begins,
    while scheduled_arrival_ms also includes client executor/scheduler delay.


## scripts/measure_vision_engineering.py

[Open module](../../scripts/measure_vision_engineering.py)


Run bounded synthetic vision experiments and retain explicit evidence.

The default CLI path trains a tiny grayscale CNN, checks held-out fixtures, and
compares sequential CPU/CUDA calls. --load-probe instead runs a random-weight
ResNet-18-shaped RGB service under scheduled concurrent localhost traffic.
Neither path downloads weights/packages or approves production use. Raw reports
and checkpoints must be written to a new directory outside this Git checkout.
The module adds the checkout root to sys.path at import, but experiments start
only when main is called or the file is executed as a script.


### run_load_probe

    run_load_probe(args, output)

[Function source](../../scripts/measure_vision_engineering.py#L36)

Measure the random-weight RGB probe under one bounded offered-load point.


**Parameters**

    args (argparse.Namespace): CLI-validated configuration with seed (int),
        require_cuda (bool), gpu_memory_mib (int, 64..512 MiB allocator cap),
        input_size (int, 32..224 pixels per side), small_batch (int, 1..8 images),
        warmup (int, calls per worker), queue_size (int, 1..256 requests),
        batch_wait_ms (float, 0..5 ms after first dequeue), target_rps (int,
        1..200 requests/second), and duration (float, 1..30 seconds per case).
    output (pathlib.Path): Existing writable empty evidence directory, validated
        and created by main outside the source checkout.


**Returns**

    int: 0 when all requested configurations complete their request accounting,
    1 for a partial case failure, or 2 for a caught setup/execution failure.


**Raises**

    Evidence-write/cleanup failures can escape if they occur while handling
    another error; otherwise Exception instances are recorded in report.json.


**Side effects and ownership**

    Sets process-global PyTorch thread/cuDNN settings and optional CUDA allocator
    fraction. Builds random model weights and eight seeded RGB fixtures. Runs
    CPU FP32/one-worker, then CUDA FP32/FP16 with one/two workers when available.
    Creates and closes owned loopback servers and model-worker threads, prints
    per-case JSON, saves incremental evidence, and collects unused allocations.


**Limitations**

    Completion means accounting completed, not that latency, model quality or a
    release threshold passed. Random outputs have no detection meaning.
    Model/driver overhead and other processes fall outside the allocator cap.
    Settings are not restored for reuse as a general in-process application API.


### run_load_probe.save

    run_load_probe.save()

[Function source](../../scripts/measure_vision_engineering.py#L84)

Write the captured current report as UTF-8 indented JSON.


**Returns**

    None.


**Raises**

    Serialization or filesystem errors propagate to the surrounding run logic.


**Side effects and ownership**

    Replaces this run's report.json, allowing partial progress to be retained.
    The caller selected an initially empty directory; historical runs elsewhere
    are not overwritten. Captures report and path; accepts no parameters.


### run_load_probe.run_case

    run_load_probe.run_case(device, precision, workers)

[Function source](../../scripts/measure_vision_engineering.py#L151)

Warm and measure one device/precision/worker configuration.


**Parameters**

    device (str): cpu or cuda:0 from the captured configuration matrix.
    precision (str): fp32, or fp16 on CUDA.
    workers (int): Number of independent model copies/streams, one or two.


**Returns**

    dict: Device configuration, actual weight hash, finite-output diagnostic,
    request accounting, batch metrics, optional CUDA memory counters and status.
    Caught execution errors become status=failed with capped error details.


**Raises**

    Server/service cleanup errors in finally can propagate to the outer run.
    They are not converted into a successful or apparently closed case.


**Side effects and ownership**

    Captures base model, synthetic payloads, torch and CLI settings. Allocates
    worker-specific copies/streams, executes warm-ups and one diagnostic call,
    then serves uncached synthetic RGB requests on an ephemeral loopback port.
    Finally shuts down the owned HTTP server and batch service and records
    closure flags. The diagnostic checks finiteness, never class accuracy.


### main

    main()

[Function source](../../scripts/measure_vision_engineering.py#L244)

Parse the vision CLI, enforce experiment bounds and retain measured evidence.


**Parameters**

    None. Reads process sys.argv using argparse. --output is required; remaining
    options select seed, train/test fixture counts, epochs, warm-up/repeat counts,
    small-batch size, GPU allocator MiB, optional HTTP timing/persistent localhost
    serving, CUDA requirement, and the separate open-loop load-probe settings.


**Returns**

    int: Default tiny-model path returns 0 for its synthetic checks, 1 when a
    synthetic check fails, or 2 for a caught runtime failure. --load-probe returns
    that path's code: 0 completed, 1 partial failure, 2 caught failure.


**Raises**

    SystemExit: argparse help/usage/validation exits, normally 0 or 2.
    Filesystem errors during output validation/creation or error-report writing
    can propagate. KeyboardInterrupt stops an optional owned --serve loop.


**Side effects and ownership**

    Resolves and creates a fresh output directory outside the checkout. The tiny
    path changes global PyTorch thread/cuDNN settings, sets an optional 64..512 MiB
    allocator fraction, generates separate seeded train/test fixtures, rejects
    image-hash overlap, trains on CPU, and writes model checkpoint/report JSON.
    It measures CPU and available CUDA routes, optionally times an ephemeral
    loopback server, prints a summary, and can keep --serve running until stopped.
    Garbage collection runs on exit from the tiny-model try/finally.


**Limitations**

    Passing means only the declared synthetic fixture/cache/consistency checks
    passed. No production approval, YOLO/TensorRT/Triton measurement, GPU-in-OKD
    deployment or sustainable concurrent-capacity claim follows. The load probe
    uses random weights and must not be interpreted as a trained detector.


## tests/test_engineering_vision.py

[Open module](../../tests/test_engineering_vision.py)


Verify synthetic vision input, cache, concurrency and optional device behavior.

Base tests validate boundaries/cache identities and deliberately overload owned
localhost services with scripted callbacks. PyTorch tests are skipped when torch
is absent; CUDA consistency is skipped when no CUDA device is accessible.
Tests use fictional fixtures only, but concurrency cases create local sockets
and threads, and optional model cases allocate tensors/change torch settings.
A passing suite does not validate image-model quality or production capacity.


### InputAndCacheTests

    InputAndCacheTests

[Class source](../../tests/test_engineering_vision.py#L25)

Check pure input validation and cache identity without requiring PyTorch.

unittest creates a fresh instance and invokes setUp for each case. Tests require
no model download, HTTP service, GPU, real user identity or external dataset.


### InputAndCacheTests.setUp

    InputAndCacheTests.setUp(self)

[Function source](../../tests/test_engineering_vision.py#L31)

Prepare one seeded grayscale image for each pure boundary/cache test.


**Returns**

    None.


**Side effects and ownership**

    Stores a SyntheticImage pixel tuple on self.image using two fixtures seeded
    with 123. Fixture-generation/validation errors propagate as setup failures.


### InputAndCacheTests.test_rejects_wrong_shape_and_non_integer_pixels

    InputAndCacheTests.test_rejects_wrong_shape_and_non_integer_pixels(self)

[Function source](../../tests/test_engineering_vision.py#L43)

Assert rejection of malformed shapes and invalid grayscale pixel types/ranges.


**Returns**

    None.


**Raises**

    AssertionError: A 31-row/31-column image, boolean pixel, value 256 or float
    pixel fails to raise ValueError.


**Side effects and ownership**

    Runs separate subtests over deliberately invalid in-memory arrays.
    It verifies representative invalid boundaries, not every possible input type.


### InputAndCacheTests.test_cache_invalidates_on_weights_preprocessing_and_input

    InputAndCacheTests.test_cache_invalidates_on_weights_preprocessing_and_input(self)

[Function source](../../tests/test_engineering_vision.py#L63)

Assert that changing any cache-identity component yields a miss.


**Returns**

    None.


**Raises**

    AssertionError: The original key does not retrieve its stored value, or a
    changed weight label, preprocessing label or pixel still hits that entry.


**Side effects and ownership**

    Creates a two-entry cache, inserts a synthetic result and reads it under
    several keys. Mutates a copied image only. This checks key isolation, not
    model execution, authorization or a new preprocessing implementation.


### InputAndCacheTests.test_held_out_images_are_disjoint_from_training

    InputAndCacheTests.test_held_out_images_are_disjoint_from_training(self)

[Function source](../../tests/test_engineering_vision.py#L89)

Check pixel-hash disjointness for the two selected synthetic fixture seeds.


**Returns**

    None.


**Raises**

    AssertionError: Any of 32 training hashes equals a held-out hash.


**Side effects and ownership**

    Generates two in-memory fixture sets seeded with 47 and 10047.
    The result applies to these selected sets; it is not a universal guarantee
    of statistical independence or absence of real-data leakage.


### InputAndCacheTests.test_missing_optional_runtime_has_a_clear_error

    InputAndCacheTests.test_missing_optional_runtime_has_a_clear_error(self)

[Function source](../../tests/test_engineering_vision.py#L107)

Verify the explanatory exception when this interpreter lacks PyTorch.


**Returns**

    None.


**Raises**

    unittest.SkipTest: PyTorch can already be located in this interpreter.
    AssertionError: require_torch does not raise VisionDependencyUnavailable.


**Side effects and ownership**

    Performs import discovery and, on the missing-runtime path, an attempted
    import. It neither installs a package nor checks CUDA capability.


### ConcurrentServiceTests

    ConcurrentServiceTests

[Class source](../../tests/test_engineering_vision.py#L127)

Exercise bounded queue overload and localhost accounting with fake models.

The callbacks simulate service time; they do not execute PyTorch or prove model
latency. Each test owns and closes its worker threads and any HTTP socket.


### ConcurrentServiceTests.test_bounded_queue_rejects_and_closes_workers

    ConcurrentServiceTests.test_bounded_queue_rejects_and_closes_workers(self)

[Function source](../../tests/test_engineering_vision.py#L133)

Verify nonblocking overload rejection, payload validation and worker cleanup.


**Returns**

    None.


**Raises**

    AssertionError: Expected future values, queue metrics, rejection, input
    failure or final stopped-worker condition differ from the fixture contract.


**Side effects and ownership**

    Starts one batching worker with a one-slot queue. A gated fake callback holds
    the first request while a second waits; the third must raise ProbeQueueFull.
    Finally releases the event and closes the service even after an assertion.


### ConcurrentServiceTests.test_bounded_queue_rejects_and_closes_workers.blocked_call

    ConcurrentServiceTests.test_bounded_queue_rejects_and_closes_workers.blocked_call(worker, payloads)

[Function source](../../tests/test_engineering_vision.py#L150)

Hold a scripted model batch until released or its three-second wait expires.


**Parameters**

    worker (int): Worker index supplied by the service; unused by this fixture.
    payloads (Sequence[bytes]): Synthetic images whose count determines results.


**Returns**

    tuple[list[dict], dict]: One value=1 result per payload and model_ms=0.0.


**Side effects and ownership**

    Sets the captured entered event and waits up to three seconds on release.
    The zero timing is deliberate test metadata, not measured inference time.
    Unexpected callback errors propagate into the service's future handling.


### ConcurrentServiceTests.test_http_open_loop_accounts_for_overload_and_queue_time

    ConcurrentServiceTests.test_http_open_loop_accounts_for_overload_and_queue_time(self)

[Function source](../../tests/test_engineering_vision.py#L190)

Verify that scheduled arrivals retain rejections and queue latency under overload.


**Returns**

    None.


**Raises**

    AssertionError: The run loses any of 40 offered requests, reports no HTTP 429,
    serves none, hides expected waiting, or leaves its serving thread alive.


**Side effects and ownership**

    Starts a fake one-worker, two-slot service and an ephemeral loopback server.
    Offers 40 requests/second for one second against an 80-millisecond callback,
    checks response/queue timing, and finally shuts down sockets and all workers.
    Timing thresholds are behavioral checks; this is not a performance benchmark.


### ConcurrentServiceTests.test_http_open_loop_accounts_for_overload_and_queue_time.slow_call

    ConcurrentServiceTests.test_http_open_loop_accounts_for_overload_and_queue_time.slow_call(worker, payloads)

[Function source](../../tests/test_engineering_vision.py#L206)

Simulate a slow model batch for the overload test.


**Parameters**

    worker (int): Service worker index, unused by the fixture.
    payloads (Sequence[bytes]): Synthetic bodies; one result is made per body.


**Returns**

    tuple[list[dict], dict]: finite=True results and a reported model_ms of 80.0.


**Side effects and ownership**

    Sleeps approximately 0.08 seconds. The reported duration is a fixed fixture
    value, not a timer measurement or evidence of GPU/model performance.


### PredictionTests

    PredictionTests

[Class source](../../tests/test_engineering_vision.py#L245)

Run optional real PyTorch snapshot, shape and CPU/CUDA consistency checks.

The entire class is skipped if torch cannot be located. Tests use untrained
models and synthetic fixtures; output consistency does not establish detection
quality. The CUDA-specific case has its own device-availability skip.


### PredictionTests.setUpClass

    PredictionTests.setUpClass(cls)

[Function source](../../tests/test_engineering_vision.py#L253)

Create shared untrained model/fixtures in the optional PyTorch interpreter.


**Parameters**

    cls (type[PredictionTests]): unittest class receiving torch/model/images.


**Returns**

    None.


**Raises**

    VisionDependencyUnavailable: PyTorch import fails despite discovery.
    PyTorch or fixture setup errors propagate as class setup failures.


**Side effects and ownership**

    Sets global torch intra-op threads to one, disables cuDNN, constructs a
    seeded tiny CNN and four synthetic images. These settings are not restored.


### PredictionTests.test_realistic_probe_shape_without_quality_claim

    PredictionTests.test_realistic_probe_shape_without_quality_claim(self)

[Function source](../../tests/test_engineering_vision.py#L276)

Check the larger random probe's parameter count and finite output shape.


**Returns**

    None.


**Raises**

    AssertionError: Parameter count is not 11689512, output is not (1,1000), or
    any output is non-finite.


**Side effects and ownership**

    Allocates a random ResNet-shaped CPU model and one zero RGB input of shape
    (1,3,224,224), then executes inference_mode. No weights are downloaded and
    no expected class label or model-quality threshold is asserted.


### PredictionTests.test_cache_hit_and_new_weight_snapshot

    PredictionTests.test_cache_hit_and_new_weight_snapshot(self)

[Function source](../../tests/test_engineering_vision.py#L298)

Check repeated-image reuse and cache isolation after an actual weight change.


**Returns**

    None.


**Raises**

    AssertionError: Cold/warm cache flags, compute count or changed fingerprint
    contradict the expected snapshot behavior.


**Side effects and ownership**

    Creates predictors sharing one cache, performs synthetic inference and
    perturbs a separate model copy by 0.001 in one parameter. The class's original
    model is not mutated. A new weight snapshot must not reuse an old prediction.


### PredictionTests.test_cpu_cuda_consistency_for_same_weights

    PredictionTests.test_cpu_cuda_consistency_for_same_weights(self)

[Function source](../../tests/test_engineering_vision.py#L328)

Compare synthetic predictions of equal model snapshots on CPU and CUDA.


**Returns**

    None.


**Raises**

    unittest.SkipTest: This torch interpreter has no available CUDA device.
    AssertionError: Weight fingerprints/classes differ or any corresponding
    probability differs by more than 0.00001.
    PyTorch allocation/device errors propagate as test failures.


**Side effects and ownership**

    Sets this process's CUDA tensor-allocator fraction to a 256 MiB budget,
    allocates CPU/GPU model copies and executes four synthetic images. The cap
    excludes CUDA context/driver memory and other processes; it is not restored.
    This proves fixture consistency within tolerance, not GPU production quality.


## Reference verification

This documentation covers every explicit function, method, nested helper, class and module in the three listed files. Python-generated dataclass/base-class methods are described by their owning class rather than inserted as executable definitions. Definitions were parsed/compiled, and executable ASTs compared equal to the source checkout after removing docstrings. No measurements or new model executions are implied by that documentation check.
