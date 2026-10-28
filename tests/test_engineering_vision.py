"""Verify synthetic vision input, cache, concurrency and optional device behavior.

Base tests validate boundaries/cache identities and deliberately overload owned
localhost services with scripted callbacks. PyTorch tests are skipped when torch
is absent; CUDA consistency is skipped when no CUDA device is accessible.
Tests use fictional fixtures only, but concurrency cases create local sockets
and threads, and optional model cases allocate tensors/change torch settings.
A passing suite does not validate image-model quality or production capacity.
"""
import copy
import importlib.util
import threading
import time
import unittest

from lab.engineering_vision import (
    PredictionCache, VisionDependencyUnavailable, VisionPredictor, build_tiny_cnn,
    generate_synthetic_examples, make_cache_key, optional_runtime_status,
    require_torch, validate_image, weights_fingerprint,
    BoundedBatchVisionService, ProbeQueueFull, build_resnet18_probe,
    create_batched_http_server, run_open_loop_http,
)


class InputAndCacheTests(unittest.TestCase):
    """Check pure input validation and cache identity without requiring PyTorch.

    unittest creates a fresh instance and invokes setUp for each case. Tests require
    no model download, HTTP service, GPU, real user identity or external dataset.
    """
    def setUp(self):
        """Prepare one seeded grayscale image for each pure boundary/cache test.

        Returns:
            None.

        Side effects:
            Stores a SyntheticImage pixel tuple on self.image using two fixtures seeded
            with 123. Fixture-generation/validation errors propagate as setup failures.
        """
        self.image = generate_synthetic_examples(2, 123)[0].pixels

    def test_rejects_wrong_shape_and_non_integer_pixels(self):
        """Assert rejection of malformed shapes and invalid grayscale pixel types/ranges.

        Returns:
            None.

        Raises:
            AssertionError: A 31-row/31-column image, boolean pixel, value 256 or float
            pixel fails to raise ValueError.

        Side effects:
            Runs separate subtests over deliberately invalid in-memory arrays.
            It verifies representative invalid boundaries, not every possible input type.
        """
        for pixels in ([[0] * 32] * 31, [[0] * 31] * 32,
                       [[False] * 32] * 32, [[256] * 32] * 32,
                       [[0.1] * 32] * 32):
            with self.subTest(pixels=str(pixels)[:30]), self.assertRaises(ValueError):
                validate_image(pixels)

    def test_cache_invalidates_on_weights_preprocessing_and_input(self):
        """Assert that changing any cache-identity component yields a miss.

        Returns:
            None.

        Raises:
            AssertionError: The original key does not retrieve its stored value, or a
            changed weight label, preprocessing label or pixel still hits that entry.

        Side effects:
            Creates a two-entry cache, inserts a synthetic result and reads it under
            several keys. Mutates a copied image only. This checks key isolation, not
            model execution, authorization or a new preprocessing implementation.
        """
        cache = PredictionCache(max_entries=2)
        key = make_cache_key(self.image, "weights-v1", "prep-v1")
        cache.put(key, {"result": 1})
        self.assertEqual(cache.get(key), {"result": 1})
        modified = [list(row) for row in self.image]
        modified[0][0] += 1
        for changed_key in (make_cache_key(self.image, "weights-v2", "prep-v1"),
                            make_cache_key(self.image, "weights-v1", "prep-v2"),
                            make_cache_key(modified, "weights-v1", "prep-v1")):
            self.assertIsNone(cache.get(changed_key))

    def test_held_out_images_are_disjoint_from_training(self):
        """Check pixel-hash disjointness for the two selected synthetic fixture seeds.

        Returns:
            None.

        Raises:
            AssertionError: Any of 32 training hashes equals a held-out hash.

        Side effects:
            Generates two in-memory fixture sets seeded with 47 and 10047.
            The result applies to these selected sets; it is not a universal guarantee
            of statistical independence or absence of real-data leakage.
        """
        train = {make_cache_key(x.pixels, "w", "p")[2] for x in generate_synthetic_examples(32, 47, "train")}
        held_out = {make_cache_key(x.pixels, "w", "p")[2] for x in generate_synthetic_examples(32, 10047, "test")}
        self.assertFalse(train.intersection(held_out))

    def test_missing_optional_runtime_has_a_clear_error(self):
        """Verify the explanatory exception when this interpreter lacks PyTorch.

        Returns:
            None.

        Raises:
            unittest.SkipTest: PyTorch can already be located in this interpreter.
            AssertionError: require_torch does not raise VisionDependencyUnavailable.

        Side effects:
            Performs import discovery and, on the missing-runtime path, an attempted
            import. It neither installs a package nor checks CUDA capability.
        """
        if optional_runtime_status()["torch_installed"]:
            self.skipTest("This interpreter already has the optional runtime")
        with self.assertRaises(VisionDependencyUnavailable):
            require_torch()


class ConcurrentServiceTests(unittest.TestCase):
    """Exercise bounded queue overload and localhost accounting with fake models.

    The callbacks simulate service time; they do not execute PyTorch or prove model
    latency. Each test owns and closes its worker threads and any HTTP socket.
    """
    def test_bounded_queue_rejects_and_closes_workers(self):
        """Verify nonblocking overload rejection, payload validation and worker cleanup.

        Returns:
            None.

        Raises:
            AssertionError: Expected future values, queue metrics, rejection, input
            failure or final stopped-worker condition differ from the fixture contract.

        Side effects:
            Starts one batching worker with a one-slot queue. A gated fake callback holds
            the first request while a second waits; the third must raise ProbeQueueFull.
            Finally releases the event and closes the service even after an assertion.
        """
        entered, release = threading.Event(), threading.Event()

        def blocked_call(worker, payloads):
            """Hold a scripted model batch until released or its three-second wait expires.

            Args:
                worker (int): Worker index supplied by the service; unused by this fixture.
                payloads (Sequence[bytes]): Synthetic images whose count determines results.

            Returns:
                tuple[list[dict], dict]: One value=1 result per payload and model_ms=0.0.

            Side effects:
                Sets the captured entered event and waits up to three seconds on release.
                The zero timing is deliberate test metadata, not measured inference time.
                Unexpected callback errors propagate into the service's future handling.
            """
            entered.set()
            release.wait(3)
            return [{"value": 1} for _ in payloads], {"model_ms": 0.0}

        service = BoundedBatchVisionService(blocked_call, image_size=32, max_batch=1,
                                            max_wait_ms=0, queue_size=1)
        payload = bytes(32 * 32 * 3)
        try:
            first = service.submit(payload)
            self.assertTrue(entered.wait(1))
            second = service.submit(payload)
            with self.assertRaises(ProbeQueueFull):
                service.submit(payload)
            with self.assertRaises(ValueError):
                service.submit(b"incorrect length")
            release.set()
            self.assertEqual(first.result(2)["value"], 1)
            self.assertEqual(second.result(2)["value"], 1)
            self.assertEqual(service.metrics()["queue_rejections"], 1)
            self.assertLessEqual(service.metrics()["max_queue_depth"], 1)
        finally:
            release.set()
            service.close()
        self.assertFalse(any(t.is_alive() for t in service.worker_threads))

    def test_http_open_loop_accounts_for_overload_and_queue_time(self):
        """Verify that scheduled arrivals retain rejections and queue latency under overload.

        Returns:
            None.

        Raises:
            AssertionError: The run loses any of 40 offered requests, reports no HTTP 429,
            serves none, hides expected waiting, or leaves its serving thread alive.

        Side effects:
            Starts a fake one-worker, two-slot service and an ephemeral loopback server.
            Offers 40 requests/second for one second against an 80-millisecond callback,
            checks response/queue timing, and finally shuts down sockets and all workers.
            Timing thresholds are behavioral checks; this is not a performance benchmark.
        """
        def slow_call(worker, payloads):
            """Simulate a slow model batch for the overload test.

            Args:
                worker (int): Service worker index, unused by the fixture.
                payloads (Sequence[bytes]): Synthetic bodies; one result is made per body.

            Returns:
                tuple[list[dict], dict]: finite=True results and a reported model_ms of 80.0.

            Side effects:
                Sleeps approximately 0.08 seconds. The reported duration is a fixed fixture
                value, not a timer measurement or evidence of GPU/model performance.
            """
            time.sleep(.08)
            return [{"finite": True} for _ in payloads], {"model_ms": 80.0}

        service = BoundedBatchVisionService(slow_call, image_size=32, max_batch=1,
                                            max_wait_ms=0, queue_size=2)
        server = create_batched_http_server(service)
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        try:
            result = run_open_loop_http(server, [bytes(32 * 32 * 3)], 40, 1)
            self.assertEqual(result["offered_requests"], 40)
            self.assertTrue(result["accounting_complete"])
            self.assertGreater(result["http_status_counts"].get("429", 0), 0)
            self.assertGreater(result["successful_requests"], 0)
            self.assertGreater(result["queue_wait_latency"]["p95_ms"], 20)
            self.assertGreater(result["http_end_to_end_latency"]["p95_ms"], 80)
        finally:
            server.shutdown()
            server.server_close()
            serving.join(3)
            service.close()
        self.assertFalse(serving.is_alive())


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch is not installed")
class PredictionTests(unittest.TestCase):
    """Run optional real PyTorch snapshot, shape and CPU/CUDA consistency checks.

    The entire class is skipped if torch cannot be located. Tests use untrained
    models and synthetic fixtures; output consistency does not establish detection
    quality. The CUDA-specific case has its own device-availability skip.
    """
    @classmethod
    def setUpClass(cls):
        """Create shared untrained model/fixtures in the optional PyTorch interpreter.

        Args:
            cls (type[PredictionTests]): unittest class receiving torch/model/images.

        Returns:
            None.

        Raises:
            VisionDependencyUnavailable: PyTorch import fails despite discovery.
            PyTorch or fixture setup errors propagate as class setup failures.

        Side effects:
            Sets global torch intra-op threads to one, disables cuDNN, constructs a
            seeded tiny CNN and four synthetic images. These settings are not restored.
        """
        cls.torch = require_torch()
        cls.torch.set_num_threads(1)
        cls.torch.backends.cudnn.enabled = False
        cls.model = build_tiny_cnn(47)
        cls.images = generate_synthetic_examples(4, 103)[0:4]

    def test_realistic_probe_shape_without_quality_claim(self):
        """Check the larger random probe's parameter count and finite output shape.

        Returns:
            None.

        Raises:
            AssertionError: Parameter count is not 11689512, output is not (1,1000), or
            any output is non-finite.

        Side effects:
            Allocates a random ResNet-shaped CPU model and one zero RGB input of shape
            (1,3,224,224), then executes inference_mode. No weights are downloaded and
            no expected class label or model-quality threshold is asserted.
        """
        model = build_resnet18_probe(47)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11689512)
        with self.torch.inference_mode():
            result = model(self.torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(result.shape), (1, 1000))
        self.assertTrue(bool(self.torch.isfinite(result).all()))

    def test_cache_hit_and_new_weight_snapshot(self):
        """Check repeated-image reuse and cache isolation after an actual weight change.

        Returns:
            None.

        Raises:
            AssertionError: Cold/warm cache flags, compute count or changed fingerprint
            contradict the expected snapshot behavior.

        Side effects:
            Creates predictors sharing one cache, performs synthetic inference and
            perturbs a separate model copy by 0.001 in one parameter. The class's original
            model is not mutated. A new weight snapshot must not reuse an old prediction.
        """
        cache = PredictionCache()
        first = VisionPredictor(self.model, cache=cache)
        cold = first.predict(self.images[0].pixels)
        calls = first.compute_calls
        warm = first.predict(self.images[0].pixels)
        self.assertFalse(cold["cache_hit"])
        self.assertTrue(warm["cache_hit"])
        self.assertEqual(first.compute_calls, calls)
        changed = copy.deepcopy(self.model)
        with self.torch.no_grad():
            next(changed.parameters()).reshape(-1)[0] += 0.001
        second = VisionPredictor(changed, cache=cache)
        self.assertNotEqual(first.weights_version, second.weights_version)
        self.assertFalse(second.predict(self.images[0].pixels)["cache_hit"])

    def test_cpu_cuda_consistency_for_same_weights(self):
        """Compare synthetic predictions of equal model snapshots on CPU and CUDA.

        Returns:
            None.

        Raises:
            unittest.SkipTest: This torch interpreter has no available CUDA device.
            AssertionError: Weight fingerprints/classes differ or any corresponding
            probability differs by more than 0.00001.
            PyTorch allocation/device errors propagate as test failures.

        Side effects:
            Sets this process's CUDA tensor-allocator fraction to a 256 MiB budget,
            allocates CPU/GPU model copies and executes four synthetic images. The cap
            excludes CUDA context/driver memory and other processes; it is not restored.
            This proves fixture consistency within tolerance, not GPU production quality.
        """
        if not self.torch.cuda.is_available():
            self.skipTest("No accessible CUDA device in this interpreter")
        total = self.torch.cuda.get_device_properties(0).total_memory
        self.torch.cuda.set_per_process_memory_fraction(256 * 1024**2 / total, 0)
        cpu = VisionPredictor(self.model, device="cpu")
        gpu = VisionPredictor(self.model, device="cuda:0")
        self.assertEqual(cpu.weights_version, gpu.weights_version)
        cpu_results = cpu.predict_batch([x.pixels for x in self.images])
        gpu_results = gpu.predict_batch([x.pixels for x in self.images])
        for left, right in zip(cpu_results, gpu_results):
            self.assertEqual(left["predicted_class"], right["predicted_class"])
            for a, b in zip(left["probabilities"], right["probabilities"]):
                self.assertAlmostEqual(a, b, delta=0.00001)


if __name__ == "__main__":
    unittest.main()
