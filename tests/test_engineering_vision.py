"""Verify synthetic vision input, cache, concurrency and optional device behavior.

Base tests validate boundaries/cache identities and deliberately overload owned
localhost services with scripted callbacks. Ramp tests check load accounting,
stop/capacity/sizing rules, the result cache and spawned client processes.
PyTorch tests are skipped when torch is absent; CUDA consistency is skipped when
no CUDA device is accessible. Packaging tests train only tiny CPU runs.
Tests use fictional fixtures only, but concurrency cases create local sockets
and threads, and optional model cases allocate tensors/change torch settings.
A passing suite does not validate image-model quality or production capacity.
"""
import copy
import http.client
import importlib.util
import json
import threading
import time
import unittest

from lab.engineering_vision import (
    PredictionCache, VisionDependencyUnavailable, VisionPredictor, build_tiny_cnn,
    generate_synthetic_examples, make_cache_key, optional_runtime_status,
    require_torch, validate_image, weights_fingerprint,
    BoundedBatchVisionService, ProbeQueueFull, build_resnet18_probe,
    create_batched_http_server, run_open_loop_http,
    PACKAGING_CLASS_NAMES, classification_metrics, cpu_usage_between,
    generate_packaging_images, make_resnet_batch_runtime, open_loop_request_plan,
    packaging_images_to_uint8, packaging_split_report, parse_nvidia_smi_sample,
    predict_packaging_logits, ramp_should_stop, read_cpu_snapshot, replicas_for_target,
    run_open_loop_ramp_step, select_sustainable_capacity, summarize_gpu_samples,
    summarize_open_loop_step, train_packaging_classifier,
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


def _record(index, due, sent, ended, status, image_index=0, predicted=0, kind="cycle", cache_hit=False):
    """Build one synthetic client record for summary tests.

    Args:
        index (int): Scheduled request index.
        due (float): Scheduled arrival offset in seconds.
        sent (float | None): Send offset in seconds, or None for not_sent.
        ended (float | None): Completion offset in seconds.
        status (int | str): HTTP status, client_error or not_sent.
        image_index (int): Payload-file image index.
        predicted (int | None): Predicted class returned by the service.
        kind (str): cycle, repeat or unique.
        cache_hit (bool): Whether the service reported a cache hit.

    Returns:
        dict: A record shaped like run_open_loop_ramp_step output, with a fixed
        2 ms queue wait and 6 ms service time for HTTP 200 records.
    """
    ok = status == 200
    return {"index": index, "image_index": image_index, "kind": kind, "due": due, "sent": sent,
            "ended": ended, "status": status, "predicted_class": predicted if ok else None,
            "queue_wait_ms": 2.0 if ok else None, "service_ms": 6.0 if ok else None,
            "batch_size": 1 if ok else None, "cache_hit": cache_hit if ok else None,
            "error_type": None if ok or status == 429 else (
                "client_in_flight_cap" if status == "not_sent" else "TimeoutError")}


class RampAnalysisTests(unittest.TestCase):
    """Check pure load-ramp accounting, stop, capacity and sizing rules.

    These tests need neither PyTorch nor sockets. They use hand-built records,
    so they verify arithmetic and classification rules, not service speed.
    """
    def test_summary_keeps_every_failure_visible(self):
        """Verify outcome classes, failure share, within-SLO share and p99 flag.

        Returns:
            None.

        Raises:
            AssertionError: A count, share or flag differs from the hand calculation.

        Side effects:
            None. Builds 200 measured records (100 req/s for 2 s after 1 s warm-up):
            185 fast answers, 3 answers at 150 ms, 2 answers after the 1 s
            deadline, 5 rejections, 3 client errors and 2 not_sent requests, plus
            one warm-up record that must be excluded.
        """
        records = [_record(-1, 0.5, 0.5, 0.51, 200)]
        for i in range(200):
            due = 1.0 + i / 100
            if i < 185:
                records.append(_record(i, due, due, due + 0.010, 200, image_index=i % 4, predicted=i % 4))
            elif i < 188:
                records.append(_record(i, due, due, due + 0.150, 200, image_index=1, predicted=2))
            elif i < 190:
                records.append(_record(i, due, due, due + 1.200, 200, image_index=1, predicted=1))
            elif i < 195:
                records.append(_record(i, due, due, due + 0.002, 429))
            elif i < 198:
                records.append(_record(i, due, due, due + 5.0, "client_error"))
            else:
                records.append(_record(i, due, None, None, "not_sent"))
        summary = summarize_open_loop_step(records, target_rps=100, warmup_seconds=1, measure_seconds=2,
                                           labels=[0, 1, 2, 3], offline_predictions=[0, 1, 2, 3])
        self.assertEqual(summary["offered_requests"], 200)
        self.assertEqual(summary["outcomes"], {"ok": 188, "deadline_miss": 2, "rejected": 5,
                                               "error": 3, "not_sent": 2})
        self.assertAlmostEqual(summary["failure_share"], 12 / 200)
        self.assertAlmostEqual(summary["within_slo_share_of_offered"], 185 / 200)
        self.assertEqual(summary["p99_observations"], 190)
        self.assertFalse(summary["p99_reliable"])
        self.assertGreater(summary["latency_from_scheduled_arrival_ms"]["max_ms"], 1000)
        self.assertFalse(summary["generator_limited"])
        self.assertFalse(summary["slo_met"])
        self.assertEqual(summary["quality_under_load"]["predictions"], 190)
        self.assertEqual(summary["quality_under_load"]["disagreements"], 3)

    def test_generator_limit_is_flagged_with_reason(self):
        """Verify that a send rate under 95 % marks the point unusable for capacity.

        Returns:
            None.

        Raises:
            AssertionError: The generator flag, reason or SLO verdict is wrong.

        Side effects:
            None. Builds 100 offered records of which only 90 were sent, all fast.
        """
        records = []
        for i in range(100):
            due = i / 100
            sent = due if i < 90 else None
            records.append(_record(i, due, sent, None if sent is None else due + 0.005,
                                   200 if sent is not None else "not_sent"))
        summary = summarize_open_loop_step(records, target_rps=100, warmup_seconds=0, measure_seconds=1)
        self.assertTrue(summary["generator_limited"])
        self.assertEqual(summary["generator_limited_reason"], "client_in_flight_cap_reached")
        self.assertFalse(summary["slo_met"])

    def test_stop_rule_and_capacity_selection(self):
        """Verify the 20 % stop rule and the ascending all-runs-pass capacity rule.

        Returns:
            None.

        Raises:
            AssertionError: Stop decisions or the selected/first-failing rate differ.

        Side effects:
            None. A repeat that fails at 200 req/s must lower capacity to 100 req/s
            even though the ladder run at 200 req/s passed.
        """
        def step(rate, met, share=0.0, limited=False):
            """Build a minimal step summary for the capacity rule.

            Args:
                rate (int): Offered rate.
                met (bool): Whether the run met the SLO.
                share (float): Failure share.
                limited (bool): Whether the run was generator-limited.

            Returns:
                dict: Fields read by select_sustainable_capacity and ramp_should_stop.
            """
            return {"target_rps": rate, "slo_met": met, "failure_share": share, "generator_limited": limited,
                    "successful_throughput_rps": rate * (1 - share), "p99_reliable": True,
                    "latency_from_scheduled_arrival_ms": {"p50_ms": 10, "p95_ms": 20, "p99_ms": 30 + rate / 10}}

        self.assertEqual(ramp_should_stop(step(400, False, 0.25)), (True, "failure_share_above_0.2"))
        self.assertEqual(ramp_should_stop(step(400, False, 0.05)), (False, None))
        self.assertEqual(ramp_should_stop(step(400, True, 0.0, True)), (True, "generator_limited"))
        steps = [step(50, True), step(100, True), step(200, True), step(400, False, 0.3),
                 step(200, False, 0.02), step(100, True), step(800, True)]
        capacity = select_sustainable_capacity(steps)
        self.assertEqual(capacity["max_sustainable_offered_rps"], 100)
        self.assertEqual(capacity["first_failing_offered_rps"], 200)
        self.assertEqual(capacity["runs_at_max"], 2)
        self.assertIsNone(select_sustainable_capacity([step(10, False, 0.5)])["max_sustainable_offered_rps"])

    def test_replica_formula(self):
        """Verify the sizing formula, N+1 and zone-loss variants on fixed numbers.

        Returns:
            None.

        Raises:
            AssertionError: A computed replica count differs from the hand result.

        Side effects:
            None. 100,000 / (800 * 0.7) = 178.57, so 179 base replicas.
        """
        result = replicas_for_target(100000, 800, headroom=0.3, zones=3)
        self.assertEqual(result["base_replicas"], 179)
        self.assertEqual(result["n_plus_1_replicas"], 180)
        self.assertEqual(result["replicas_per_zone_surviving_one_zone_loss"], 90)
        self.assertEqual(result["zone_redundant_total_replicas"], 270)
        self.assertFalse(result["validated_measurement"])
        with self.assertRaises(ValueError):
            replicas_for_target(100000, 800, headroom=1.0)

    def test_request_plan_repeat_share_and_unique_markers(self):
        """Verify cycling, the repeat share and unique markers of the request plan.

        Returns:
            None.

        Raises:
            AssertionError: Cycling order, the observed repeat share over 10,000
            requests (expected 0.45..0.55) or marker uniqueness is wrong.

        Side effects:
            None.
        """
        self.assertEqual([open_loop_request_plan(i, 3)[0] for i in range(5)], [0, 1, 2, 0, 1])
        plans = [open_loop_request_plan(i, 1000, 0.5, list(range(32))) for i in range(10000)]
        repeats = [p for p in plans if p[1] == "repeat"]
        uniques = [p for p in plans if p[1] == "unique"]
        self.assertTrue(0.45 <= len(repeats) / len(plans) <= 0.55)
        self.assertTrue(all(p[0] < 32 and p[2] is None for p in repeats))
        self.assertEqual(len({p[2] for p in uniques}), len(uniques))

    def test_gpu_and_cpu_sample_parsing(self):
        """Verify nvidia-smi line parsing, window summaries and CPU arithmetic.

        Returns:
            None.

        Raises:
            AssertionError: Parsed values, window filtering or CPU shares differ.

        Side effects:
            Uses literal snapshots for the arithmetic, then takes one real
            read_cpu_snapshot to check that it returns without error. No subprocess.
        """
        self.assertEqual(parse_nvidia_smi_sample("17, 4715"), {"utilization_percent": 17.0, "memory_used_mib": 4715.0})
        self.assertIsNone(parse_nvidia_smi_sample("[N/A], 4715"))
        samples = [{"t": t, "utilization_percent": u, "memory_used_mib": m}
                   for t, u, m in ((0.5, 90, 6000), (1.0, 40, 5000), (2.0, 60, 5200), (3.5, 99, 7000))]
        summary = summarize_gpu_samples(samples, 1.0, 3.0)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["utilization_percent_mean"], 50)
        self.assertEqual(summary["memory_used_mib_max"], 5200)
        usage = cpu_usage_between({"t": 0.0, "busy_jiffies": 100, "total_jiffies": 1000, "process_cpu_seconds": 1.0,
                                   "all_processes_cpu_seconds": 10.0},
                                  {"t": 2.0, "busy_jiffies": 400, "total_jiffies": 2000, "process_cpu_seconds": 4.0,
                                   "all_processes_cpu_seconds": 17.0})
        self.assertAlmostEqual(usage["system_busy_share_tick_sampled"], 0.3)
        self.assertAlmostEqual(usage["server_process_cores"], 1.5)
        self.assertAlmostEqual(usage["all_linux_processes_cores"], 3.5)
        snapshot = read_cpu_snapshot()
        self.assertIn("process_cpu_seconds", snapshot)


class RampServiceTests(unittest.TestCase):
    """Exercise the cache path and the multi-process client against fake models.

    Each test owns and closes its loopback server and batching workers. Fake
    callbacks simulate model time and prove accounting, not model latency.
    """
    def test_cache_hit_skips_model_execution(self):
        """Verify that an identical body is answered from the cache without the model.

        Returns:
            None.

        Raises:
            AssertionError: Cache flags, callback count or a missing-scope error differ.

        Side effects:
            Starts a one-worker fake service and loopback server, posts the same body
            twice and a different body once, then closes everything.
        """
        calls = []

        def counting_call(worker, payloads):
            """Count model calls and return one fixed prediction per payload.

            Args:
                worker (int): Worker index, unused.
                payloads (Sequence[bytes]): Bodies in the batch.

            Returns:
                tuple[list[dict], dict]: predicted_class=1 results and model_ms=0.0.

            Side effects:
                Appends the batch size to the captured calls list.
            """
            calls.append(len(payloads))
            return [{"predicted_class": 1} for _ in payloads], {"model_ms": 0.0}

        with self.assertRaises(ValueError):
            create_batched_http_server(None, cache=PredictionCache(4))
        service = BoundedBatchVisionService(counting_call, image_size=32, max_batch=1, max_wait_ms=0)
        server = create_batched_http_server(service, cache=PredictionCache(8), cache_scope=("w", "p"))
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()

        def post(body):
            """POST one body and return the decoded JSON response.

            Args:
                body (bytes): Exact 32 x 32 RGB body.

            Returns:
                dict: Parsed response JSON.

            Side effects:
                Opens and closes one loopback HTTP connection.
            """
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            try:
                connection.request("POST", "/predict-rgb", body, {"Content-Type": "application/octet-stream"})
                return json.loads(connection.getresponse().read())
            finally:
                connection.close()

        try:
            body = bytes(32 * 32 * 3)
            first, second = post(body), post(body)
            third = post(b"\x01" + body[1:])
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(second["batch_size"], 0)
            self.assertFalse(third["cache_hit"])
            self.assertEqual(sum(calls), 2)
        finally:
            server.shutdown()
            server.server_close()
            serving.join(3)
            service.close()

    def test_multiprocess_open_loop_step_accounts_every_request(self):
        """Verify that spawned clients return one record per scheduled request.

        Returns:
            None.

        Raises:
            AssertionError: Record count, measured cohort, outcomes or achieved rate
            are inconsistent with a light 40 req/s load on a 2 ms fake model.

        Side effects:
            Writes a temporary payload file, starts a fake service and loopback
            server, spawns two client processes for 1.5 s of scheduled traffic,
            then removes the file and closes all resources.
        """
        import os
        import tempfile

        def quick_call(worker, payloads):
            """Sleep 2 ms and return predicted_class 0 for each payload.

            Args:
                worker (int): Worker index, unused.
                payloads (Sequence[bytes]): Bodies in the batch.

            Returns:
                tuple[list[dict], dict]: One result per payload and model_ms=2.0.
            """
            time.sleep(.002)
            return [{"predicted_class": 0} for _ in payloads], {"model_ms": 2.0}

        service = BoundedBatchVisionService(quick_call, image_size=32, max_batch=4, max_wait_ms=1)
        server = create_batched_http_server(service)
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        handle, payload_path = tempfile.mkstemp(suffix=".u8")
        os.close(handle)
        try:
            with open(payload_path, "wb") as file:
                file.write(bytes(32 * 32 * 3) * 2)
            raw = run_open_loop_ramp_step(server.server_address[1], payload_path, image_bytes=32 * 32 * 3,
                                          image_count=2, target_rps=40, warmup_seconds=0.5, measure_seconds=1,
                                          process_count=2, start_delay_seconds=1.5)
            self.assertEqual(len(raw["records"]), raw["scheduled_requests"])
            self.assertEqual(raw["client_errors"], [])
            summary = summarize_open_loop_step(raw["records"], target_rps=40, warmup_seconds=0.5,
                                               measure_seconds=1, labels=[0, 0], offline_predictions=[0, 0])
            self.assertEqual(summary["offered_requests"], 40)
            self.assertEqual(summary["outcomes"]["ok"], 40)
            self.assertFalse(summary["generator_limited"])
            self.assertEqual(summary["quality_under_load"]["accuracy_under_load"], 1.0)
        finally:
            os.remove(payload_path)
            server.shutdown()
            server.server_close()
            serving.join(3)
            service.close()


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch is not installed")
class PackagingModelTests(unittest.TestCase):
    """Check the procedural dataset, metrics, training loop and trained runtime.

    The class is skipped without torch. Tests run on CPU at small image sizes;
    they check contracts and determinism, not model quality or GPU speed.
    """
    def test_generation_is_deterministic_balanced_and_disjoint(self):
        """Verify byte determinism, class balance and the split overlap report.

        Returns:
            None.

        Raises:
            AssertionError: Same-seed hashes differ, classes are unbalanced, disjoint
            seeds overlap, or an injected duplicate is not counted.

        Side effects:
            Generates small 64 x 64 CPU fixtures.
        """
        first = generate_packaging_images(8, 5, "a", 64)
        again = generate_packaging_images(8, 5, "a", 64)
        other = generate_packaging_images(8, 6, "b", 64)
        self.assertEqual([x.sha256 for x in first], [x.sha256 for x in again])
        self.assertEqual(len(first[0].payload), 64 * 64 * 3)
        report = packaging_split_report({"train": first, "test": other})
        self.assertEqual(report["cross_split_overlap_total"], 0)
        self.assertEqual(report["splits"]["train"]["class_counts"],
                         {"good": 2, "missing_cap": 2, "scratch": 2, "misaligned_label": 2})
        leaked = packaging_split_report({"train": first, "test": other + [first[0], first[0]]})
        self.assertEqual(leaked["pairwise_overlap"]["train|test"], 1)
        self.assertEqual(leaked["splits"]["test"]["duplicates_within_split"], 1)
        with self.assertRaises(ValueError):
            generate_packaging_images(8, 5, "a", 16)

    def test_classification_metrics_by_hand(self):
        """Verify accuracy, per-class precision/recall and confusion orientation.

        Returns:
            None.

        Raises:
            AssertionError: A metric differs from the hand calculation.

        Side effects:
            None.
        """
        metrics = classification_metrics([0, 0, 1, 1, 2, 3], [0, 1, 1, 1, 2, 2])
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertEqual(metrics["confusion_matrix"][0], [1, 1, 0, 0])
        self.assertAlmostEqual(metrics["per_class"]["missing_cap"]["precision"], 2 / 3)
        self.assertEqual(metrics["per_class"]["missing_cap"]["recall"], 1.0)
        self.assertEqual(metrics["per_class"]["misaligned_label"]["recall"], 0.0)
        self.assertIsNone(metrics["per_class"]["misaligned_label"]["precision"])
        with self.assertRaises(ValueError):
            classification_metrics([0], [0, 1])

    def test_training_selects_an_epoch_and_runtime_returns_labels(self):
        """Run a two-epoch CPU training and serve the result through the runtime.

        Returns:
            None.

        Raises:
            AssertionError: History length, selected epoch, logits shape or the
            labelled runtime result contract differ.

        Side effects:
            Sets torch to one thread, trains a ResNet-18 shape for two epochs on 16
            images of 32 x 32 pixels on CPU and runs one labelled batch call.
        """
        torch = require_torch()
        torch.set_num_threads(1)
        train = generate_packaging_images(16, 11, "train", 32)
        validation = generate_packaging_images(8, 12, "validation", 32)
        model, history = train_packaging_classifier(train, validation, image_size=32, epochs=2, batch_size=8)
        self.assertEqual(len(history["history"]), 2)
        self.assertIn(history["selected_epoch"], (1, 2))
        logits = predict_packaging_logits(model, packaging_images_to_uint8(validation, 32), batch_size=4)
        self.assertEqual(tuple(logits.shape), (8, 4))
        single = predict_packaging_logits(model, packaging_images_to_uint8(validation[:1], 32), batch_size=1)
        batch_call, _, _ = make_resnet_batch_runtime(model, "cpu", "fp32", 1, 32, class_names=PACKAGING_CLASS_NAMES)
        results, timing = batch_call(0, [validation[0].payload])
        self.assertIn(results[0]["label"], PACKAGING_CLASS_NAMES)
        self.assertEqual(results[0]["predicted_class"], int(single[0].argmax()))
        self.assertFalse(results[0]["random_weights"])
        self.assertIn("model_ms", timing)


if __name__ == "__main__":
    unittest.main()
