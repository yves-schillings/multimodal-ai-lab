"""Meaningful cache, input-boundary and optional CPU/CUDA checks."""
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
    def setUp(self):
        self.image = generate_synthetic_examples(2, 123)[0].pixels

    def test_rejects_wrong_shape_and_non_integer_pixels(self):
        for pixels in ([[0] * 32] * 31, [[0] * 31] * 32,
                       [[False] * 32] * 32, [[256] * 32] * 32,
                       [[0.1] * 32] * 32):
            with self.subTest(pixels=str(pixels)[:30]), self.assertRaises(ValueError):
                validate_image(pixels)

    def test_cache_invalidates_on_weights_preprocessing_and_input(self):
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
        train = {make_cache_key(x.pixels, "w", "p")[2] for x in generate_synthetic_examples(32, 47, "train")}
        held_out = {make_cache_key(x.pixels, "w", "p")[2] for x in generate_synthetic_examples(32, 10047, "test")}
        self.assertFalse(train.intersection(held_out))

    def test_missing_optional_runtime_has_a_clear_error(self):
        if optional_runtime_status()["torch_installed"]:
            self.skipTest("This interpreter already has the optional runtime")
        with self.assertRaises(VisionDependencyUnavailable):
            require_torch()


class ConcurrentServiceTests(unittest.TestCase):
    def test_bounded_queue_rejects_and_closes_workers(self):
        entered, release = threading.Event(), threading.Event()

        def blocked_call(worker, payloads):
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
        def slow_call(worker, payloads):
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
    @classmethod
    def setUpClass(cls):
        cls.torch = require_torch()
        cls.torch.set_num_threads(1)
        cls.torch.backends.cudnn.enabled = False
        cls.model = build_tiny_cnn(47)
        cls.images = generate_synthetic_examples(4, 103)[0:4]

    def test_realistic_probe_shape_without_quality_claim(self):
        model = build_resnet18_probe(47)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11689512)
        with self.torch.inference_mode():
            result = model(self.torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(result.shape), (1, 1000))
        self.assertTrue(bool(self.torch.isfinite(result).all()))

    def test_cache_hit_and_new_weight_snapshot(self):
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
