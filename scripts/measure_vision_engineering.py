"""Train and benchmark a tiny synthetic CNN on CPU and, when available, CUDA.

No weights are downloaded. Evidence and checkpoints stay outside the checkout.
The optional server binds only to 127.0.0.1 and processes synthetic fixture data.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.engineering_vision import (
    CLASS_NAMES, PREPROCESS_VERSION, VisionDependencyUnavailable, VisionPredictor,
    create_local_http_server, demonstrate_cache, evaluate_model,
    generate_synthetic_examples, image_digest, measure_engine, measure_http,
    require_torch, train_tiny_model, weights_fingerprint,
    BoundedBatchVisionService, build_resnet18_probe, create_batched_http_server,
    fast_weights_fingerprint, make_resnet_batch_runtime, run_open_loop_http,
)


def run_load_probe(args, output):
    report = {"schema_version": "vision-open-loop-v1", "purpose": "performance_probe",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "production_approval": False, "release_eligible": False,
              "weights_downloaded": False, "packages_installed": False,
              "runtimes_not_exercised": ["YOLO", "TensorRT", "Triton", "OKD GPU"],
              "cases": [], "gpu_tensor_allocator_budget_mib": args.gpu_memory_mib,
              "limitations": ["Random weights and synthetic RGB inputs: no model-quality or object-detection claim.",
                              "Local host/WSL experiment, not a cluster or production capacity acceptance.",
                              "One short offered-load point per configuration does not locate maximum sustainable throughput.",
                              "HTTP service has no authentication and only binds to 127.0.0.1.",
                              "Allocator cap excludes CUDA context/driver memory and other processes.",
                              "Each request opens an HTTP connection. No external network or production image decoder is included."]}
    path = output / "report.json"

    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        torch = require_torch()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        cuda = torch.cuda.is_available()
        if args.require_cuda and not cuda:
            raise RuntimeError("CUDA required but unavailable in the selected interpreter")
        report["runtime"] = {"torch": torch.__version__, "compiled_cuda": torch.version.cuda,
                             "python": platform.python_version(), "platform": platform.system(),
                             "cpu_threads_per_call": 1, "interop_threads": 1,
                             "cudnn_enabled": True, "cudnn_benchmark": False,
                             "cuda_available": cuda, "cluster_execution_claimed": False}
        if cuda:
            props = torch.cuda.get_device_properties(0)
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_mib * 1024**2 / props.total_memory, 0)
            free, total = torch.cuda.mem_get_info()
            report["runtime"].update(gpu=props.name, compute_capability=[props.major, props.minor],
                                     total_device_memory_mib=total / 1024**2,
                                     free_device_memory_before_mib=free / 1024**2)
            if free < (args.gpu_memory_mib + 256) * 1024**2:
                raise RuntimeError("Insufficient currently free device memory for the bounded probe plus safety margin")
        model = build_resnet18_probe(args.seed)
        report["model"] = {"architecture": "ResNet-18-shaped: stem7x7, BasicBlock stages [2,2,2,2], channels [64,128,256,512], 1000 logits",
                           "parameters": sum(p.numel() for p in model.parameters()),
                           "random_weights": True, "trained": False, "quality_evaluated": False,
                           "quality_claimed": False, "seed": args.seed,
                           "base_fp32_weights_sha256": fast_weights_fingerprint(model),
                           "input_shape": [args.input_size, args.input_size, 3],
                           "preprocessing": "Validate exact RGB uint8 body; HWC to CHW; float /255; stack; transfer to requested device/dtype",
                           "postprocessing": "FP32 softmax, maximum class index and score; no semantic class names"}
        rng = random.Random(args.seed + 123)
        payloads = [rng.randbytes(args.input_size * args.input_size * 3) for _ in range(8)]
        report["inputs"] = {"synthetic_only": True, "count": len(payloads),
                            "payload_sha256": [hashlib.sha256(p).hexdigest() for p in payloads],
                            "bytes_per_image": len(payloads[0]), "cache_enabled": False}
        report["protocol"] = {"duration_seconds_per_case": args.duration, "offered_rps": args.target_rps,
                              "queue_capacity_requests": args.queue_size, "maximum_batch_size": args.small_batch,
                              "maximum_collection_wait_ms": args.batch_wait_ms, "warmup_calls_per_worker": args.warmup,
                              "warmup_batch_size": args.small_batch,
                              "one_worker": "One model snapshot and inference thread; one CUDA stream when CUDA selected",
                              "two_workers": "Two model snapshots and inference threads; two CUDA streams; shared bounded queue",
                              "clock": "perf_counter for wall time; CUDA events for model forward",
                              "concurrency": "ThreadingHTTPServer and open-loop scheduler, up to 128 client threads/256 pending requests"}
        matrix = [("cpu", "fp32", 1)]
        if cuda:
            matrix += [("cuda:0", precision, workers) for precision in ("fp32", "fp16") for workers in (1, 2)]
        report["matrix_requested"] = [{"device": d, "precision": p, "workers": w} for d, p, w in matrix]
        save()

        def run_case(device, precision, workers):
            case = {"device": device, "precision": precision, "model_instances": workers,
                    "workers": workers, "cache_enabled": False, "quality_evaluated": False}
            service, server, server_thread = None, None, None
            try:
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(0)
                batch_call, models, streams = make_resnet_batch_runtime(model, device, precision, workers, args.input_size)
                case["actual_weights_sha256"] = fast_weights_fingerprint(models[0])
                for worker in range(workers):
                    for _ in range(args.warmup):
                        batch_call(worker, payloads[:args.small_batch])
                # Diagnostic output is not a class-quality assessment.
                diagnostic, _ = batch_call(0, payloads[:1])
                case["finite_output_diagnostic"] = diagnostic[0]
                service = BoundedBatchVisionService(batch_call, image_size=args.input_size, workers=workers,
                                                    max_batch=args.small_batch, max_wait_ms=args.batch_wait_ms,
                                                    queue_size=args.queue_size)
                server = create_batched_http_server(service)
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                case["load"] = run_open_loop_http(server, payloads, args.target_rps, args.duration)
                case["batching"] = service.metrics()
                case["status"] = "completed" if case["load"]["accounting_complete"] else "failed"
                if device.startswith("cuda"):
                    case["gpu_memory"] = {"peak_tensor_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                           "peak_allocator_reserved_bytes": torch.cuda.max_memory_reserved(0),
                                           "scope": "PyTorch allocator only, excludes context/driver/other processes"}
            except Exception as exc:
                case.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1500])
            finally:
                if server is not None:
                    server.shutdown()
                    server.server_close()
                    server_thread.join(timeout=5)
                if service is not None:
                    service.close()
                case["owned_server_closed"] = server is None or (not server_thread.is_alive())
                case["owned_batch_workers_closed"] = service is None or not any(t.is_alive() for t in service.worker_threads)
            return case

        for device, precision, workers in matrix:
            case = run_case(device, precision, workers)
            report["cases"].append(case)
            save()
            print(json.dumps({"case_completed": {k: case[k] for k in ("device", "precision", "workers", "status")},
                              "http": case.get("load", {}).get("http_end_to_end_latency"),
                              "successful_requests": case.get("load", {}).get("successful_requests"),
                              "errors": case.get("load", {}).get("error_requests"),
                              "error": case.get("error")}), flush=True)
            gc.collect()
            if cuda:
                # Releases this process's unused cached blocks, never another application's memory.
                torch.cuda.empty_cache()
        report["status"] = "completed" if all(x["status"] == "completed" for x in report["cases"]) else "partial_failure"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        if cuda:
            free, _ = torch.cuda.mem_get_info()
            report["runtime"]["free_device_memory_after_mib"] = free / 1024**2
        save()
        print(json.dumps({"status": report["status"], "report": str(path), "release_eligible": False}), flush=True)
        return 0 if report["status"] == "completed" else 1
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1500],
                      completed_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps({"status": "failed", "error": str(exc), "report": str(path)}), flush=True)
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--small-batch", type=int, default=8)
    parser.add_argument("--gpu-memory-mib", type=int, default=256)
    parser.add_argument("--http", action="store_true", help="Also measure a temporary localhost HTTP service")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--serve", action="store_true", help="After measurement, keep a synthetic localhost server running")
    parser.add_argument("--port", type=int, default=0, help="Local demo port, 0 chooses a free port")
    parser.add_argument("--load-probe", action="store_true", help="Run random-weight ResNet-18 open-loop concurrent service probe instead of tiny trained model")
    parser.add_argument("--duration", type=float, default=10, help="Open-loop offered-load seconds per case")
    parser.add_argument("--target-rps", type=int, default=100)
    parser.add_argument("--queue-size", type=int, default=64)
    parser.add_argument("--batch-wait-ms", type=float, default=5)
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[1]
    if output == checkout or checkout in output.parents:
        parser.error("Keep raw evidence and model checkpoints outside the Git checkout")
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory; existing evidence is preserved")
    if not 1 <= args.small_batch <= 32 or not 5 <= args.repeats <= 1000 or not 1 <= args.warmup <= 100:
        parser.error("Use batch 1..32, repeats 5..1000 and warm-up 1..100")
    if not 64 <= args.gpu_memory_mib <= 512:
        parser.error("GPU tensor allocator budget must be between 64 and 512 MiB")
    if args.test_count < args.small_batch:
        parser.error("The held-out set must contain at least the requested small batch")
    if not 0 <= args.port <= 65535:
        parser.error("The localhost port must be between 0 and 65535")
    if args.load_probe and (not 1 <= args.duration <= 30 or not 1 <= args.target_rps <= 200
                           or not 1 <= args.queue_size <= 256 or not 0 <= args.batch_wait_ms <= 5
                           or not 32 <= args.input_size <= 224 or args.small_batch > 8 or args.serve):
        parser.error("Load probe: duration 1..30 s, rate 1..200, queue 1..256, wait 0..5 ms, image 32..224, batch <=8, no persistent --serve")
    output.mkdir(parents=True, exist_ok=True)
    if args.load_probe:
        return run_load_probe(args, output)
    report = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Synthetic vision engineering demonstration, not a production quality or capacity acceptance",
              "weights_downloaded": False, "packages_installed": False,
              "cluster_execution_claimed": False, "gpu_tensor_allocator_budget_mib": args.gpu_memory_mib}
    try:
        torch = require_torch()
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        # Bound this tiny exercise. Production cuDNN/TensorRT tuning is a separate experiment.
        torch.backends.cudnn.enabled = False
        cuda = torch.cuda.is_available()
        if args.require_cuda and not cuda:
            raise RuntimeError("This run requires CUDA, but the selected interpreter cannot access it")
        report["runtime"] = {"python": platform.python_version(), "platform": platform.system(),
                             "torch": torch.__version__, "torch_compiled_cuda": torch.version.cuda,
                             "cuda_available": cuda, "cpu_threads": 1, "cudnn_enabled": False}
        if cuda:
            props = torch.cuda.get_device_properties(0)
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_mib * 1024**2 / props.total_memory, 0)
            report["runtime"].update(gpu=props.name, gpu_total_memory_mib=props.total_memory / 1024**2,
                                     gpu_compute_capability=[props.major, props.minor])
        train = generate_synthetic_examples(args.train_count, args.seed, "train")
        held_out = generate_synthetic_examples(args.test_count, args.seed + 10000, "held-out")
        train_hashes = {image_digest(x.pixels) for x in train}
        test_hashes = {image_digest(x.pixels) for x in held_out}
        overlap = train_hashes.intersection(test_hashes)
        if overlap:
            raise RuntimeError("Training and held-out images overlap")
        report["dataset"] = {"synthetic_only": True, "image_format": "32x32 grayscale unsigned 8-bit arrays",
                             "labels": CLASS_NAMES, "training_count": len(train), "held_out_count": len(held_out),
                             "training_seed": args.seed, "held_out_seed": args.seed + 10000,
                             "image_hash_overlap": len(overlap), "task": "Distinguish continuous and interrupted synthetic seal stripes"}
        model, training = train_tiny_model(train, seed=args.seed, epochs=args.epochs)
        report["training"] = training
        version = weights_fingerprint(model)
        checkpoint = output / "tiny_synthetic_seal_cnn.pt"
        torch.save(model.state_dict(), checkpoint)
        report["model"] = {"architecture": "Conv2d(1,4)-ReLU-MaxPool-Conv2d(4,8)-ReLU-MaxPool-Linear(512,2)",
                           "parameters": sum(p.numel() for p in model.parameters()),
                           "weights_sha256": version, "checkpoint_file": checkpoint.name,
                           "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                           "preprocessing_version": PREPROCESS_VERSION, "quality_threshold": 0.95}
        report["quality"] = {}
        report["benchmarks"] = []
        report["http_benchmarks"] = []
        predictions_by_device = {}
        last_engine = None
        for device in (["cpu", "cuda:0"] if cuda else ["cpu"]):
            engine = VisionPredictor(model, device=device)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(0)
            quality, predictions = evaluate_model(engine, held_out)
            report["quality"][device] = quality
            predictions_by_device[device] = predictions
            for batch_size in sorted({1, args.small_batch}):
                report["benchmarks"].append(measure_engine(engine, held_out, batch_size, args.warmup, args.repeats))
            if args.http:
                report["http_benchmarks"].append(measure_http(engine, held_out[0].pixels, 5, min(args.repeats, 30)))
            if device == "cpu":
                # Use a fresh cache for the cache experiment, independently of quality/timing calls.
                report["cache_demonstration"] = demonstrate_cache(VisionPredictor(model), held_out[0].pixels)
            if device.startswith("cuda"):
                report["gpu_memory"] = {"peak_tensor_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                        "peak_allocator_reserved_bytes": torch.cuda.max_memory_reserved(0),
                                        "scope": "PyTorch allocator only; excludes driver/context overhead and other processes"}
            last_engine = engine
        if cuda:
            cpu, gpu = predictions_by_device["cpu"], predictions_by_device["cuda:0"]
            agreement = sum(a["predicted_class"] == b["predicted_class"] for a, b in zip(cpu, gpu))
            error = max(abs(a - b) for left, right in zip(cpu, gpu) for a, b in zip(left["probabilities"], right["probabilities"]))
            report["cpu_cuda_comparison"] = {"same_weights": True, "samples": len(cpu),
                                             "classification_agreement": agreement,
                                             "max_absolute_probability_difference": error,
                                             "probability_tolerance": 0.00001,
                                             "passed": agreement == len(cpu) and error <= 0.00001}
        else:
            report["cpu_cuda_comparison"] = {"passed": False, "status": "not_run", "reason": "CUDA unavailable in selected interpreter"}
        report["timing_protocol"] = {"warmup_calls": args.warmup, "measured_calls_per_case": args.repeats,
                                     "clock": "perf_counter_ns, synchronized at CUDA measurement boundaries",
                                     "batch_sizes": sorted({1, args.small_batch}),
                                     "load": "One process, sequential calls. This does not establish concurrent peak service capacity.",
                                     "cache_enabled_during_timings": False}
        report["fixture_examples"] = [{"id": item.image_id, "expected_class": item.label,
                                       "pixels": item.pixels, "prediction": predictions_by_device["cpu"][index]}
                                      for index, item in enumerate(held_out[:4])]
        quality_passed = all(x["accuracy"] >= 0.95 for x in report["quality"].values())
        comparison_passed = not cuda or report["cpu_cuda_comparison"]["passed"]
        report["synthetic_acceptance"] = {"quality_passed": quality_passed,
                                           "cache_checks_passed": report["cache_demonstration"]["passed"],
                                           "cpu_cuda_consistency_passed": comparison_passed if cuda else None,
                                           "production_approval": False}
        report["limitations"] = [
            "Fictional stripe images and a tiny trained CNN do not validate real packaging inspection.",
            "This run does not execute YOLO, TensorRT or Triton.",
            "The CUDA route is the host interpreter; it does not provision a GPU inside OKD.",
            "Batching and cache effects are workload-specific. No speed-up is guaranteed.",
            "The optional localhost HTTP demo has no authentication and must not serve sensitive or production data.",
            "PyTorch allocator limits exclude CUDA driver/context overhead.",
            "The memory-bounded exercise disables cuDNN. It does not measure an optimized production inference server.",
        ]
        report["status"] = "passed" if quality_passed and comparison_passed and report["cache_demonstration"]["passed"] else "failed"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary = {"status": report["status"], "report": str(output / "report.json"),
                   "cuda_executed": cuda, "quality": report["quality"],
                   "weights_sha256": version, "synthetic_only": True}
        print(json.dumps(summary))
        if args.serve and report["status"] == "passed":
            server = create_local_http_server(last_engine, args.port)
            print(json.dumps({"local_demo_url": f"http://127.0.0.1:{server.server_address[1]}/health",
                              "scope": "Synthetic localhost demo only"}), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        return 0 if report["status"] == "passed" else 1
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      completed_at_utc=datetime.now(timezone.utc).isoformat())
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": "failed", "error": str(exc), "report": str(output / "report.json")}))
        return 2
    finally:
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
