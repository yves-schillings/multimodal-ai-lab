"""Run bounded synthetic vision experiments and retain explicit evidence.

The default CLI path trains a tiny grayscale CNN, checks held-out fixtures, and
compares sequential CPU/CUDA calls. --load-probe instead runs a random-weight
ResNet-18-shaped RGB service under scheduled concurrent localhost traffic.
--train-packaging trains the same ResNet-18 shape on procedural packaging-line
images and evaluates it on an untouched test split. --ramp serves those trained
weights through the batched service and climbs an open-loop offered-load ladder
from separate client processes for one named configuration.
No path downloads weights/packages or approves production use. Raw reports
and checkpoints must be written to a new directory outside this Git checkout.
The module adds the checkout root to sys.path at import, but experiments start
only when main is called or the file is executed as a script.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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
    PACKAGING_CLASS_NAMES, PACKAGING_PREPROCESS_VERSION, NvidiaSmiSampler, PredictionCache,
    classification_metrics, generate_packaging_images, packaging_images_to_uint8,
    packaging_split_report, predict_packaging_logits, ramp_should_stop,
    run_open_loop_ramp_step, select_sustainable_capacity, summarize_gpu_samples,
    summarize_open_loop_step, train_packaging_classifier,
)

# Each GPU configuration changes one lever relative to gpu_fp32_dynamic. Thread
# settings: None keeps the PyTorch default (all physical cores on the test host);
# an integer is passed to torch.set_num_threads / set_num_interop_threads.
RAMP_CONFIGURATIONS = {
    "cpu_fp32_dynamic": {"device": "cpu", "precision": "fp32", "workers": 1, "max_batch": 8,
                         "batch_wait_ms": 2.0, "cache_repeat_share": None,
                         "intra_op_threads": None, "interop_threads": None,
                         "description": "CPU, FP32, one worker, dynamic batching up to 8 with a 2 ms collection wait; PyTorch default threads (all physical cores)"},
    "cpu_fp32_dynamic_1thread": {"device": "cpu", "precision": "fp32", "workers": 1, "max_batch": 8,
                                 "batch_wait_ms": 2.0, "cache_repeat_share": None,
                                 "intra_op_threads": 1, "interop_threads": 1,
                                 "description": "Historical comparison only: the 27 October probe's one intra-op and one inter-op thread, otherwise cpu_fp32_dynamic"},
    "gpu_fp32_batch1": {"device": "cuda:0", "precision": "fp32", "workers": 1, "max_batch": 1,
                        "batch_wait_ms": 0.0, "cache_repeat_share": None,
                        "intra_op_threads": 4, "interop_threads": 1,
                        "description": "GPU, FP32, one worker, batch size 1 (batching lever off)"},
    "gpu_fp32_dynamic": {"device": "cuda:0", "precision": "fp32", "workers": 1, "max_batch": 8,
                         "batch_wait_ms": 2.0, "cache_repeat_share": None,
                         "intra_op_threads": 4, "interop_threads": 1,
                         "description": "GPU, FP32, one worker, dynamic batching up to 8 with a 2 ms collection wait (reference)"},
    "gpu_fp16_dynamic": {"device": "cuda:0", "precision": "fp16", "workers": 1, "max_batch": 8,
                         "batch_wait_ms": 2.0, "cache_repeat_share": None,
                         "intra_op_threads": 4, "interop_threads": 1,
                         "description": "GPU, FP16 weights and inputs, otherwise the reference configuration"},
    "gpu_fp32_dynamic_2workers": {"device": "cuda:0", "precision": "fp32", "workers": 2, "max_batch": 8,
                                  "batch_wait_ms": 2.0, "cache_repeat_share": None,
                                  "intra_op_threads": 4, "interop_threads": 1,
                                  "description": "GPU, FP32, two model copies and CUDA streams in one process on one GPU; not multi-node replication"},
    "gpu_fp32_dynamic_cache50": {"device": "cuda:0", "precision": "fp32", "workers": 1, "max_batch": 8,
                                 "batch_wait_ms": 2.0, "cache_repeat_share": 0.5,
                                 "intra_op_threads": 4, "interop_threads": 1,
                                 "description": "Reference configuration plus an exact-body result cache; 50 % of requests repeat 32 hot images, the rest are byte-unique"},
}
INPUT_SIZE_JUSTIFICATION = (
    "224 x 224 RGB keeps per-image compute and the 150,528-byte body identical to the 27 October "
    "random-weight probe, so the two series are comparable; trained weights do not change dense "
    "convolution cost. The thinnest scratches are 0.9 to 1.8 px wide at 224 px and would fall below "
    "one pixel at 112 px.")


def run_load_probe(args, output):
    """Measure the random-weight RGB probe under one bounded offered-load point.

    Args:
        args (argparse.Namespace): CLI-validated configuration with seed (int),
            require_cuda (bool), gpu_memory_mib (int, 64..512 MiB allocator cap),
            input_size (int, 32..224 pixels per side), small_batch (int, 1..8 images),
            warmup (int, calls per worker), queue_size (int, 1..256 requests),
            batch_wait_ms (float, 0..5 ms after first dequeue), target_rps (int,
            1..200 requests/second), and duration (float, 1..30 seconds per case).
        output (pathlib.Path): Existing writable empty evidence directory, validated
            and created by main outside the source checkout.

    Returns:
        int: 0 when all requested configurations complete their request accounting,
        1 for a partial case failure, or 2 for a caught setup/execution failure.

    Raises:
        Evidence-write/cleanup failures can escape if they occur while handling
        another error; otherwise Exception instances are recorded in report.json.

    Side effects:
        Sets process-global PyTorch thread/cuDNN settings and optional CUDA allocator
        fraction. Builds random model weights and eight seeded RGB fixtures. Runs
        CPU FP32/one-worker, then CUDA FP32/FP16 with one/two workers when available.
        Creates and closes owned loopback servers and model-worker threads, prints
        per-case JSON, saves incremental evidence, and collects unused allocations.

    Limitations:
        Completion means accounting completed, not that latency, model quality or a
        release threshold passed. Random outputs have no detection meaning.
        Model/driver overhead and other processes fall outside the allocator cap.
        Settings are not restored for reuse as a general in-process application API.
    """
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
        """Write the captured current report as UTF-8 indented JSON.

        Returns:
            None.

        Raises:
            Serialization or filesystem errors propagate to the surrounding run logic.

        Side effects:
            Replaces this run's report.json, allowing partial progress to be retained.
            The caller selected an initially empty directory; historical runs elsewhere
            are not overwritten. Captures report and path; accepts no parameters.
        """
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
            """Warm and measure one device/precision/worker configuration.

            Args:
                device (str): cpu or cuda:0 from the captured configuration matrix.
                precision (str): fp32, or fp16 on CUDA.
                workers (int): Number of independent model copies/streams, one or two.

            Returns:
                dict: Device configuration, actual weight hash, finite-output diagnostic,
                request accounting, batch metrics, optional CUDA memory counters and status.
                Caught execution errors become status=failed with capped error details.

            Raises:
                Server/service cleanup errors in finally can propagate to the outer run.
                They are not converted into a successful or apparently closed case.

            Side effects:
                Captures base model, synthetic payloads, torch and CLI settings. Allocates
                worker-specific copies/streams, executes warm-ups and one diagnostic call,
                then serves uncached synthetic RGB requests on an ephemeral loopback port.
                Finally shuts down the owned HTTP server and batch service and records
                closure flags. The diagnostic checks finiteness, never class accuracy.
            """
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


def query_gpu_once():
    """Read one device-wide nvidia-smi sample for GPU 0.

    Returns:
        dict | None: utilization_percent, memory_used_mib and memory_total_mib, or
        None when nvidia-smi is missing, times out or prints an unexpected line.

    Raises:
        Nothing for a missing tool or malformed output.

    Side effects:
        Runs one short nvidia-smi subprocess with a 15-second timeout. The values
        include other processes and Windows desktop applications.
    """
    import subprocess
    try:
        completed = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    parts = [part.strip() for part in completed.stdout.strip().split(",")]
    try:
        return {"utilization_percent": float(parts[0]), "memory_used_mib": float(parts[1]),
                "memory_total_mib": float(parts[2])}
    except (ValueError, IndexError):
        return None


def cpu_topology():
    """Describe the CPUs visible to this process.

    Returns:
        dict: logical_cpus (os.cpu_count), cpus_in_affinity_mask, physical_cores
        counted as unique (physical id, core id) pairs in /proc/cpuinfo (None when
        unavailable) and the model name.

    Raises:
        Nothing for a missing /proc/cpuinfo or affinity API.

    Limitations:
        Under WSL these are the VM's virtual CPUs; the Windows host shares them.
    """
    pairs, model, physical = set(), None, None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key == "physical id":
                    physical = value
                elif key == "core id":
                    pairs.add((physical, value))
                elif key == "model name" and model is None:
                    model = value
    except OSError:
        pass
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {"logical_cpus": os.cpu_count(), "cpus_in_affinity_mask": affinity,
            "physical_cores": len(pairs) or None, "model_name": model}


def raise_open_file_limit(wanted=65536):
    """Raise this process's soft open-file limit toward a requested value.

    Args:
        wanted (int): Desired soft limit; the hard limit is never exceeded.

    Returns:
        list[int] | None: [soft, hard] after the attempt, or None where the
        resource module is unavailable.

    Raises:
        Nothing: ValueError/OSError from setrlimit leave the limit unchanged.

    Side effects:
        Changes a process-wide resource limit used by server sockets.
    """
    try:
        import resource
    except ImportError:
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, wanted)
    if soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ValueError, OSError):
            pass
    return list(resource.getrlimit(resource.RLIMIT_NOFILE))


def run_packaging_training(args, output):
    """Generate the packaging splits, train the ResNet-18 shape and test it once.

    Args:
        args (argparse.Namespace): CLI-validated settings: seed (int, dataset and
            model seed), train_per_class/val_per_class/test_per_class (int images
            per class), input_size (int pixels), epochs (int), training_batch
            (int), learning_rate (float), cpu_threads (int), require_cuda (bool)
            and train_gpu_memory_mib (int allocator cap for training).
        output (pathlib.Path): Existing empty evidence directory outside the
            checkout; receives training-report.json, the weights file and its
            metadata sidecar.

    Returns:
        int: 0 when training and test evaluation completed, 2 for a caught failure.

    Raises:
        Evidence-write failures while recording another error can escape.

    Side effects:
        Sets process-global torch thread/cuDNN settings and an optional CUDA
        allocator fraction. Generates train, validation and test splits from
        disjoint seeds (seed, seed + 100000, seed + 200000), refuses any image-hash
        overlap, trains on the GPU when available, selects the epoch on validation
        only, then evaluates the untouched test split once in FP32 and, on CUDA,
        FP16. Writes weights with SHA-256 and prints one line per epoch.

    Limitations:
        Procedural images and fictional defect rules: test accuracy describes
        this generator only, not a real packaging line.
    """
    report = {"schema_version": "vision-packaging-training-v1", "purpose": "trained_synthetic_inspection_model",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "synthetic_only": True,
              "personal_data": False, "production_approval": False, "weights_downloaded": False,
              "packages_installed": False}
    path = output / "training-report.json"

    def save():
        """Write the captured training report as UTF-8 indented JSON.

        Returns:
            None.

        Raises:
            Serialization or filesystem errors propagate.

        Side effects:
            Replaces this run's training-report.json so partial progress survives.
        """
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        torch = require_torch()
        torch.set_num_threads(args.cpu_threads)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        cuda = torch.cuda.is_available()
        if args.require_cuda and not cuda:
            raise RuntimeError("CUDA required but unavailable in the selected interpreter")
        device = "cuda:0" if cuda else "cpu"
        report["runtime"] = {"torch": torch.__version__, "compiled_cuda": torch.version.cuda,
                             "python": platform.python_version(), "platform": platform.system(),
                             "torch_get_num_threads": torch.get_num_threads(),
                             "torch_get_num_interop_threads": torch.get_num_interop_threads(),
                             "cpu_topology": cpu_topology(), "cudnn_benchmark": False, "device": device,
                             "gpu_before_training": query_gpu_once() if cuda else None}
        if cuda:
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info()
            report["runtime"].update(gpu=props.name, compute_capability=[props.major, props.minor],
                                     total_device_memory_mib=total / 1024**2,
                                     free_device_memory_before_mib=free / 1024**2,
                                     training_allocator_cap_mib=args.train_gpu_memory_mib)
            if free < (args.train_gpu_memory_mib + 512) * 1024**2:
                raise RuntimeError("Insufficient currently free device memory for the training cap plus margin")
            torch.cuda.set_per_process_memory_fraction(args.train_gpu_memory_mib * 1024**2 / props.total_memory, 0)
            torch.cuda.reset_peak_memory_stats(0)
        seeds = {"train": args.seed, "validation": args.seed + 100000, "test": args.seed + 200000}
        counts = {"train": 4 * args.train_per_class, "validation": 4 * args.val_per_class,
                  "test": 4 * args.test_per_class}
        started = time.perf_counter()
        splits = {name: generate_packaging_images(counts[name], seeds[name], name, args.input_size) for name in seeds}
        check = packaging_split_report(splits)
        report["dataset"] = {"task": "Classify a procedural bottle image as good, missing_cap, scratch or misaligned_label",
                             "class_names": list(PACKAGING_CLASS_NAMES), "image_size": args.input_size,
                             "input_size_justification": INPUT_SIZE_JUSTIFICATION,
                             "image_format": "Raw interleaved HWC RGB uint8, no compression or decoder",
                             "generator": "lab.engineering_vision.draw_packaging_image", "seeds": seeds,
                             "generation_seconds": time.perf_counter() - started, "split_check": check}
        save()
        if check["cross_split_overlap_total"] or any(x["duplicates_within_split"] for x in check["splits"].values()):
            raise RuntimeError("Split image hashes overlap or repeat")

        def progress(record):
            """Print one epoch record as a JSON line.

            Args:
                record (dict): Epoch metrics produced by train_packaging_classifier.

            Returns:
                None.
            """
            print(json.dumps({"epoch": record}), flush=True)

        model, training = train_packaging_classifier(
            splits["train"], splits["validation"], image_size=args.input_size, epochs=args.epochs,
            batch_size=args.training_batch, seed=args.seed, device=device,
            learning_rate=args.learning_rate, progress=progress)
        report["training"] = training
        report["training"]["total_seconds"] = sum(x["duration_seconds"] for x in training["history"])
        if cuda:
            report["training"]["peak_tensor_allocated_mib"] = torch.cuda.max_memory_allocated(0) / 1024**2
            report["training"]["peak_allocator_reserved_mib"] = torch.cuda.max_memory_reserved(0) / 1024**2
        save()
        test_tensor = packaging_images_to_uint8(splits["test"], args.input_size)
        labels = [item.label for item in splits["test"]]
        evaluations, predictions = {}, {"labels": labels}
        for precision in (("fp32", "fp16") if cuda else ("fp32",)):
            logits = predict_packaging_logits(model, test_tensor, device, precision, 64)
            predicted = logits.argmax(dim=1).tolist()
            predictions[precision] = predicted
            evaluations[precision] = {"device": device, "batch_size": 64,
                                      "metrics": classification_metrics(labels, predicted)}
        if cuda:
            agree = sum(a == b for a, b in zip(predictions["fp32"], predictions["fp16"]))
            evaluations["fp16_vs_fp32"] = {
                "agreement": agree / len(labels), "disagreements": len(labels) - agree,
                "accuracy_change": evaluations["fp16"]["metrics"]["accuracy"] - evaluations["fp32"]["metrics"]["accuracy"]}
        report["test_evaluation"] = evaluations
        report["test_predictions"] = predictions
        weights_path = output / f"packaging_resnet18_{args.input_size}.pt"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights_path)
        file_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        metadata = {"architecture": "ResNet-18-shaped (lab.engineering_vision.build_resnet18_probe)",
                    "class_names": list(PACKAGING_CLASS_NAMES), "num_classes": len(PACKAGING_CLASS_NAMES),
                    "input_size": args.input_size, "input_layout": "HWC RGB uint8 body",
                    "preprocessing_version": PACKAGING_PREPROCESS_VERSION, "model_seed": args.seed,
                    "weights_file": weights_path.name, "weights_file_sha256": file_sha,
                    "state_fingerprint_sha256": fast_weights_fingerprint(model.cpu()),
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "selected_epoch": training["selected_epoch"],
                    "test_split": {"seed": seeds["test"], "count": counts["test"],
                                   "aggregate_sha256": check["splits"]["test"]["aggregate_sha256"]}}
        weights_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        report["model"] = metadata
        report["status"] = "completed"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["limitations"] = [
            "Procedural images with fictional defect rules; accuracy describes this generator only.",
            "Training is not bit-reproducible on GPU; the weights file hash identifies the evaluated weights.",
            "Random horizontal flip is the only augmentation; no real camera, lens or lighting model is simulated.",
            "Test split was evaluated once after epoch selection on validation."]
        save()
        print(json.dumps({"status": "completed", "report": str(path), "weights": str(weights_path),
                          "weights_sha256": file_sha,
                          "test_accuracy": {k: v["metrics"]["accuracy"] for k, v in evaluations.items() if "metrics" in v}}),
              flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1500],
                      completed_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps({"status": "failed", "error": str(exc), "report": str(path)}), flush=True)
        return 2


def run_packaging_ramp(args, output):
    """Serve the trained weights for one configuration and climb the load ladder.

    Args:
        args (argparse.Namespace): CLI-validated settings: ramp_config (str key of
            RAMP_CONFIGURATIONS), trained_weights (Path with a .json sidecar),
            weights_sha256 (optional expected file hash), ramp_steps (list[int]
            offered req/s), duration (float measured seconds per step),
            ramp_warmup_seconds (float), key_point_repeats (int total runs at key
            points), slo_p99_ms, slo_max_failure_share, stop_failure_share,
            deadline_ms (floats), queue_size (int), warmup (int model warm-up
            calls per batch size), gpu_memory_mib (int allocator cap),
            cache_hot_images (int), step_cooldown_seconds (float). Torch thread
            counts come from the configuration, not from args.cpu_threads.
        output (pathlib.Path): Existing empty evidence directory outside the
            checkout; receives ramp-<config>.json and the test payload file.

    Returns:
        int: 0 when every step executed, 1 when a step failed to execute, 2 for a
        caught setup failure. SLO failures are results, not execution failures.

    Raises:
        Evidence-write failures while recording another error can escape.

    Side effects:
        Verifies the weights file hash and regenerates the test split, refusing a
        mismatch. Sets process-global torch settings, raises the open-file limit,
        allocates model copies, starts one loopback server and batching service per
        step, spawns client processes and an nvidia-smi sampler, and closes them.
        Climbs the ladder until a step's failure share exceeds the stop share, it
        is generator-limited or the ladder ends; then reruns the highest passing
        and first failing rates until each has key_point_repeats runs.
    """
    config = RAMP_CONFIGURATIONS[args.ramp_config]
    report = {"schema_version": "vision-trained-ramp-v1", "purpose": "trained_model_load_ramp",
              "configuration_name": args.ramp_config, "configuration": config,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "synthetic_only": True,
              "production_approval": False, "weights_downloaded": False, "packages_installed": False,
              "steps": []}
    path = output / f"ramp-{args.ramp_config}.json"

    def save():
        """Write the captured ramp report as UTF-8 indented JSON.

        Returns:
            None.

        Raises:
            Serialization or filesystem errors propagate.

        Side effects:
            Replaces this run's ramp JSON so partial progress survives a failure.
        """
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        report["open_file_limit"] = raise_open_file_limit()
        weights = args.trained_weights.expanduser().resolve()
        metadata = json.loads(weights.with_suffix(".json").read_text(encoding="utf-8"))
        file_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        if file_sha != metadata["weights_file_sha256"] or (args.weights_sha256 and file_sha != args.weights_sha256):
            raise RuntimeError("Trained weights file hash does not match its metadata or the expected value")
        cuda_config = config["device"].startswith("cuda")
        gpu_before = query_gpu_once() if cuda_config else None
        torch = require_torch()
        default_threads = {"intra_op": torch.get_num_threads(), "interop": torch.get_num_interop_threads()}
        if config["intra_op_threads"] is not None:
            torch.set_num_threads(config["intra_op_threads"])
        if config["interop_threads"] is not None:
            torch.set_num_interop_threads(config["interop_threads"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        if cuda_config and not torch.cuda.is_available():
            raise RuntimeError("This configuration needs CUDA, which the selected interpreter cannot access")
        size = metadata["input_size"]
        class_names = tuple(metadata["class_names"])
        test = generate_packaging_images(metadata["test_split"]["count"], metadata["test_split"]["seed"], "test", size)
        aggregate = packaging_split_report({"test": test})["splits"]["test"]["aggregate_sha256"]
        if aggregate != metadata["test_split"]["aggregate_sha256"]:
            raise RuntimeError("Regenerated test split differs from the split recorded at training time")
        payloads = [item.payload for item in test]
        labels = [item.label for item in test]
        payload_path = output / "test_payloads.u8"
        payload_path.write_bytes(b"".join(payloads))
        model = build_resnet18_probe(metadata["model_seed"], num_classes=len(class_names))
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval()
        weights_version = fast_weights_fingerprint(model)
        report["runtime"] = {"torch": torch.__version__, "compiled_cuda": torch.version.cuda,
                             "python": platform.python_version(), "platform": platform.system(),
                             "torch_default_threads_before_setting": default_threads,
                             "torch_get_num_threads": torch.get_num_threads(),
                             "torch_get_num_interop_threads": torch.get_num_interop_threads(),
                             "thread_setting": "PyTorch default" if config["intra_op_threads"] is None else "explicit",
                             "cpu_topology": cpu_topology(), "cudnn_benchmark": False,
                             "gpu_before_model_load": gpu_before}
        report["model"] = {"weights_file": weights.name, "weights_file_sha256": file_sha,
                           "state_fingerprint_sha256": weights_version, "class_names": list(class_names),
                           "input_size": size, "selected_epoch": metadata.get("selected_epoch"),
                           "trained": True, "random_weights": False}
        report["inputs"] = {"test_split_seed": metadata["test_split"]["seed"], "test_images": len(test),
                            "test_split_aggregate_sha256": aggregate, "bytes_per_image": len(payloads[0]),
                            "class_counts": {n: labels.count(i) for i, n in enumerate(class_names)}}
        if cuda_config:
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info()
            if free < (args.gpu_memory_mib + 256) * 1024**2:
                raise RuntimeError("Insufficient currently free device memory for the bounded service plus margin")
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_mib * 1024**2 / props.total_memory, 0)
            torch.cuda.reset_peak_memory_stats(0)
            report["runtime"].update(gpu=props.name, compute_capability=[props.major, props.minor],
                                     total_device_memory_mib=total / 1024**2,
                                     free_device_memory_before_mib=free / 1024**2,
                                     allocator_cap_mib=args.gpu_memory_mib)
        batch_call, models, streams = make_resnet_batch_runtime(model, config["device"], config["precision"],
                                                                config["workers"], size, class_names=class_names)
        warm_started = time.perf_counter()
        for worker in range(config["workers"]):
            for batch in range(1, config["max_batch"] + 1):
                for _ in range(args.warmup):
                    batch_call(worker, payloads[:batch])
        offline = []
        for offset in range(0, len(payloads), config["max_batch"]):
            results, _ = batch_call(0, payloads[offset:offset + config["max_batch"]])
            offline.extend(result["predicted_class"] for result in results)
        report["offline_quality"] = {"runtime": "Same batch_call, worker 0, batch size max_batch, before serving",
                                     "batch_size": config["max_batch"], "predictions": offline,
                                     "metrics": classification_metrics(labels, offline, class_names)}
        report["warmup"] = {"calls_per_batch_size_per_worker": args.warmup,
                            "seconds_including_offline_predictions": time.perf_counter() - warm_started}
        if cuda_config:
            report["memory_after_warmup"] = {
                "peak_tensor_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
                "peak_allocator_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
                "device_memory_used_by_all_processes_mib": (query_gpu_once() or {}).get("memory_used_mib"),
                "scope": "PyTorch allocator figures cover this process only; device figure includes desktop applications"}
        hot_images = []
        if config["cache_repeat_share"] is not None:
            stride = max(1, len(payloads) // args.cache_hot_images)
            hot_images = [i * stride for i in range(args.cache_hot_images)]
        report["protocol"] = {
            "ladder_rps": args.ramp_steps, "warmup_seconds_per_step": args.ramp_warmup_seconds,
            "measured_seconds_per_step": args.duration, "key_point_total_runs": args.key_point_repeats,
            "slo": {"p99_end_to_end_ms_max": args.slo_p99_ms, "failure_share_max": args.slo_max_failure_share,
                    "deadline_ms": args.deadline_ms,
                    "latency_origin": "Scheduled arrival time, so client lag counts against the service"},
            "stop_rule": f"Stop climbing after a step with failure share above {args.stop_failure_share}, a generator-limited step or a step execution failure",
            "generator_limited_rule": "Achieved send rate in the measured window below 95 % of the offered rate; such points are not used for capacity",
            "p99_reliability_rule": "p99 flagged unreliable below 1,000 observations",
            "queue_capacity_requests": args.queue_size, "max_batch": config["max_batch"],
            "batch_wait_ms": config["batch_wait_ms"], "workers": config["workers"],
            "cache_repeat_share": config["cache_repeat_share"], "cache_hot_images": hot_images,
            "client": "Spawned asyncio processes, min(4, ceil(rate / 500)); raw HTTP/1.0, one connection per request, 256 open requests per process, 5 s request timeout",
            "server": "One process: ThreadingHTTPServer (listen backlog 128) plus BoundedBatchVisionService worker thread(s)",
            "server_torch_threads": {"intra_op": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                                     "configured_intra_op": config["intra_op_threads"],
                                     "configured_interop": config["interop_threads"]},
            "step_cooldown_seconds": args.step_cooldown_seconds}
        save()

        def run_step(rate, run_kind, repeat_index):
            """Serve and measure one offered-load step on a fresh server/service pair.

            Args:
                rate (int): Offered requests per second.
                run_kind (str): ladder or key_point_repeat.
                repeat_index (int): 0 for the ladder run, then 1, 2, ... for repeats.

            Returns:
                dict: summarize_open_loop_step output plus run metadata, client
                process figures, CPU and GPU usage over the measured window,
                batching metrics, cache counters and allocator peaks; or a failed
                record with error details.

            Raises:
                Cleanup errors from server or service closure can propagate.

            Side effects:
                Starts and stops a loopback server, batching service, optional cache,
                nvidia-smi sampler and client processes; sleeps for the cool-down.
            """
            step = {"target_rps": rate, "run_kind": run_kind, "repeat_index": repeat_index,
                    "started_at_utc": datetime.now(timezone.utc).isoformat()}
            service = server = server_thread = None
            sampler = NvidiaSmiSampler() if cuda_config else None
            cache = PredictionCache(4096) if config["cache_repeat_share"] is not None else None
            try:
                if cuda_config:
                    torch.cuda.reset_peak_memory_stats(0)
                service = BoundedBatchVisionService(batch_call, image_size=size, workers=config["workers"],
                                                    max_batch=config["max_batch"], max_wait_ms=config["batch_wait_ms"],
                                                    queue_size=args.queue_size)
                server = create_batched_http_server(service, cache=cache,
                                                    cache_scope=(weights_version, PACKAGING_PREPROCESS_VERSION) if cache else None)
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                if sampler is not None:
                    sampler.start()
                raw = run_open_loop_ramp_step(server.server_address[1], payload_path, image_bytes=len(payloads[0]),
                                              image_count=len(payloads), target_rps=rate,
                                              warmup_seconds=args.ramp_warmup_seconds, measure_seconds=args.duration,
                                              repeat_share=config["cache_repeat_share"], hot_images=hot_images,
                                              plan_seed=rate * 10 + repeat_index)
                samples = sampler.stop() if sampler is not None else []
                step.update(summarize_open_loop_step(
                    raw["records"], target_rps=rate, warmup_seconds=args.ramp_warmup_seconds,
                    measure_seconds=args.duration, slo_p99_ms=args.slo_p99_ms,
                    slo_max_failure_share=args.slo_max_failure_share, deadline_ms=args.deadline_ms,
                    labels=labels, offline_predictions=offline))
                window = [raw["t0"] + x for x in raw["window_offsets_seconds"]]
                step["client"] = {"processes": raw["process_count"], "scheduled_requests_including_warmup": raw["scheduled_requests"],
                                  "returned_records": len(raw["records"]),
                                  "accounting_complete": len(raw["records"]) == raw["scheduled_requests"],
                                  "cpu_seconds_per_process": raw["client_cpu_seconds"], "errors": raw["client_errors"]}
                step["cpu_window"] = raw["cpu_window"]
                step["gpu_window"] = summarize_gpu_samples(samples, *window) if cuda_config else None
                metrics = service.metrics()
                step["batching_whole_step"] = {k: metrics[k] for k in ("batches", "images", "batch_occupancy_histogram",
                                                                         "mean_batch_size", "queue_rejections",
                                                                         "max_queue_depth", "batch_model_latency",
                                                                         "batch_pipeline_latency")}
                step["batching_whole_step"]["scope"] = "Whole step including warm-up seconds"
                if cache is not None:
                    lookups = cache.hits + cache.misses
                    step["cache_counters_whole_step"] = {"hits": cache.hits, "misses": cache.misses,
                                                         "hit_rate": cache.hits / lookups if lookups else None,
                                                         "entries_capacity": cache.max_entries}
                if cuda_config:
                    step["gpu_allocator"] = {"peak_tensor_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
                                             "peak_allocator_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2}
                step["status"] = "completed" if step["client"]["accounting_complete"] and not raw["client_errors"] else "failed"
            except Exception as exc:
                step.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1500])
            finally:
                if sampler is not None:
                    sampler.stop()
                if server is not None:
                    server.shutdown()
                    server.server_close()
                    server_thread.join(timeout=5)
                if service is not None:
                    service.close()
                gc.collect()
                time.sleep(args.step_cooldown_seconds)
            return step

        def announce(step):
            """Print a one-line progress summary for a finished step.

            Args:
                step (dict): run_step output.

            Returns:
                None.
            """
            latency = step.get("latency_from_scheduled_arrival_ms") or {}
            print(json.dumps({"config": args.ramp_config, "rps": step["target_rps"], "kind": step["run_kind"],
                              "status": step.get("status"), "slo_met": step.get("slo_met"),
                              "failure_share": step.get("failure_share"), "p50": latency.get("p50_ms"),
                              "p99": latency.get("p99_ms"), "achieved_send": step.get("achieved_send_rate_rps"),
                              "generator_limited": step.get("generator_limited"),
                              "gpu_util": (step.get("gpu_window") or {}).get("utilization_percent_mean"),
                              "error": step.get("error")}), flush=True)

        for rate in args.ramp_steps:
            step = run_step(rate, "ladder", 0)
            report["steps"].append(step)
            save()
            announce(step)
            stop, reason = ramp_should_stop(step, args.stop_failure_share)
            if stop:
                report["ladder_stopped"] = {"at_rps": rate, "reason": reason}
                break
        else:
            report["ladder_stopped"] = {"at_rps": None, "reason": "ladder_exhausted"}
        first = select_sustainable_capacity([s for s in report["steps"] if s.get("status") == "completed"] or report["steps"])
        key_rates = [r for r in (first["max_sustainable_offered_rps"], first["first_failing_offered_rps"]) if r is not None]
        report["key_point_rates"] = key_rates
        for rate in key_rates:
            ladder_step = next(s for s in report["steps"] if s["target_rps"] == rate)
            if ladder_step.get("generator_limited") or ladder_step.get("status") == "failed":
                continue
            for repeat in range(1, args.key_point_repeats):
                step = run_step(rate, "key_point_repeat", repeat)
                report["steps"].append(step)
                save()
                announce(step)
        report["capacity_at_slo"] = select_sustainable_capacity(report["steps"])
        executed = all(s.get("status") == "completed" for s in report["steps"])
        report["status"] = "completed" if executed else "partial_failure"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["limitations"] = [
            "Procedural synthetic images; served accuracy describes this generator only.",
            "One WSL host: server and client processes share the same 8-core/16-thread CPU and Windows desktop load.",
            "The GPU also drives the Windows desktop; nvidia-smi values are device-wide.",
            "Raw RGB bodies over loopback: no image decoding, TLS, authentication or real network.",
            "Capacity resolution is limited to the ladder rates; the true limit lies between the highest passing and first failing rate.",
            "Two workers share one GPU and one Python process; this is not multi-node replication."]
        save()
        print(json.dumps({"status": report["status"], "report": str(path),
                          "max_sustainable_offered_rps": report["capacity_at_slo"]["max_sustainable_offered_rps"]}), flush=True)
        return 0 if executed else 1
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1500],
                      completed_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps({"status": "failed", "error": str(exc), "report": str(path)}), flush=True)
        return 2


def main():
    """Parse the vision CLI, enforce experiment bounds and retain measured evidence.

    Parameters:
        None. Reads process sys.argv using argparse. --output is required; remaining
        options select seed, train/test fixture counts, epochs, warm-up/repeat counts,
        small-batch size, GPU allocator MiB, optional HTTP timing/persistent localhost
        serving, CUDA requirement, and the separate open-loop load-probe settings.
        --train-packaging and --ramp select the trained packaging-model paths and
        their own options (split sizes, training batch/rate/allocator cap, trained
        weights, configuration, ladder, SLO, stop share, deadline, CPU threads).

    Returns:
        int: Default tiny-model path returns 0 for its synthetic checks, 1 when a
        synthetic check fails, or 2 for a caught runtime failure. --load-probe returns
        that path's code: 0 completed, 1 partial failure, 2 caught failure.
        --train-packaging returns 0 or 2; --ramp returns 0, 1 or 2 as documented
        in run_packaging_training and run_packaging_ramp.

    Raises:
        SystemExit: argparse help/usage/validation exits, normally 0 or 2.
        Filesystem errors during output validation/creation or error-report writing
        can propagate. KeyboardInterrupt stops an optional owned --serve loop.

    Side effects:
        Resolves and creates a fresh output directory outside the checkout. The tiny
        path changes global PyTorch thread/cuDNN settings, sets an optional 64..512 MiB
        allocator fraction, generates separate seeded train/test fixtures, rejects
        image-hash overlap, trains on CPU, and writes model checkpoint/report JSON.
        It measures CPU and available CUDA routes, optionally times an ephemeral
        loopback server, prints a summary, and can keep --serve running until stopped.
        Garbage collection runs on exit from the tiny-model try/finally.

    Limitations:
        Passing means only the declared synthetic fixture/cache/consistency checks
        passed. No production approval, YOLO/TensorRT/Triton measurement, GPU-in-OKD
        deployment or sustainable concurrent-capacity claim follows. The load probe
        uses random weights and must not be interpreted as a trained detector.
    """
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
    parser.add_argument("--train-packaging", action="store_true", help="Train and test the ResNet-18 shape on procedural packaging images")
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--val-per-class", type=int, default=250)
    parser.add_argument("--test-per-class", type=int, default=250)
    parser.add_argument("--training-batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-gpu-memory-mib", type=int, default=4096)
    parser.add_argument("--cpu-threads", type=int, default=4, help="torch intra-op threads for --train-packaging; --ramp uses each configuration's own thread setting")
    parser.add_argument("--ramp", action="store_true", help="Serve trained weights and climb an open-loop load ladder")
    parser.add_argument("--trained-weights", type=Path, help="State-dict file written by --train-packaging (with its .json sidecar)")
    parser.add_argument("--weights-sha256", help="Optional expected SHA-256 of the trained weights file")
    parser.add_argument("--ramp-config", choices=sorted(RAMP_CONFIGURATIONS))
    parser.add_argument("--ramp-steps", default="10,25,50,100,200,400,800,1600,3200")
    parser.add_argument("--ramp-warmup-seconds", type=float, default=5)
    parser.add_argument("--key-point-repeats", type=int, default=3)
    parser.add_argument("--slo-p99-ms", type=float, default=100)
    parser.add_argument("--slo-max-failure-share", type=float, default=0.01)
    parser.add_argument("--stop-failure-share", type=float, default=0.20)
    parser.add_argument("--deadline-ms", type=float, default=1000)
    parser.add_argument("--cache-hot-images", type=int, default=32)
    parser.add_argument("--step-cooldown-seconds", type=float, default=2)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[1]
    if output == checkout or checkout in output.parents:
        parser.error("Keep raw evidence and model checkpoints outside the Git checkout")
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory; existing evidence is preserved")
    if sum(map(bool, (args.load_probe, args.train_packaging, args.ramp))) > 1:
        parser.error("Choose at most one of --load-probe, --train-packaging and --ramp")
    if args.train_packaging and not 1 <= args.cpu_threads <= 16:
        parser.error("Use 1..16 CPU threads")
    if args.train_packaging and (not 1 <= args.train_per_class <= 5000 or not 1 <= args.val_per_class <= 2500
                                 or not 1 <= args.test_per_class <= 2500 or not 2 <= args.training_batch <= 256
                                 or not 1 <= args.epochs <= 100 or not 512 <= args.train_gpu_memory_mib <= 8192
                                 or not 32 <= args.input_size <= 224 or not 0 < args.learning_rate <= 1):
        parser.error("Packaging training: 1..5000 train and 1..2500 val/test per class, batch 2..256, epochs 1..100, cap 512..8192 MiB, image 32..224, rate 0..1")
    if args.ramp:
        try:
            args.ramp_steps = [int(x) for x in args.ramp_steps.split(",")]
        except ValueError:
            parser.error("--ramp-steps must be comma-separated integers")
        if (args.ramp_config is None or args.trained_weights is None
                or not args.trained_weights.expanduser().is_file()
                or not args.trained_weights.expanduser().with_suffix(".json").is_file()):
            parser.error("--ramp needs --ramp-config and an existing --trained-weights file with its .json sidecar")
        if (args.ramp_steps != sorted(set(args.ramp_steps)) or not all(1 <= x <= 10000 for x in args.ramp_steps)
                or not 5 <= args.duration <= 60 or not 0 <= args.ramp_warmup_seconds <= 30
                or not 1 <= args.key_point_repeats <= 5 or not 1 <= args.queue_size <= 256
                or not 0 < args.slo_p99_ms <= args.deadline_ms or not 0 <= args.slo_max_failure_share < 1
                or not 0 < args.stop_failure_share < 1 or not 1 <= args.cache_hot_images <= 256
                or not 0 <= args.step_cooldown_seconds <= 30):
            parser.error("Ramp: ascending unique rates 1..10000, duration 5..60 s, warm-up 0..30 s, repeats 1..5, queue 1..256, 0 < SLO <= deadline, shares in 0..1, hot images 1..256, cool-down 0..30 s")
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
    if args.train_packaging:
        return run_packaging_training(args, output)
    if args.ramp:
        return run_packaging_ramp(args, output)
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
