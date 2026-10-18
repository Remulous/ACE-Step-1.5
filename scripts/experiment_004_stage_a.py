"""Run LM-only nano-vLLM teardown/rebuild trials in isolated staging."""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import weakref
from pathlib import Path

import torch


def device_snapshot(label, handler):
    """Capture synchronized CUDA allocator and device-wide memory counters."""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    free, total = torch.cuda.mem_get_info()
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    runner = getattr(getattr(handler, "llm", None), "model_runner", None)
    return {
        "label": label,
        "monotonic": time.monotonic(),
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cuda_free_bytes": free,
        "cuda_total_bytes": total,
        "nvidia_smi_used_mib": int(smi[0]),
        "nvidia_smi_total_mib": int(smi[1]),
        "compute_processes": processes,
        "engine_present": runner is not None,
        "graph_enabled": None if runner is None else not runner.enforce_eager,
        "kv_cache_bytes": None if runner is None or runner.kv_cache is None
        else runner.kv_cache.numel() * runner.kv_cache.element_size(),
    }


def capture_timer(original):
    """Wrap graph capture to separate its duration from model construction."""
    def measured(runner):
        start = time.perf_counter()
        result = original(runner)
        runner._stage_a_graph_capture_seconds = time.perf_counter() - start
        return result
    return measured


def initialize(handler, checkpoint_dir, model_name):
    """Use ACE-Step's normal LM initializer and return phase durations."""
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.sampling_params import SamplingParams

    original = ModelRunner.capture_cudagraph
    ModelRunner.capture_cudagraph = capture_timer(original)
    construction_started = time.perf_counter()
    try:
        status, ok = handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=model_name,
            backend="vllm",
            device="cuda",
            offload_to_cpu=False,
            dtype=torch.bfloat16,
        )
    finally:
        ModelRunner.capture_cudagraph = original
    if not ok or handler.llm_backend != "vllm":
        raise RuntimeError(f"ACE-Step nano-vLLM initialization failed: {status}")
    runner = handler.llm.model_runner
    construction_seconds = time.perf_counter() - construction_started
    if runner.enforce_eager:
        raise RuntimeError("Expected CUDA graph capture; initializer selected eager mode")
    canary_started = time.perf_counter()
    output = handler.llm.generate(
        ["Write one short musical phrase."],
        SamplingParams(temperature=0.01, max_tokens=8),
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    canary_seconds = time.perf_counter() - canary_started
    if not output or not output[0].get("token_ids"):
        raise RuntimeError("nano-vLLM restoration canary produced no tokens")
    return {
        "reconstruction_seconds": construction_seconds,
        "graph_capture_seconds": getattr(runner, "_stage_a_graph_capture_seconds", None),
        "enforce_eager": runner.enforce_eager,
        "canary_seconds": canary_seconds,
        "canary_tokens": len(output[0]["token_ids"]),
        "status": status,
    }


def main():
    """Run an initial load plus repeated close/rebuild cycles to a JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="/app/checkpoints")
    parser.add_argument("--model", default="acestep-5Hz-lm-4B")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=os.environ.get("ACESTEP_STAGE_A_REVISION", "unknown"))
    args = parser.parse_args()
    if args.cycles < 1 or not torch.cuda.is_available():
        raise SystemExit("cycles must be positive and CUDA must be available")
    if os.environ.get("ACESTEP_STAGE_A_CONFIRM") != "isolated-staging":
        raise SystemExit("Set ACESTEP_STAGE_A_CONFIRM=isolated-staging in staging only")
    if subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                       "--format=csv,noheader"], check=True,
                      capture_output=True, text=True).stdout.strip():
        raise SystemExit("Refusing to run: a GPU compute process is already active")

    sys.path.insert(0, "/app")
    from acestep.llm_inference import LLMHandler

    handler = LLMHandler()
    records = []
    try:
        initial_gpu = device_snapshot("before_initialization", handler)
        for cycle in range(args.cycles + 1):
            init = initialize(handler, args.checkpoint_dir, args.model)
            records.append({"cycle": cycle, "initialize": init,
                            "after_rebuild": device_snapshot("after_rebuild", handler)})
            engine = handler.llm
            runner = engine.model_runner
            if runner.world_size != 1 or not engine.is_finished():
                raise RuntimeError("Stage A harness requires an idle tensor-parallel-1 engine")
            references = {
                "engine": weakref.ref(engine),
                "runner": weakref.ref(runner),
                "model": weakref.ref(runner.model),
                "kv_cache": weakref.ref(runner.kv_cache),
            }
            torch.cuda.synchronize()
            start = time.perf_counter()
            engine.close()
            teardown_seconds = time.perf_counter() - start
            release_started = time.perf_counter()
            handler.llm = None
            handler.llm_initialized = False
            handler.llm_backend = None
            del engine, runner
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            records[-1]["teardown_seconds"] = teardown_seconds
            records[-1]["release_and_gc_seconds"] = time.perf_counter() - release_started
            dangling = [name for name, reference in references.items()
                        if reference() is not None]
            records[-1]["dangling_python_references"] = dangling
            if dangling:
                raise RuntimeError(f"Engine resources remain referenced after close: {dangling}")
            records[-1]["after_teardown"] = device_snapshot("after_teardown", handler)
            torch.cuda.reset_peak_memory_stats()
        payload = {
            "schema": "ace-step-exp004-stage-a-v1",
            "commit": args.revision,
            "gpu_name": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "checkpoint_dir": args.checkpoint_dir,
            "model": args.model,
            "cycles": args.cycles,
            "before_initialization": initial_gpu,
            "records": records,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    finally:
        if handler.llm is not None:
            handler.llm.close()
            handler.llm = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
