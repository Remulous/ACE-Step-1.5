"""Read-only CUDA model and nano-vLLM storage inventory for opt-in diagnostics."""

import weakref

import torch

_LM = None


def register_lm(host):
    """Keep a weak reference, without extending the LM engine lifetime."""
    global _LM
    _LM = weakref.ref(host)


def storage_bytes(tensors):
    """Count unique underlying storage bytes by device/dtype, including views once."""
    seen, totals = set(), {}
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        label = f"{tensor.device}/{tensor.dtype}"
        totals[label] = totals.get(label, 0) + storage.nbytes()
    return totals


def model_inventory(model):
    """Report registered parameter/buffer storage without reading tensor contents."""
    if model is None or not hasattr(model, "parameters"):
        return None
    tensors = list(model.parameters()) + list(model.buffers())
    return {"storage": storage_bytes(tensors),
            "attention": getattr(getattr(model, "config", None), "_attn_implementation", None)}


def lm_inventory():
    """Inspect persistent weights/cache and live scheduler occupancy without mutation."""
    host = _LM() if _LM is not None else None
    engine = getattr(host, "llm", None)
    runner = getattr(engine, "model_runner", None)
    if runner is None:
        return {"initialized": bool(getattr(host, "llm_initialized", False)), "runner": None}
    cache = getattr(runner, "kv_cache", None)
    block_mgr = getattr(getattr(engine, "scheduler", None), "block_manager", None)
    result = {
        "initialized": True, "weights": model_inventory(runner.model),
        "cache_storage": storage_bytes([cache]),
        "cache_shape": list(cache.shape) if cache is not None else None,
        "cache_dtype": str(cache.dtype) if cache is not None else None,
        "block_size": runner.block_size,
        "blocks": getattr(runner.config, "num_kvcache_blocks", None),
        "max_model_len": runner.config.max_model_len,
        "enforce_eager": runner.enforce_eager,
        "peak_used_blocks_since_init": getattr(engine.scheduler, "_profile_peak_used_blocks", None),
        "used_blocks": len(block_mgr.used_block_ids) if block_mgr else None,
        "free_blocks": len(block_mgr.free_block_ids) if block_mgr else None,
        "runner_direct_tensors": storage_bytes(vars(runner).values()),
    }
    return result


def inventory(host):
    """Collect component storage and DiT nesting state at stable boundaries."""
    return {"dit": model_inventory(getattr(host, "model", None)),
            "vae": model_inventory(getattr(host, "vae", None)),
            "text_encoder": model_inventory(getattr(host, "text_encoder", None)),
            "lm": lm_inventory(),
            "lease_depth": getattr(getattr(host, "_dit_residency", None), "depth", None)}


def track_cache_occupancy(host):
    """Track scheduler high-water occupancy after scheduling, without changing decisions."""
    engine = getattr(host, "llm", None)
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is None or hasattr(scheduler, "_profile_original_schedule"):
        return
    original = scheduler.schedule
    scheduler._profile_original_schedule = original
    scheduler._profile_peak_used_blocks = 0

    def schedule():
        result = original()
        used = len(scheduler.block_manager.used_block_ids)
        scheduler._profile_peak_used_blocks = max(scheduler._profile_peak_used_blocks, used)
        return result
    scheduler.schedule = schedule
