"""Opt-in structured CUDA residency measurements; never controls model execution."""

import functools
import json
import os
import time
from contextlib import contextmanager

import torch
from loguru import logger


def trace(host, event: str, **fields) -> None:
    """Log CUDA allocator and whole-device memory at a request/phase boundary."""
    if os.getenv("ACESTEP_RESIDENCY_TRACE") != "1":
        return
    data = {"event": event, "time": time.time(), **fields}
    try:
        if str(host.device).startswith("cuda") and torch.cuda.is_available():
            device = torch.device(host.device)
            free, total = torch.cuda.mem_get_info(device)
            data.update(
                allocated=torch.cuda.memory_allocated(device),
                reserved=torch.cuda.memory_reserved(device),
                peak_allocated=torch.cuda.max_memory_allocated(device),
                peak_reserved=torch.cuda.max_memory_reserved(device),
                device_used=total - free, device_total=total,
            )
    except (RuntimeError, ValueError) as exc:
        data["measurement_error"] = str(exc)
    logger.info("[residency] {}", json.dumps(data, sort_keys=True))


@contextmanager
def measured_phase(host, phase):
    """Measure each explicit GPU phase; tracing is opt-in and synchronizes endpoints."""
    active = (os.getenv("ACESTEP_RESIDENCY_TRACE") == "1"
              and str(getattr(host, "device", "")).startswith("cuda"))
    if not active:
        yield
        return
    from .residency_request import request_id

    torch.cuda.synchronize(host.device)
    torch.cuda.reset_peak_memory_stats(host.device)
    start = time.perf_counter()
    trace(host, "operation_start", request=request_id(), phase=phase)
    try:
        yield
    finally:
        torch.cuda.synchronize(host.device)
        trace(host, "operation_end", request=request_id(), phase=phase,
              seconds=time.perf_counter() - start)


def measured(phase):
    """Add phase telemetry to an existing handler operation without changing its API."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(host, *args, **kwargs):
            with measured_phase(host, phase):
                return function(host, *args, **kwargs)
        return wrapped
    return decorate
