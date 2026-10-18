"""Default-off memory profiling with worker-owned, nested peak interval accounting."""

import functools
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager

import torch
from loguru import logger

from .memory_inventory import inventory, register_lm, track_cache_occupancy

_WRITE_LOCK = threading.Lock()


def mode():
    """Return the explicitly selected mode; unknown values disable profiling."""
    value = os.getenv("ACESTEP_MEMORY_PROFILE", "off")
    return value if value in ("timing", "attribution") else "off"


def enabled(host):
    """Only enable CUDA instrumentation when explicitly requested."""
    return mode() != "off" and str(getattr(host, "device", "")).startswith("cuda")


def emit(event, **fields):
    """Append a versioned record; failures are visible and never suppress inference errors."""
    path = os.getenv("ACESTEP_MEMORY_PROFILE_PATH")
    record = {"schema": 1, "event": event, "time": time.time(),
              "monotonic": time.perf_counter(), "pid": os.getpid(),
              "thread": threading.get_ident(), "mode": mode(), **fields}
    try:
        line = json.dumps(record, sort_keys=True, default=str)
        with _WRITE_LOCK:
            if path:
                with open(path, "a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            else:
                logger.info("[memory-profile] {}", line)
    except (OSError, TypeError, ValueError) as exc:
        logger.error("Memory profiling output failed: {}", exc)


def sample(device):
    """Read allocator and driver accounting; driver usage includes all CUDA processes."""
    free, total = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    return {"allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
            "device_used": total - free, "device_total": total,
            "allocator": {key: stats.get(key) for key in (
                "inactive_split_bytes.all.current", "inactive_split_bytes.all.peak",
                "active_bytes.all.current", "segment.all.current", "num_alloc_retries",
                "num_ooms", "requested_bytes.all.current")}}


class Profile:
    """Accumulate disjoint allocator intervals into every live parent span."""

    def __init__(self, host, request):
        """Initialize before request work; request admission excludes competing resets."""
        self.host, self.request = host, request
        self.spans, self.counter = [], 0
        self.lock = threading.RLock()
        torch.cuda.synchronize(host.device)
        torch.cuda.reset_peak_memory_stats(host.device)
        from .memory_diagnostics import install_forward_probes
        self.handles = install_forward_probes(host, self)
        self.start = time.perf_counter()
        self.peaks = [0, 0]
        emit("request_start", request=request, **sample(host.device), models=inventory(host))

    def boundary(self):
        """Fold the prior interval into parents before any nested reset occurs."""
        if mode() == "attribution":
            torch.cuda.synchronize(self.host.device)
        values = sample(self.host.device)
        for target in [self.peaks] + [span["peaks"] for span in self.spans]:
            target[0] = max(target[0], values["peak_allocated"])
            target[1] = max(target[1], values["peak_reserved"])
        if mode() == "attribution":
            torch.cuda.reset_peak_memory_stats(self.host.device)
        return values

    @contextmanager
    def phase(self, name):
        """Measure a nested operation, preserving errors and worker ownership."""
        with self.lock:
            values = self.boundary()
            self.counter += 1
            span = {"id": self.counter, "phase": name, "peaks": [values["allocated"],
                     values["reserved"]], "start": time.perf_counter()}
            self.spans.append(span)
            emit("phase_start", request=self.request, span=span["id"], phase=name, **values)
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            with self.lock:
                values = self.boundary()
                self.spans.remove(span)
                emit("phase_end", request=self.request, span=span["id"], phase=name,
                     seconds=time.perf_counter() - span["start"], error=error,
                     interval_aggregate_peak_allocated=span["peaks"][0],
                     interval_aggregate_peak_reserved=span["peaks"][1], **values,
                     models=inventory(self.host) if name in (
                         "diffusion", "dit_prepare_condition", "lm", "vae_decode") else None)

    def finish(self):
        """Read final peaks only after the caller and all actual workers release ownership."""
        with self.lock:
            for handle in self.handles:
                handle.remove()
            torch.cuda.synchronize(self.host.device)
            values = self.boundary()
            emit("request_end", request=self.request, seconds=time.perf_counter() - self.start,
                 request_peak_allocated=self.peaks[0], request_peak_reserved=self.peaks[1],
                 **values, models=inventory(self.host))


def current():
    """Return the profiler inherited by the live service worker, if present."""
    from .residency_request import _CURRENT
    return getattr(_CURRENT.get(), "profile", None)


@contextmanager
def phase(host, name):
    """Be a no-op with profiling disabled or outside a request."""
    profile = current() if enabled(host) else None
    if profile is None:
        yield
    else:
        with profile.phase(name):
            yield


def measured(name):
    """Decorate a handler operation without changing its public contract."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(host, *args, **kwargs):
            with phase(host, name):
                return function(host, *args, **kwargs)
        return wrapped
    return decorate


def initialize_lm(function):
    """Register the initialized engine for a read-only process-lifetime census."""
    @functools.wraps(function)
    def wrapped(host, *args, **kwargs):
        if mode() == "off":
            return function(host, *args, **kwargs)
        register_lm(host)
        result = function(host, *args, **kwargs)
        if not enabled(host):
            return result
        track_cache_occupancy(host)
        emit("lm_initialized", models=inventory(host), **sample(host.device))
        return result
    return wrapped



def milestone(name):
    """Record component initialization endpoints when no request profiler exists."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(host, *args, **kwargs):
            if not enabled(host):
                return function(host, *args, **kwargs)
            start = time.perf_counter()
            emit(name + "_start", **sample(host.device))
            result = function(host, *args, **kwargs)
            emit(name + "_end", seconds=time.perf_counter() - start,
                 models=inventory(host), **sample(host.device))
            return result
        return wrapped
    return decorate
