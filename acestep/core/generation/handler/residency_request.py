"""Request admission retained until CUDA worker completion, including UI timeouts."""

import contextvars
import functools
import os
import threading
import uuid
from contextlib import contextmanager

from .residency_telemetry import trace

_CURRENT = contextvars.ContextVar("acestep_residency_request", default=None)
_INIT_LOCK = threading.Lock()
_PROFILE_GATE = threading.Lock()


def enabled(host) -> bool:
    """Enable the experimental policy only on CUDA with existing DiT offload."""
    return (
        os.getenv("ACESTEP_DIT_RESIDENCY") == "phase"
        and str(getattr(host, "device", "")).startswith("cuda")
        and getattr(host, "offload_to_cpu", False)
        and getattr(host, "offload_dit_to_cpu", False)
    )


def gate(host):
    """Lazily create one admission lock per shared DiT handler."""
    with _INIT_LOCK:
        if "_residency_admission" not in vars(host):
            host._residency_admission = threading.Lock()
        return host._residency_admission


class Request:
    """Count caller/worker ownership without requiring same-thread lock release."""

    def __init__(self, host):
        """Start with one caller reference; worker references are retained explicitly."""
        self.host = host
        self.id = uuid.uuid4().hex[:12]
        self.lock = threading.Lock()
        self.users = 1
        self.profile = None
        self.profile_gate = False

    def retain(self):
        """Register a worker before starting its thread."""
        with self.lock:
            self.users += 1

    def release(self):
        """Release admission only when caller and every worker have finished."""
        with self.lock:
            self.users -= 1
            if self.users == 0:
                try:
                    if self.profile is not None:
                        self.profile.finish()
                    trace(self.host, "request_end", request=self.id)
                finally:
                    gate(self.host).release()
                    if self.profile_gate:
                        _PROFILE_GATE.release()


@contextmanager
def request_scope(host):
    """Serialize a request, reusing its token in nested calls and workers."""
    current = _CURRENT.get()
    if current is not None and current.host is host:
        yield current
        return
    from .memory_telemetry import enabled as profile_enabled
    profiling = profile_enabled(host)
    if profiling:
        if current is not None:
            raise RuntimeError("Nested profiling across different handlers is unsupported")
        _PROFILE_GATE.acquire()
    try:
        gate(host).acquire()
    except BaseException:
        if profiling:
            _PROFILE_GATE.release()
        raise
    request = Request(host)
    request.profile_gate = profiling
    token = _CURRENT.set(request)
    trace(host, "request_start", request=request.id)
    try:
        from .memory_telemetry import Profile
        if profile_enabled(host):
            request.profile = Profile(host, request.id)
        yield request
    finally:
        _CURRENT.reset(token)
        request.release()


def guarded(function):
    """Guard handler entrypoints only when the experimental policy is enabled."""
    @functools.wraps(function)
    def wrapped(host, *args, **kwargs):
        from .memory_telemetry import enabled as profile_enabled
        if (not enabled(host) and os.getenv("ACESTEP_RESIDENCY_TRACE") != "1"
                and not profile_enabled(host)):
            return function(host, *args, **kwargs)
        with request_scope(host):
            return function(host, *args, **kwargs)
    return wrapped


def worker_target(target):
    """Copy request ownership into a worker; return target and start-failure cleanup."""
    request = _CURRENT.get()
    if request is None:
        return target, lambda: None
    request.retain()
    context = contextvars.copy_context()

    def run():
        try:
            return context.run(target)
        finally:
            request.release()
    return run, request.release


def request_id():
    """Return a correlation identifier for structured logs."""
    request = _CURRENT.get()
    return request.id if request is not None else None
