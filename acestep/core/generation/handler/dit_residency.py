"""Serialized, nested DiT leases with retention between consecutive phase users."""

import functools
import threading
import time
from contextlib import contextmanager

import torch

from .residency_request import enabled, request_id, request_scope
from .residency_telemetry import trace

_INIT_LOCK = threading.Lock()


def manager(host):
    """Return the handler's single residency manager."""
    with _INIT_LOCK:
        if "_dit_residency" not in vars(host):
            host._dit_residency = DiTResidency(host)
        return host._dit_residency


class DiTResidency:
    """Hold a reentrant execution lock and offload only after the final lease."""

    def __init__(self, host):
        """Create state for one handler; model mutation requires this execution lock."""
        self.host = host
        self.lock = threading.RLock()
        self.depth = 0
        self.phases = 0
        self.poisoned = False

    def _on_device(self, device):
        """Inspect every parameter and buffer rather than trusting cached placement."""
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        tensors = list(self.host.model.parameters()) + list(self.host.model.buffers())
        return all(tensor.device == target for tensor in tensors)

    def move(self, device):
        """Move only when necessary; poison the manager after failed restoration."""
        if self.poisoned:
            raise RuntimeError("DiT residency is unsafe after a failed transfer; restart required")
        if self._on_device(device):
            return
        trace(self.host, "transfer_start", request=request_id(), destination=str(device))
        start = time.perf_counter()
        try:
            self.host._synchronize()
            self.host._recursive_to_device(
                self.host.model, device, self.host.dtype if str(device) != "cpu" else None
            )
            self.host._synchronize()
            if not self._on_device(device):
                raise RuntimeError(f"DiT parameters/buffers did not reach {device}")
        except BaseException:
            self.poisoned = True
            trace(self.host, "transfer_failed", request=request_id(), destination=str(device))
            raise
        elapsed = time.perf_counter() - start
        self.host.current_offload_cost += elapsed
        if str(device) != "cpu" and hasattr(self.host, "silence_latent"):
            self.host.silence_latent = self.host.silence_latent.to(device).to(self.host.dtype)
        self.host._release_system_memory()
        trace(self.host, "transfer_end", request=request_id(), destination=str(device),
              seconds=elapsed, cleanup_inclusive_seconds=time.perf_counter() - start)

    @contextmanager
    def lease(self):
        """Serialize execution and keep the outer model resident across nested leases."""
        with self.lock:
            self.move(self.host.device)
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1
                if self.depth == 0 and self.phases == 0:
                    self.move("cpu")

    @contextmanager
    def phase(self):
        """Retain DiT after helper leases, releasing it in the actual worker finally."""
        with self.lock:
            self.phases += 1
            trace(self.host, "phase_start", request=request_id(), phase="conditioning_diffusion")
            try:
                yield
            finally:
                self.phases -= 1
                if self.phases == 0:
                    if self.depth:
                        raise RuntimeError("DiT phase ended with active leases")
                    self.move("cpu")
                trace(self.host, "phase_end", request=request_id(), phase="conditioning_diffusion")

    def before_vae(self):
        """Ensure VAE work never starts while a DiT execution lease is active."""
        if self.depth:
            raise RuntimeError("VAE encoding/decoding requested inside an active DiT lease")
        self.move("cpu")


def service_phase(function):
    """Keep residency in the service worker, never in the timeout-waiting caller."""
    @functools.wraps(function)
    def wrapped(host, *args, **kwargs):
        if not enabled(host):
            return function(host, *args, **kwargs)
        with request_scope(host), manager(host).phase():
            return function(host, *args, **kwargs)
    return wrapped
