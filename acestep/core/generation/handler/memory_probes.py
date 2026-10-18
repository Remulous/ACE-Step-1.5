"""Leaf-call metadata and actual model transfer diagnostics, disabled by default."""

import torch

from .memory_inventory import model_inventory
from .memory_telemetry import current, emit, enabled, phase


def call(host, name, function, *args, **kwargs):
    """Observe a leaf operation without retaining its inputs or outputs."""
    if enabled(host):
        profile = current()
        if profile is not None:
            emit("call_metadata", request=profile.request, phase=name,
                 tensors=[{"shape": list(value.shape), "dtype": str(value.dtype),
                           "device": str(value.device)} for value in args
                          if isinstance(value, torch.Tensor)])
    with phase(host, name):
        return function(*args, **kwargs)


def transfer(function):
    """Describe an actual requested transfer, including whether storage placement changes."""
    import functools

    @functools.wraps(function)
    def wrapped(host, model, device, dtype=None):
        if not enabled(host) or current() is None:
            return function(host, model, device, dtype)
        component = next((key for key in ("model", "vae", "text_encoder")
                          if getattr(host, key, None) is model), "other")
        before = model_inventory(model)
        with phase(host, f"transfer_{component}_to_{device}"):
            result = function(host, model, device, dtype)
        emit("transfer", request=current().request, component=component,
             destination=str(device), before=before, after=model_inventory(model))
        return result
    return wrapped
