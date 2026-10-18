"""Request-scoped forward probes and optional heavy allocator diagnostics."""

import os
import pickle
from pathlib import Path

import torch


def install_forward_probes(host, profile):
    """Observe actual DiT decoder forwards without modifying checkpoint source files."""
    decoder = getattr(getattr(host, "model", None), "decoder", None)
    if decoder is None:
        return []
    stack, index = [], [0]
    trigger = os.getenv("ACESTEP_MEMORY_DIAGNOSTIC_TRIGGER")
    diagnostic = bool(trigger and Path(trigger).exists())
    if diagnostic:
        Path(trigger).unlink()
    capture = [None]

    def before(module, inputs):
        index[0] += 1
        if diagnostic and index[0] == 1:
            capture[0] = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True, record_shapes=True, with_stack=False)
            capture[0].__enter__()
        scope = profile.phase(f"diffusion_forward_{index[0]}")
        scope.__enter__()
        stack.append(scope)

    def after(module, inputs, output):
        if stack:
            stack.pop().__exit__(None, None, None)
        if capture[0] is not None:
            capture[0].__exit__(None, None, None)
            directory = Path(os.environ["ACESTEP_MEMORY_SNAPSHOT_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            capture[0].export_chrome_trace(str(directory / f"{profile.request}-forward.json"))
            capture[0] = None
            snapshot(host, profile.request)

    return [decoder.register_forward_pre_hook(before),
            decoder.register_forward_hook(after, always_call=True)]


def snapshot(host, name):
    """Write a separately requested diagnostic snapshot, never in normal timing mode."""
    directory = os.getenv("ACESTEP_MEMORY_SNAPSHOT_DIR")
    if not directory:
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize(host.device)
    with open(Path(directory) / f"{name}.pickle", "wb") as stream:
        pickle.dump(torch.cuda.memory_snapshot(), stream)
    (Path(directory) / f"{name}.txt").write_text(torch.cuda.memory_summary(host.device))
