# Experimental phase-aware DiT residency

This opt-in CUDA policy keeps the existing XL Turbo and LM weights and uses the
existing CPU offload infrastructure. It does not change nano-vLLM cache sizing,
attention kernels, sampler settings, VAE tiling policy, or model selection.

Enable both existing offload options, then set:

```sh
ACESTEP_DIT_RESIDENCY=phase
```

For the Gradio Docker entrypoint in release 0.1.8, pass the existing CLI options:

```sh
ACESTEP_EXTRA_ARGS="--offload_to_cpu true --offload_dit_to_cpu true"
```

The API's existing offload environment options continue to work as before.
The phase policy is inactive unless both offload settings are enabled on CUDA.
Unset the new flag to restore the existing behavior. MPS, XPU, CPU and persistent
DiT operation retain their existing paths.

## Execution

Reference VAE encoding and prompt/lyric embedding preparation run before the
DiT is needed. Target preparation loads the VAE only for real source encoding;
code-only and silent targets do not need VAE weights. Consecutive audio-code
helper calls and diffusion share DiT residency inside the service worker.
The final phase exit moves DiT to CPU before output VAE decoding.

Every model context uses the same reentrant execution lock in phase mode. DiT
leases count nesting and verify every parameter and buffer's actual device.
A VAE context requires zero active DiT leases and releases retained DiT residency
before loading VAE. Mixed conditioning inputs can still need multiple windows
when dependency ordering requires additional VAE work.

Request admission spans inference (including LM work), handler generation, and
postprocessing. The service worker receives an explicit retained request token.
A caller timeout or cancellation does not release that token while the worker
is alive. Context cleanup runs in the worker's `finally`; a subsequent request
waits for actual completion. This deliberately does not kill CUDA worker threads.
An indeterminate interruption during thread startup conservatively retains
admission rather than risk admitting work alongside a potentially started worker.
A stuck worker or indeterminate startup can require restarting the process.

An execution exception restores CPU DiT residency. A failed device transfer
marks the manager unsafe and rejects subsequent transfers until process restart.
This avoids continuing with partially moved weights.

## Instrumentation

Set `ACESTEP_RESIDENCY_TRACE=1` to emit structured `[residency]` JSON events for
requests, operations, components and actual DiT transfers. Values include time,
request ID, allocated/reserved CUDA memory, peak allocator values, and whole-device
used/total capacity from CUDA. Use an external NVML/nvidia-smi sampler alongside
these logs for GPU-level peaks.

Tracing synchronizes measured operation endpoints and resets allocator peak
counters at operation start. Therefore those peaks describe the operation, not
the entire process lifetime; aggregate operation/transfer records to reconstruct
request maxima. The existing process peak debug display is affected while tracing
is enabled. Disable tracing for ordinary use and benchmark it consistently across
conditions. Transfer logs distinguish copy/synchronization time from time including
GC, allocator cache release and host heap cleanup.

## Validation and limits

Use fixed audio codes as well as fixed diffusion seeds for paired comparisons:
LM sampling alone is not made deterministic by the diffusion seed. Compare saved
latents separately from final WAVs, because existing adaptive VAE tile selection
can change when memory headroom changes.

Validate nested leases, concurrent callers, computation failures, cancellation,
thread startup failures and caller timeout with a continuing worker. Validate normal
LM generation, code replay, source/reference conditioning, batches and full-duration
repaint on the target deployment. Serialize access to the GPU across containers.
The admission guard is per handler/process; it does not coordinate other processes
or authorize concurrent training or model/adapter mutation during generation.

Keep production on its preserved image/configuration until the experimental build
has been reviewed and explicitly adopted. Roll back staging by stopping it before
restarting the preserved production container. Source overlays are experimental and
must be recorded alongside the underlying image digest and source revision.
