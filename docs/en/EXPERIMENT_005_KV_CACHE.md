# Experiment 005: nano-vLLM KV-cache sizing

The optional `ACESTEP_NANOVLLM_MAX_KV_BLOCKS` environment variable sets a maximum
number of nano-vLLM KV blocks during LM initialization. It is unset by default, so the
existing memory-derived sizing behavior is unchanged. It applies only to the nano-vLLM
backend. The configured block maximum is an upper bound: the allocator still limits the
selected count to the memory-derived safe count.

The limit is applied once, before CUDA graphs are captured. KV storage is not resized or
moved after initialization. If memory permits fewer than one block, initialization fails.
If the resulting pool cannot satisfy a request, nano-vLLM reports an admission error; the
LM context length is not silently changed. A block pool smaller than the request's total
sequence requirement can lower batched concurrency or reject a long request.

When `ACESTEP_MEMORY_PROFILE=timing` or `attribution` is set, request-end model inventory
includes `lm.generate_cache_profiles`. Each nano-vLLM generation call records prompt token
lengths, maximum scheduled token length per sequence, completion token lengths, peak
simultaneously occupied physical blocks, pool capacity, final occupancy, and any exception
type. This adds Python bookkeeping at scheduler boundaries and does not synchronize CUDA.
Profiling remains disabled by default.

Candidate caps must be benchmarked with the actual prompts, generated durations, CFG mode,
batch sizes, and supported long-audio workflows. The observed E003 high-water mark is not
by itself a safe production capacity: those traces did not preserve token lengths per call.

## RTX PRO 4000 Blackwell planning values

E003 measured a 36 MiB byte size per block for the 4B 5Hz LM. Using that measured size,
theoretical pool reductions from a 64-block control are:

| Maximum blocks | Pool capacity | Theoretical pool-memory reduction |
|---:|---:|---:|
| 48 | 12,288 tokens | 576 MiB |
| 32 | 8,192 tokens | 1,152 MiB |
| 16 | 4,096 tokens | 1,728 MiB |

These are capacity arithmetic, not measured device-wide savings. The staged benchmark must
report actual allocated/reserved and GPU-wide deltas and validate admission capacity.
