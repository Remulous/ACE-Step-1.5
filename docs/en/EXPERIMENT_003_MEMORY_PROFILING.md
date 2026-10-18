# Experiment 003 memory diagnostics

This feature only observes inference. It changes no model weights, precision, scheduler,
cache capacity, generation settings or cleanup policy. It is disabled by default.

Set `ACESTEP_MEMORY_PROFILE=timing` to record boundary allocator/driver statistics with
one request peak reset and synchronization at request start and final owner completion.
Set `ACESTEP_MEMORY_PROFILE=attribution` to synchronize and reset allocator peaks at
phase boundaries. Nested peaks are reconstructed from disjoint allocator intervals:
each boundary folds the prior interval maximum into every active parent and the request.
The `request_peak_*` and `interval_aggregate_peak_*` fields are the authoritative aggregates.
A raw `peak_*` field in attribution mode covers only the interval since the prior reset.
Timing mode phase peaks inherit the request high-water mark and must not be interpreted
as isolated phase peaks.

Set `ACESTEP_MEMORY_PROFILE_PATH=/writable/path/profile.jsonl` to append JSONL records.
Without a path, records use Loguru. Output errors are logged; missing data must invalidate
an experimental run rather than being treated as zero memory. Files are opened per record;
no background sampler, tensor inputs/outputs or worker thread is retained by the logger.

Profiling serializes requests across handlers in a process to protect device-global peak
counters. Nested profiling across different handlers is rejected. It does not coordinate
other processes; the benchmark operator must verify exclusive GPU use. Default-disabled
inference has no additional admission lock, probes or CUDA synchronization. Existing
phase residency admission and transfer synchronization remain unchanged.

The actual service worker inherits request identity and retains the request until its own
completion, including caller timeout. Temporary decoder hooks are removed only after all
owners finish. Decoder hooks measure individual forwards, not sampler arithmetic or complete
Heun iterations; the encompassing diffusion span includes those allocations. VAE encode and
decode leaf calls record tile shapes/device/dtype; encode distribution sampling occurs just
after the encode leaf and remains covered by the encompassing source/reference phase.

Component inventories enumerate unique registered parameter/buffer storages by device/dtype.
LM inventories separately expose cache shape/storage, runner direct tensors, configured
blocks, current free/used blocks, and peak used blocks since engine initialization. These
categories can alias (especially runner direct tensors and cache); do not sum them blindly.
A weak LM reference does not retain a prior engine. Scheduler occupancy observation wraps
`schedule()` only when profiling was explicitly enabled during LM initialization. No
scheduler decisions or tensor contents change.

`device_used` is CUDA driver total-minus-free for the device, not process-local NVML use.
An external NVML sampler records GPU-wide memory independently. Driver-used minus allocator
reserved is an accounting residual, not a measured named external allocation. Cached
reserved-minus-allocated memory is not proof of fragmentation. Inspect allocator counters
and snapshots before proposing allocator tuning.

For a separate heavy diagnostic request only, set `ACESTEP_MEMORY_DIAGNOSTIC_TRIGGER` to a
file path and `ACESTEP_MEMORY_SNAPSHOT_DIR` to a writable directory. Create the trigger
before the request. The first decoder forward captures a PyTorch CUDA/CPU profiler trace
and a current allocator snapshot/summary. The trigger is consumed once. Exclude this request
from timing and ordinary peak comparisons; record-shape profiling can retain inputs.

Use staging with a pinned image and isolated output mounts. Preserve production config and
rollback instructions, run one GPU workload at a time, and compare identical inputs under
profiling off/timing/attribution. Keep VAE tile settings equal for residency comparisons.
