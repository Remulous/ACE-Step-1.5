# Docker production profiles for RTX PRO 4000 Blackwell

These small Compose overrides select the validated XL Turbo + 4B model family and
backend while leaving the deployment's volumes, networks, secrets, GPU reservation,
health check, labels, and restart policy in the base Compose file.

## Default: Performance (Profile A)

`deploy/profiles/compose.performance.yaml` is the documented default profile:

- DiT: `acestep-v15-xl-turbo`
- LM: `acestep-5Hz-lm-4B`
- nano-vLLM backend and FlashAttention 2
- CPU offloading disabled; DiT remains resident on CUDA
- Original model precision; no quantization
- Turbo UI/API default is eight diffusion steps. Steps and batch size are request
  parameters. The profile uses the supported `--batch_size 1` Gradio default; an
  existing workload may explicitly request a different batch. Use eight steps unless
  the request explicitly overrides them.

The offload environment variables are supported by image `0.1.8`. The performance
override explicitly sets them to false so the model initialization does not apply the
4B LM auto-offload policy.

**Measurement caveat:** Experiment 008's Profile A result (13.907-second short-run
median) used `ACESTEP_OFFLOAD_TO_CPU=true`, while keeping `ACESTEP_OFFLOAD_DIT_TO_CPU=false`.
This repository override explicitly disables general CPU offload as requested here,
so 13.907 seconds is a reference point, not a measurement of this exact override.
Disabling the 4B LM auto-offload may increase peak memory; this profile has not been
benchmarked in that state. Do not deploy it on the 24 GB card until that exact setting
has been validated.

## Optional: Low Memory (Profile D)

`deploy/profiles/compose.low-memory.yaml` is opt-in only. It selects the same XL Turbo
and 4B LM at original precision, enables phase-aware DiT residency and sets
`ACESTEP_NANOVLLM_MAX_KV_BLOCKS=32`. Experiment 008 measured 16.964-second median
short generation, 10.188 GiB lower startup idle VRAM, and 3.064 GiB lower sampled peak
VRAM than its Profile A run.

The installed production image `ghcr.io/ace-step/ace-step-1.5:0.1.8` does **not**
contain the `ACESTEP_DIT_RESIDENCY` or `ACESTEP_NANOVLLM_MAX_KV_BLOCKS` implementation.
The environment values would be ignored by that image; the override must not be applied
to it. The measured Profile D used source revision `1e9fc1df3b7ad64650c9c84e64f35090bf1b4719`
mounted into the same image. This file is a configuration record for a compatible,
reviewed build; the compatible image has not been built or deployed.

## Selecting and reverting a profile

Use the existing production Compose file first and add exactly one profile override.
For the current docker-server stack, keep the existing base file at
`/etc/komodo/stacks/ace-step/docker-server/ace-step/compose.yaml`. Set
`ACE_STEP_REPO` to the checkout containing this repository's `deploy/profiles` files.

Render the resolved configuration before any deployment:

```sh
ACE_STEP_REPO=/path/to/ACE-Step-1.5
ACE_BASE=/etc/komodo/stacks/ace-step/docker-server/ace-step/compose.yaml

docker compose -f "$ACE_BASE" \
  -f "$ACE_STEP_REPO/deploy/profiles/compose.performance.yaml" config
```

To select the low-memory profile on a compatible image, replace
`compose.performance.yaml` with `compose.low-memory.yaml`. Never combine the two
profile files. Confirm the selected image contains both implementations before using
Profile D.

A profile switch takes effect only after `docker compose ... up -d acestep`, which
recreates the service. This command was **not** run as part of this configuration task.

To restore the current deployment's Compose-defined behavior, render and apply the
base file alone:

```sh
docker compose -f "$ACE_BASE" config
docker compose -f "$ACE_BASE" up -d acestep
```

These deployment commands require explicit approval. No production Compose file,
container, image, mount, or runtime setting was changed by this task.
