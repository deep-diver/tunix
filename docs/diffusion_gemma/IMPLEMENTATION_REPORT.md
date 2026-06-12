# DiffusionGemma Hackable Diffusion Wrapper Report

Date: 2026-06-12

Branch: `codex/diffusion-gemma-integration`

## Scope

This report covers only the current Hackable Diffusion backend wrapper in this
fork. The wrapper keeps the official DeepMind Gemma Diffusion, Hackable
Diffusion, and Kauldron SFT implementation as the source of truth, and exposes
it through Tunix-side entrypoints for controlled configuration and validation.

It does not claim that the native Tunix NNX/Qwix DiffusionGemma path is a
drop-in replacement for the official recipe.

## Relevant Files

- `tunix/models/diffusion_gemma/hackable_adapter.py`
- `scripts/run_diffusion_gemma_official_backend.py`
- `scripts/run_diffusion_gemma_official_reference.py`
- `scripts/run_diffusion_gemma_h100x2_comparison_job.sh`
- `scripts/summarize_diffusion_gemma_training.py`

## What The Wrapper Does

The official backend remains responsible for model construction, PubMedQA/Sudoku
recipe construction, token corruption, diffusion timestep sampling, denoising
loss, self-conditioning, encoder loss, LoRA wrapping, optimizer construction,
and sharding.

The Tunix wrapper is responsible for importing that backend inside this fork,
applying run-environment overrides, launching comparable official-vs-wrapper
runs, and collecting concise logs plus GPU memory telemetry.

## Local Parity Evidence

The local parity checks pass against official source checkouts:

- `scripts/verify_diffusion_gemma_official_parity.py`
  - positions: max diff `0.0`
  - causal prefill mask: max diff `0.0`
  - decoder attention mask: max diff `0.0`
  - cache `end_index`: max diff `0.0`
  - uniform time sampling: max diff `0.0`
  - categorical corruption tokens/mask: max diff `0.0`
  - unweighted discrete loss mean: max diff `0.0`
- `scripts/verify_diffusion_gemma_official_logits.py`
  - official Gemma revision: `682e412`
  - copied tiny-model parameter leaves: `19`, all copied with max diff `0.0`
  - `encode_logits`: max diff `0.0`
  - plain logits: max diff `2.8405338525772095e-08`
  - self-conditioned logits: max diff `0.0`

These checks validate the deterministic helper and tiny-model math paths. They
do not prove full 26B training completion.

## H100 x2 Comparison

Two H100 x2 JarvisLabs VMs were used for side-by-side validation:

- Official reference VM: `425566`
- Tunix wrapper VM: `425567`
- Both VMs were destroyed after the run. Final `jl status --json` reported
  `running_instances: 0` and `running_vms: 0`.

Both runs used the same public DiffusionGemma checkpoint, official PubMedQA
recipe geometry, LoRA rank `4`, batch size `2`, JAX `0.10.1`, CUDA 12 wheels,
and NCCL `2.30.7`.

The official reference runner and the Tunix wrapper both reached dependency
setup, dataset preparation, JAX two-device visibility, `pmap` psum validation,
config construction, checkpoint restore, and model load. Both discarded the
same 17 checkpoint-only vision/mm keys.

An asynchronous 1-step hybrid run returned completion markers for both sides:

- Official reference: `r_be358b08`
- Tunix wrapper: `r_3f109764`

That is not sufficient proof of completed training, because JAX execution is
asynchronous. A stricter rerun forced synchronization after the train step:

- Official reference strict sync: `r_3b9a2ca7`
- Tunix wrapper strict sync: `r_b4405817`

Both strict-sync runs failed at the same point:

```text
jax.errors.JaxRuntimeError: INTERNAL: NCCL operation ncclAllGather(...) failed:
invalid argument ... 'lib wrapper not initialized.' [executable_name='jit_step']
```

The failing call was train-step synchronization, not Tunix-native NNX code. The
same failure appeared in the standalone official-reference runner and in the
Tunix wrapper that delegates to the official backend.

## Current Verdict

The wrapper correctly imports and configures the official DiffusionGemma backend
and matches official helper/logit parity checks, but the H100 x2 synchronized
26B training step is not verified in the current JarvisLabs runtime.

This is not evidence that the Tunix wrapper has diverged from the official
backend. It is evidence that the official Hackable Diffusion train step, when
driven under this H100 x2 JAX/NCCL setup and forced to synchronize, fails inside
`jit_step` NCCL `allGather`.

Treat this integration as a reference wrapper and parity scaffold, not as a
ready H100 x2 training path.

## Next Work

- Re-run the strict official reference with a JAX/NCCL version matrix instead of
  only `jax[cuda12]==0.10.1`.
- Compare against the official Kauldron loop with an explicit synchronization
  point that cannot be satisfied by asynchronous dispatch alone.
- Keep loss/metric logging separate from train-step completion checks, because
  host loss materialization can trigger additional collectives.
- Use this official wrapper as the oracle while native NNX/Qwix work continues.
