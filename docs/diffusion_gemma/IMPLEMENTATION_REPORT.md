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

One H100 x2 JarvisLabs VM was used for sequential official-vs-wrapper
validation:

- VM: `425813`, H100 80GB x2, region `IN2`.
- The VM was destroyed after the run.

Both runs used the same public DiffusionGemma checkpoint, official PubMedQA
recipe geometry, LoRA rank `4`, batch size `2`, prompt length `1024`, two
canvases of size `128`, JAX `jax[cuda13]==0.10.1`, and the same official Gemma
and Hackable Diffusion revisions.

The runner follows the official DiffusionGemma runtime guidance for CUDA13 JAX:

- initialize JAX and call `jax.devices()` before TensorFlow/Kauldron imports;
- set `XLA_FLAGS="--xla_disable_hlo_passes=constant_folding"`;
- set `NCCL_ALGO=Ring`, `NCCL_PROTO=LL128`, `NCCL_NVLS_ENABLE=0`, and
  `NCCL_CUMEM_ENABLE=0`;
- synchronize by reading addressable loss shards instead of forcing a full
  post-step state all-gather.

The official reference and Tunix wrapper both completed one synchronized 26B
PubMedQA LoRA train step:

| Path | Run id | Exit | Total loss | Diffusion loss | Encoder loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Official reference | `r_a7c3063e` | `0` | `13.671875` | `6.421875` | `7.25` |
| Tunix wrapper | `r_f9bdce4c` | `0` | `13.40625` | `6.15625` | `7.25` |

The loss values are finite but not expected to be bit-identical because the two
runs were independent stochastic runs, not a shared-seed deterministic parity
job. Peak sampled HBM was `61459/61363` MiB for the official reference and
`63005/62979` MiB for the Tunix wrapper.

Evidence is stored in
`evidence/diffusion_gemma/h100x2_cuda13_preinit_2026-06-12.md` and the sibling
JSON/CSV artifacts.

## Current Verdict

The wrapper correctly imports, configures, and executes the official
DiffusionGemma backend on H100 x2 when the official CUDA13/JAX runtime guidance
is applied. It matches official helper/logit parity checks locally and completes
a real 26B PubMedQA LoRA train step through the Tunix wrapper path.

Treat this integration as a practical official-backend wrapper plus parity
scaffold. It is not yet a claim that the native Tunix NNX/Qwix training graph
has matched the official 2-GPU memory profile.

## Next Work

- Add a shared-seed deterministic GPU comparison if exact stochastic loss parity
  is required.
- Compare the hybrid wrapper against a longer official Kauldron run now that the
  CUDA13/JAX preinit setup is known-good.
- Keep the official wrapper as the oracle while native NNX/Qwix memory work
  continues.
