# DiffusionGemma Hackable Diffusion Wrapper Report

Date: 2026-06-13

Branch: `codex/diffusion-gemma-native-port`

## Scope

This report covers only the current Hackable Diffusion backend wrapper in this
fork. The wrapper keeps the official DeepMind Gemma Diffusion, Hackable
Diffusion, and Kauldron SFT implementation as the source of truth, and exposes
it through Tunix-side entrypoints for controlled configuration and validation.

It does not claim that the native Tunix NNX/Qwix DiffusionGemma path is a
drop-in replacement for the official recipe. The current H100 x2 practical path
is the official Hackable Diffusion backend wrapped by Tunix.

## Relevant Files

- `tunix/models/diffusion_gemma/hackable_adapter.py`
- `scripts/run_diffusion_gemma_official_backend.py`
- `scripts/run_diffusion_gemma_official_reference.py`
- `scripts/run_diffusion_gemma_h100x2_comparison_job.sh`
- `scripts/generate_diffusion_gemma_official_backend.py`
- `scripts/render_diffusion_gemma_trace_gif.py`
- `scripts/summarize_diffusion_gemma_training.py`

## What The Wrapper Does

The official backend remains responsible for model construction, PubMedQA/Sudoku
recipe construction, token corruption, diffusion timestep sampling, denoising
loss, self-conditioning, encoder loss, LoRA wrapping, optimizer construction,
and sharding.

The Tunix wrapper is responsible for importing that backend inside this fork,
applying run-environment overrides, launching comparable official-vs-wrapper
runs, and collecting concise logs plus GPU memory telemetry.

The wrapper can preserve the official LoRA layer or replace just that layer with
Tunix/Qwix LoRA via `lora_backend="qwix_lora"`. Qwix QLoRA is not exposed for
DiffusionGemma; `lora_backend="qwix_qlora"` is rejected.

In `train_loop=hybrid` mode the wrapper still uses the official model, dataset,
optimizer, loss, train step, checkpoint restore, and sharding. Tunix only drives
the outer step loop and reads addressable loss shards so that GPU validation can
avoid Kauldron post-step metric/final-sync paths that are fragile on some
multi-GPU Jarvis runtimes.

Minimal Tunix-facing usage:

```python
from tunix.models.diffusion_gemma import (
    OfficialDiffusionGemmaTrainer,
    OfficialSFTConfig,
)

trainer = OfficialDiffusionGemmaTrainer(
    OfficialSFTConfig(
        recipe="pubmedqa",
        gemma_ref="/home/ubuntu/gemma_official_reference",
        hackable_diffusion_ref="/home/ubuntu/hackable_diffusion_reference",
        checkpoint_path="/home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it",
        workdir="/home/ubuntu/diffusion_gemma_pubmedqa_wrapper",
        num_train_steps=2000,
        lora_rank=4,
        lora_backend="qwix_lora",
        train_loop="hybrid",
        sync_after_step="losses",
        log_losses=True,
        save_final_checkpoint=True,
        disable_evals=True,
    )
)
trainer.train()
```

Equivalent CLI usage for the validated H100 x2 official-LoRA recipe:

```bash
SAVE_FINAL_CHECKPOINT=true \
TRAIN_LOOP=hybrid \
LOG_LOSSES=true \
SYNC_AFTER_STEP=losses \
XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
bash scripts/run_diffusion_gemma_h100x2_comparison_job.sh \
  --mode tunix \
  --max_runtime_seconds 10800 \
  --num_train_steps 2000 \
  --lora_backend official \
  --log_losses true \
  --sync_after_step losses \
  --jax_package_spec "jax[cuda13]==0.10.1" \
  --run_name tunix_wrapper_2000step
```

Qwix LoRA-only validation uses the same official backend and swaps only the
LoRA layer:

```bash
LORA_RANK=4 LOG_PARAM_SUMMARY=true \
bash scripts/run_diffusion_gemma_h100x2_comparison_job.sh \
  --mode tunix \
  --run_name qwix_lora_only_10step \
  --lora_backend qwix_lora \
  --run_steps 10 \
  --num_train_steps 2000 \
  --max_runtime_seconds 7200 \
  --sync_after_step losses \
  --log_losses true \
  --encoder_loss_token_chunk_size 128 \
  --gpu_poll_seconds 10
```

The latest Qwix LoRA-only H100x2 validation succeeded:

- Run: `r_05256f15` on machine `426095` (`H100`, `IN2`, 2 GPUs, VM), exit `0`.
- Steps: `10/10` with finite loss values.
- Total loss: step 1 `13.511953353881836`, step 10
  `14.651562213897705`, min/max `12.34370231628418` /
  `14.945141792297363`.
- Qwix LoRA inventory after restore: `1092` LoRA leaves,
  `0.02474355697631836GiB`; dense frozen base leaves: `608`,
  `47.033628053963184GiB`.
- Peak HBM: GPU0 `65665MiB`, GPU1 `65635MiB`.

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

## H100 x2 One-Step Bring-Up

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

## H100 x2 2000-Step Comparison

The longer comparison was run under matched H100 80GB x2 conditions with the
same official Gemma and Hackable Diffusion revisions:

- Official Gemma revision:
  `a7e33454206ca24984565992f5c903191baecc22`
- Hackable Diffusion revision:
  `03ce88ed0acec3b17f4f284502a6053c5f2b3a15`
- JAX package: `jax[cuda13]==0.10.1`
- Training loop: `hybrid`
- Loss synchronization: `sync_after_step=losses`
- Public recipe: PubMedQA, batch size `2`, LoRA rank `4`, prompt length
  `1024`, two canvases of size `128`, `2000` train steps

Both runs completed all `2000` steps with finite losses and no OOM, NCCL,
traceback, or runtime-error events in the logs:

| Path | Run id | Steps | First total | Final total | Last-50 total mean | Peak HBM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Official reference | `r_5c5a20f2` | `2000` | `13.921875` | `2.07421875` | `1.872822265625` | `65661/65631` MiB |
| Tunix wrapper | `r_014dadd5` | `2000` | `13.921875` | `2.099609375` | `1.87337890625` | `65659/65629` MiB |

The first loss is identical. The final total-loss delta is `0.025390625`
(`~1.22%` of the official final loss), while the last-50 total-loss means differ
by only `0.000556640625`. Peak memory is effectively identical.

The runs are independent stochastic training jobs, so exact step-by-step loss
identity is not expected. The result is practical training parity for the
official-backend wrapper: same recipe, same backend revisions, same runtime
knobs, same 2-GPU HBM profile, same successful 2000-step completion.

Local artifacts for this comparison are under
`evidence/diffusion_gemma/h100x2_2000step_comparison_2026-06-13/`.

## Generation From A Tuned Checkpoint

The wrapper includes a generation helper for checkpoints produced by the
official backend path:

```bash
python scripts/generate_diffusion_gemma_official_backend.py \
  --gemma_ref /home/ubuntu/gemma_official_reference \
  --hackable_diffusion_ref /home/ubuntu/hackable_diffusion_reference \
  --workdir /home/ubuntu/diffusion_gemma_run/workdir \
  --checkpoint_path /home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it \
  --step 2000 \
  --denoising_steps 8 \
  --max_num_canvases 1 \
  --tokenizer_path /home/ubuntu/checkpoints/tokenizers/tokenizer_gemma4.model \
  --output_json generation.json \
  --trace_json generation_trace.json
```

`generation.json` records the decoded prompt and generated text. The optional
`generation_trace.json` can be rendered as a GIF:

```bash
python scripts/render_diffusion_gemma_trace_gif.py generation_trace.json \
  --output diffusion_gemma_tuned_generation.gif \
  --width 1280 \
  --height 720 \
  --final_hold_frames 3
```

The trace uses the generated token ids from the restored checkpoint. Intermediate
frames are a denoising-style reveal visualization because the current official
AR sampler returns final generated tokens but does not expose every internal
diffusion trajectory frame.

## Current Verdict

The wrapper correctly imports, configures, and executes the official
DiffusionGemma backend on H100 x2 when the official CUDA13/JAX runtime guidance
is applied. It matches official helper/logit parity checks locally and completes
matched 2000-step 26B PubMedQA LoRA training through the Tunix wrapper path with
near-identical loss and memory behavior to the official reference runner.

Treat this integration as a practical official-backend wrapper plus parity
scaffold. For H100 x2 use, prefer the wrapper path. The native NNX/Qwix path is
useful for model-family work and parity development, but it is still
experimental for multi-step public 26B training on H100 80GB x2.

## Next Work

- Add a shared-seed deterministic GPU comparison if exact stochastic loss parity
  is required.
- Keep the official wrapper as the oracle while native NNX/Qwix memory work
  continues.
