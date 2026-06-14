# DiffusionGemma Tunix Status and Runbook

Date: 2026-06-14

This document summarizes the current DiffusionGemma work in this fork: what is
implemented, what has been measured on GPU, how to use it, and what remains
before this becomes a fully native Tunix model family.

## Quick Start

The intended user-facing path is a Tunix-style trainer config, while the heavy
DiffusionGemma math still comes from the official Hackable Diffusion backend:

```python
from tunix.models.diffusion_gemma import (
    DiffusionGemmaOfficialLossConfig,
    DiffusionGemmaQwixLoRAConfig,
    OfficialDiffusionGemmaTrainer,
)

trainer = OfficialDiffusionGemmaTrainer.from_pubmedqa(
    checkpoint_path="/home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it",
    workdir="/home/ubuntu/diffusion_gemma_runs/pubmedqa_qwix_lora",
    num_train_steps=2000,
    run_steps=2000,
    save_final_checkpoint=True,
    peft_config=DiffusionGemmaQwixLoRAConfig(rank=4),
    loss_config=DiffusionGemmaOfficialLossConfig(
        train_loop="hybrid",
        sync_after_step="losses",
        log_losses=True,
        encoder_loss_token_chunk_size=128,
    ),
)

trainer.train()

ckpt = trainer.checkpoint_info("latest")
generation_kwargs = trainer.generation_kwargs("latest")
print(ckpt.as_dict())
print(generation_kwargs)
```

`from_pubmedqa()` selects the built-in official DiffusionGemma PubMedQA SFT
recipe. The official `gemma` and `hackable_diffusion` packages are expected to
already be importable in the runtime. If they are not installed as packages,
pass local source checkout paths explicitly:

```python
trainer = OfficialDiffusionGemmaTrainer.from_pubmedqa(
    gemma_ref="/path/to/google-deepmind/gemma",
    hackable_diffusion_ref="/path/to/hackable_diffusion",
    checkpoint_path="/path/to/diffusiongemma-26B-A4B-it",
    workdir="/path/to/run",
    peft_config=DiffusionGemmaQwixLoRAConfig(rank=4),
    loss_config=DiffusionGemmaOfficialLossConfig(),
)
```

For custom data, provide normal prompt/response records and let the
DiffusionGemma data adapter build `prompt`, `canvas`, `canvas_id`,
`canvas_mask`, and encoder targets:

```python
from tunix.models.diffusion_gemma import data as dg_data

batch_config = dg_data.DiffusionGemmaTextBatchConfig(
    prompt_len=1024,
    canvas_size=128,
    num_canvases=2,
    pad_id=0,
    eos_id=1,
    bos_id=2,
)

train_ds = dg_data.make_sft_dataset_from_jsonl(
    "train.jsonl",
    tokenizer=tokenizer,
    config=batch_config,
    batch_size=2,
    as_model_inputs=True,
)
```

## Current Verdict

The validated H100 x2 path is:

```text
Tunix wrapper -> official Hackable Diffusion backend -> Qwix LoRA bridge
```

This path keeps the official DeepMind Flax/Linen model, Hackable Diffusion
losses, Kauldron recipe objects, optimizer, sharding, checkpoint restore, and
sampling logic. Tunix owns the user-facing config, run entrypoints, telemetry,
checkpoint/generation helpers, data-adapter surface, and the optional Qwix LoRA
replacement.

It is usable for PubMedQA LoRA SFT on H100 80GB x2. It is not yet a full native
NNX rewrite of the official DiffusionGemma recipe.

## Implementation Layers

| Layer | Status | Use it for |
| --- | --- | --- |
| Upstream official runner | Validated | Reference behavior and official baseline runs |
| Tunix official-backend wrapper, official LoRA | Validated | Tunix-managed runs while preserving official LoRA exactly |
| Tunix official-backend wrapper, Qwix LoRA | Validated | Current recommended Tunix-facing LoRA path |
| Tunix-native NNX model/loss/data/generation | In progress | Parity work, small tests, future model-family integration |

The practical distinction:

- **Official upstream** answers "does the official recipe run here?"
- **Tunix wrapper + official LoRA** answers "can Tunix manage the official
  recipe without changing model math?"
- **Tunix wrapper + Qwix LoRA** answers "can Tunix replace the adapter layer
  while preserving the official DiffusionGemma training stack?"
- **Native NNX path** is the long-term target, but H100 x2 production training
  should still use the official-backend wrapper today.

## User-Facing API

### Official Backend With Qwix LoRA

```python
from tunix.models.diffusion_gemma import (
    DiffusionGemmaOfficialLossConfig,
    DiffusionGemmaQwixLoRAConfig,
    OfficialDiffusionGemmaTrainer,
)

trainer = OfficialDiffusionGemmaTrainer.from_pubmedqa(
    gemma_ref="/home/ubuntu/gemma_official_reference",
    hackable_diffusion_ref="/home/ubuntu/hackable_diffusion_reference",
    checkpoint_path="/home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it",
    workdir="/home/ubuntu/diffusion_gemma_runs/qwix_lora_pubmedqa",
    num_train_steps=2000,
    run_steps=2000,
    save_final_checkpoint=True,
    peft_config=DiffusionGemmaQwixLoRAConfig(rank=4),
    loss_config=DiffusionGemmaOfficialLossConfig(
        train_loop="hybrid",
        sync_after_step="losses",
        log_losses=True,
        encoder_loss_token_chunk_size=128,
    ),
)

trainer.train()

print(trainer.checkpoint_info("latest").as_dict())
print(trainer.generation_kwargs("latest"))
```

Equivalent CLI:

```bash
LORA_RANK=4 LOG_PARAM_SUMMARY=true \
./scripts/run_diffusion_gemma_h100x2_comparison_job.sh \
  --mode tunix \
  --run_name qwix_lora_pubmedqa_2000step \
  --lora_backend qwix_lora \
  --run_steps 2000 \
  --num_train_steps 2000 \
  --max_runtime_seconds 14400 \
  --sync_after_step losses \
  --log_losses true \
  --encoder_loss_token_chunk_size 128 \
  --gpu_poll_seconds 30
```

### Official Backend Generation With Measured Trace

Use this for a tuned official-backend checkpoint. With `--trace_mode measured`,
the script records the actual official canvas sampler trajectory:

```bash
XLA_FLAGS=--xla_disable_hlo_passes=constant_folding \
python scripts/generate_diffusion_gemma_official_backend.py \
  --recipe pubmedqa \
  --gemma_ref /home/ubuntu/gemma_official_reference \
  --hackable_diffusion_ref /home/ubuntu/hackable_diffusion_reference \
  --workdir /home/ubuntu/diffusion_gemma_runs/qwix_lora_pubmedqa \
  --checkpoint_path /home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it \
  --step latest \
  --num_train_steps 2000 \
  --checkpoint_every_n_steps 1000 \
  --lora_rank 4 \
  --lora_backend qwix_lora \
  --encoder_loss_token_chunk_size 128 \
  --dataset_batch_size 2 \
  --eval_num_batches 1 \
  --denoising_steps 8 \
  --max_num_canvases 1 \
  --max_trace_tokens 36 \
  --trace_mode measured \
  --output_json generation.json \
  --trace_json generation_trace.json

python scripts/verify_diffusion_gemma_generation_trace.py \
  --generation_json generation.json \
  --trace_json generation_trace.json \
  --output_json generation_trace_parity.json

python scripts/render_diffusion_gemma_trace_gif.py \
  generation_trace.json \
  --output diffusion_gemma_generation.gif \
  --width 1280 \
  --height 720 \
  --final_hold_frames 3
```

Measured trace semantics:

- `settings.measurement` is
  `DiffusionSampler(store_trajectory=True).trajectory.xt`.
- Final text and trace come from the same official sampling pass.
- The displayed accepted mask is derived from token stability against the final
  canvas across later measured frames. It is not a separate official internal
  accept/reject mask unless the official sampler exposes that later.

### Tunix Data Adapter

DiffusionGemma SFT does not consume plain causal-LM `input_ids`/`labels`.
Records are converted into prompt plus fixed denoising canvases:

```python
from tunix.models.diffusion_gemma import data as dg_data

batch_config = dg_data.DiffusionGemmaTextBatchConfig(
    prompt_len=1024,
    canvas_size=128,
    num_canvases=2,
    pad_id=0,
    eos_id=1,
    bos_id=2,
)

records = [
    {
        "prompt": "Context: ...\n\nQuestion: ...",
        "response": "Brief explanation. The answer is: yes",
    }
]

train_ds = dg_data.make_sft_dataset(
    records,
    tokenizer=tokenizer,
    config=batch_config,
    batch_size=1,
    rng_seed=0,
    as_model_inputs=True,
)

first_batch = next(iter(train_ds))
print(dg_data.describe_sft_batch(first_batch).as_dict())
```

JSONL files can use `prompt`, `input`, or `question` for the prompt and
`response`, `completion`, `target`, or `answer` for the response:

```python
train_ds = dg_data.make_sft_dataset_from_jsonl(
    "train.jsonl",
    tokenizer=tokenizer,
    config=batch_config,
    batch_size=2,
    rng_seed=0,
    as_model_inputs=True,
)
```

The resulting batch fields are:

```text
prompt                int32[batch, prompt_len]
canvas                int32[batch, canvas_size * num_canvases, 1]
canvas_id             int32[batch, canvas_size * num_canvases]
canvas_mask           bool[batch, canvas_size * num_canvases]
encoder_target        int32[batch, prompt_len + canvas_size * num_canvases]
encoder_target_mask   float32[batch, prompt_len + canvas_size * num_canvases]
rng                   PRNGKey
```

### Native Tunix PeftTrainer Hooks

The native path installs DiffusionGemma-specific data and loss hooks into
`PeftTrainer`:

```python
from tunix.models.diffusion_gemma import sft as dg_sft

sft_config = dg_sft.DiffusionGemmaSFTConfig(
    prompt_len=1024,
    canvas_size=128,
    num_canvases=2,
    vocab_size=tokenizer.vocab_size,
    decoder_implementation="cached_selected_canvas_slice",
    encoder_loss_chunk_size=128,
)

trainer = dg_sft.configure_peft_trainer_for_diffusion_gemma_sft(
    trainer,
    sft_config,
)
```

This hook path implements prompt/canvas prefill, timestep sampling, corruption,
selected-canvas denoising loss, optional self-conditioning, and encoder AR
loss. It is covered by unit and parity tests, but the public 26B multi-step H100
x2 recommendation remains the official-backend wrapper.

## GPU Evidence

### Matched 2000-Step H100 x2 Runs

All three runs completed `2000/2000` PubMedQA LoRA SFT steps on H100 80GB x2
without OOM, traceback, NaN, or timeout.

| Run | Run id | Steps | First total | Final total | Last-50 mean | Peak HBM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Upstream official | `r_5c5a20f2` | 2000 | 13.9219 | 2.0742 | 1.8728 | 65661 MiB |
| Tunix wrapper, official LoRA | `r_014dadd5` | 2000 | 13.9219 | 2.0996 | 1.8734 | 65659 MiB |
| Tunix wrapper, Qwix LoRA | `r_7ea13ca4` | 2000 | 13.5599 | 2.1066 | 1.8847 | 65663 MiB |

The Qwix LoRA run ended within `+0.0324` total loss of upstream official and
`+0.0070` of the non-Qwix Tunix wrapper. Peak HBM was effectively identical.

Detailed comparison: `docs/diffusion_gemma/QWIX_LORA_COMPARISON.md`.

### 500-Step API, Checkpoint, and Generation Run

The newer Tunix-facing API surface was validated with a 500-step H100 x2 run:

```text
train_run_id: r_f68ec248
generation_run_id: r_1dd04e43
machine_id: 426600
workdir: /home/ubuntu/diffusion_gemma_compare/qwix_lora_tunix_apiux_500step_20260614T020042Z/workdir
checkpoint: checkpoints/ckpt_500
```

Training summary:

```text
loss_count: 500
first_total_loss: 13.577728271484375
final_total_loss: 1.7979092597961426
min_total_loss: 1.0636097192764282
max_total_loss: 16.80182647705078
peak_hbm_gpu0: 65741 MiB
peak_hbm_gpu1: 65711 MiB
```

Measured generation summary:

```text
trace_token_parity.equal: true
measurement: DiffusionSampler(store_trajectory=True).trajectory.xt
acceptance_trace: measured_stability_from_official_diffusion_trajectory
measured_frames: 8
final_frame_accepted_tokens: 36/36
local_gif: /tmp/diffusion_gemma_500step/diffusion_gemma_500step_generation_measured_singlepass.gif
```

Example generated answer:

```text
Our analysis of PCD-in vivo mitochondrial dynamics suggest that mitochondria are
active to the process of PCD in plants, and the role of mitochondria is
potential in opening the PTP. The answer is: yes
```

## Local Verification

DiffusionGemma-specific checks exercised during this work:

```bash
python -m py_compile \
  tunix/models/diffusion_gemma/data.py \
  tunix/models/diffusion_gemma/generation.py \
  tunix/models/diffusion_gemma/sft.py \
  tunix/models/diffusion_gemma/hackable_adapter.py \
  scripts/generate_diffusion_gemma_official_backend.py \
  scripts/render_diffusion_gemma_trace_gif.py

python -m pytest tests/models/diffusion_gemma_test.py -q --import-mode=importlib
python scripts/verify_diffusion_gemma_official_parity.py
python scripts/verify_diffusion_gemma_official_logits.py
python scripts/verify_diffusion_gemma_generation_trace.py \
  --generation_json generation.json \
  --trace_json generation_trace.json \
  --output_json generation_trace_parity.json
```

Full repository `pytest` is not a clean signal in this environment because
unrelated optional serving backends and local native tokenizer crashes are
present. Use the DiffusionGemma-specific checks above for this feature.

## What Changed In Code

Primary package files:

- `tunix/models/diffusion_gemma/__init__.py`
- `tunix/models/diffusion_gemma/hackable_adapter.py`
- `tunix/models/diffusion_gemma/linen_qwix_lora.py`
- `tunix/models/diffusion_gemma/lora_inventory.py`
- `tunix/models/diffusion_gemma/sft.py`
- `tunix/models/diffusion_gemma/data.py`
- `tunix/models/diffusion_gemma/generation.py`
- `tunix/models/diffusion_gemma/model.py`
- `tunix/models/diffusion_gemma/params.py`
- `tunix/models/diffusion_gemma/params_safetensors.py`

Primary scripts:

- `scripts/run_diffusion_gemma_official_reference.py`
- `scripts/run_diffusion_gemma_official_backend.py`
- `scripts/run_diffusion_gemma_h100x2_comparison_job.sh`
- `scripts/run_diffusion_gemma_pubmedqa_tunix.py`
- `scripts/generate_diffusion_gemma_official_backend.py`
- `scripts/render_diffusion_gemma_trace_gif.py`
- `scripts/verify_diffusion_gemma_generation_trace.py`
- `scripts/verify_diffusion_gemma_official_parity.py`
- `scripts/verify_diffusion_gemma_official_logits.py`
- `scripts/summarize_diffusion_gemma_training.py`

The most important implementation choices:

- The official backend remains the oracle for public 26B H100 x2 training.
- Qwix LoRA is applied as a Linen-compatible bridge over the resolved official
  model and AR sampler networks.
- The hybrid loop synchronizes addressable loss shards rather than forcing full
  host state materialization after every step.
- Encoder AR loss can use token/vocab chunking to reduce peak intermediate
  memory without changing the exact objective.
- Checkpoint metadata and generation kwargs are exposed through the wrapper.
- Measured generation GIFs now come from actual official sampler trajectories,
  not a synthetic prefix reveal.

## Boundaries

Current supported recommendation:

```text
H100 80GB x2 + official-backend wrapper + Qwix LoRA + PubMedQA-style SFT
```

Not claimed yet:

- Full native NNX/Qwix replacement for the official training recipe.
- Full-weight tuning.
- Long production training quality benchmarks.
- vLLM serving, DPO, GRPO, or rollout serving.
- Bitwise-identical stochastic trajectories across independent training runs.

## Next Work

1. Promote the official-backend wrapper API into a cleaner Tunix recipe surface
   with less script glue.
2. Add a first-class DiffusionGemma dataset config object for common custom-data
   flows, including prompt templates and answer post-processing.
3. Keep checkpoint UX split between frozen base checkpoint and adapter
   checkpoint, with export/import helpers for LoRA-only artifacts.
4. Add official evaluator parity checks that compare generated text and measured
   trajectories under controlled seeds.
5. Continue native NNX work only after the official-backend wrapper remains the
   stable oracle.
