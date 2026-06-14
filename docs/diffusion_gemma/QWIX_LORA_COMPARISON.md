# DiffusionGemma Official, Tunix Wrapper, and Qwix LoRA Comparison

Date: 2026-06-14

This note compares three H100 x2 DiffusionGemma PubMedQA SFT runs:

1. The upstream DeepMind implementation, run directly.
2. The Tunix wrapper around the official Hackable Diffusion backend, using the official LoRA wrapper.
3. The same Tunix wrapper, replacing only the LoRA layer with Qwix LoRA.

The purpose is not to claim bitwise-identical training trajectories. These are
independent stochastic training jobs. The purpose is to verify that the Qwix
LoRA surface trains in the same practical regime as the official implementation:
same recipe shape, same official model/loss/trainstep stack, finite losses,
same H100 x2 memory profile, and similar throughput.

## Result Summary

All three runs completed `2000/2000` PubMedQA LoRA SFT steps on H100 80GB x2
without OOM, traceback, NaN, or timeout.

| Run | Steps | Total first | Total final | Total last-50 mean | Peak HBM | Steps/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official upstream | 2000 | 13.9219 | 2.0742 | 1.8728 | 65661 MiB | 1444.8 |
| Tunix wrapper, official LoRA | 2000 | 13.9219 | 2.0996 | 1.8734 | 65659 MiB | 1427.6 |
| Tunix wrapper, Qwix LoRA | 2000 | 13.5599 | 2.1066 | 1.8847 | 65663 MiB | 1435.1 |

Final loss breakdown:

| Run | Diffusion final | Encoder final | Total final |
| --- | ---: | ---: | ---: |
| Official upstream | 1.0352 | 1.0391 | 2.0742 |
| Tunix wrapper, official LoRA | 1.0527 | 1.0469 | 2.0996 |
| Tunix wrapper, Qwix LoRA | 1.0625 | 1.0441 | 2.1066 |

The Qwix LoRA run ended within `+0.0324` total loss of the upstream official
run and `+0.0070` of the non-Qwix Tunix wrapper. Last-50 total-loss mean was
within about `0.012` of both baselines. Peak HBM was effectively identical
across all runs, around `65.6 GiB` per GPU.

## Shared Conditions

The matched 2000-step runs used:

- Recipe: official PubMedQA SFT.
- Public DiffusionGemma checkpoint: `DiffusionGemma_A26B_A4B`.
- LoRA rank: `4`.
- Dataset batch size: official default batch size `2`.
- Prompt length: `1024`.
- Canvas geometry: `num_canvases=2`, `canvas_size=128`.
- Train steps: `2000`.
- Training loop: `hybrid`.
- Loss synchronization: `sync_after_step=losses`.
- XLA flag: `--xla_disable_hlo_passes=constant_folding`.
- Official Gemma revision: `a7e33454206ca24984565992f5c903191baecc22`.
- Hackable Diffusion revision: `03ce88ed0acec3b17f4f284502a6053c5f2b3a15`.

Evidence for the earlier upstream official and non-Qwix wrapper runs is stored
under:

```text
evidence/diffusion_gemma/h100x2_2000step_comparison_2026-06-13/
```

The Qwix LoRA 2000-step run was:

```text
run_id: r_7ea13ca4
machine_id: 426095
workdir: /home/ubuntu/diffusion_gemma_compare/qwix_lora_only_2000step_20260613T145146Z/workdir
```

## 500-Step API, Checkpoint, and Generation Validation

A follow-up H100 x2 run validated the newer Tunix-facing API surface,
checkpoint UX, official evaluator generation path, and GIF trace rendering.

```text
train_run_id: r_f68ec248
generation_run_id: r_ef06d8d9
measured_generation_run_id: r_1dd04e43
machine_id: 426600
workdir: /home/ubuntu/diffusion_gemma_compare/qwix_lora_tunix_apiux_500step_20260614T020042Z/workdir
```

The training run used the same PubMedQA recipe family and 2000-step schedule as
the matched 2000-step runs, but stopped after `run_steps=500` and saved the
final checkpoint:

```text
loss_count: 500
first_total_loss: 13.577728271484375
final_total_loss: 1.7979092597961426
min_total_loss: 1.0636097192764282
max_total_loss: 16.80182647705078
peak_hbm_gpu0: 65741 MiB
peak_hbm_gpu1: 65711 MiB
checkpoint: checkpoints/ckpt_500
```

Generation was then restored with `--step latest`, which resolved to
`ckpt_500`, and ran through the official Hackable Diffusion AR evaluator:

```text
official_backend_path: official_hackable_diffusion_ar_evaluator
evaluator_class: gemma.diffusion.hackable_diffusion_adapter.hd.sft_model.GemmaSamplingEvaluator
sampler_class: gemma.diffusion.hackable_diffusion_adapter.hd.sft_model.GemmaKDARSampler
lora_backend: qwix_lora
token_ids_equal: true
output_text_equal: true
num_generation_tokens: 40
num_trace_tokens: 40
```

The generated response for the first PubMedQA sample was:

```text
Our observations of PCD-in vivo mitochondrial dynamics suggest that mitochondria are involved in the development of PCD in plants plants and the role of mitochondria is potential in forming the PTP. The answer is: yes
```

The current generated trace was rendered from the official sampler's measured
denoising trajectory, not from a synthetic reveal order:

```text
/tmp/diffusion_gemma_500step/diffusion_gemma_500step_generation_measured_singlepass.gif
settings.measurement: DiffusionSampler(store_trajectory=True).trajectory.xt
settings.acceptance_trace: measured_stability_from_official_diffusion_trajectory
```

The measured trace used a single official sampling pass for both the final text
sample and the GIF source. This avoids stochastic mismatch between "final
sample" and "trajectory" outputs. Its final frame accepted all traced tokens and
its intermediate token changes were not prefix ordered:

```text
trace_token_parity.equal: true
measured_frames: 8
final_frame_accepted_tokens: 36/36
example_intermediate_accepted_indices:
  step 2: [11, 27]
  step 3: [2, 11, 19, 27, 28, 29, 33]
```

## How To Run Each Path

### 1. Upstream Official Implementation

This path runs the official DeepMind DiffusionGemma recipe without importing
Tunix. It is the reference behavior.

```bash
LORA_RANK=4 \
./scripts/run_diffusion_gemma_h100x2_comparison_job.sh \
  --mode upstream \
  --run_name upstream_official_2000step \
  --run_steps 2000 \
  --num_train_steps 2000 \
  --max_runtime_seconds 14400 \
  --sync_after_step losses \
  --log_losses true \
  --gpu_poll_seconds 30
```

Internally, `--mode upstream` dispatches to:

```bash
python scripts/run_diffusion_gemma_official_reference.py \
  --recipe pubmedqa \
  --train_loop hybrid \
  --sync_after_step losses
```

Use this path when the question is whether the official implementation itself
works under the current runtime.

### 2. Tunix Wrapper With Official LoRA

This path imports Tunix, but keeps the official Hackable Diffusion model,
dataset, loss, LoRA wrapper, optimizer, trainstep, checkpoint restore, and
sharding.

CLI:

```bash
LORA_RANK=4 \
./scripts/run_diffusion_gemma_h100x2_comparison_job.sh \
  --mode tunix \
  --run_name tunix_wrapper_official_lora_2000step \
  --lora_backend official \
  --run_steps 2000 \
  --num_train_steps 2000 \
  --max_runtime_seconds 14400 \
  --sync_after_step losses \
  --log_losses true \
  --gpu_poll_seconds 30
```

Python:

```python
from tunix.models.diffusion_gemma import (
    OfficialDiffusionGemmaTrainer,
    OfficialSFTConfig,
)

config = OfficialSFTConfig(
    recipe="pubmedqa",
    gemma_ref="/home/ubuntu/gemma_official_reference",
    hackable_diffusion_ref="/home/ubuntu/hackable_diffusion_reference",
    checkpoint_path="/home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it",
    workdir="/home/ubuntu/diffusion_gemma_runs/tunix_official_lora",
    num_train_steps=2000,
    run_steps=2000,
    lora_rank=4,
    lora_backend="official",
    train_loop="hybrid",
    sync_after_step="losses",
    log_losses=True,
    disable_evals=True,
)

trainer = OfficialDiffusionGemmaTrainer(config)
trainer.train()
```

Use this path when the goal is a Tunix-managed entrypoint and telemetry while
preserving the official LoRA implementation.

### 3. Tunix Wrapper With Qwix LoRA

This path is identical to the Tunix official-LoRA wrapper except for one piece:
the resolved official Linen model's LoRA module is replaced by the Tunix/Qwix
Linen LoRA bridge.

CLI:

```bash
LORA_RANK=4 LOG_PARAM_SUMMARY=true \
./scripts/run_diffusion_gemma_h100x2_comparison_job.sh \
  --mode tunix \
  --run_name qwix_lora_only_2000step \
  --lora_backend qwix_lora \
  --run_steps 2000 \
  --num_train_steps 2000 \
  --max_runtime_seconds 14400 \
  --sync_after_step losses \
  --log_losses true \
  --encoder_loss_token_chunk_size 128 \
  --gpu_poll_seconds 30
```

Python:

```python
from tunix.models.diffusion_gemma import (
    DiffusionGemmaOfficialLossConfig,
    DiffusionGemmaQwixLoRAConfig,
    OfficialDiffusionGemmaTrainer,
)

trainer = OfficialDiffusionGemmaTrainer.from_official_backend(
    recipe="pubmedqa",
    gemma_ref="/home/ubuntu/gemma_official_reference",
    hackable_diffusion_ref="/home/ubuntu/hackable_diffusion_reference",
    checkpoint_path="/home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it",
    workdir="/home/ubuntu/diffusion_gemma_runs/tunix_qwix_lora",
    num_train_steps=2000,
    run_steps=2000,
    log_param_summary=True,
    disable_evals=True,
    peft_config=DiffusionGemmaQwixLoRAConfig(rank=4),
    loss_config=DiffusionGemmaOfficialLossConfig(
        train_loop="hybrid",
        sync_after_step="losses",
        log_losses=True,
        encoder_loss_token_chunk_size=128,
    ),
)
trainer.train()
```

Use this path when the goal is to expose DiffusionGemma through a Tunix/Qwix
LoRA surface while keeping the official DiffusionGemma training semantics.

### DiffusionGemma Data Format

DiffusionGemma SFT data is not the usual causal-LM `input_ids` plus `labels`
format. The response is split into fixed-size clean canvases, and the encoder
AR objective has its own shifted target:

```python
from tunix.models.diffusion_gemma import data as dg_data

records = [
    {"prompt": "Context: ...\n\nQuestion: ...", "response": "The answer is: yes"},
]

batch_config = dg_data.config_from_tokenizer(
    tokenizer,
    prompt_len=1024,
    canvas_size=256,
    num_canvases=4,
)

train_ds = dg_data.make_sft_batches_from_text_examples(
    records,
    tokenizer=tokenizer,
    config=batch_config,
    batch_size=1,
    rng_seed=0,
    as_model_inputs=True,
)
```

JSONL files can use the same prompt/response surface:

```jsonl
{"prompt": "Context: ...\n\nQuestion: ...", "response": "The answer is: yes"}
{"question": "Does the intervention help?", "answer": "The answer is: maybe"}
```

```python
from tunix.models.diffusion_gemma import data as dg_data

train_ds = dg_data.make_sft_dataset_from_jsonl(
    "train.jsonl",
    tokenizer=tokenizer,
    config=batch_config,
    batch_size=2,
    rng_seed=0,
    as_model_inputs=True,
)

first_batch = next(iter(train_ds))
print(dg_data.describe_sft_batch(first_batch).as_dict())
```

Each yielded batch contains:

```text
prompt                int32[batch, prompt_len]
canvas                int32[batch, canvas_size * num_canvases, 1]
canvas_id             int32[batch, canvas_size * num_canvases]
canvas_mask           bool[batch, canvas_size * num_canvases]
encoder_target        int32[batch, prompt_len + canvas_size * num_canvases]
encoder_target_mask   float32[batch, prompt_len + canvas_size * num_canvases]
rng                   PRNGKey
```

The default record parser accepts `prompt`, `prompts`, `input`, or `question`
as the prompt field and `response`, `completion`, `target`, `targets`, or
`answer` as the response field. Pass `prompt_fn` and `response_fn` when a custom
dataset needs richer formatting.

## Checkpoint and Adapter UX

The wrapper can now report saved checkpoint metadata from the workdir and turn
it back into generation kwargs:

```python
from tunix.models.diffusion_gemma import (
    DiffusionGemmaOfficialLossConfig,
    DiffusionGemmaQwixLoRAConfig,
    OfficialDiffusionGemmaTrainer,
)

trainer = OfficialDiffusionGemmaTrainer.from_official_backend(
    recipe="pubmedqa",
    gemma_ref="/home/ubuntu/gemma_official_reference",
    hackable_diffusion_ref="/home/ubuntu/hackable_diffusion_reference",
    checkpoint_path="/home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it",
    workdir="/home/ubuntu/diffusion_gemma_runs/qwix_lora_500",
    num_train_steps=500,
    run_steps=500,
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

info = trainer.checkpoint_info("latest")
print(info.as_dict())
print(trainer.generation_kwargs("latest"))
```

For scripts, `generate_diffusion_gemma_official_backend.py` accepts
`--step latest`, so a saved wrapper workdir can be restored without manually
checking the `checkpoints/ckpt_*` directory.

## Tunix-Native Evaluation API

For the NNX DiffusionGemma path, evaluation can use the same SFT loss wrapper
that is installed into `PeftTrainer`:

```python
from tunix.models.diffusion_gemma import evaluate_sft_loss
from tunix.models.diffusion_gemma import sft as dg_sft

sft_config = dg_sft.DiffusionGemmaSFTConfig(
    prompt_len=1024,
    canvas_size=256,
    num_canvases=4,
    vocab_size=tokenizer.vocab_size,
    pad_token=tokenizer.pad_id(),
    decoder_implementation="cached_selected_canvas_slice",
    encoder_loss_chunk_size=128,
)

result = evaluate_sft_loss(
    model,
    eval_ds,
    sft_config,
    max_batches=16,
)

print(result.num_batches)
print(result.metrics["total_loss"])
print(result.metrics["decoder_loss"])
print(result.metrics["encoder_loss"])
```

This is intentionally a DiffusionGemma evaluator rather than a causal-LM
perplexity helper. It aggregates the diffusion denoising loss, encoder AR loss,
corruption rate, selected-canvas index, timestep mean, and self-conditioning
rate returned by the Tunix-native SFT loss.

## Tunix-Native Generation Trace API

The native NNX model also exposes a lightweight denoising trace API:

```python
from tunix.models.diffusion_gemma import generation as dg_generation

gen_config = dg_generation.DiffusionGemmaGenerationConfig(
    prompt_len=1024,
    canvas_size=256,
    num_canvases=4,
    vocab_size=tokenizer.vocab_size,
    pad_token=tokenizer.pad_id(),
    num_steps=16,
    temperature=0.0,
    use_self_conditioning=True,
    entropy_budget=8.0,
    renoise_rejected_tokens=True,
)

trace = dg_generation.generate_tokens(
    model,
    prompt_ids,  # int32[batch, prompt_len]
    gen_config,
)

final_canvas_ids = trace.final_tokens
decoded_frames = dg_generation.decode_trace(tokenizer, trace, batch_index=0)
```

`trace.frames[0]` is the initial canvas, and each later frame is the canvas
after one selected-canvas denoising pass. `trace.selected_canvas_idx` records
which canvas was updated at each step, and `trace.changed_fraction` gives a
small per-step sanity signal that is useful for logs and animations.
`entropy_budget` enables the DiffusionGemma-style confidence rule: low-entropy
positions are accepted first, while rejected positions can either keep their
previous value or be re-noised with `renoise_rejected_tokens=True`.

Current boundary: this sampler is a Tunix-native validation and visualization
surface. It reuses the same NNX encoder prefill, KV cache end-index handling,
selected-canvas decoder path, and optional self-conditioning path used by SFT.
It is not the official DeepMind inference sampler ported byte-for-byte.

For official evaluator parity, use the official-backend generation script. It
restores a Tunix-wrapper checkpoint, applies the selected LoRA backend, invokes
the official AR evaluator, and emits a token trace that can be rendered as GIF:

```bash
XLA_FLAGS=--xla_disable_hlo_passes=constant_folding \
python scripts/generate_diffusion_gemma_official_backend.py \
  --recipe pubmedqa \
  --gemma_ref /home/ubuntu/gemma_official_reference \
  --hackable_diffusion_ref /home/ubuntu/hackable_diffusion_reference \
  --checkpoint_path /home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it \
  --workdir /home/ubuntu/diffusion_gemma_runs/tunix_qwix_lora \
  --step latest \
  --lora_rank 4 \
  --lora_backend qwix_lora \
  --denoising_steps 8 \
  --max_num_canvases 1 \
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

`canvas_id` identifies which canvas each response token belongs to.
`canvas_mask` marks valid clean-canvas positions, including EOS fill inside the
last partially used canvas. This is the surface consumed by the Tunix
DiffusionGemma loss hooks.

## Internal Differences

### Runner Dispatch

The H100 comparison script uses the same environment, checkpoint, dataset
conversion, GPU polling, and summary logic for all paths. The main branch point
is the runner.

```bash
if [[ "${MODE}" == "upstream" ]]; then
  python scripts/run_diffusion_gemma_official_reference.py "${COMMON_ARGS[@]}"
else
  python scripts/run_diffusion_gemma_official_backend.py "${COMMON_ARGS[@]}"
fi
```

The upstream path imports official modules directly. The Tunix paths construct
an `OfficialSFTConfig` and instantiate `OfficialDiffusionGemmaTrainer`.

### Tunix Wrapper Configuration

The wrapper exposes official recipe controls plus Tunix-specific override
points:

```python
@dataclasses.dataclass(frozen=True, kw_only=True)
class OfficialSFTConfig:
    recipe: str = "pubmedqa"
    checkpoint_path: str | pathlib.Path | None = None
    num_train_steps: int | None = None
    run_steps: int | None = None
    lora_rank: int | None = None
    lora_backend: str = "official"
    lora_alpha: float | None = None
    train_loop: str = "kauldron"
    sync_after_step: str = "state"
    log_losses: bool = True
    log_param_summary: bool = False
    encoder_loss_token_chunk_size: int | None = None
```

`lora_backend="official"` returns without changing the resolved official LoRA
model. `lora_backend="qwix_lora"` patches the resolved official model after the
official recipe has been built.

The more Tunix-like Qwix path now composes separate PEFT and loss configs before
constructing the official backend config:

```python
from tunix.models.diffusion_gemma import (
    DiffusionGemmaOfficialLossConfig,
    DiffusionGemmaQwixLoRAConfig,
    make_official_tunix_sft_config,
)

config = make_official_tunix_sft_config(
    recipe="pubmedqa",
    peft_config=DiffusionGemmaQwixLoRAConfig(rank=4),
    loss_config=DiffusionGemmaOfficialLossConfig(
        train_loop="hybrid",
        sync_after_step="losses",
        encoder_loss_token_chunk_size=128,
    ),
)
```

This keeps the official recipe fields available, but moves the user-facing LoRA
and loss choices toward the same conceptual split used by Tunix PEFT trainers.

### Official LoRA Path

For the official LoRA wrapper path, Tunix does not replace the LoRA module:

```python
def _replace_resolved_lora_backend(trainer, config):
    if config.lora_backend == "official":
        return
```

The rest of the wrapper is still useful because it standardizes:

- official reference checkout discovery,
- checkpoint path overrides,
- controlled `run_steps`,
- `hybrid` loop execution,
- loss logging from addressable shards,
- GPU memory telemetry,
- run summaries.

### Qwix LoRA Path

For the Qwix path, the wrapper loads the Linen Qwix bridge, converts the
official target-module description into a Qwix `module_path`, and replaces the
resolved model's `gemma_network`.

```python
bridge = _load_linen_qwix_lora_module()
base_network = getattr(gemma_network, "model", gemma_network)
module_path = _qwix_module_path_from_official_targets(
    bridge,
    target_modules,
    override=config.qwix_lora_module_path,
)
replacement = bridge.apply_lora_to_linen_model(
    base_network,
    rank=int(rank),
    alpha=float(alpha),
    module_path=module_path,
    methods=bridge.DIFFUSION_GEMMA_LINEN_LORA_METHODS,
)

for model in model_candidates:
    object.__setattr__(model, "gemma_network", replacement)
```

The validated Qwix target pattern is the official `all-linear` target set
translated into Linen/Qwix names:

```text
(?:(.*/)?attn/(q_einsum|kv_einsum|k_einsum|attn_vec_einsum)$)
|(?:(.*/)?(mlp|mlp2)$)
|(?:(.*/)?(mlp|mlp2)/(gating_einsum|linear|router_logits)$)
|(?:(.*/)?self_conditioner/ffw$)
|(?:(.*/)?self_conditioner/ffw/(gating_einsum|linear)$)
```

The 2000-step Qwix run logged:

```text
LoRA leaves: 1092
LoRA storage: 0.02474355697631836 GiB
Dense frozen base leaves: 608
Dense frozen base storage: 47.033628053963184 GiB
```

### Memory-Safe Encoder Loss Path

The Qwix 2000-step run used `encoder_loss_token_chunk_size=128`. This keeps the
official encoder AR loss semantics, but avoids materializing a full
batch-by-sequence-by-vocabulary logits tensor for the encoder loss.

Conceptually:

```python
encoder_hidden = gemma_network.encoder_hidden_call(...)
encoder_loss = exact_cross_entropy_by_decoding_hidden_in_token_chunks(
    gemma_network,
    encoder_hidden,
    encoder_target,
    encoder_target_mask,
    chunk_size=128,
)
preds["encoder_loss"] = encoder_loss
train_losses["encoder_loss"] = EncoderLossValue("preds.encoder_loss")
```

This is why Qwix LoRA can run under the same H100 x2 memory envelope while
keeping exact encoder-loss values. The improvement comes from reducing the
full-vocab encoder-loss peak-memory path.

### Optimizer Update Mask

The Qwix path patches update application so only official/Qwix LoRA leaves are
updated:

```python
def apply_updates(params, updates):
    def _apply_one(path, param, update):
        path_str = _jax_key_path_to_string(path)
        if not _is_qwix_or_official_lora_path(path_str):
            return param
        return (param + update).astype(param.dtype)

    return jax.tree_util.tree_map_with_path(_apply_one, params, updates)
```

This is the practical distinction from a plain official wrapper: Tunix owns the
LoRA surface and update filter, while the official backend still owns the model
family semantics.

## Interpretation

The Qwix LoRA result is successful.

What it proves:

- Qwix LoRA can replace the official Linen LoRA wrapper inside the official
  DiffusionGemma SFT path.
- The run completes `2000/2000` H100 x2 steps with finite diffusion, encoder,
  and total losses.
- The loss curve is in the same range as both the upstream official and
  non-Qwix Tunix wrapper runs.
- The memory profile and throughput are effectively the same as the baselines.
- The Qwix LoRA inventory and update path are visible through Tunix.

What it does not prove:

- It is not bitwise parity. Independent stochastic runs are expected to differ.
- It is not a native NNX DiffusionGemma model-family implementation.
- It does not prove full fine-tuning or DPO/GRPO support.

Recommended wording:

> DiffusionGemma PubMedQA LoRA SFT is validated on H100 80GB x2 through the
> official Hackable Diffusion backend wrapped by Tunix. The Qwix LoRA variant
> swaps only the LoRA surface and matches the official and non-Qwix wrapper
> practical training envelope over 2000 steps.
