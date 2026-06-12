# DiffusionGemma Native Tunix H100x2 Validation

Date: 2026-06-13

## Scope

This note records the native Tunix NNX/Qwix validation path for
DiffusionGemma 26B PubMedQA LoRA SFT. It is separate from the Hackable
Diffusion wrapper evidence, which delegates the model/loss/trainstep to the
official Flax/Linen + Kauldron stack.

Official source references used for this validation:

- [`gemma/diffusion/hackable_diffusion_adapter/hd/lora.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/diffusion/hackable_diffusion_adapter/hd/lora.py)
- [`gemma/peft/_lora.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/peft/_lora.py)
- [`gemma/peft/_einsum_utils.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/peft/_einsum_utils.py)
- [`gemma/gm/nn/gemma4/_moe.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_moe.py)

The official LoRA wrapper applies to supported linear modules such as
`_layers.Einsum`. In the official ragged MoE path, `router_logits` is an
Einsum, while expert `gating_einsum` and `linear` weights are raw `_Weight`
providers used by `ragged_dot`. The native Tunix default mirrors the
official-compatible ragged path by applying Qwix to ordinary projections and a
manual MoE LoRA hook to `router_logits`. Raw expert LoRA remains available as an
explicit experimental target set.

## Native H100x2 Run

- Machine: JarvisLabs `425962`, H100 80GB x2, region `IN2`.
- Repo commit: `c95a2ff0` for the final default native run. Earlier baseline
  runs at `7007c657` are retained below for comparison.
- Checkpoint: `/home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it`.
- Tokenizer: `/home/ubuntu/checkpoints/tokenizers/tokenizer_gemma4.model`.
- Geometry: `prompt_len=1024`, `canvas_size=128`, `num_canvases=2`.
- Batch: `batch_size=1`, `gradient_accumulation_steps=2`.
- Mesh: `mesh_fsdp=1`, `mesh_tp=2`.
- Runtime knobs: `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `NCCL_ALGO=Ring`, `NCCL_PROTO=LL128`, `NCCL_NVLS_ENABLE=0`,
  `NCCL_CUMEM_ENABLE=0`,
  `XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=constant_folding"`.
- Tunix knobs: `cached_selected_canvas_slice`, cuDNN local attention, full
  remat, split encoder/decoder gradients, separate loss JITs,
  `encoder_loss_chunk_size=8`, `--skip_initial_loss`.

Run `r_f30bde99` validated the `c95a2ff0` default load-only path:

- LoRA module path:
  `.*(q_einsum|kv_einsum|k_einsum|attn_vec_einsum|gate_proj|up_proj|down_proj)$`.
- MoE LoRA targets: `router_logits`.
- Trainable state: 426 leaves, 4,777,216 elements, 9,554,432 bytes.
- Frozen state: 638 leaves, 25,250,986,784 elements, 50,501,973,576 bytes.
- Minimal state:
  `/home/ubuntu/diffusion_gemma_native_load_h100x2_default_c95a2ff/minimal_state.json`.

Run `r_3ca6917b` validated the `c95a2ff0` default 1-step train path with the
same official geometry and runtime:

- Step loss: `18.10016632080078`.
- Decoder loss: `7.257199287414551`.
- Encoder loss: `10.842966079711914`.
- Grad norm: `56.25`.
- Update norm: `0.150390625`.
- Final sampled loss: `17.54163932800293`.
- Final sampled decoder loss: `6.385552883148193`.
- Final sampled encoder loss: `11.156085968017578`.
- LoRA norm delta: `0.000293731689453125`.
- LoRA checksum delta: `0.028049468994140625`.
- LoRA max absolute delta: `0.00010013580322265625`.
- Base checksum delta: `0.0`.
- Base max absolute delta: `0.0`.
- Trainable state: 426 leaves, 4,777,216 elements, 9,554,432 bytes.
- Frozen state: 638 leaves, 25,250,986,784 elements, 50,501,973,576 bytes.
- Peak sampled HBM: GPU0 `78255` MiB, GPU1 `78253` MiB.
- Minimum sampled headroom: `3304` MiB.
- Minimal state:
  `/home/ubuntu/diffusion_gemma_native_train_h100x2_default_c95a2ff/minimal_state.json`.

Run `r_d416d3f5` completed one train step with `--no-moe_lora` before the
default router-only cleanup. It is retained as the memory-safe predecessor
baseline:

- Step loss: `17.26303482055664`.
- Decoder loss: `6.420068264007568`.
- Encoder loss: `10.842966079711914`.
- Grad norm: `95.5`.
- Update norm: `0.150390625`.
- Final sampled loss: `17.80583381652832`.
- LoRA checksum delta: `0.02187347412109375`.
- Base checksum delta: `0.0`.
- Trainable state: 546 leaves, 5,131,456 elements, 10,262,912 bytes.
- Frozen state: 638 leaves, 25,250,986,784 elements, 50,501,973,576 bytes.
- Peak sampled HBM: GPU0 `78401` MiB, GPU1 `78395` MiB.
- Minimum sampled headroom: `3158` MiB.
- Minimal state:
  `/home/ubuntu/diffusion_gemma_native_train_h100x2_official_geometry_effb2_nomoe_cudnn_local_fullremat_skipinit_bd1527c4/minimal_state.json`.

Run `r_a9b4d177` is the router-only MoE LoRA follow-up using the same geometry
and runtime. It exited successfully with code 0:

- Step loss: `17.26303482055664`.
- Decoder loss: `6.420068264007568`.
- Encoder loss: `10.842966079711914`.
- Grad norm: `127.0`.
- Update norm: `0.150390625`.
- Final sampled loss: `18.02530288696289`.
- Final sampled decoder loss: `7.147544860839844`.
- Final sampled encoder loss: `10.87775707244873`.
- LoRA norm delta: `0.0002288818359375`.
- LoRA checksum delta: `0.026342391967773438`.
- LoRA max absolute delta: `0.00010013580322265625`.
- Base checksum delta: `0.0`.
- Base max absolute delta: `0.0`.
- Trainable state: 546 leaves, 5,131,456 elements, 10,262,912 bytes.
- Frozen state: 638 leaves, 25,250,986,784 elements, 50,501,973,576 bytes.
- Peak sampled HBM: GPU0 `78365` MiB, GPU1 `78359` MiB.
- Minimum sampled headroom: `3194` MiB.
- Minimal state:
  `/home/ubuntu/diffusion_gemma_native_train_h100x2_official_geometry_moe_router_cudnn_fullremat_skipinit_7007c65/minimal_state.json`.

## Raw Expert LoRA Probe

Run `r_cf14b59d` enabled the broader experimental MoE target set
`router_logits,gating_einsum,linear` under the same H100x2 geometry. It failed
before the first encoder phase finished:

```text
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 65.85GiB.
[executable_name='jit_encoder_grad_step']
```

That run had 666 trainable LoRA leaves, 38,247,616 trainable elements, and
76,495,232 trainable bytes. The result shows that raw expert LoRA is not the
H100x2 default path for this NNX/Qwix implementation. It remains an opt-in
engineering target for larger meshes or a future more memory-efficient expert
LoRA implementation.

## Local Regression Checks

- `python -m pytest tests/models/diffusion_gemma_test.py -q`: 26 passed.
- `python -m pytest tests/models/gemma4/model_test.py -q`: 5 passed.

The new LoRA tests verify:

- default DiffusionGemma MoE LoRA creates router LoRA leaves only;
- no recursive LoRA-on-LoRA leaves are created;
- raw expert `gating_einsum` and `linear` LoRA leaves are opt-in and still
  change the MoE output when enabled.
