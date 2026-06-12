# DiffusionGemma Tunix Fork Verification Report

Date: 2026-06-12

Branch: `codex/diffusion-gemma-integration`

Implementation baseline commit under report: `0faad277`

## Executive Summary

This fork contains two complementary DiffusionGemma paths:

1. **Native Tunix MVP path**
   - Files: `tunix/models/diffusion_gemma/model.py`,
     `tunix/models/diffusion_gemma/sft.py`,
     `scripts/smoke_diffusion_gemma_pubmedqa_tunix.py`
   - Goal: make DiffusionGemma tunable through Tunix/NNX/Qwix/PeftTrainer.

2. **Official Hackable Diffusion backend wrapper**
   - Files: `tunix/models/diffusion_gemma/hackable_adapter.py`,
     `scripts/smoke_diffusion_gemma_official_backend.py`
   - Goal: keep the official DeepMind Gemma Diffusion + Hackable Diffusion +
     Kauldron implementation intact and expose it through a Tunix-controlled
     entrypoint for resource, parity, and debugging checks.

The fork does **not** claim that the native Tunix NNX/Qwix path is already a
drop-in production replacement for the official Kauldron recipe on every GPU
shape. The evidence supports a narrower but useful claim:

> The fork implements the official DiffusionGemma SFT semantics in Tunix enough
> to pass local parity/unit tests, tiny official-logits parity, LoRA-only update
> checks, real PubMedQA smoke training, and public-checkpoint 26B LoRA smoke
> training on 8 GPUs. It also includes a reference official backend wrapper that
> runs the official Kauldron recipe under Tunix orchestration and cleanly
> completes the same 2-GPU official smoke on H100 x2 and RTX PRO 6000 x2.

## What Is Implemented

Native Tunix code covers the core DiffusionGemma SFT mechanics:

- DiffusionGemma model family registration.
- `DiffusionGemma_A26B_A4B` NNX model based on the Tunix Gemma4 A26B/A4B
  backbone.
- Prompt plus clean-canvas prefill.
- KV cache construction and cache `end_index` handling.
- Diffusion timestep sampling with the official `SafeSpan(1e-4)` semantics.
- Token corruption.
- Selected-canvas denoising loss.
- Self-conditioning second pass.
- Encoder autoregressive loss.
- LoRA SFT path through Qwix/Tunix.
- Official-style cached selected-canvas decoder path, plus a lower-memory
  `cached_selected_canvas_slice` smoke path.
- Public checkpoint loading for the Gemma4-compatible DiffusionGemma backbone.
- Public PubMedQA smoke trainer with official-style prompt/canvas formatting.

The official backend wrapper covers:

- Official Gemma Diffusion recipe import.
- Official Hackable Diffusion SFT recipe config construction.
- Official Flax/Linen model, loss, corruption, sampling, data factories, LoRA,
  optimizer, FSDP sharding, and Kauldron trainer.
- Tunix-side path/workdir/step-count/config overrides.
- A `kauldron` mode that calls the official `trainer.train()`.
- A `hybrid` smoke mode that reuses official model/data/loss/trainstep objects
  while bypassing selected Kauldron metric/final-sync paths for debugging.

## Evidence Matrix

| Area | Evidence | Result |
| --- | --- | --- |
| Local DiffusionGemma tests | `python -m pytest tests/models/diffusion_gemma_test.py -q --import-mode=importlib` | `20 passed` |
| Official helper semantics | Unit tests for positions, masks, corruption, timestep sampling, selected-canvas sampling, loss normalization, self-conditioning post-norm | Pass |
| Full tiny official logits parity | `scripts/verify_diffusion_gemma_official_logits.py` | `plain_logits max_abs_diff=2.8405338525772095e-08`; encode/self-conditioned logits `0.0` |
| LoRA-only local update | Tiny smoke and unit tests | LoRA params update, sampled non-LoRA params unchanged |
| Real PubMedQA local tiny smoke | `scripts/smoke_diffusion_gemma_pubmedqa_tunix.py --tiny ...` | Finite loss; loss decreased from `11.092823028564453` to `11.091582298278809`; LoRA checksum changed; non-LoRA unchanged |
| Public 26B native Tunix smoke | 8x RTX PRO 6000 96GB, official-length PubMedQA shape | Full-loss 1-step LoRA smoke succeeded; final loss `11.146772384643555`; LoRA checksum delta nonzero; sampled non-LoRA delta `0.0` |
| Official backend config/build | H100/RTX/A100 official wrapper config runs | Official deps and PubMedQA configs build with `gemma==4.0.1`, `hackable_diffusion==1.0.1`, `kauldron==1.4.4` |
| Official backend 2-GPU H100 | H100 x2 VM, CUDA12, official Kauldron loop via Tunix wrapper | `train: 100%|2/2`, `official_backend_train_complete`, exit `0` |
| Official backend 2-GPU RTX PRO 6000 | RTX PRO 6000 x2 container, CUDA13, official Kauldron loop via Tunix wrapper | `train: 100%|2/2`, `official_backend_train_complete`, exit `0` |
| Official backend A100 x2 failure characterization | A100-80GB x2 container | Official train loop reached `train: 100%|2/2`, but final Kauldron `_sync()` failed with fatal NCCL `ncclAllReduce invalid argument` |

## Official Parity Details

The fork includes two classes of parity checks.

### Helper And Loss Semantics

The native Tunix path is checked against official behavior for:

- Prompt/canvas positions.
- Prefill and decoder masks.
- Cache `end_index`.
- Diffusion timestep sampling.
- Categorical corruption.
- Selected-canvas sampling.
- Unweighted discrete loss normalization.
- Self-conditioning post-norm behavior.
- Split decoder/encoder loss gradients.

These checks live primarily in:

- `tests/models/diffusion_gemma_test.py`
- `scripts/verify_diffusion_gemma_official_parity.py`

### Tiny Full-Model Logits Parity

The strongest local model-equivalence check is:

```bash
python scripts/verify_diffusion_gemma_official_logits.py
```

Recorded result:

- Official Gemma revision: `682e412`
- Copied parameters: 19 leaves
- `max_abs_diff_after_copy=0.0`
- `encode_logits`: shape `[2, 4, 8]`, `max_abs_diff=0.0`
- `plain_logits`: shape `[2, 4, 32]`,
  `max_abs_diff=2.8405338525772095e-08`
- `self_conditioned_logits`: shape `[2, 4, 32]`, `max_abs_diff=0.0`

Interpretation: the direct transformer forward path of the native Tunix model
matches the official Flax/Linen model to numerical tolerance on a tiny
non-MoE configuration after copying official parameters into the Tunix model.

## Training Evidence

### Local Tiny Training

Local tiny SFT checks verify that the Tunix trainer path is trainable at small
scale:

- Loss is finite.
- Gradients can be computed.
- LoRA parameters update.
- Sampled non-LoRA parameters remain unchanged.
- Minimal training state/checkpoint artifact is written.

### Real PubMedQA Formatting

The native PubMedQA smoke mirrors the official DeepMind PubMedQA recipe data
shape and prompt/target conventions:

- Source: `pubmedqa/pubmedqa`
- Train examples: 500 after removing official test IDs
- Test examples: 500
- Official-style medical research assistant prompt
- Long-answer target ending with `The answer is: yes|no|maybe<turn|>`
- Gemma4 tokenizer/canvas layout
- Prompt length/canvas length controls matching the official recipe

Latest local tiny result:

- First PubMed ID: `24785562`
- Initial loss: `11.092823028564453`
- Final loss: `11.091582298278809`
- LoRA checksum delta: `0.0013999984366819263`
- Sampled non-LoRA checksum delta: `0.0`

### Public 26B Native Tunix Smoke

The native Tunix public-checkpoint path was tested on 8x RTX PRO 6000 96GB:

- Checkpoint: public `diffusiongemma-26B-A4B-it`
- Shape: prompt length 1024, canvas size 128, two canvases
- Batch size: 1
- Decoder path: `cached_selected_canvas_slice`
- Decoder remat enabled
- Encoder AR loss enabled
- LoRA rank 4, alpha 8.0
- Mesh: `mesh_fsdp=4`, `mesh_tp=2`

Full-loss result:

- Initial total loss: `12.994644165039062`
- Initial decoder loss: `5.365616798400879`
- Initial encoder loss: `7.629027843475342`
- Final total loss: `11.146772384643555`
- Final decoder loss: `4.296473503112793`
- Final encoder loss: `6.850298881530762`
- LoRA coverage: 366 leaves / 4,423,936 elements
- LoRA checksum delta: `0.029854297637939453`
- Sampled non-LoRA checksum delta: `0.0`
- Artifact: `minimal_state.json`

Interpretation: the native Tunix path can load the public 26B checkpoint, run
real PubMedQA SFT loss, update only LoRA parameters, and save a minimal state
artifact on an 8-GPU high-memory setup.

## Official Backend Evidence

The official backend wrapper is deliberately a reference path, not a native
rewrite. It lets the fork run official DeepMind training logic under Tunix
orchestration.

The important distinction:

- The **model, data factories, loss, LoRA wrapping, optimizer, sharding, and
  Kauldron trainer** come from the official implementation.
- The **entrypoint, path/config overrides, checkpoint downloader, logging,
  smoke step count, and selected debug switches** come from this Tunix fork.

This gives the fork a reliable oracle for:

- resource comparisons,
- official-vs-native loss/training behavior,
- GPU runtime diagnosis,
- reproducible smoke commands,
- evidence capture.

### Cross-GPU Official Backend Smoke

Same official PubMedQA smoke:

- Official Kauldron loop
- Batch size 2
- LoRA rank 4
- Prompt length 1024
- Canvas size 128
- Two canvases
- Step metrics skipped to avoid writer-side materialization noise

Results:

- RTX PRO 6000 x2: `r_6f265dee`, succeeded, exit `0`.
- H100 x2: `r_2200993f`, succeeded, exit `0`.
- A100-80GB x2: official train loop reached `train: 100%|2/2`, but final
  Kauldron `_sync()` failed with fatal NCCL error.

Interpretation: the official backend works through this fork on at least two
2-GPU high-memory runtimes. The A100 failure is environment/runtime-specific,
not evidence that the official implementation is generally broken.

Evidence files:

- `evidence/diffusion_gemma/official_backend_cross_gpu_2026-06-12.md`
- `evidence/diffusion_gemma/a100x2_official_backend_hybrid_2026-06-12.md`

## What This Fork Proves

This fork demonstrates:

- DiffusionGemma has a Tunix model family entrypoint.
- The core official SFT semantics have been ported into the native Tunix path.
- Tiny full-model logits parity against official Flax/Linen is achieved for
  direct transformer forward behavior.
- The native Tunix SFT path performs finite-loss LoRA training and updates only
  LoRA parameters.
- Real PubMedQA formatting and loss construction match the official recipe
  closely enough for public-checkpoint smoke training.
- A public 26B checkpoint can be loaded and trained for a 1-step LoRA smoke on
  8 high-memory GPUs.
- The fork can also invoke the official DeepMind implementation as a reference
  backend and complete official 2-GPU Kauldron smokes on H100 x2 and RTX PRO
  6000 x2.

## What This Fork Does Not Yet Prove

The report should not be read as claiming:

- Full native Tunix production parity with official Kauldron on 2xH100.
- Full cached SFT logits parity against the complete official Flax/Linen stack.
- Long training convergence.
- DPO/GRPO support.
- vLLM or serving support.
- Fully upstream-ready API polish.
- That the official backend wrapper is a pure upstream DeepMind run without
  Tunix involvement.

Known native Tunix gap:

- The official Kauldron recipe can run the 2-GPU H100 smoke, but the native
  NNX/Qwix/PeftTrainer path still OOMs on H100 x2 for the public 26B train
  step. It currently requires the 8-GPU high-memory setup for full-loss 26B
  smoke training.

## Why The Official Wrapper Still Matters

Even though it is not a full native Tunix integration, the wrapper has concrete
engineering value:

- It gives Tunix a reference oracle for official DiffusionGemma behavior.
- It lets native Tunix work be compared against official loss/resource behavior
  using the same scripts and evidence format.
- It standardizes GPU smoke commands, checkpoint mirroring, dataset paths, and
  config overrides.
- It makes environment problems obvious: for example, A100 x2 failed while
  H100 x2 and RTX PRO 6000 x2 completed the same official smoke.
- It provides a lower-risk execution path while native NNX/Qwix memory
  efficiency catches up.

In short: the wrapper is not the final destination, but it is the measurement
rig that makes the native port credible.

## Recommended Next Gates

1. Add cached SFT logits parity against the full official Flax/Linen stack.
2. Reduce native Tunix train-step memory so the public 26B path fits on H100 x2.
3. Run a multi-step native 26B smoke after first-JIT warmup to measure
   steady-state throughput.
4. Add a pure upstream DeepMind-repo runbook for a no-Tunix control experiment.
5. Convert the current smoke/evidence commands into CI-friendly optional GPU
   jobs.
