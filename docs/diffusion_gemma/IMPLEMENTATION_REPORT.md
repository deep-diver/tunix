# DiffusionGemma Hackable Diffusion Wrapper Report

Date: 2026-06-12

Branch: `codex/diffusion-gemma-integration`

Wrapper implementation baseline: `0faad277`

## Claim

This fork provides a Tunix-hosted wrapper for the official DeepMind
DiffusionGemma SFT recipe. The wrapper keeps the official Gemma Diffusion,
Hackable Diffusion, and Kauldron training implementation as the source of
truth, while adding a Tunix-side entrypoint for setup, configuration overrides,
controlled runs, and repeatable GPU validation.

This report is intentionally scoped to the current Hackable Diffusion backend
wrapper. It does not claim that the native Tunix NNX/Qwix implementation is a
production-ready replacement for the official recipe.

## Files

The relevant implementation files are:

- `tunix/models/diffusion_gemma/hackable_adapter.py`
- `scripts/run_diffusion_gemma_official_backend.py`
- `scripts/setup_diffusion_gemma_official_backend.sh`

The wrapper entrypoint can be inspected with:

```bash
python scripts/run_diffusion_gemma_official_backend.py --help
```

## Integration Shape

The official implementation remains responsible for:

- DiffusionGemma model construction.
- Hackable Diffusion SFT model and loss.
- PubMedQA/Sudoku recipe config construction.
- Token corruption, diffusion timestep sampling, denoising loss, and
  self-conditioning behavior.
- LoRA wrapping.
- Optimizer and sharding setup.
- Kauldron training loop.

The Tunix fork is responsible for:

- Importing and launching the official recipe from inside the Tunix repository.
- Installing or preparing the official backend dependencies.
- Overriding work directories, checkpoint paths, step counts, LoRA rank, batch
  size, and selected recipe constants for validation runs.
- Exposing a stable script interface for GPU validation.
- Capturing concise run output that can be compared across GPU environments.

This means the wrapper is not a reimplementation of the official algorithm. It
is an adapter that lets Tunix run the official DiffusionGemma SFT backend in a
controlled, reproducible way.

## Supported Modes

The wrapper currently exposes two execution modes:

- `kauldron`: calls the official Kauldron trainer loop.
- `hybrid`: reuses official model, data, loss, and train-step objects while
  giving Tunix more control over selected validation mechanics.

The main compatibility signal is the `kauldron` mode, because it exercises the
official training loop directly.

## Validation Summary

The wrapper was validated with the official PubMedQA LoRA configuration:

- Recipe: `pubmedqa`
- Train loop: `kauldron`
- LoRA rank: 4
- Dataset batch size: 2
- Prompt length: official recipe shape
- Canvas layout: official recipe shape
- Training length: 2 validation steps
- Step metrics: skipped for validation-run stability

Passing GPU validation results:

| GPU runtime | Result |
| --- | --- |
| H100 x2 | Completed `2/2` training steps and exited with status `0` |
| RTX PRO 6000 x2 | Completed `2/2` training steps and exited with status `0` |

Both passing runs reached the wrapper completion marker:

```text
official_backend_train_complete
```

This is the strongest current evidence that the fork can launch and complete
the official DiffusionGemma Hackable Diffusion SFT path through the Tunix
wrapper on real multi-GPU machines.

## Example Commands

Prepare the official backend environment:

```bash
bash scripts/setup_diffusion_gemma_official_backend.sh
```

Build the official recipe config without launching training:

```bash
python scripts/run_diffusion_gemma_official_backend.py \
  --recipe pubmedqa \
  --build_config_only \
  --workdir /tmp/diffusion_gemma_official_backend
```

Run a short official-backend validation:

```bash
python scripts/run_diffusion_gemma_official_backend.py \
  --recipe pubmedqa \
  --train_loop kauldron \
  --num_train_steps 2 \
  --dataset_batch_size 2 \
  --lora_rank 4 \
  --skip_step_metrics \
  --workdir /tmp/diffusion_gemma_official_backend
```

For JarvisLabs, the same script was launched through `jl run` on the selected
GPU machine, with the repository as the run directory.

## What This Proves

The current fork demonstrates that:

- Tunix can host a stable entrypoint for the official DiffusionGemma SFT recipe.
- The official Hackable Diffusion and Kauldron backend can be imported,
  configured, and launched from the Tunix repository.
- The official PubMedQA LoRA path completes on real 2-GPU H100 and RTX
  PRO 6000 runtimes.
- The integration preserves the official backend as the behavioral authority
  instead of silently replacing it with a partial native rewrite.

## What This Does Not Claim

This report does not claim:

- Full native Tunix NNX/Qwix parity with the official implementation.
- Long training convergence.
- Full fine-tuning instead of LoRA validation.
- Serving support.
- DPO, GRPO, or other post-training methods beyond the official SFT path.

## Why This Wrapper Is Useful

The wrapper gives Tunix a practical bridge to DiffusionGemma today:

- It provides a known-good official reference path inside the Tunix fork.
- It makes GPU validation repeatable with one script.
- It gives future native Tunix work a concrete oracle for behavior and resource
  comparison.
- It keeps the current integration honest: the official backend is still doing
  the model and training work, while Tunix supplies orchestration and
  compatibility glue.

The next engineering step is to use this wrapper as the reference while
incrementally moving more behavior into native Tunix only when parity and
memory behavior can be proven.
