# DiffusionGemma Tunix MVP

This directory is the runbook for the Tunix DiffusionGemma MVP integration.

## What Works

- Adds `diffusion_gemma` as a Tunix model family.
- Adds `DiffusionGemma_A26B_A4B` as an NNX model that reuses Tunix Gemma4 A26B/A4B and adds a self-conditioning block.
- Adds a DiffusionGemma SFT adapter around `PeftTrainer.with_gen_model_input_fn()` and `with_loss_fn(has_aux=True)`.
- Implements prompt + clean canvas prefill, KV cache construction, diffusion timestep sampling, token corruption, selected canvas denoising loss, self-conditioning second pass, encoder AR loss, and LoRA SFT.
- Matches official helper semantics for positions, prefill/decoder masks, cache `end_index`, `SafeSpan(1e-4)` timestep sampling, categorical corruption, selected-canvas sampling, unweighted discrete loss normalization, and unscaled self-conditioning post-norm.
- Matches official full-model logits on a tiny non-MoE DiffusionGemma config by copying official Flax/Linen weights into the Tunix NNX model and comparing complete-sequence plain logits plus self-conditioning logits.
- Includes a tiny synthetic smoke script that checks finite loss, LoRA-only updates, and checkpoint directory creation.

## Current Limitations

- The NNX MVP uses a full-sequence denoising compatibility path for the decoder loss. It still builds the prefilled KV cache and selected-canvas `end_index`, but Tunix Gemma4 does not yet expose the official multi-token canvas attention-over-prefilled-cache path used by the Flax Linen implementation.
- The full-model logits parity script covers the direct transformer forward path without cache. The cached official SFT decoder path is still distinct from the MVP full-sequence denoising compatibility path because Tunix Gemma4 cache updates do not yet implement the official multi-token canvas write-at-`end_index` behavior.
- Upstream Orbax checkpoint loading maps the Gemma4-compatible backbone and known `self_conditioner` leaves, but a real public-checkpoint alignment run is still needed once the full DiffusionGemma checkpoint is available in the target environment.
- Sampling/eval is not ported yet; this MVP is SFT smoke training only.

## Local Commands

```bash
python -m pyink tunix/models/diffusion_gemma tests/models/diffusion_gemma_test.py scripts/smoke_diffusion_gemma_tunix.py tunix/models/automodel.py tunix/models/naming.py
python -m pytest tests/models/diffusion_gemma_test.py -q --import-mode=importlib
python scripts/verify_diffusion_gemma_official_parity.py
python scripts/verify_diffusion_gemma_official_logits.py
python -m pytest tests/models/registry_test.py -q --import-mode=importlib
python -m pytest tests/models/naming_test.py -q --import-mode=importlib -k 'not model_id_exists_on_huggingface'
python scripts/smoke_diffusion_gemma_tunix.py --steps 3 --use_lora
python -m pytest
```

`scripts/verify_diffusion_gemma_official_parity.py` expects local reference
checkouts at `/tmp/gemma-diffusion-reference` and
`/tmp/hackable-diffusion-reference` by default. It imports the official helper
modules directly and fails if any compared intermediate differs outside
tolerance.

`scripts/verify_diffusion_gemma_official_logits.py` expects the official Gemma
checkout at `/tmp/gemma-diffusion-reference` by default. It stubs only the
Kauldron import surface needed on local machines, runs the official
Flax/Linen DiffusionGemma model, copies its tiny-model parameters into the Tunix
NNX model, and fails if complete-sequence logits differ outside tolerance.

Latest verified official logits parity:

- Official Gemma revision: `682e412`
- Copied parameters: 19 leaves, all `max_abs_diff_after_copy=0.0`
- `encode_logits`: shape `[2, 4, 8]`, `max_abs_diff=0.0`
- `plain_logits`: shape `[2, 4, 32]`, `max_abs_diff=2.8405338525772095e-08`
- `self_conditioned_logits`: shape `[2, 4, 32]`, `max_abs_diff=0.0`

`python -m pytest` currently requires optional serving backends (`vllm`,
`sgl_jax`) for some generation tests. In the local environment used for this
MVP, the full suite also aborts in `tests/generate/tokenizer_adapter_test.py`
inside SentencePiece native code. With that crashing file ignored, the latest
run reached `1528 passed`, `34 failed`, and `105 errors`; the remaining failures
are unrelated optional-backend imports, JAX CPU-device ordering setup, and
pre-existing sampler/SFT/RL/smoke-test environment failures. The
DiffusionGemma-specific tests and parity scripts above pass.

## JarvisLabs Smoke

```bash
jl status --json
jl gpus --json
jl resources --json
jl create --gpu H100 --storage 200 --template pytorch --yes --json
jl run . --script scripts/smoke_diffusion_gemma_tunix.py --on <machine_id> --requirements requirements-gpu.txt --json --yes -- --steps 5 --use_lora
sleep 15 && jl run logs <run_id> --tail 30
sleep 120 && jl run logs <run_id> --tail 50
jl destroy <machine_id> --yes --json
```

If H100 is unavailable, choose the best available H200/H100-class GPU reported by `jl gpus --json`; do not guess the exact SKU.

For a faster upload, stage only the files needed by the smoke test:

```bash
mkdir -p /tmp/tunix-dg-smoke/scripts
rsync -a --delete --exclude __pycache__/ tunix/ /tmp/tunix-dg-smoke/tunix/
rsync -a pyproject.toml README.md requirements-gpu.txt /tmp/tunix-dg-smoke/
rsync -a scripts/smoke_diffusion_gemma_tunix.py /tmp/tunix-dg-smoke/scripts/
cd /tmp/tunix-dg-smoke
jl run . --script scripts/smoke_diffusion_gemma_tunix.py --on <machine_id> --requirements requirements-gpu.txt --json --yes -- --steps 5 --use_lora
```

Latest verified H100 run:

- Machine: `424801` (`H100`, `EU1`), destroyed after the run.
- Run: `r_220b9ed9`
- Result: `cuda:0`, finite initial loss, 5 training steps, checkpoint directory created, `base_params_unchanged=true`, `lora_params_changed=true`.

## References

- Official DiffusionGemma implementation: <https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion>
- Official SFT recipe: <https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion/hackable_diffusion_adapter>
- Tunix SFT API: <https://tunix.readthedocs.io/en/latest/api/api_sft.html>
