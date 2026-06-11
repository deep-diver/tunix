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
- Includes a no-tuning generation demo with official-style confidence selection, annealed temperature, token-stability plus entropy early stopping, JSON trace export, and a self-contained HTML animation of every denoising frame.

## Current Limitations

- The NNX MVP uses a full-sequence denoising compatibility path for the decoder loss. It still builds the prefilled KV cache and selected-canvas `end_index`, but Tunix Gemma4 does not yet expose the official multi-token canvas attention-over-prefilled-cache path used by the Flax Linen implementation.
- The full-model logits parity script covers the direct transformer forward path without cache. The cached official SFT decoder path is still distinct from the MVP full-sequence denoising compatibility path because Tunix Gemma4 cache updates do not yet implement the official multi-token canvas write-at-`end_index` behavior.
- Upstream Orbax checkpoint loading maps the Gemma4-compatible backbone and known `self_conditioner` leaves. The public `diffusiongemma-26B-A4B-it` checkpoint has been loaded successfully on a 4xH100 JAX mesh.
- The no-tuning generation demo uses the full-sequence no-cache path covered by logits parity, not the official cached production sampler. Treat it as a load/denoise visibility smoke test, not a quality benchmark.

## Local Commands

```bash
python -m pyink tunix/models/diffusion_gemma tests/models/diffusion_gemma_test.py scripts/smoke_diffusion_gemma_tunix.py tunix/models/automodel.py tunix/models/naming.py
python -m pytest tests/models/diffusion_gemma_test.py -q --import-mode=importlib
python scripts/verify_diffusion_gemma_official_parity.py
python scripts/verify_diffusion_gemma_official_logits.py
python scripts/generate_diffusion_gemma_tunix.py --tiny --max_new_tokens 4 --canvas_length 4 --denoising_steps 3 --prompt "Hi" --animation_output /tmp/diffusion_gemma_tiny_trace.html --trace_json /tmp/diffusion_gemma_tiny_trace.json
python scripts/render_diffusion_gemma_trace_gif.py /tmp/diffusion_gemma_tiny_trace.json --output /tmp/diffusion_gemma_tiny_trace.gif --width 900 --height 520
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

## Public Checkpoint Generation Smoke

The public checkpoint is stored in Google Cloud Storage and can be mirrored with
the small HTTP downloader in this branch:

```bash
mkdir -p /tmp/tunix-dg-generate/scripts
rsync -a --delete --exclude __pycache__/ tunix/ /tmp/tunix-dg-generate/tunix/
rsync -a pyproject.toml README.md requirements-gpu.txt /tmp/tunix-dg-generate/
rsync -a scripts/generate_diffusion_gemma_tunix.py scripts/download_public_gcs_prefix.py /tmp/tunix-dg-generate/scripts/

jl create --gpu H100 --region IN2 --num-gpus 4 --storage 300 --vm --yes --json
jl run /tmp/tunix-dg-generate --script scripts/download_public_gcs_prefix.py --on <machine_id> --requirements /tmp/tunix-dg-generate/requirements-gpu.txt --json --yes -- --bucket gemma-data --prefix checkpoints/diffusiongemma-26B-A4B-it/ --dest /home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it --workers 8
jl exec <machine_id> --json -- sh -lc 'mkdir -p /home/ubuntu/checkpoints/tokenizers && curl -L https://storage.googleapis.com/gemma-data/tokenizers/tokenizer_gemma4.model -o /home/ubuntu/checkpoints/tokenizers/tokenizer_gemma4.model'
jl run /tmp/tunix-dg-generate --script scripts/generate_diffusion_gemma_tunix.py --on <machine_id> --requirements /tmp/tunix-dg-generate/requirements-gpu.txt --json --yes -- --checkpoint /home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it --tokenizer /home/ubuntu/checkpoints/tokenizers/tokenizer_gemma4.model --prompt "What are diffusion LLMs? Answer in two concise sentences." --max_new_tokens 16 --canvas_length 16 --denoising_steps 4 --seed 7 --mesh_fsdp 4 --mesh_tp 1 --animation_output /home/ubuntu/diffusion_gemma_trace.html --trace_json /home/ubuntu/diffusion_gemma_trace.json
jl run logs <run_id> --tail 120
jl download <machine_id> /home/ubuntu/diffusion_gemma_trace.html /tmp/diffusion_gemma_trace.html
jl download <machine_id> /home/ubuntu/diffusion_gemma_trace.json /tmp/diffusion_gemma_trace.json
python scripts/render_diffusion_gemma_trace_gif.py /tmp/diffusion_gemma_trace.json --output /tmp/diffusion_gemma_trace.gif
jl destroy <machine_id> --yes --json
```

Latest verified 4xH100 generation run:

- Machine: `424943` (`H100`, `IN2`, 4 GPUs, VM), destroyed after the run.
- JAX devices: `cuda:0`, `cuda:1`, `cuda:2`, `cuda:3`.
- Checkpoint mirror: 31 files, 37.633 GiB.
- Checkpoint download run: `r_8683c723`, completed in 351.254 seconds.
- Step-visible generation run: `r_3c3c26ee`, 32-token canvas, 8 denoising steps, completed in 79.06 seconds.
- Animation artifact: `/tmp/diffusion_gemma_trace.html`, with 9 frames (initial canvas plus 8 denoising steps).
- GIF artifact: `/tmp/diffusion_gemma_trace.gif`, 1280x720, 36 frames, with local token-cell denoising noise.
- Final visible output: `<|channel>thought\n<channel|>Diffusion language models are tapered that generate text by iteratively refining noisy data into ... sequences through a reverse diffusion process.<eos>`.

## References

- Official DiffusionGemma implementation: <https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion>
- Official SFT recipe: <https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion/hackable_diffusion_adapter>
- Tunix SFT API: <https://tunix.readthedocs.io/en/latest/api/api_sft.html>
