# DiffusionGemma Tunix MVP

This directory is the runbook for the Tunix DiffusionGemma MVP integration.

## What Works

- Adds `diffusion_gemma` as a Tunix model family.
- Adds `DiffusionGemma_A26B_A4B` as an NNX model that reuses Tunix Gemma4 A26B/A4B and adds a self-conditioning block.
- Adds a DiffusionGemma SFT adapter around `PeftTrainer.with_gen_model_input_fn()` and `with_loss_fn(has_aux=True)`.
- Implements prompt + clean canvas prefill, KV cache construction, diffusion timestep sampling, token corruption, selected canvas denoising loss, self-conditioning second pass, encoder AR loss, and LoRA SFT.
- Implements an official-style cached selected-canvas decoder path plus a `cached_selected_canvas_slice` smoke path that decodes only the selected canvas loss window.
- Matches official helper semantics for positions, prefill/decoder masks, cache `end_index`, `SafeSpan(1e-4)` timestep sampling, categorical corruption, selected-canvas sampling, unweighted discrete loss normalization, and unscaled self-conditioning post-norm.
- Matches official full-model logits on a tiny non-MoE DiffusionGemma config by copying official Flax/Linen weights into the Tunix NNX model and comparing complete-sequence plain logits plus self-conditioning logits.
- Includes a tiny synthetic smoke script that checks finite loss, LoRA-only updates, and checkpoint directory creation.
- Includes a PubMedQA real-data LoRA SFT smoke script that mirrors the official DeepMind PubMedQA split and prompt/answer formatting without importing Kauldron or Grain.
- Includes a no-tuning generation demo with official-style confidence selection, annealed temperature, token-stability plus entropy early stopping, JSON trace export, and a self-contained HTML animation of every denoising frame.

## Current Limitations

- The NNX MVP now has cached selected-canvas decoding, including per-example cache `end_index` updates. The `cached_selected_canvas_slice` mode is a lower-memory smoke path for the selected loss window; it is parity-checked against the full cached path when the selected canvas is the first canvas.
- The full-model logits parity script covers the direct transformer forward path without cache. Cached SFT logits parity against the full official Flax/Linen stack is still a next gate.
- Upstream Orbax checkpoint loading maps the Gemma4-compatible backbone and known `self_conditioner` leaves. The public `diffusiongemma-26B-A4B-it` checkpoint has been loaded successfully on 4xH100 and 8x96GB JAX meshes.
- The no-tuning generation demo uses the full-sequence no-cache path covered by logits parity, not the official cached production sampler. Treat it as a load/denoise visibility smoke test, not a quality benchmark.
- Public 26B PubMedQA LoRA tuning now passes 1-step official-length smokes on an 8x RTX PRO 6000 96GB container with `mesh_fsdp=4, mesh_tp=2`, `prompt_len=1024`, `num_canvases=2`, `canvas_size=128`, batch size 1, long answers, `cached_selected_canvas_slice`, decoder remat, self-conditioning, encoder AR loss, full backbone LoRA coverage, LoRA-only update, base-parameter non-update, VRAM telemetry, and minimal state save.
- The official PubMedQA SFT recipe uses the same prompt/canvas geometry, batch size 2, long answers, encoder AR loss, LoRA rank 4, and 2000 train steps. A direct official Kauldron run on 2xH100 successfully saved `ckpt_0` and completed train step 1 at batch size 2. The Tunix MVP does not yet match that 2xH100 memory profile: `mesh_fsdp=2, mesh_tp=1` still OOMs during `jit__train_step`, even with selected-canvas slice decoding and decoder remat.
- For multi-GPU public-checkpoint runs, `scripts/smoke_diffusion_gemma_pubmedqa_tunix.py` now defaults to an official-style FSDP-first mesh (`mesh_fsdp=jax.device_count(), mesh_tp=1`) when both mesh flags are omitted. Tensor parallelism can still be requested explicitly, but TP-only is not the right H100x2 comparison for the official recipe.
- Decoder rematerialization is compatible with Qwix LoRA materialization in this MVP. The SFT adapter temporarily disables decoder remat while Qwix discovers LoRA targets, then restores remat for the actual forward/train path; GPU logs verify the same 366 LoRA leaves with `--remat_decoder`.
- For public 26B smoke runs, `scripts/smoke_diffusion_gemma_pubmedqa_tunix.py` defaults to a minimal `minimal_state.json` proof artifact instead of Tunix/Orbax optimizer checkpointing. Use `--orbax_checkpoint` only when the shape is known to fit; the public 26B optimizer checkpoint path can exceed memory.

## Local Commands

```bash
python -m pyink tunix/models/diffusion_gemma tests/models/diffusion_gemma_test.py scripts/smoke_diffusion_gemma_tunix.py tunix/models/automodel.py tunix/models/naming.py
python -m pytest tests/models/diffusion_gemma_test.py -q --import-mode=importlib
python scripts/verify_diffusion_gemma_official_parity.py
python scripts/verify_diffusion_gemma_official_logits.py
python scripts/generate_diffusion_gemma_tunix.py --tiny --max_new_tokens 4 --canvas_length 4 --denoising_steps 3 --prompt "Hi" --animation_output /tmp/diffusion_gemma_tiny_trace.html --trace_json /tmp/diffusion_gemma_tiny_trace.json
python scripts/render_diffusion_gemma_trace_gif.py /tmp/diffusion_gemma_tiny_trace.json --output /tmp/diffusion_gemma_tiny_trace.gif --width 900 --height 520
python scripts/smoke_diffusion_gemma_pubmedqa_tunix.py --tiny --steps 1 --batch_size 1 --gradient_accumulation_steps 2 --prompt_len 128 --canvas_size 32 --num_canvases 1 --max_examples 4 --max_context_chars 600 --tiny_vocab_size 256 --decoder_implementation cached_selected_canvas_slice
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

## PubMedQA SFT Smoke

The official DeepMind DiffusionGemma SFT recipe includes PubMedQA data setup in
`gemma/diffusion/hackable_diffusion_adapter/configs/sft_pubmedqa.py` and
`.../data/pubmedqa`. The Tunix smoke mirrors the relevant pieces:

- data source: `pubmedqa/pubmedqa`
- train split: `ori_pqal.json` minus IDs in `test_ground_truth.json`
- train examples: 500; test examples: 500
- prompt: official medical research assistant system prompt plus PubMedQA context/question
- target: long answer by default, ending with `The answer is: yes|no|maybe<turn|>`
- tokenization/canvas layout: Gemma4 tokenizer, BOS prompt token, response chunks, EOS-filled final valid canvas, `canvas_id`, `canvas_mask`, and shifted encoder target

Local tiny real-data command:

```bash
python scripts/smoke_diffusion_gemma_pubmedqa_tunix.py --tiny --steps 2 --batch_size 1 --prompt_len 128 --canvas_size 32 --num_canvases 1 --max_examples 4 --max_context_chars 600 --tiny_vocab_size 256
```

Latest verified local tiny PubMedQA result:

- selected examples: 4 from official train split; first PubMed ID `24785562`
- initial loss: `11.092823028564453`
- final loss after 1 LoRA step with 2 microbatches: `11.091582298278809`
- LoRA checksum delta: `0.0013999984366819263`
- sampled non-LoRA checksum delta: `0.0`
- artifact: `minimal_state.json` in the run checkpoint directory

Public 26B PubMedQA GPU commands used for the latest H100 attempts:

```bash
jl run --on <machine_id> --json --yes -- sh -lc 'cd /home/ubuntu/tunix-dg-pubmedqa && . .venv/bin/activate && python scripts/smoke_diffusion_gemma_pubmedqa_tunix.py --steps 1 --batch_size 1 --prompt_len 64 --canvas_size 8 --num_canvases 1 --max_examples 2 --max_context_chars 300 --no-use_long_answer --checkpoint /home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it --tokenizer /home/ubuntu/checkpoints/tokenizers/tokenizer_gemma4.model --mesh_fsdp 4 --mesh_tp 1 --restore_concurrent_gb 16 --checkpoint_dir /home/ubuntu/diffusion_gemma_pubmedqa_state_short_v2 --lora_rank 4 --lora_alpha 8.0 --fast_uniform_corruption'
jl run --on <machine_id> --json --yes -- sh -lc 'cd /home/ubuntu/tunix-dg-pubmedqa-tp && . .venv/bin/activate && python scripts/smoke_diffusion_gemma_pubmedqa_tunix.py --steps 1 --batch_size 1 --prompt_len 64 --canvas_size 8 --num_canvases 1 --max_examples 2 --max_context_chars 300 --no-use_long_answer --checkpoint /home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it --tokenizer /home/ubuntu/checkpoints/tokenizers/tokenizer_gemma4.model --mesh_fsdp 1 --mesh_tp 4 --restore_concurrent_gb 16 --checkpoint_dir /home/ubuntu/diffusion_gemma_pubmedqa_state_tp4 --lora_rank 4 --lora_alpha 8.0 --fast_uniform_corruption'
jl run --on <machine_id> --json --yes -- sh -lc 'cd /home/ubuntu/tunix-dg-pubmedqa-tp && . .venv/bin/activate && python scripts/smoke_diffusion_gemma_pubmedqa_tunix.py --steps 1 --batch_size 1 --prompt_len 64 --canvas_size 8 --num_canvases 1 --max_examples 2 --max_context_chars 300 --no-use_long_answer --checkpoint /home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it --tokenizer /home/ubuntu/checkpoints/tokenizers/tokenizer_gemma4.model --mesh_fsdp 2 --mesh_tp 2 --restore_concurrent_gb 16 --checkpoint_dir /home/ubuntu/diffusion_gemma_pubmedqa_state_fsdp2_tp2 --lora_rank 4 --lora_alpha 8.0 --fast_uniform_corruption'
```

H100x2 official-vs-Tunix verification:

```bash
jl create --gpu H100 --num-gpus 2 --storage 300 --vm --region IN2 --yes --json
jl run /tmp/tunix-dg-h100x2 --script scripts/download_public_gcs_prefix.py --on <machine_id> --requirements /tmp/tunix-dg-h100x2/requirements-gpu.txt --json --yes -- --bucket gemma-data --prefix checkpoints/diffusiongemma-26B-A4B-it/ --dest /home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it --workers 16
jl exec <machine_id> --json -- sh -lc 'mkdir -p /home/ubuntu/checkpoints/tokenizers && curl -L https://storage.googleapis.com/gemma-data/tokenizers/tokenizer_gemma4.model -o /home/ubuntu/checkpoints/tokenizers/tokenizer_gemma4.model'
jl run --on <machine_id> --json --yes -- sh -lc 'cd /home/ubuntu/tunix-dg-h100x2 && . .venv/bin/activate && env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" NCCL_ALGO=Ring NCCL_PROTO=LL128 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 python3 scripts/smoke_diffusion_gemma_pubmedqa_tunix.py --steps 1 --batch_size 1 --prompt_len 1024 --canvas_size 128 --num_canvases 2 --max_examples 4 --max_context_chars 4000 --checkpoint /home/ubuntu/checkpoints/diffusiongemma-26B-A4B-it --tokenizer /home/ubuntu/checkpoints/tokenizers/tokenizer_gemma4.model --mesh_fsdp 2 --mesh_tp 1 --restore_concurrent_gb 16 --checkpoint_dir /home/ubuntu/diffusion_gemma_pubmedqa_state_h100x2_fsdp2_encoder_b1 --lora_rank 4 --lora_alpha 8.0 --fast_uniform_corruption --decoder_implementation cached_selected_canvas_slice --remat_decoder --gpu_memory_poll_seconds 2'
jl destroy <machine_id> --yes --json
```

Latest verified H100x2 comparison:

- Machine: `425205` (`H100`, `IN2`, 2 GPUs, VM), destroyed after the run. `jl status --json` reported `running_instances: 0` and `running_vms: 0` after destroy.
- Checkpoint mirror: `r_d0ffa7fc`, 31 objects, 37.633 GiB, 207.234 seconds.
- Official Gemma checkout: `8fb37ee37e43d23165fe8919c4b1bb9c6e57492a`, Python 3.12, CUDA 13 JAX wheel (`jax==0.10.1`), both H100 devices visible to JAX.
- Official PubMedQA source patch used for the smoke was path-only plus one enum alias: local checkpoint/tokenizer paths and `DIFFUSIONGEMMA_26B_A4B_IT = DIFFUSIONGEMMA_A26B_A4B_IT`. No training/model logic was changed.
- Official config evidence: `kd.sharding.ShardingStrategy(params=kd.sharding.FSDPSharding(), opt_state=kd.sharding.FSDPSharding())`, `use_lora=True`, `target_modules="all-linear"`, train `batch_size=2`, `num_train_steps=2_000`.
- Official run: `r_e97874e5`; `ckpt_0` saved a 47.1 GiB checkpoint, then train step 1 completed with `losses/diffusion_loss=6.328125`, `losses/encoder_loss=7.25`, `losses/total=13.578125`, `perf_stats/train/avg_time_sec=259.697`, and about `65.6GB` used / `15.4GB` free on each H100. The run was manually stopped after first-step success, so JarvisLabs records exit 143.
- Tunix TP-only comparison: `r_d83c9316`, `mesh_fsdp=1, mesh_tp=2`, batch size 1. Initial loss was finite (`total=12.563678741455078`, decoder `4.934838771820068`, encoder `7.62883996963501`), LoRA coverage was 366 leaves / 4,423,936 elements, but train step failed with `RESOURCE_EXHAUSTED` while allocating `32.42GiB`; XLA remat reduced the module to `58.67GiB` from `74.46GiB`.
- Tunix official-style FSDP comparison: `r_4b8ace5f`, `mesh_fsdp=2, mesh_tp=1`, batch size 1, `cached_selected_canvas_slice`, decoder remat. Initial loss was finite (`total=12.444439888000488`, decoder `4.84384298324585`, encoder `7.600596904754639`) and LoRA coverage matched, but train step failed with `RESOURCE_EXHAUSTED` while allocating `21.97GiB`; XLA remat reduced the module to `40.54GiB` from `57.80GiB`.
- Analysis: Tunix now matches the official FSDP direction by default, and the SFT math path matches the official clean-canvas prefill, cache `end_index`, corruption, selected-canvas loss, self-conditioning second pass, encoder AR loss, and LoRA update mask. The remaining H100x2 gap is the NNX/Qwix/PeftTrainer train graph and sharding memory profile: Tunix OOMs even when decoding only the selected canvas slice, while official Flax Linen/Kauldron succeeds decoding the full canvas batch. Treat H100x2 support as not ready until the train step is made as memory efficient as the official Kauldron path.

Repeatable 8-GPU public-checkpoint command with VRAM telemetry:

```bash
jl create --gpu RTX-PRO6000 --region IN1 --num-gpus 8 --storage 300 --template pytorch --yes --json
jl run /tmp/tunix-dg-efficient --script scripts/download_public_gcs_prefix.py --on <machine_id> --requirements /tmp/tunix-dg-efficient/requirements-gpu.txt --json --yes -- --bucket gemma-data --prefix checkpoints/diffusiongemma-26B-A4B-it/ --dest /root/checkpoints/diffusiongemma-26B-A4B-it --workers 16
jl exec <machine_id> --json -- sh -lc 'mkdir -p /root/checkpoints/tokenizers && curl -L https://storage.googleapis.com/gemma-data/tokenizers/tokenizer_gemma4.model -o /root/checkpoints/tokenizers/tokenizer_gemma4.model'
jl run /tmp/tunix-dg-efficient --script scripts/smoke_diffusion_gemma_pubmedqa_tunix.py --on <machine_id> --requirements /tmp/tunix-dg-efficient/requirements-gpu.txt --json --yes -- --steps 1 --batch_size 1 --prompt_len 1024 --canvas_size 128 --num_canvases 2 --max_examples 4 --max_context_chars 4000 --checkpoint /root/checkpoints/diffusiongemma-26B-A4B-it --tokenizer /root/checkpoints/tokenizers/tokenizer_gemma4.model --mesh_fsdp 4 --mesh_tp 2 --restore_concurrent_gb 16 --checkpoint_dir /root/diffusion_gemma_pubmedqa_state_8gpu_remat_official_denoiser_b1_prefillskip --lora_rank 4 --lora_alpha 8.0 --fast_uniform_corruption --decoder_implementation cached_selected_canvas_slice --encoder_loss_weight 0.0 --stop_gradient_from_denoiser_to_encoder --remat_decoder --gpu_memory_poll_seconds 2
jl run /tmp/tunix-dg-efficient --script scripts/smoke_diffusion_gemma_pubmedqa_tunix.py --on <machine_id> --requirements /tmp/tunix-dg-efficient/requirements-gpu.txt --json --yes -- --steps 1 --batch_size 1 --prompt_len 1024 --canvas_size 128 --num_canvases 2 --max_examples 4 --max_context_chars 4000 --checkpoint /root/checkpoints/diffusiongemma-26B-A4B-it --tokenizer /root/checkpoints/tokenizers/tokenizer_gemma4.model --mesh_fsdp 4 --mesh_tp 2 --restore_concurrent_gb 16 --checkpoint_dir /root/diffusion_gemma_pubmedqa_state_8gpu_remat_official_encoder_b1 --lora_rank 4 --lora_alpha 8.0 --fast_uniform_corruption --decoder_implementation cached_selected_canvas_slice --remat_decoder --gpu_memory_poll_seconds 2
jl destroy <machine_id> --yes --json
```

Latest verified 4xH100 PubMedQA public-checkpoint result:

- Machine: `424974` (`H100`, `IN2`, 4 GPUs, VM), destroyed after the run.
- Checkpoint mirror: `r_027dbcd5`, 31 objects, 37.633 GiB.
- Run: `r_64b7e672`.
- Public checkpoint loaded and PubMedQA loss was finite: total `17.086498260498047`, decoder `8.215320587158203`, encoder `8.871176719665527`.
- Train step failed: XLA reported `RESOURCE_EXHAUSTED` while allocating `88.20GiB` in `jit__train_step` after rematerialization reduced the module only to about `100.75GiB`.
- Model-parallel follow-up machine: `424999` (`H100`, `IN2`, 4 GPUs, VM), destroyed after the run.
- Follow-up checkpoint mirror: `r_cc1f0740`, 31 objects, 37.633 GiB, 352.863 seconds.
- `mesh_fsdp=1, mesh_tp=4` run: `r_f3f3ff49`, failed at checkpoint load because a weight with shape `(2, 2816, 512)` cannot shard axis 0 over TP 4.
- `mesh_fsdp=2, mesh_tp=2` run: `r_13800793`, public checkpoint loaded and PubMedQA loss was finite: total `17.48516845703125`, decoder `8.583871841430664`, encoder `8.901296615600586`.
- `mesh_fsdp=2, mesh_tp=2` train step failed: XLA reported `RESOURCE_EXHAUSTED` while allocating `37.05GiB` in `jit__train_step`; rematerialization reduced the module to about `56.47GiB`, down from `57.09GiB`.
- 4xH100 verdict: this proves real PubMedQA preprocessing and public-checkpoint forward/loss compatibility, but 4 GPUs are not enough for the current 26B LoRA train step. The next required work for smaller GPU counts is a lower-memory train path, likely by avoiding the full-sequence compatibility decoder path during SFT and implementing the official cached selected-canvas decoder update in Tunix Gemma4.

Latest verified 8x RTX PRO 6000 PubMedQA public-checkpoint result:

- Machine: `425165` (`RTX-PRO6000`, `IN1`, 8 GPUs, 96GB each, pytorch container), destroyed after the run. `jl status --json` reported `running_instances: 0` after destroy.
- H200 8-GPU creation was attempted first and failed with `H200 not available at this moment`; RTX PRO 6000 was the best available 8-GPU high-memory option returned by JarvisLabs.
- Checkpoint mirror: `r_f075dbb9`, 31 objects, 37.633 GiB, 208.567 seconds.
- Denoiser-only official-length run: `r_4f323c33`, `mesh_fsdp=4`, `mesh_tp=2`, decoder remat, `cached_selected_canvas_slice`, prompt length 1024, canvas size 128, two canvases, batch size 1, `prefill_decode_only_last_token=true`.
- Denoiser-only result: initial total/decoder loss `4.789341926574707`, final total/decoder loss `4.481716632843018`, LoRA norm delta `0.00029754638671875`, LoRA checksum delta `0.029350757598876953`, sampled non-LoRA checksum delta `0.0`.
- Denoiser-only VRAM telemetry: peak used MiB `74627` on GPU 0 and `58219-58221` on GPUs 1-7; minimum free MiB `23260`.
- Full-loss official-length run: `r_0f069fcd`, same mesh/shape/remat/slice path, encoder AR loss enabled, `prefill_decode_only_last_token=false`.
- Full-loss result: initial total loss `12.994644165039062`, decoder `5.365616798400879`, encoder `7.629027843475342`; final total loss `11.146772384643555`, decoder `4.296473503112793`, encoder `6.850298881530762`.
- Full-loss update proof: LoRA coverage 366 leaves / 4,423,936 elements / 8,847,872 bytes; LoRA norm delta `0.00029754638671875`; LoRA checksum delta `0.029854297637939453`; sampled non-LoRA checksum delta `0.0`.
- Full-loss artifact: `/root/diffusion_gemma_pubmedqa_state_8gpu_remat_official_encoder_b1/minimal_state.json`.
- Full-loss VRAM telemetry: peak used MiB `76963` on GPU 0 and `60553-60555` on GPUs 1-7; minimum free MiB `20924`.
- First JIT compile remains expensive. The encoder-loss train step emitted XLA constant-folding warnings in `jit(_train_step)/jvp()/reduce_max` and took about 14-15 minutes wall time including restore/compile/run. Batch size 2/effective batch 2 and steady-state multi-step throughput are still separate scale-out checks.

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
- GIF artifact: `/tmp/diffusion_gemma_trace.gif`, 1280x720, 56 frames, with a readable token-chip denoising canvas, smooth 70ms transition frames, short 80ms step holds, and 3 final hold frames.
- Final visible output: `<|channel>thought\n<channel|>Diffusion language models are tapered that generate text by iteratively refining noisy data into ... sequences through a reverse diffusion process.<eos>`.

## References

- Official DiffusionGemma implementation: <https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion>
- Official SFT recipe: <https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion/hackable_diffusion_adapter>
- Tunix SFT API: <https://tunix.readthedocs.io/en/latest/api/api_sft.html>
