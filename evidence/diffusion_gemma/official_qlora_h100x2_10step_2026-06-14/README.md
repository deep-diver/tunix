# DiffusionGemma Official QLoRA H100x2 10-Step Run

This evidence captures the first successful H100x2 run of the official
Hackable-Diffusion DiffusionGemma SFT path with an official-layout QLoRA
backend added in Tunix.

## Result

- Run id: `r_f10546ff`
- Machine id: `426600`
- GPU: `H100` x2, 80 GB each, JarvisLabs `IN2`
- Recipe: `pubmedqa`
- Backend: `official_qlora`
- QLoRA base storage: packed int4 `qvalue` + `scale`
- MoE expert weights quantized: `true`
- Official Gemma4 block remat: `true`
- Denoiser-to-encoder stop-gradient: `true`
- Output chunk sizes: `einsum=16`, `ragged=16`
- Encoder loss chunking: `token=128`, `vocab=8192`
- Completed steps: `10`
- Exit code: `0`

Peak GPU memory from `gpu_memory.csv`:

| GPU | Peak used MiB | Peak used GiB | Peak util |
| --- | ---: | ---: | ---: |
| 0 | 68,979 | 67.36 | 100% |
| 1 | 68,941 | 67.33 | 100% |

The training log contains:

- `official_backend_official_lora_patched_for_qlora`
- `official_backend_official_qlora_checkpoint_loader_patched`
- `official_backend_official_qlora_ready`
- `official_backend_gemma4_block_remat_patched`
- `official_backend_memory_safe_sft_model_replaced`
- `official_backend_memory_safe_encoder_loss_replaced`
- `official_backend_hybrid_train_complete` with `num_steps=10`
- `official_backend_train_complete` with `run_steps=10`

## Reproduction Command

```bash
jl run . \
  --script scripts/run_diffusion_gemma_h100x2_comparison_job.sh \
  --on <H100X2_MACHINE_ID> \
  --json --yes -- \
  --mode tunix \
  --run_name qlora_remat_moe_chunk16_10step_retry_20260614 \
  --lora_backend official_qlora \
  --official_qlora_quantize_moe_weights true \
  --official_qlora_einsum_output_chunk_size 16 \
  --official_qlora_ragged_output_chunk_size 16 \
  --official_remat_blocks true \
  --stop_gradient_from_denoiser_to_encoder true \
  --dataset_batch_size 1 \
  --num_train_steps 2000 \
  --run_steps 10 \
  --max_runtime_seconds 7200 \
  --log_losses false \
  --sync_after_step state \
  --log_param_summary false \
  --gpu_poll_seconds 5 \
  --xla_python_client_mem_fraction 0.90 \
  --xla_python_client_preallocate false \
  --tf_gpu_allocator cuda_malloc_async \
  --encoder_loss_token_chunk_size 128 \
  --encoder_loss_vocab_chunk_size 8192
```

## Files

- `train.log`: raw training log and JSON events
- `run_summary.json`: parsed event and scalar summary
- `gpu_memory.csv`: GPU memory/utilization samples
