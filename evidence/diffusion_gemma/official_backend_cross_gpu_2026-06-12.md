# DiffusionGemma Official Backend Cross-GPU Check

Date: 2026-06-12

Goal: compare the same official DeepMind DiffusionGemma PubMedQA Kauldron smoke
on RTX PRO 6000 x2 and H100 x2 after A100-80GB x2 showed fatal NCCL final-sync
failures.

Both machines were destroyed after the run. Final `jl status --json` reported
`running_instances: 0` and `running_vms: 0`.

## RTX PRO 6000 x2

- Machine: `425551`, `RTX-PRO6000` x2, `IN1`, pytorch container.
- Config/setup: `r_d5baac99`, succeeded.
- Checkpoint mirror: `r_04367c75`, 32 objects, 37.633 GiB, 83.988 seconds.
- Train: `r_6f265dee`, succeeded with exit code 0.
- Runtime: CUDA13 JAX, `NCCL_IB_DISABLE=1`, official Kauldron loop,
  `--skip_step_metrics`, PubMedQA batch size 2.
- Result: reached `train: 100%|2/2` and printed
  `official_backend_train_complete`.
- Warnings: TensorFlow PTX JIT warning for Blackwell compute capability 12.0,
  TensorFlow allocator warning for a 33.57 GiB allocation, and non-fatal NCCL
  `corrupted comm object detected` warnings.

## H100 x2

- Machine: `425553`, `H100` x2, `IN2`, VM.
- Initial setup failure: `r_ee34574d` used the default Python 3.14 environment;
  Kauldron/TensorFlow could not be installed because TensorFlow wheels were not
  available for Python 3.14.
- Valid setup/config: `r_06c091e1`, explicit Python 3.13 venv, succeeded.
- Checkpoint mirror: `r_aaab1b1b`, 32 objects, 37.633 GiB, 77.306 seconds.
- CUDA13 train attempt: `r_467172a5`, stopped because JAX CUDA13 plugin fell
  back to CPU due a cuBLAS plugin/library mismatch.
- CUDA12 pmap baseline: `r_b7da99b3`, succeeded on both H100s.
- Train: `r_2200993f`, CUDA12 JAX, `NCCL_IB_DISABLE=1`, official Kauldron loop,
  `--skip_step_metrics`, PubMedQA batch size 2, succeeded with exit code 0.
- Result: reached `train: 100%|2/2` and printed
  `official_backend_train_complete`.
- Warnings: CUDA VMM permission fallback warnings and non-fatal NCCL
  `corrupted comm object detected` warnings.

## Conclusion

The A100-80GB x2 fatal final-sync failure is not universal. The same official
backend smoke completes on RTX PRO 6000 x2 and H100 x2. The likely culprit is
the A100 container runtime/JAX-NCCL combination, not the official DiffusionGemma
implementation in general.
