# DiffusionGemma A100x2 Official Backend Follow-up

Date: 2026-06-12

Machine `425531` was an `A100-80GB` x2 JarvisLabs `IN2` pytorch container. It
was destroyed after the run; `jl status --json` reported `running_instances: 0`
and `running_vms: 0`.

## Runs

- Config-only official backend: `r_0192fe97`, succeeded. Deps were
  `gemma==4.0.1`, `hackable_diffusion==1.0.1`, `kauldron==1.4.4`.
- Public checkpoint mirror: `r_aeac19fb`, succeeded. Downloaded 32 objects,
  37.633 GiB, in 96.474 seconds.
- JAX pmap baseline with CUDA13 + `NCCL_IB_DISABLE=1`: `r_dfc8cf19`, succeeded.
- JAX pmap baseline with CUDA12 + `NCCL_IB_DISABLE=1`: `r_4bf8abdb`, succeeded.
- Hybrid official loop, CUDA13: `r_b68bffeb`, failed after
  `official_backend_hybrid_step_complete` with async `jit_step` NCCL
  `ncclAllGather ... invalid argument`.
- Hybrid official loop, CUDA13 + `NCCL_IB_DISABLE=1`: `r_391e403e`, same
  failure.
- Hybrid official loop, CUDA12 + `NCCL_IB_DISABLE=1`: `r_ef15db9d`, same
  failure.
- Direct official Kauldron loop, CUDA12 + `NCCL_IB_DISABLE=1`: `r_3d606f48`.
  The official train progress reached `train: 100%|2/2`, then failed in
  Kauldron final `_sync()` with `ncclAllReduce ... invalid argument`.

## Observations

- The official recipe imports, config build, PubMedQA paths, public checkpoint
  restore, and LoRA parameter discovery all work.
- Hybrid pre-step checksum saw 364 LoRA leaves / 8,856,064 elements, and a
  sampled base checksum over 8 leaves / 1,280,317,952 elements.
- Runtime memory was about 61.3 GiB per A100 for the hybrid path and about
  61.7 GiB per A100 during the direct official Kauldron path.
- Basic two-GPU JAX collectives work, but the official DiffusionGemma trainstep
  and/or Kauldron final sync corrupts the NCCL communicator in this container.

## Verdict

This is not a clean 2-GPU pass. The wrapper now gets as far as official
DiffusionGemma trainstep execution, and the direct official loop also reaches
two train iterations, but final synchronization fails on this Jarvis A100x2
container. Treat A100x2 clean training as unverified until the NCCL issue is
resolved on the target runtime or the official sync path is replaced with a
safe production checkpoint/verification path.
