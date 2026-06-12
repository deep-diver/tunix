# DiffusionGemma H100x2 Official-vs-Wrapper Sync Comparison

Date: 2026-06-12

## Setup

- Official reference VM: `425566`, H100 80GB x2.
- Tunix wrapper VM: `425567`, H100 80GB x2.
- Both VMs used the same public `diffusiongemma-26B-A4B-it` checkpoint, the
  official PubMedQA recipe shape, batch size `2`, LoRA rank `4`, JAX `0.10.1`,
  CUDA 12 wheels, and NCCL `2.30.7`.
- Both VMs were destroyed after the experiment. Final `jl status --json`
  reported `running_instances: 0` and `running_vms: 0`.

## Async Dispatch Check

An initial 1-step hybrid run returned completion markers without forcing a
post-step synchronization:

- Official reference: `r_be358b08`, exit `0`.
- Tunix wrapper: `r_3f109764`, exit `0`.

This is not sufficient evidence of a completed train step because JAX dispatch
is asynchronous.

## Strict Synchronization Check

A follow-up run forced synchronization after the official train step:

- Official reference: `r_3b9a2ca7`, exit `1`.
- Tunix wrapper: `r_b4405817`, exit `1`.

Both failed at the same boundary:

```text
jax.errors.JaxRuntimeError: INTERNAL: NCCL operation ncclAllGather(...) failed:
invalid argument ... 'lib wrapper not initialized.' [executable_name='jit_step']
```

The failure happened in the official `jit_step` path for both the standalone
official-reference runner and the Tunix wrapper that delegates to the official
backend.

## Interpretation

The result does not isolate a Tunix-wrapper divergence. It shows that this
H100x2 JarvisLabs JAX/NCCL runtime cannot currently prove synchronized
completion of the official DiffusionGemma 26B PubMedQA LoRA train step.

The local deterministic parity checks still pass, but H100x2 synchronized
training should be treated as not verified.
