# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DiffusionGemma API."""

from tunix.models.diffusion_gemma import data
from tunix.models.diffusion_gemma import generation
from tunix.models.diffusion_gemma import hackable_adapter
from tunix.models.diffusion_gemma import linen_qwix_lora
from tunix.models.diffusion_gemma import lora_inventory
from tunix.models.diffusion_gemma import model
from tunix.models.diffusion_gemma import params
from tunix.models.diffusion_gemma import params_safetensors
from tunix.models.diffusion_gemma import sft
from tunix.models.diffusion_gemma.generation import (
    DiffusionGemmaGenerationConfig,
    DiffusionGemmaGenerationTrace,
    decode_trace,
    generate_tokens,
)
from tunix.models.diffusion_gemma.hackable_adapter import (
    DiffusionGemmaOfficialLossConfig,
    DiffusionGemmaQwixLoRAConfig,
    OfficialCheckpointInfo,
    OfficialBackendDependencyError,
    OfficialDiffusionGemmaTrainer,
    OfficialSFTConfig,
    get_official_checkpoint_info,
    list_official_checkpoint_steps,
    make_official_tunix_sft_config,
    resolve_official_checkpoint_step,
)
from tunix.models.diffusion_gemma.sft import (
    DiffusionGemmaEvalResult,
    DiffusionGemmaSFTLoss,
    configure_peft_trainer_for_diffusion_gemma_sft,
    evaluate_sft_loss,
)

__all__ = [
    "DiffusionGemmaGenerationConfig",
    "DiffusionGemmaGenerationTrace",
    "DiffusionGemmaOfficialLossConfig",
    "DiffusionGemmaQwixLoRAConfig",
    "DiffusionGemmaEvalResult",
    "DiffusionGemmaSFTLoss",
    "OfficialBackendDependencyError",
    "OfficialCheckpointInfo",
    "OfficialDiffusionGemmaTrainer",
    "OfficialSFTConfig",
    "configure_peft_trainer_for_diffusion_gemma_sft",
    "data",
    "decode_trace",
    "evaluate_sft_loss",
    "generation",
    "get_official_checkpoint_info",
    "generate_tokens",
    "hackable_adapter",
    "linen_qwix_lora",
    "lora_inventory",
    "list_official_checkpoint_steps",
    "make_official_tunix_sft_config",
    "model",
    "params",
    "params_safetensors",
    "resolve_official_checkpoint_step",
    "sft",
]
