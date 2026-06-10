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

"""SafeTensors loading for DiffusionGemma-compatible checkpoints."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models.diffusion_gemma import model as model_lib
from tunix.models.gemma4 import params_safetensors as gemma4_safetensors


def create_model_from_safe_tensors(
    file_dir: str,
    config,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
):
  """Loads DiffusionGemma from Gemma4-compatible SafeTensors.

  Public DiffusionGemma checkpoints are currently published as upstream Orbax
  checkpoints, so this path is primarily for converted checkpoints.
  """
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.DiffusionGemma_A26B_A4B,
      config=config,
      key_mapping=gemma4_safetensors._get_key_and_transform_mapping,  # pylint: disable=protected-access
      mesh=mesh,
      preprocess_fn=gemma4_safetensors._make_preprocess_fn(config),  # pylint: disable=protected-access
      dtype=dtype,
      mode=mode,
  )
