# Copyright 2026 The RLinf Authors.
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
"""Policy transforms for real-world Franka PnP data: 7-dim state, dual view (image + wrist_image), 7-dim action.

Follows the same pattern as libero_policy.py — the only difference is that the state
dimension is 7 (tcp_pose 6 + gripper 1) instead of Libero's 8.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_realworld_example() -> dict:
    """Creates a random input example for the real-world Franka policy."""
    return {
        "observation/state": np.random.rand(7).astype(np.float32),
        "observation/image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(128, 128, 3), dtype=np.uint8
        ),
        "prompt": "Pick and place the object.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RealworldInputs(transforms.DataTransformFn):
    """Converts inputs to the format expected by the model for real-world Franka PnP.

    Dual view: observation/image (third-person) + observation/wrist_image (wrist cam).
    State: 7-dim [tcp_pose(6) + gripper(1)].
    Action: 7-dim [delta_xyz(3) + delta_rpy(3) + gripper(1)].
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/extra_view_images"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RealworldOutputs(transforms.DataTransformFn):
    """Converts model outputs back to dataset format: 7-dim actions [dx, dy, dz, drx, dry, drz, gripper]."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
