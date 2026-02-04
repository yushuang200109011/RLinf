# Copyright 2025 The USER Authors.
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

from enum import Enum


class SupportedEnvType(Enum):
    MANISKILL = "maniskill"
    LIBERO = "libero"
    ROBOTWIN = "robotwin"
    ISAACLAB = "isaaclab"
    METAWORLD = "metaworld"
    BEHAVIOR = "behavior"
    CALVIN = "calvin"
    ROBOCASA = "robocasa"
    REALWORLD = "realworld"
    FRANKASIM = "frankasim"
    HABITAT = "habitat"
    OPENSORAWM = "opensora_wm"


def get_env_cls(env_type: str, env_cfg=None, enable_offload=False):
    """
    Get environment class based on environment type.

    Args:
        env_type: Type of environment (e.g., "maniskill", "libero", "isaaclab", etc.)
        env_cfg: Optional environment configuration. Required for "isaaclab" environment type.

    Returns:
        Environment class corresponding to the environment type.
    """

    env_type = SupportedEnvType(env_type)

    if env_type == SupportedEnvType.MANISKILL:
        if not enable_offload:
            from user.envs.maniskill.maniskill_env import ManiskillEnv
        else:
            from user.envs.maniskill.maniskill_offload_env import (
                ManiskillOffloadEnv as ManiskillEnv,
            )

        return ManiskillEnv
    elif env_type == SupportedEnvType.LIBERO:
        from user.envs.libero.libero_env import LiberoEnv

        return LiberoEnv
    elif env_type == SupportedEnvType.REALWORLD:
        from user.envs.realworld.realworld_env import RealWorldEnv

        return RealWorldEnv
    elif env_type == SupportedEnvType.FRANKASIM:
        from user.envs.frankasim.frankasim_env import FrankaSimEnv

        return FrankaSimEnv
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented")
