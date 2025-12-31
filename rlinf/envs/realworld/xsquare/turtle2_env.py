# Copyright 2025 The RLinf Authors.
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

import copy
import queue
import time
from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional
from PIL import Image

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.camera import Camera, CameraInfo
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import (
    FrankaHWInfo,
    WorkerInfo,
)
from rlinf.utils.logging import get_logger

from rlinf.envs.realworld.xsquare.turtle2_robot_state import Turtle2RobotState


@dataclass
class Turtle2RobotConfig:
    use_camera_ids: list[int] = field(default_factory=lambda: [0, 2]) # [0, 1, 2]
    use_arm_ids: list[int] = field(default_factory=lambda: [0, 1])  # [0, 1]
    
    is_dummy: bool = False
    use_dense_reward: bool = False
    step_frequency: float = 10.0  # Max number of steps per second
    smooth_frequency: int = 50  # Frequency for smooth controller

    # Positions are stored in eular angles (xyz for position, rzryrx for orientation)
    # It will be converted to quaternions internally
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([[0, 0, 0, 0, 0, 0],[0.3, 0.0, 0.1, 0.0, 1, 0.0]])
    )
    reset_ee_pose: np.ndarray = field(default_factory=lambda: np.array(
        [[0.3, 0, 0.0, 0.2, 0, 0], [0.2, 0, 0.1, 0, 0.8, 0.0]]
    ))
    
    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros((2, 6)))
    action_scale: np.ndarray = field(
        default_factory=lambda: np.ones((2, 3))
    )  # [xyz move scale, orientation scale, gripper scale]
    enable_random_reset: bool = True

    random_xy_range: float = 0.05
    random_rz_range: float = np.pi / 10

    # Robot parameters
    # Same as the position arrays: first 3 are position limits, last 3 are orientation limits
    ee_pose_limit_min: np.ndarray = field(default_factory=lambda: np.zeros((2, 6)))
    ee_pose_limit_max: np.ndarray = field(default_factory=lambda: np.zeros((2, 6)))
    gripper_width_limit_min: float = 0.0
    gripper_width_limit_max: float = 5.0
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1
    save_video_path: Optional[str] = None


class Turtle2Env(gym.Env):
    """Turtle2 robot arm environment."""

    def __init__(
        self,
        config: Turtle2RobotConfig,
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[FrankaHWInfo],
        env_idx: int,
    ):
        self._logger = get_logger()
        self.config = config
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        assert len(self.config.use_arm_ids) > 0 and len(self.config.use_arm_ids)<=2, "please choose arm IDs from [0, 1]."
        assert len(self.config.use_camera_ids) > 0 and len(self.config.use_camera_ids)<=3, "please choose camera IDs from [0, 1, 2]."
        self._turtle2_state = Turtle2RobotState()
        self._num_steps = 0

        if not self.config.is_dummy:
            self._setup_hardware()

        # Init action and observation spaces
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Wait for the first frame
        self._reset_arms()
        self._turtle2_state = self._controller.get_state().wait()[0]

        # Init cameras
        self._check_cameras()
        # Video player for displaying camera frames

    def _setup_hardware(self):
        from .turtle2_smooth_controller import Turtle2SmoothController

        assert self.env_idx >= 0, "env_idx must be set for Turtle2Env."

        # Launch Turtle controller
        self._controller = Turtle2SmoothController.launch_controller(
            freq=self.config.smooth_frequency,
            env_idx=self.env_idx,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
        )

    def _init_action_obs_spaces(self):
        """Initialize action and observation spaces, including arm safety box."""
        self._xyz_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[self.config.use_arm_ids, :3].flatten(),
            high=self.config.ee_pose_limit_max[self.config.use_arm_ids, :3].flatten(),
            dtype=np.float64,
        )
        self._rpy_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[self.config.use_arm_ids, 3:].flatten(),
            high=self.config.ee_pose_limit_max[self.config.use_arm_ids, 3:].flatten(),
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            np.ones((len(self.config.use_arm_ids) * 7), dtype=np.float32) * -1,
            np.ones((len(self.config.use_arm_ids) * 7), dtype=np.float32),
        )

        obs_dim_per_arm = 7  # xyz(3) + rpy(3) + gripper(1)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(len(self.config.use_arm_ids) * obs_dim_per_arm,)
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        f"wrist_{k + 1}": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for k in range(len(self.config.use_camera_ids))
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _reset_arms(self):
        if self.config.is_dummy:
            return
        
        if self.config.enable_random_reset:
            random_xy1 = np.random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range, (2,)
            )
            random_xy2 = np.random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range, (2,)
            )
            random_euler1 = np.random.uniform(
                -self.config.random_rz_range, self.config.random_rz_range, (3, )
            )
            random_euler2 = np.random.uniform(
                -self.config.random_rz_range, self.config.random_rz_range, (3, )
            )
        else:
            random_xy1 = np.zeros(2)
            random_xy2 = np.zeros(2)
            random_euler1 = np.zeros(3)
            random_euler2 = np.zeros(3)
        
        if 0 in self.config.use_arm_ids:
            left_arm_reset_pose = self.config.reset_ee_pose[0]
            left_arm_reset_pose[:2] += random_xy1
            left_arm_reset_pose[3:6] += random_euler1
            left_arm_reset_pose = left_arm_reset_pose.tolist()
            left_arm_reset_pose.append(0.0)
        else:
            left_arm_reset_pose = [0, 0, 0, 0, 0, 0, 0]
        if 1 in self.config.use_arm_ids:
            right_arm_reset_pose = self.config.reset_ee_pose[1]
            right_arm_reset_pose[:2] += random_xy2
            right_arm_reset_pose[3:6] += random_euler2
            right_arm_reset_pose = right_arm_reset_pose.tolist()
            right_arm_reset_pose.append(0.0)
        else:
            right_arm_reset_pose = [0, 0, 0, 0, 0, 0, 0]

        print("going to reset:", left_arm_reset_pose, right_arm_reset_pose)
        
        self._controller.move_arm(left_arm_reset_pose, right_arm_reset_pose).wait()

        reach = False
        start_time = time.time()
        while not reach:
            state = self._controller.get_state().wait()[0]
            left_pos = state.follow1_pos
            right_pos = state.follow2_pos
            # print("current right pos:", right_pos)
            left_reach = np.linalg.norm(left_pos[:6] - np.array(left_arm_reset_pose)[:6]) < 0.03 if 0 in self.config.use_arm_ids else True
            right_reach = np.linalg.norm(right_pos[:6] - np.array(right_arm_reset_pose)[:6]) < 0.03 if 1 in self.config.use_arm_ids else True
            # print("left err:", np.linalg.norm(left_pos[:6] - np.array(left_arm_reset_pose)[:6]))
            # print("right err:", np.linalg.norm(right_pos[:6] - np.array(right_arm_reset_pose)[:6]))
            # print("lr reach:", left_reach, right_reach)
            reach = left_reach and right_reach
            if time.time() - start_time > 10.0:
                raise ValueError("Reset arms timeout.")
            
            time.sleep(0.1)
        time.sleep(0.5)
        return
    
    def _check_cameras(self):
        if self.config.is_dummy:
            return
        
        cam1_ok, cam2_ok, cam3_ok = self._controller.check_cams().wait()[0]
        if 0 in self.config.use_camera_ids and not cam1_ok:
            raise ValueError("Camera 1 not available.")
        if 1 in self.config.use_camera_ids and not cam2_ok:
            raise ValueError("Camera 2 not available.")
        if 2 in self.config.use_camera_ids and not cam3_ok:
            raise ValueError("Camera 3 not available.")

    def reset(self, *, seed=None, options=None):
        if self.config.is_dummy:
            observation = self._get_observation()
            return observation, {}

        # Reset
        self._reset_arms()
        self._num_steps = 0
        self._turtle2_state = self._controller.get_state().wait()[0]
        observation = self._get_observation()
        for key in observation["frames"].keys():
            img = Image.fromarray(observation["frames"][key])
            img.save(f'{key}.jpg')

        return observation, {}

    def transform_action_ee_to_base(self, action):
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action

    def step(self, action: np.ndarray):
        """Take a step in the environment.

        action (np.ndarray): The action to take, in shape (2, 7).
        """
        start_time = time.time()

        # if self.use_rel_frame:
        #     action = self.transform_action_ee_to_base(action)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.next_position = self._franka_state.tcp_pose.copy()
        self.next_position[:3] = (
            self.next_position[:3] + xyz_delta * self.config.action_scale[0]
        )

        if not self.config.is_dummy:
            self.next_position[3:] = (
                R.from_euler("xyz", action[3:6] * self.config.action_scale[1])
                * R.from_quat(self._franka_state.tcp_pose[3:].copy())
            ).as_quat()

            gripper_action = action[6] * self.config.action_scale[2]

            is_gripper_action_effective = self._gripper_action(gripper_action)
            self._move_action(self._clip_position_to_safety_box(self.next_position))
        else:
            is_gripper_action_effective = True

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._franka_state = self._controller.get_state().wait()[0]
        else:
            self._franka_state = self._franka_state
        observation = self._get_observation()
        reward = self._calc_step_reward(observation, is_gripper_action_effective)
        terminated = reward == 1
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self):
        return self._num_steps

    def _calc_step_reward(
        self,
        observation: dict[str, np.ndarray],
        is_gripper_action_effective: bool = False,
    ) -> float:
        """Compute the reward for the current observation, namely the robot state and camera frames.

        Args:
            observation (Dict[str, np.ndarray]): The current observation from the environment.
            is_gripper_action_effective (bool): Whether the gripper action was effective (i.e., the gripper state changed).
        """
        if not self.config.is_dummy:
            # Convert orientation to euler angles
            euler_angles = np.abs(
                R.from_quat(self._franka_state.tcp_pose[3:].copy()).as_euler("xyz")
            )
            position = np.hstack([self._franka_state.tcp_pose[:3], euler_angles])
            target_delta = np.abs(position - self.config.target_ee_pose)
            is_success = np.all(target_delta[:3] <= self.config.reward_threshold[:3])
            if is_success:
                reward = 1.0
            else:
                if self.config.use_dense_reward:
                    reward = np.exp(-500 * np.sum(np.square(target_delta[:3])))
                else:
                    reward = 0.0
                self._logger.debug(
                    f"Does not meet success criteria. Target delta: {target_delta}, "
                    f"Success threshold: {self.config.reward_threshold}, "
                    f"Current reward={reward}",
                )

            if self.config.enable_gripper_penalty and is_gripper_action_effective:
                reward -= self.config.gripper_penalty

            return reward
        else:
            return 0.0

    def _open_cameras(self):
        self._cameras: list[Camera] = []
        if self.config.camera_serials is None:
            return
        camera_infos = [
            CameraInfo(name=f"wrist_{i + 1}", serial_number=n)
            for i, n in enumerate(self.config.camera_serials)
        ]
        for info in camera_infos:
            camera = Camera(info)
            if not self.config.is_dummy:
                camera.open()
            self._cameras.append(camera)

    def _close_cameras(self):
        for camera in self._cameras:
            camera.close()
        self._cameras = []

    def _crop_frame(
        self, frame: np.ndarray, reshape_size: tuple[int, int]
    ) -> np.ndarray:
        """Crop the frame to the desired resolution."""
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped_frame = frame[
            start_y : start_y + crop_size, start_x : start_x + crop_size
        ]
        resized_frame = cv2.resize(cropped_frame, reshape_size)
        return cropped_frame, resized_frame

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Get frames from all cameras."""
        frames = {}
        display_frames = {}
        for camera in self._cameras:
            try:
                frame = camera.get_frame()
                reshape_size = self.observation_space["frames"][
                    camera._camera_info.name
                ].shape[:2][::-1]
                cropped_frame, resized_frame = self._crop_frame(frame, reshape_size)
                frames[camera._camera_info.name] = resized_frame[
                    ..., ::-1
                ]  # Convert RGB to BGR
                display_frames[camera._camera_info.name] = (
                    resized_frame  # Original RGB for display
                )
                display_frames[f"{camera._camera_info.name}_full"] = (
                    cropped_frame  # Non-resized version
                )
            except queue.Empty:
                self._logger.warning(
                    f"Camera {camera._camera_info.name} is not producing frames. Wait 5 seconds and try again."
                )
                time.sleep(5)
                camera.close()
                self._open_cameras()
                return self._get_camera_frames()

        self.camera_player.put_frame(display_frames)
        return frames

    # Robot actions

    def _clip_position_to_safety_box(self, position: np.ndarray) -> np.ndarray:
        """Clip the position array to be within the safety box."""
        position[:3] = np.clip(
            position[:3], self._xyz_safe_space.low, self._xyz_safe_space.high
        )
        euler = R.from_quat(position[3:].copy()).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self._rpy_safe_space.low[0],
                self._rpy_safe_space.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self._rpy_safe_space.low[1:], self._rpy_safe_space.high[1:]
        )
        position[3:] = R.from_euler("xyz", euler).as_quat()

        return position

    def _clear_error(self):
        self._controller.clear_errors().wait()

    def _gripper_action(self, position: float, is_binary: bool = True):
        if is_binary:
            if (
                position <= -self.config.binary_gripper_threshold
                and self._franka_state.gripper_open
            ):
                # Close gripper
                self._controller.close_gripper().wait()
                time.sleep(0.6)
                return True
            elif (
                position >= self.config.binary_gripper_threshold
                and not self._franka_state.gripper_open
            ):
                # Open gripper
                self._controller.open_gripper().wait()
                time.sleep(0.6)
                return True
            else:  # No change
                return False
        else:
            raise NotImplementedError("Non-binary gripper action not implemented.")

    def _interpolate_move(self, pose: np.ndarray, timeout: float = 1.5):
        num_steps = int(timeout * self.config.step_frequency)
        self._franka_state: FrankaRobotState = self._controller.get_state().wait()[0]
        pos_path = np.linspace(
            self._franka_state.tcp_pose[:3], pose[:3], int(num_steps) + 1
        )
        quat_path = quat_slerp(
            self._franka_state.tcp_pose[3:], pose[3:], int(num_steps) + 1
        )

        for pos, quat in zip(pos_path[1:], quat_path[1:]):
            pose = np.concatenate([pos, quat])
            self._move_action(pose.astype(np.float32))
            time.sleep(1.0 / self.config.step_frequency)

        self._franka_state: FrankaRobotState = self._controller.get_state().wait()[0]

    def _move_action(self, position: np.ndarray):
        if not self.config.is_dummy:
            self._clear_error()
            self._controller.move_arm(position.astype(np.float32)).wait()
        else:
            print(f"Executing dummy action towards {position=}.")

    def _get_observation(self) -> dict:
        if not self.config.is_dummy:
            frames = self._controller.get_cams(self.config.use_camera_ids).wait()[0]
            assert len(frames) == len(self.config.use_camera_ids), "get frames failed."
            tcp_pose = []
            if 0 in self.config.use_arm_ids:
                tcp_pose.append(self._turtle2_state.follow1_pos)
            if 1 in self.config.use_arm_ids:
                tcp_pose.append(self._turtle2_state.follow2_pos)
            tcp_pose = np.concatenate(tcp_pose, axis=0)
            state = {
                "tcp_pose": tcp_pose,
            }
            frames_dict = {}
            for k in range(len(self.config.use_camera_ids)):
                frames_dict[f"wrist_{k+1}"] = frames[k]
            
            observation = {
                "state": state,
                "frames": frames_dict,
            }
            return copy.deepcopy(observation)
        else:
            obs = self._base_observation_space.sample()
            return obs


def main():
    env = Turtle2Env(
        config=Turtle2RobotConfig(),
        worker_info=None,
        hardware_info=None,
        env_idx=0,
    )
    obs, _ = env.reset()
    done = False
    print("obs.keys(): ", obs.keys())
    for key in obs.keys():
        if isinstance(obs[key], dict):
            for subkey in obs[key].keys():
                print(f"{key}.{subkey}: ", obs[key][subkey].shape)
        else:
            print(f"{key}: ", obs[key].shape)

if __name__ == "__main__":
    main()