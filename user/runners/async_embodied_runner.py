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

import asyncio
import time
from typing import TYPE_CHECKING

from omegaconf.dictconfig import DictConfig

from user.runners.embodied_runner import EmbodiedRunner
from user.scheduler import Channel
from user.scheduler import WorkerGroupFuncResult as Handle
from user.utils.metric_utils import (
    compute_env_metrics_per_env_worker,
    compute_evaluate_metrics,
)
from user.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from user.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from user.workers.env.async_env_worker import AsyncEnvWorker
    from user.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )


class AsyncEmbodiedRunner(EmbodiedRunner):
    def __init__(
        self,
        cfg: DictConfig,
        actor: "AsyncEmbodiedSACFSDPPolicy",
        rollout: "AsyncMultiStepRolloutWorker",
        env: "AsyncEnvWorker",
        critic=None,
        reward=None,
        run_timer=None,
    ):
        super().__init__(cfg, actor, rollout, env, critic, reward, run_timer)

        # Data channels
        self.env_metric_channel = Channel.create("EnvMetric")
        self.rollout_metric_channel = Channel.create("RolloutMetric")
        self.replay_channel = Channel.create("ReplayBuffer")

    def get_env_metrics(self):
        try:
            result = self.env_metric_channel.get_nowait()
        except asyncio.QueueEmpty:
            return None
        all_workers_env_metrics = compute_env_metrics_per_env_worker([result])
        env_metrics = compute_evaluate_metrics(
            [
                env_metrics,
            ]
        )
        return all_workers_env_metrics, env_metrics

    def run(self):
        start_step = self.global_step
        start_time = time.time()
        self.update_rollout_weights()

        env_handle: Handle = self.env.interact(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
            metric_channel=self.env_metric_channel,
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
            replay_channel=self.replay_channel,
            metric_channel=self.rollout_metric_channel,
        )
        actor_handle: Handle = self.actor.recv_rollout_trajectories(
            input_channel=self.replay_channel
        )

        train_step = start_step
        while train_step < self.max_steps:
            skip_step = False
            with self.timer("step"):
                actor_training_handle: Handle = self.actor.run_training()
                actor_result = actor_training_handle.wait()
                if not actor_result[0]:
                    skip_step = True

                if not skip_step:
                    train_step += 1
                    self.update_rollout_weights()

            training_metrics = {f"train/{k}": v for k, v in actor_result[0].items()}
            self.metric_logger.log(training_metrics, train_step)

            env_metrics_result = self.get_env_metrics()
            if env_metrics_result is not None:
                all_workers_env_metrics, env_metrics = env_metrics_result
                rollout_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
                env_worker_metrics = {
                    f"env/worker_{rank_id}/{k}": v
                    for rank_id, worker_env_metrics in all_workers_env_metrics.items()
                    for k, v in worker_env_metrics.items()
                }
                self.metric_logger.log(rollout_metrics, train_step)
                self.metric_logger.log(env_worker_metrics, train_step)
            
            self.global_step = train_step
            _, save_model, _ = check_progress(
                self.global_step,
                self.max_steps,
                self.cfg.runner.val_check_interval,
                self.cfg.runner.save_interval,
                1.0,
                run_time_exceeded=False,
            )
            if save_model:
                self._save_checkpoint()

        self.env.stop().wait()
        self.rollout.stop().wait()
        self.actor.stop().wait()
        env_handle.wait()
        rollout_handle.wait()
        actor_handle.wait()

        self._save_checkpoint()
