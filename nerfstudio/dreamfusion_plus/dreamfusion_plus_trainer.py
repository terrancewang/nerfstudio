# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dreamfusion trainer"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch

from nerfstudio.dreamfusion_plus.dreamfusion_plus_pipeline import (
    DreamfusionPlusPipeline,
    DreamfusionPlusPipelineConfig,
)
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils import profiler


@dataclass
class DreamfusionPlusTrainerConfig(TrainerConfig):
    """Configuration for trainer instantiation"""

    _target: Type = field(default_factory=lambda: DreamfusionTrainer)
    """target class to instantiate"""
    pipeline: DreamfusionPlusPipelineConfig = DreamfusionPlusPipelineConfig()
    """specifies the pipeline config"""
    gradient_accumulation_steps: int = 4
    """number of gradient accumulation steps before optimizer step"""


class DreamfusionTrainer(Trainer):
    """Dreamfusion trainer"""

    pipeline: DreamfusionPlusPipeline
    config: DreamfusionPlusTrainerConfig

    def __init__(self, config: DreamfusionPlusTrainerConfig, local_rank: int = 0, world_size: int = 1):
        assert isinstance(config, DreamfusionPlusTrainerConfig)
        Trainer.__init__(self, config=config, local_rank=local_rank, world_size=world_size)

    @profiler.time_function
    def train_iteration(self, step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        self.optimizers.zero_grad_all()
        _loss_dict = self.pipeline.get_train_loss_dict_input_img(step)
        loss = _loss_dict["rgb_loss"]
        if self.config.pipeline.datamanager.use_input_transient:
            loss = loss + _loss_dict["transient_quantized_loss"] + _loss_dict["transient_loss"]
        self.grad_scaler.scale(loss).backward()  # type: ignore
        self.grad_scaler.step(self.optimizers.optimizers["fields"])
        # self.grad_scaler.step(self.optimizers.optimizers["proposal_networks"])
        self.grad_scaler.update()
        self.optimizers.zero_grad_all()

        # # Original before refactor:
        # self.grad_scaler.scale(loss).backward()  # type: ignore
        # self.optimizers.optimizer_scaler_step_all(self.grad_scaler)
        # self.grad_scaler.update()
        # self.optimizers.scheduler_step_all(step)

        model_outputs, loss_dict, _metrics_dict = super().train_iteration(step)
        metrics_dict = {
            "rgb_loss": _loss_dict["rgb_loss"],
            **_metrics_dict,
        }
        if self.config.pipeline.datamanager.use_input_transient:
            metrics_dict["transient_loss"] = loss_dict["transient_loss"]
            metrics_dict["transient_quantized_loss"] = loss_dict["transient_quantized_loss"]
        return model_outputs, loss_dict, metrics_dict
