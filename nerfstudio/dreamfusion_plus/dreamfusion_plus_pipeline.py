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

"""Dreamfusion Pipeline and trainer"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.dreamfusion_plus.dreamfusion_plus_datamanager import (
    DreamFusionPlusDataManager,
    DreamFusionPlusDataManagerConfig,
)
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.generative.stable_diffusion import StableDiffusion
from nerfstudio.models.dreamfusion import DreamFusionModel, DreamFusionModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.pipelines.dreamfusion_pipeline import (
    DreamfusionPipeline,
    DreamfusionPipelineConfig,
)


@dataclass
class DreamfusionPlusPipelineConfig(DreamfusionPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: DreamfusionPlusPipeline)
    """target class to instantiate"""
    datamanager: DreamFusionPlusDataManagerConfig = DreamFusionPlusDataManagerConfig()
    """specifies the datamanager config"""

    rgb_scale: float = 0.5
    """scale for the rgb loss"""
    rgb_scale_decay: float = 0.98
    """decay for the rgb scale"""
    transient_loss_scale: float = 10
    """scale for the transient loss"""
    transient_quantized_loss_scale: float = 0
    """scale for the transient quantized loss"""


class DreamfusionPlusPipeline(DreamfusionPipeline):
    """Dreamfusion pipeline"""

    config: DreamfusionPlusPipelineConfig
    datamanager: DreamFusionPlusDataManager
    model: DreamFusionModel

    def __init__(
        self,
        config: DreamfusionPlusPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"],
        world_size: int,
        local_rank: int,
        grad_scaler,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        self.rgb_loss = torch.nn.MSELoss()
        self.rgb_scale = self.config.rgb_scale

    def get_train_loss_dict_input_img(self, step: int):
        ray_bundle, batch = self.datamanager.next_input(step)

        if self.model.collider is not None:
            ray_bundle = self.model.collider(ray_bundle)

        model_outputs = self.model.get_outputs(ray_bundle)
        if self.config.datamanager.use_input_transient:
            rgb_loss = (
                self.rgb_loss(
                    batch["image"].to(self.device) * batch["input_mask"].unsqueeze(-1),
                    model_outputs["rgb"] * batch["input_mask"].unsqueeze(-1),
                )
                * self.rgb_scale
            )
            Image.fromarray(255 * np.concatenate([np.ones((10, 60)), np.zeros((50, 60))], axis=0)).convert("RGB").save(
                "outputs/test.jpg"
            )
            data = torch.sigmoid(self.datamanager.input_mask.detach()).cpu().numpy() * 255
            Image.fromarray(data).convert("RGB").save("outputs/input_mask.png")
        else:
            rgb_loss = self.rgb_loss(batch["image"].to(self.device), model_outputs["rgb"]) * self.config.rgb_scale
        loss_dict = {"rgb_loss": rgb_loss}
        if self.config.datamanager.use_input_transient:
            loss_dict["transient_quantized_loss"] = (
                torch.exp(-((4 * batch["input_mask"] - 2) ** 2)).mean() * self.config.transient_quantized_loss_scale
            )  # Lowest when mask is 0 or 1
            # When our transients are small (ie: everything is just masked from the input image), then
            # the mean mask / transient value will be small, and when we negate it, the model will try and make the
            # mean mask value larger
            loss_dict["transient_loss"] = (
                -torch.minimum(batch["input_mask"].mean(), torch.tensor(self.model.target_transmittance))
                * self.config.transient_loss_scale
            )

        Image.fromarray(255 * model_outputs["alphas_loss"].detach().reshape(64, 64).cpu().numpy()).convert("RGB").save(
            "outputs/alphas_loss.jpg"
        )

        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""

        def anneal_rgb_scale(
            self, training_callback_attributes: TrainingCallbackAttributes, step: int  # pylint: disable=unused-argument
        ):
            self.rgb_scale *= self.config.rgb_scale_decay

        pipeline_callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=10,
                func=anneal_rgb_scale,
                args=[self, training_callback_attributes],
            ),
        ]
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks + pipeline_callbacks
        return callbacks
