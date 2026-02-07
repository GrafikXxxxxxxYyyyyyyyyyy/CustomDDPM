# /usr/bin/env python3
import torch 
import numpy as np
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusers.utils import BaseOutput
from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor

from .models import UNetDenoisingModel



class InferencePipelineOutput(BaseOutput):
    images: np.ndarray



class InferencePipeline:
    def __init__(
        self,
        denoiser: UNetDenoisingModel,
        scheduler: Optional[DDPMScheduler] = None,
        device: str = 'mps'
    ):
        self.denoiser = denoiser
        self.scheduler = scheduler or DDPMScheduler()
        self.device = device


    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple]:
        
        if isinstance(self.denoiser.sample_size, int):
            image_shape = (
                batch_size,
                self.denoiser.in_channels,
                self.denoiser.sample_size,
                self.denoiser.sample_size,
            )
        else:
            image_shape = (batch_size, self.denoiser.in_channels, *self.denoiser.sample_size)

        image = randn_tensor(image_shape, generator=generator, device=self.device)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        for t in tqdm(self.scheduler.timesteps):
            model_output = self.denoiser(image, t).sample

            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if not return_dict:
            return (image,)
        
        return InferencePipelineOutput(images=image)