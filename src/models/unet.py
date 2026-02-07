import torch
import torch.nn as nn

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Optional, Tuple, Union

from .blocks.mid_blocks import MidBlock
from .blocks.up_blocks import get_up_block
from .blocks.down_blocks import get_down_block
from .blocks.embeddings import Timesteps, TimestepEmbedding



@dataclass
class UNetDenoisingModelOutput(BaseOutput):
    sample: torch.Tensor



class UNetDenoisingModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        time_embedding_dim: Optional[int] = None,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = ("DownBlock", "AttnDownBlock", "AttnDownBlock", "AttnDownBlock"),
        mid_block_type: Optional[str] = "MidBlock",
        up_block_types: Tuple[str, ...] = ("AttnUpBlock", "AttnUpBlock", "AttnUpBlock", "UpBlock"),
        block_out_channels: Tuple[int, ...] = (64, 128, 256, 512),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "scale_shift",
        add_attention: bool = True,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.in_channels = in_channels
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )
        
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )
        
        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type is None:
            self.mid_block = None
        else:
            self.mid_block = MidBlock(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
                resnet_groups=norm_num_groups,
                attn_groups=attn_norm_num_groups,
                add_attention=add_attention,
            )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)

        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)


    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
    ) -> Union[UNetDenoisingModelOutput, Tuple]:
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        t_emb = self.time_proj(timesteps)
        temb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=temb)
            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(sample, temb)

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            
            sample = upsample_block(sample, res_samples, temb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)
        
        return UNetDenoisingModelOutput(sample=sample)