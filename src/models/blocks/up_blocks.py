import torch
import torch.nn as nn

from .resnet import ResNetBlock
from .attention import Attention
from .upsampling import Upsample2D
from typing import Tuple, Optional



def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float = 1e-6,
    resnet_time_scale_shift: str = "scale_shift",
    resnet_groups: Optional[int] = None,
    attention_head_dim: Optional[int] = None,
    upsample_type: Optional[str] = None,
    dropout: float = 0.0,
    cross_attention_dim: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
):
    if up_block_type == "UpBlock":
        return UpBlock(
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,  
            resnet_groups=resnet_groups or min(prev_output_channel // 4, 32),
            add_upsample=add_upsample,
        )
        
    elif up_block_type == "AttnUpBlock":
        if add_upsample is False:
            effective_upsample_type = None
        else:
            effective_upsample_type = upsample_type or "conv" 
            
        return AttnUpBlock(
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift, 
            resnet_groups=resnet_groups or min(prev_output_channel // 4, 32),
            attention_head_dim=attention_head_dim,
            upsample_type=effective_upsample_type or "conv",
        )

    else:
        raise NotImplementedError(
            f"Up block type '{up_block_type}' is not implemented. "
            f"Available types: 'UpBlock', 'AttnUpBlock'"
        )



class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "scale_shift",
        resnet_groups: int = 32,
        add_upsample: bool = True,
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResNetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        use_conv=True,
                    )
                ]
            )
        else:
            self.upsamplers = None

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            # Извлекаем skip-connection из энкодера (в обратном порядке)
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                
            # Конкатенируем текущее состояние с skip-connection по каналам
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # Применяем ResNet-блок
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
    


class AttnUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "scale_shift",
        resnet_groups: int = 32,
        attention_head_dim: int = 1,
        upsample_type: str = "conv",
    ):
        super().__init__()

        resnets = []
        attentions = []
        
        if attention_head_dim is None:
            attention_head_dim = out_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResNetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                )
            )
            attentions.append(
                Attention(
                    query_dim=out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if upsample_type == "conv":
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        use_conv=True,
                    )
                ]
            )
        else:
            self.upsamplers = None

        
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states