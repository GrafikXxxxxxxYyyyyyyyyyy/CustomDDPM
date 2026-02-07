import torch
import torch.nn as nn

from typing import Tuple, Optional
from .resnet import ResNetBlock
from .downsampling import Downsample2D
from .attention import Attention



def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_groups: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    resnet_time_scale_shift: str = "scale_shift",
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
    cross_attention_dim: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
):
    if down_block_type == "DownBlock":
        return DownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_groups=resnet_groups or min(in_channels // 4, 32),
            add_downsample=add_downsample,
            downsample_padding=downsample_padding or 1,
        )
        
    elif down_block_type == "AttnDownBlock":
        if add_downsample is False:
            effective_downsample_type = None
        else:
            effective_downsample_type = downsample_type or "conv"
            
        return AttnDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_groups=resnet_groups or min(in_channels // 4, 32),
            attention_head_dim=attention_head_dim,
            downsample_padding=downsample_padding or 1,
            downsample_type=effective_downsample_type,
        )
        
    else:
        raise NotImplementedError(
            f"Down block type '{down_block_type}' is not implemented. "
            f"Available types: 'DownBlock', 'AttnDownBlock'"
        )



class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "scale_shift",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResNetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        use_conv=True,
                        padding=downsample_padding,
                    )
                ]
            )
        else:
            self.downsamplers = None

        
    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states



class AttnDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "scale_shift",
        resnet_groups: int = 32,
        attention_head_dim: int = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
    ):
        super().__init__()

        if attention_head_dim is None:
            attention_head_dim = out_channels

        resnets = []
        attentions = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResNetBlock(
                    in_channels=in_channels,
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

        if downsample_type == "conv":
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        use_conv=True,
                        padding=downsample_padding,
                    )
                ]
            )
        else:
            self.downsamplers = None

        
    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            
            output_states += (hidden_states,)

        return hidden_states, output_states