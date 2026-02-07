import torch
import torch.nn as nn

from typing import Optional
from .resnet import ResNetBlock
from .attention import Attention



class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "scale_shift",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        add_attention: bool = True,
        attention_head_dim: int = 1,
    ):
        super().__init__()
        
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None:
            # attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None
            attn_groups = resnet_groups

        resnets = [
            ResNetBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        query_dim=in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResNetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)


    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)

            hidden_states = resnet(hidden_states, temb)

        return hidden_states