import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional



class Upsample2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        use_conv: bool = True,
        use_conv_transpose: bool = False,
        kernel_size: Optional[int] = None,
        padding: int = 1,
        bias: bool = True,
        interpolate: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.interpolate = interpolate

        self.conv = None
        if use_conv: 
            if kernel_size is None:
                kernel_size = 3

            self.conv = nn.Conv2d(
                self.in_channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias,
            )
        elif use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4

            self.conv = nn.ConvTranspose2d(
                self.in_channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias,
            )

        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.in_channels, (
            f"Input channel mismatch: expected {self.in_channels}, got {hidden_states.shape[1]}"
        )

        if self.use_conv_transpose:
            return self.conv(hidden_states)
        
        if self.interpolate:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

        if self.use_conv and self.conv is not None:
            hidden_states = self.conv(hidden_states)

        return hidden_states