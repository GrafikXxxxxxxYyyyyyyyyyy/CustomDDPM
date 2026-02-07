import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional



class Downsample2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        use_conv: bool = True,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        stride = 2

        if use_conv:
            self.conv = nn.Conv2d(
                self.in_channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
            )
        else:
            assert self.in_channels == self.out_channels
            self.conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.in_channels, (
            f"Input channel mismatch: expected {self.in_channels}, got {hidden_states.shape[1]}"
        )

        return self.conv(hidden_states)
