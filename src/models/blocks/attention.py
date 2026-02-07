import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional



class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        eps: float = 1e-5,
        residual_connection: bool = False,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.residual_connection = residual_connection
        self.scale = dim_head ** -0.5 if scale_qk else 1.0

        # Optional GroupNorm on input
        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(
                num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True
            )
        else:
            self.group_norm = None

        # Optional normalization for cross-attention context
        if cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            self.norm_cross = nn.GroupNorm(
                num_channels=self.cross_attention_dim,
                num_groups=cross_attention_norm_num_groups,
                eps=1e-5,
                affine=True,
            )
        else:
            self.norm_cross = None

        # Projections
        self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.query_dim, bias=out_bias),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Save residual for skip connection
        residual = hidden_states

        # Apply GroupNorm if specified 
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states)

        # Input shape: (batch, channels, height, width)
        batch, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch, channels, -1).transpose(1, 2)  # (B, N, C)

        # Determine context (for cross-attention)
        if encoder_hidden_states is None:
            context = hidden_states
        else:
            context = encoder_hidden_states
            
            if self.norm_cross is not None:
                if isinstance(self.norm_cross, nn.GroupNorm):
                    context = context.transpose(1, 2)  # (B, C, N)
                    context = self.norm_cross(context)
                    context = context.transpose(1, 2)  # (B, N, C)
                else:
                    context = self.norm_cross(context)

        # Project queries, keys, values
        query = self.to_q(hidden_states)
        key = self.to_k(context)
        value = self.to_v(context)

        # Split into heads: (B, N, inner_dim) -> (B, heads, N, dim_head)
        batch_size = query.shape[0]
        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

        # Upcast attention
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            # Expand mask to attention shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attention_scores = attention_scores + attention_mask

        # Upcast softmax
        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)

        # Apply attention to values
        hidden_states = torch.matmul(attention_probs, value)

        # Merge heads: (B, heads, N, dim_head) -> (B, N, inner_dim)
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = hidden_states.view(batch_size, -1, self.heads * self.dim_head)

        # Output projection
        hidden_states = self.to_out(hidden_states)

        # Reshape back to image format: (B, N, C) -> (B, C, H, W)
        hidden_states = hidden_states.transpose(1, 2).view(batch, channels, height, width)

        # Add residual connection if specified
        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states