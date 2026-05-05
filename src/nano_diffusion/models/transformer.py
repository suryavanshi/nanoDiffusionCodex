"""Tiny Transformer denoiser used by the MVP training loop."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use nano_diffusion.models.transformer") from exc


class TinyTransformerDenoiser(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        layers: int,
        heads: int,
        max_seq_len: int,
        timesteps: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.timestep_embedding = nn.Sequential(
            nn.Embedding(timesteps + 1, dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = hidden + self.timestep_embedding(timesteps).unsqueeze(1)
        key_padding_mask = ~attention_mask.bool()
        hidden = self.blocks(hidden, src_key_padding_mask=key_padding_mask)
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)
