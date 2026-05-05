"""Discrete masking diffusion process for code tokens."""

from __future__ import annotations

import math

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use nano_diffusion.diffusion.discrete") from exc


class MaskingDiffusion:
    def __init__(
        self,
        timesteps: int,
        mask_token_id: int,
        pad_token_id: int,
        schedule: str = "cosine",
        never_mask_token_ids: tuple[int, ...] = (),
    ) -> None:
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        self.timesteps = timesteps
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.schedule = schedule
        self.never_mask_token_ids = never_mask_token_ids

    def mask_probability(self, t: torch.Tensor) -> torch.Tensor:
        progress = t.float().clamp(1, self.timesteps) / float(self.timesteps)
        if self.schedule == "linear":
            return progress
        if self.schedule == "cosine":
            return 1.0 - torch.cos(progress * math.pi / 2.0)
        raise ValueError(f"Unknown diffusion schedule: {self.schedule}")

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(1, self.timesteps + 1, (batch_size,), device=device)

    def q_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = self.mask_probability(t).view(-1, 1)
        can_mask = attention_mask & (input_ids != self.pad_token_id)
        for token_id in self.never_mask_token_ids:
            can_mask = can_mask & (input_ids != token_id)
        random_values = torch.rand(input_ids.shape, device=input_ids.device)
        mask_positions = (random_values < probs) & can_mask

        # Guarantee at least one supervised position per sample.
        empty = ~mask_positions.any(dim=1)
        if empty.any():
            for row in torch.nonzero(empty, as_tuple=False).flatten().tolist():
                valid_positions = torch.nonzero(can_mask[row], as_tuple=False).flatten()
                if len(valid_positions) > 0:
                    chosen = valid_positions[
                        torch.randint(0, len(valid_positions), (1,), device=input_ids.device)
                    ]
                    mask_positions[row, chosen] = True

        noisy = input_ids.clone()
        noisy[mask_positions] = self.mask_token_id
        return noisy, mask_positions
