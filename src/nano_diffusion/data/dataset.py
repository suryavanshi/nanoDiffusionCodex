"""PyTorch datasets for token manifests."""

from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise ImportError("Install torch to use nano_diffusion.data.dataset") from exc

from .manifest import read_token_manifest


class TokenManifestDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token manifest not found: {self.path}")
        self.records = list(read_token_manifest(self.path))
        if not self.records:
            raise ValueError(f"Token manifest is empty: {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.records[index]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.bool),
        }


def collate_token_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
    }
