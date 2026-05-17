#!/usr/bin/env python3
"""Evaluate a checkpoint on a token manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from torch.utils.data import DataLoader

    from nano_diffusion.data.dataset import TokenManifestDataset, collate_token_batch
    from nano_diffusion.diffusion.discrete import MaskingDiffusion
    from nano_diffusion.inference.sampler import load_model
    from nano_diffusion.training.loop import evaluate_metrics

    model, config, tokenizer, device = load_model(args.checkpoint, args.device)
    manifest = args.manifest or config["data"]["val_manifest"]
    max_batches = args.max_batches or int(config.get("training", {}).get("eval_batches", 8))
    dataset = TokenManifestDataset(manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_token_batch)
    diffusion = MaskingDiffusion(
        timesteps=int(config["diffusion"]["timesteps"]),
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        schedule=str(config["diffusion"].get("schedule", "cosine")),
        never_mask_token_ids=(tokenizer.bos_token_id, tokenizer.eos_token_id),
    )
    metrics = evaluate_metrics(model, loader, diffusion, device, max_batches=max_batches)
    metrics["manifest"] = manifest
    metrics["max_batches"] = max_batches
    metrics["metric_name"] = "masked_token_perplexity"
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
