#!/usr/bin/env python3
"""Run a tiny end-to-end training smoke test on Modal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import modal

app = modal.App("nano-diffusion-codex-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "datasets>=2.18", "PyYAML>=6.0", "tqdm>=4.66")
    .add_local_python_source("nano_diffusion", copy=True)
)


@app.function(image=image, gpu="T4", timeout=1800)
def run_modal_training(
    dataset_name: str = "codeparrot/codeparrot-clean",
    split: str = "train",
    max_samples: int = 128,
    val_samples: int = 32,
    total_steps: int = 20,
) -> dict:
    from datasets import load_dataset

    from nano_diffusion.data.manifest import extract_text, write_token_manifest
    from nano_diffusion.data.tokenizer import ByteTokenizer
    from nano_diffusion.training.loop import train

    workdir = Path("/tmp/nano_diffusion_codex")
    data_dir = workdir / "data"
    run_dir = workdir / "runs"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(dataset_name, split=split, streaming=True)
    rows = []
    for row in raw:
        text = extract_text(row)
        if text and len(text) >= 32:
            rows.append(row)
        if len(rows) >= max_samples + val_samples:
            break

    tokenizer = ByteTokenizer()
    val_path = data_dir / "val.jsonl"
    train_path = data_dir / "train.jsonl"
    val_count = write_token_manifest(rows[:val_samples], val_path, tokenizer, 128)
    train_count = write_token_manifest(rows[val_samples:], train_path, tokenizer, 128)

    config = {
        "project": {"name": "nano-diffusion-modal-smoke", "seed": 42},
        "model": {
            "vocab_size": tokenizer.vocab_size,
            "dim": 96,
            "layers": 2,
            "heads": 4,
            "max_seq_len": 128,
            "dropout": 0.1,
        },
        "diffusion": {"timesteps": 16, "schedule": "cosine"},
        "training": {
            "batch_size": 8,
            "learning_rate": 5.0e-4,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "total_steps": total_steps,
            "eval_interval": max(5, total_steps // 2),
            "save_interval": total_steps,
            "output_dir": str(run_dir),
            "device": "auto",
        },
        "data": {"train_manifest": str(train_path), "val_manifest": str(val_path)},
    }

    result = train(config)
    result.update(
        {
            "dataset": dataset_name,
            "train_examples": train_count,
            "val_examples": val_count,
            "cuda_available": __import__("torch").cuda.is_available(),
        }
    )
    return result


@app.local_entrypoint()
def main(
    dataset_name: str = "codeparrot/codeparrot-clean",
    split: str = "train",
    max_samples: int = 128,
    val_samples: int = 32,
    total_steps: int = 20,
) -> None:
    result = run_modal_training.remote(
        dataset_name=dataset_name,
        split=split,
        max_samples=max_samples,
        val_samples=val_samples,
        total_steps=total_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
