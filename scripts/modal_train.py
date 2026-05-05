#!/usr/bin/env python3
"""Run a tiny end-to-end training smoke test on Modal."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

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
    dataset_config: Optional[str] = None,
    split: str = "train",
    text_field: Optional[str] = None,
    max_samples: int = 128,
    val_samples: int = 32,
    max_seq_len: int = 128,
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

    if dataset_config:
        raw = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    else:
        raw = load_dataset(dataset_name, split=split, streaming=True)
    rows = []
    for row in raw:
        text = extract_text(row, text_field)
        if text and len(text) >= 32:
            rows.append(row)
        if len(rows) >= max_samples + val_samples:
            break

    tokenizer = ByteTokenizer()
    val_path = data_dir / "val.jsonl"
    train_path = data_dir / "train.jsonl"
    val_count = write_token_manifest(rows[:val_samples], val_path, tokenizer, max_seq_len, text_field=text_field)
    train_count = write_token_manifest(rows[val_samples:], train_path, tokenizer, max_seq_len, text_field=text_field)

    config = {
        "project": {"name": "nano-diffusion-modal-smoke", "seed": 42},
        "model": {
            "vocab_size": tokenizer.vocab_size,
            "dim": 96,
            "layers": 2,
            "heads": 4,
            "max_seq_len": max_seq_len,
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
    metrics_path = run_dir / "metrics.jsonl"
    metric_records = []
    if metrics_path.exists():
        metric_records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    final_record = metric_records[-1] if metric_records else {}
    result.update(
        {
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "text_field": text_field,
            "train_examples": train_count,
            "val_examples": val_count,
            "cuda_available": __import__("torch").cuda.is_available(),
            "metric_name": "masked_token_perplexity",
            "final_train_loss": final_record.get("train_loss"),
            "final_val_loss": final_record.get("val_loss"),
            "final_val_perplexity": final_record.get("val_perplexity"),
        }
    )
    return result


@app.local_entrypoint()
def main(
    dataset_name: str = "codeparrot/codeparrot-clean",
    dataset_config: Optional[str] = None,
    split: str = "train",
    text_field: Optional[str] = "content",
    max_samples: int = 128,
    val_samples: int = 32,
    max_seq_len: int = 128,
    total_steps: int = 20,
) -> None:
    result = run_modal_training.remote(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        text_field=text_field,
        max_samples=max_samples,
        val_samples=val_samples,
        max_seq_len=max_seq_len,
        total_steps=total_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
