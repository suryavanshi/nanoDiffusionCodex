#!/usr/bin/env python3
"""Run diffusion model training and evaluation jobs on Modal."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import modal

app = modal.App("nano-diffusion-codex-train")
ARTIFACT_VOLUME_NAME = "nano-diffusion-codex-artifacts"
ARTIFACT_MOUNT = Path("/vol")
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "datasets>=2.18",
        "PyYAML>=6.0",
        "tokenizers>=0.15",
        "tqdm>=4.66",
        "human-eval>=1.0.3",
    )
    .add_local_python_source("nano_diffusion", copy=True)
)


@app.function(image=image, gpu="A100", timeout=12 * 60 * 60, volumes={str(ARTIFACT_MOUNT): artifact_volume})
def run_modal_training(
    experiment_name: str = "bpe-fim-100m",
    dataset_name: str = "codeparrot/codeparrot-clean",
    dataset_config: Optional[str] = None,
    split: str = "train",
    text_field: Optional[str] = None,
    max_samples: int = 128,
    val_samples: int = 32,
    max_seq_len: int = 2048,
    tokenizer_type: str = "bpe",
    vocab_size: int = 16384,
    fim_rate: float = 0.5,
    model_dim: int = 768,
    model_layers: int = 12,
    model_heads: int = 12,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    total_steps: int = 20,
    learning_rate: float = 1.0e-4,
    min_learning_rate: float = 1.0e-5,
    warmup_steps: int = 100,
    eval_batches: int = 64,
) -> dict:
    from datasets import load_dataset

    from nano_diffusion.data.manifest import extract_text, write_token_manifest
    from nano_diffusion.data.tokenizer import ByteTokenizer, train_bpe_tokenizer
    from nano_diffusion.training.loop import train

    workdir = Path("/tmp/nano_diffusion_codex")
    data_dir = workdir / "data"
    artifact_dir = ARTIFACT_MOUNT / experiment_name
    run_dir = artifact_dir / "runs"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

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

    tokenizer_path = artifact_dir / "tokenizer.json"
    if tokenizer_type == "bpe":
        tokenizer = train_bpe_tokenizer(
            (extract_text(row, text_field) or "" for row in rows),
            tokenizer_path,
            vocab_size=vocab_size,
        )
        tokenizer_config = {"type": "bpe", "path": str(tokenizer_path)}
    elif tokenizer_type == "byte":
        tokenizer = ByteTokenizer()
        tokenizer_config = {"type": "byte"}
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")
    val_path = data_dir / "val.jsonl"
    train_path = data_dir / "train.jsonl"
    val_count = write_token_manifest(
        rows[:val_samples],
        val_path,
        tokenizer,
        max_seq_len,
        text_field=text_field,
        fim_rate=fim_rate,
    )
    train_count = write_token_manifest(
        rows[val_samples:],
        train_path,
        tokenizer,
        max_seq_len,
        text_field=text_field,
        fim_rate=fim_rate,
    )

    config = {
        "project": {"name": "nano-diffusion-modal-smoke", "seed": 42},
        "model": {
            "vocab_size": vocab_size if tokenizer_type == "bpe" else tokenizer.vocab_size,
            "dim": model_dim,
            "layers": model_layers,
            "heads": model_heads,
            "max_seq_len": max_seq_len,
            "dropout": 0.1,
        },
        "diffusion": {"timesteps": 16, "schedule": "cosine"},
        "training": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "lr_schedule": "cosine",
            "warmup_steps": warmup_steps,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "total_steps": total_steps,
            "eval_interval": max(5, total_steps // 2),
            "eval_batches": eval_batches,
            "save_interval": total_steps,
            "output_dir": str(run_dir),
            "device": "auto",
        },
        "data": {"train_manifest": str(train_path), "val_manifest": str(val_path)},
        "tokenizer": tokenizer_config,
    }

    result = train(config)
    try:
        import torch

        best_checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
        torch.save(
            {
                "model": best_checkpoint["model"],
                "config": best_checkpoint["config"],
                "step": best_checkpoint.get("step"),
                "val_loss": best_checkpoint.get("val_loss"),
            },
            run_dir / "best_model_only.pt",
        )
    except Exception as exc:
        print(f"[modal] warning: could not write model-only checkpoint: {exc}")
    config_path = artifact_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    manifest_dir = artifact_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_path, manifest_dir / "train.jsonl")
    shutil.copy2(val_path, manifest_dir / "val.jsonl")
    metrics_path = run_dir / "metrics.jsonl"
    metric_records = []
    if metrics_path.exists():
        metric_records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    final_record = metric_records[-1] if metric_records else {}
    result.update(
        {
            "experiment_name": experiment_name,
            "artifact_volume": ARTIFACT_VOLUME_NAME,
            "artifact_path": str(artifact_dir.relative_to(ARTIFACT_MOUNT)),
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "text_field": text_field,
            "tokenizer_type": tokenizer_type,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "model_vocab_size": config["model"]["vocab_size"],
            "fim_rate": fim_rate,
            "max_seq_len": max_seq_len,
            "model_dim": model_dim,
            "model_layers": model_layers,
            "model_heads": model_heads,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_steps": warmup_steps,
            "eval_batches": eval_batches,
            "train_examples": train_count,
            "val_examples": val_count,
            "cuda_available": __import__("torch").cuda.is_available(),
            "metric_name": "masked_token_perplexity",
            "final_train_loss": final_record.get("train_loss"),
            "final_val_loss": final_record.get("val_loss"),
            "final_val_perplexity": final_record.get("val_perplexity"),
        }
    )
    (artifact_dir / "training_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    artifact_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=6 * 60 * 60, volumes={str(ARTIFACT_MOUNT): artifact_volume})
def run_modal_humaneval(
    experiment_name: str = "bpe-fim-100m",
    checkpoint_name: str = "best.pt",
    max_tasks: int = 164,
    new_tokens: int = 256,
    diffusion_steps: int = 16,
    timeout_s: float = 3.0,
) -> dict:
    import os

    # HumanEval executes generated code. Keep the opt-in local to this isolated Modal job.
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    from human_eval.data import read_problems
    from human_eval.execution import check_correctness

    from nano_diffusion.inference.sampler import sample_completion

    artifact_dir = ARTIFACT_MOUNT / experiment_name
    checkpoint_path = artifact_dir / "runs" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    problems = read_problems()
    task_ids = list(problems)[:max_tasks]
    results = []
    for completion_id, task_id in enumerate(task_ids):
        problem = problems[task_id]
        generated = sample_completion(
            str(checkpoint_path),
            problem["prompt"],
            new_tokens=new_tokens,
            steps=diffusion_steps,
            device_name="auto",
        )
        completion = generated[len(problem["prompt"]) :] if generated.startswith(problem["prompt"]) else generated
        check = check_correctness(problem, completion, timeout_s, completion_id=completion_id)
        results.append(
            {
                "task_id": task_id,
                "passed": bool(check["passed"]),
                "result": check["result"],
                "completion": completion,
            }
        )

    passed = sum(1 for row in results if row["passed"])
    summary = {
        "experiment_name": experiment_name,
        "checkpoint": checkpoint_name,
        "tasks": len(results),
        "passed": passed,
        "pass_at_1": passed / max(1, len(results)),
        "new_tokens": new_tokens,
        "diffusion_steps": diffusion_steps,
        "artifact_volume": ARTIFACT_VOLUME_NAME,
        "artifact_path": str(artifact_dir.relative_to(ARTIFACT_MOUNT)),
    }
    eval_dir = artifact_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "humaneval_results.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in results) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "humaneval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    artifact_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    experiment_name: str = "bpe-fim-100m",
    dataset_name: str = "codeparrot/codeparrot-clean",
    dataset_config: Optional[str] = None,
    split: str = "train",
    text_field: Optional[str] = "content",
    max_samples: int = 128,
    val_samples: int = 32,
    max_seq_len: int = 2048,
    tokenizer_type: str = "bpe",
    vocab_size: int = 16384,
    fim_rate: float = 0.5,
    model_dim: int = 768,
    model_layers: int = 12,
    model_heads: int = 12,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    total_steps: int = 20,
    learning_rate: float = 1.0e-4,
    min_learning_rate: float = 1.0e-5,
    warmup_steps: int = 100,
    eval_batches: int = 64,
    background: bool = False,
) -> None:
    kwargs = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "text_field": text_field,
        "max_samples": max_samples,
        "val_samples": val_samples,
        "max_seq_len": max_seq_len,
        "tokenizer_type": tokenizer_type,
        "vocab_size": vocab_size,
        "fim_rate": fim_rate,
        "model_dim": model_dim,
        "model_layers": model_layers,
        "model_heads": model_heads,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "total_steps": total_steps,
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_steps": warmup_steps,
        "eval_batches": eval_batches,
    }
    if background:
        function_call = run_modal_training.spawn(**kwargs)
        result = {
            "status": "spawned",
            "function_call_id": getattr(function_call, "object_id", None),
            "experiment_name": experiment_name,
            "artifact_volume": ARTIFACT_VOLUME_NAME,
            "artifact_path": experiment_name,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    result = run_modal_training.remote(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def humaneval(
    experiment_name: str = "bpe-fim-100m",
    checkpoint_name: str = "best.pt",
    max_tasks: int = 164,
    new_tokens: int = 256,
    diffusion_steps: int = 16,
) -> None:
    result = run_modal_humaneval.remote(
        experiment_name=experiment_name,
        checkpoint_name=checkpoint_name,
        max_tasks=max_tasks,
        new_tokens=new_tokens,
        diffusion_steps=diffusion_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
