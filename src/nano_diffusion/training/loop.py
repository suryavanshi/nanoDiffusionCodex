"""Training and evaluation loops for masked discrete diffusion."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch and tqdm to use nano_diffusion.training.loop") from exc

from nano_diffusion.data.dataset import TokenManifestDataset, collate_token_batch
from nano_diffusion.data.tokenizer import ByteTokenizer
from nano_diffusion.diffusion.discrete import MaskingDiffusion
from nano_diffusion.models.transformer import TinyTransformerDenoiser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(config: dict[str, Any], tokenizer: ByteTokenizer) -> TinyTransformerDenoiser:
    model_cfg = config["model"]
    diffusion_cfg = config["diffusion"]
    vocab_size = int(model_cfg.get("vocab_size", tokenizer.vocab_size))
    if vocab_size != tokenizer.vocab_size:
        raise ValueError(f"ByteTokenizer requires vocab_size={tokenizer.vocab_size}, got {vocab_size}")
    return TinyTransformerDenoiser(
        vocab_size=vocab_size,
        dim=int(model_cfg["dim"]),
        layers=int(model_cfg["layers"]),
        heads=int(model_cfg["heads"]),
        max_seq_len=int(model_cfg["max_seq_len"]),
        timesteps=int(diffusion_cfg["timesteps"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask_positions: torch.Tensor) -> torch.Tensor:
    if not mask_positions.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask_positions], targets[mask_positions])


def loss_to_perplexity(loss: float) -> float:
    try:
        return float(math.exp(min(loss, 50.0)))
    except OverflowError:
        return float("inf")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    diffusion: MaskingDiffusion,
    device: torch.device,
    max_batches: int = 8,
) -> float:
    return evaluate_metrics(model, loader, diffusion, device, max_batches=max_batches)["loss"]


@torch.no_grad()
def evaluate_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    diffusion: MaskingDiffusion,
    device: torch.device,
    max_batches: int = 8,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    masked_tokens = 0
    batches = 0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        t = diffusion.sample_timesteps(input_ids.size(0), device)
        noisy, mask_positions = diffusion.q_sample(input_ids, attention_mask, t)
        logits = model(noisy, t, attention_mask)
        loss = masked_ce_loss(logits, input_ids, mask_positions)
        batch_masked_tokens = int(mask_positions.sum().item())
        total_nll += float(loss.item()) * batch_masked_tokens
        masked_tokens += batch_masked_tokens
        batches += 1
    model.train()
    loss = total_nll / max(1, masked_tokens)
    return {
        "loss": loss,
        "perplexity": loss_to_perplexity(loss),
        "masked_tokens": float(masked_tokens),
        "batches": float(batches),
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(config["project"].get("seed", 42)))
    tokenizer = ByteTokenizer()
    device = resolve_device(str(config["training"].get("device", "auto")))

    train_dataset = TokenManifestDataset(config["data"]["train_manifest"])
    val_dataset = TokenManifestDataset(config["data"]["val_manifest"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        drop_last=False,
        collate_fn=collate_token_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_token_batch,
    )

    model = build_model(config, tokenizer).to(device)
    diffusion = MaskingDiffusion(
        timesteps=int(config["diffusion"]["timesteps"]),
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        schedule=str(config["diffusion"].get("schedule", "cosine")),
        never_mask_token_ids=(tokenizer.bos_token_id, tokenizer.eos_token_id),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )

    total_steps = int(config["training"]["total_steps"])
    eval_interval = int(config["training"].get("eval_interval", 100))
    save_interval = int(config["training"].get("save_interval", 500))
    output_dir = Path(config["training"].get("output_dir", "runs/nano-diffusion"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.jsonl"
    best_val = float("inf")
    step = 0
    train_iter = iter(train_loader)
    progress = tqdm(total=total_steps, desc="train", leave=False)
    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        t = diffusion.sample_timesteps(input_ids.size(0), device)
        noisy, mask_positions = diffusion.q_sample(input_ids, attention_mask, t)
        logits = model(noisy, t, attention_mask)
        loss = masked_ce_loss(logits, input_ids, mask_positions)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 1.0)))
        optimizer.step()

        step += 1
        progress.update(1)
        progress.set_postfix(loss=f"{loss.item():.3f}")

        record = {"step": step, "train_loss": float(loss.item())}
        if step % eval_interval == 0 or step == total_steps:
            val_metrics = evaluate_metrics(model, val_loader, diffusion, device)
            val_loss = val_metrics["loss"]
            record["val_loss"] = val_loss
            record["val_perplexity"] = val_metrics["perplexity"]
            record["val_masked_tokens"] = val_metrics["masked_tokens"]
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(output_dir / "best.pt", model, optimizer, config, step, val_loss)
        if step % save_interval == 0 or step == total_steps:
            save_checkpoint(output_dir / "last.pt", model, optimizer, config, step, record.get("val_loss"))
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    progress.close()
    return {
        "output_dir": str(output_dir),
        "steps": step,
        "best_val_loss": best_val,
        "best_val_perplexity": loss_to_perplexity(best_val),
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
    val_loss: float | None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )
