"""Iterative mask-denoising sampler."""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use nano_diffusion.inference.sampler") from exc

from nano_diffusion.data.tokenizer import CodeTokenizer, load_tokenizer_from_config
from nano_diffusion.training.loop import build_model, resolve_device


def load_model(checkpoint_path: str, device_name: str = "auto") -> tuple[torch.nn.Module, dict[str, Any], CodeTokenizer, torch.device]:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    tokenizer = load_tokenizer_from_config(config)
    model = build_model(config, tokenizer).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config, tokenizer, device


@torch.no_grad()
def sample_completion(
    checkpoint_path: str,
    prompt: str,
    new_tokens: int = 128,
    steps: int | None = None,
    device_name: str = "auto",
) -> str:
    model, config, tokenizer, device = load_model(checkpoint_path, device_name)
    max_seq_len = int(config["model"]["max_seq_len"])
    steps = steps or int(config["diffusion"]["timesteps"])

    prompt_ids, prompt_attention = tokenizer.encode(prompt, max_seq_len)
    prompt_len = max(1, sum(prompt_attention) - 1)
    generation_end = min(max_seq_len - 1, prompt_len + new_tokens)

    ids = torch.full((1, max_seq_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((1, max_seq_len), dtype=torch.bool, device=device)
    prefix = prompt_ids[:prompt_len]
    ids[0, :prompt_len] = torch.tensor(prefix, dtype=torch.long, device=device)
    ids[0, prompt_len:generation_end] = tokenizer.mask_token_id
    ids[0, generation_end] = tokenizer.eos_token_id
    attention_mask[0, : generation_end + 1] = True

    unresolved = ids == tokenizer.mask_token_id
    for index, t_value in enumerate(range(steps, 0, -1), start=1):
        if not unresolved.any():
            break
        t = torch.tensor([min(t_value, int(config["diffusion"]["timesteps"]))], dtype=torch.long, device=device)
        logits = model(ids, t, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)
        candidates = torch.nonzero(unresolved[0], as_tuple=False).flatten()
        fill_count = max(1, math.ceil(len(candidates) / (steps - index + 1)))
        chosen_scores = confidence[0, candidates]
        chosen = candidates[torch.topk(chosen_scores, k=min(fill_count, len(candidates))).indices]
        ids[0, chosen] = predicted[0, chosen]
        unresolved[0, chosen] = False

    return tokenizer.decode(ids[0].tolist())
