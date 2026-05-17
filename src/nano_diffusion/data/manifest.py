"""JSONL manifest helpers for tokenized code samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .tokenizer import CodeTokenizer


TEXT_FIELDS = ("content", "code", "text", "func_code_string", "whole_func_string")


def extract_text(row: dict, preferred_field: str | None = None) -> str | None:
    if preferred_field:
        value = row.get(preferred_field)
        return value if isinstance(value, str) and value.strip() else None
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def write_token_manifest(
    rows: Iterable[dict],
    output_path: Path,
    tokenizer: CodeTokenizer,
    max_seq_len: int,
    text_field: str | None = None,
    min_chars: int = 32,
    fim_rate: float = 0.0,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            text = extract_text(row, text_field)
            if text is None or len(text) < min_chars:
                continue
            if fim_rate > 0.0 and _use_fim(count, fim_rate):
                ids, mask, denoise_mask = encode_fim(text, tokenizer, max_seq_len)
                mode = "fim"
            else:
                ids, mask = tokenizer.encode(text, max_seq_len)
                denoise_mask = [
                    int(
                        attention
                        and token_id
                        not in {
                            tokenizer.pad_token_id,
                            tokenizer.bos_token_id,
                            tokenizer.eos_token_id,
                        }
                    )
                    for token_id, attention in zip(ids, mask)
                ]
                mode = "standard"
            record = {
                "input_ids": ids,
                "attention_mask": mask,
                "denoise_mask": denoise_mask,
                "mode": mode,
                "chars": len(text),
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    return count


def _use_fim(index: int, fim_rate: float) -> bool:
    if fim_rate >= 1.0:
        return True
    if fim_rate <= 0.0:
        return False
    period = max(1, round(1.0 / fim_rate))
    return index % period == 0


def encode_fim(
    text: str,
    tokenizer: CodeTokenizer,
    max_seq_len: int,
) -> tuple[list[int], list[int], list[int]]:
    """Encode text as prefix/suffix-conditioned fill-in-the-middle.

    Format:
    <bos><fim_prefix>prefix<fim_suffix>suffix<fim_middle>middle<eos>

    `denoise_mask` marks only middle payload tokens as training targets.
    """

    if max_seq_len < 8:
        raise ValueError("max_seq_len is too small for FIM formatting")

    first = max(1, len(text) // 3)
    second = max(first + 1, (2 * len(text)) // 3)
    prefix = tokenizer.encode_payload(text[:first])
    middle = tokenizer.encode_payload(text[first:second])
    suffix = tokenizer.encode_payload(text[second:])

    special_budget = 5
    payload_budget = max_seq_len - special_budget
    prefix_budget = max(1, payload_budget // 4)
    suffix_budget = max(1, payload_budget // 4)
    middle_budget = max(1, payload_budget - prefix_budget - suffix_budget)

    prefix = prefix[:prefix_budget]
    suffix = suffix[:suffix_budget]
    middle = middle[:middle_budget]

    ids = [
        tokenizer.bos_token_id,
        tokenizer.fim_prefix_token_id,
        *prefix,
        tokenizer.fim_suffix_token_id,
        *suffix,
        tokenizer.fim_middle_token_id,
        *middle,
        tokenizer.eos_token_id,
    ]
    denoise_mask = [0] * len(ids)
    middle_start = 1 + 1 + len(prefix) + 1 + len(suffix) + 1
    for pos in range(middle_start, middle_start + len(middle)):
        denoise_mask[pos] = 1

    attention_mask = [1] * len(ids)
    pad = max_seq_len - len(ids)
    if pad > 0:
        ids.extend([tokenizer.pad_token_id] * pad)
        attention_mask.extend([0] * pad)
        denoise_mask.extend([0] * pad)
    else:
        ids = ids[:max_seq_len]
        attention_mask = attention_mask[:max_seq_len]
        denoise_mask = denoise_mask[:max_seq_len]

    return ids, attention_mask, denoise_mask


def read_token_manifest(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
