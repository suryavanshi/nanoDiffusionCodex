"""JSONL manifest helpers for tokenized code samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .tokenizer import ByteTokenizer


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
    tokenizer: ByteTokenizer,
    max_seq_len: int,
    text_field: str | None = None,
    min_chars: int = 32,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            text = extract_text(row, text_field)
            if text is None or len(text) < min_chars:
                continue
            ids, mask = tokenizer.encode(text, max_seq_len)
            record = {
                "input_ids": ids,
                "attention_mask": mask,
                "chars": len(text),
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_token_manifest(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
