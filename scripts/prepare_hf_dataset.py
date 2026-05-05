#!/usr/bin/env python3
"""Download a small Hugging Face code slice and tokenize it into JSONL manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_diffusion.data.manifest import extract_text, write_token_manifest
from nano_diffusion.data.tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="codeparrot/codeparrot-clean", help="Hugging Face dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--text-field", default=None, help="Optional text/code field name")
    parser.add_argument("--max-samples", type=int, default=512, help="Total usable examples to write")
    parser.add_argument("--val-samples", type=int, default=64, help="Examples reserved for validation")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Tokenized sequence length")
    parser.add_argument("--min-chars", type=int, default=32, help="Minimum raw text length")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def limited_rows(dataset_iter, max_rows: int, text_field: str | None, min_chars: int):
    accepted = 0
    for row in dataset_iter:
        text = extract_text(row, text_field)
        if text is None or len(text) < min_chars:
            continue
        yield row
        accepted += 1
        if accepted >= max_rows:
            break


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets to download from Hugging Face: pip install datasets") from exc

    total = args.max_samples + args.val_samples
    raw = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    rows = list(limited_rows(iter(raw), total, args.text_field, args.min_chars))
    if len(rows) < total:
        print(f"[prepare] warning: requested {total} rows, found {len(rows)} usable rows")

    tokenizer = ByteTokenizer()
    output_dir = Path(args.output_dir)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    val_rows = rows[: args.val_samples]
    train_rows = rows[args.val_samples :]

    train_count = write_token_manifest(
        train_rows,
        train_path,
        tokenizer,
        args.max_seq_len,
        text_field=args.text_field,
        min_chars=args.min_chars,
    )
    val_count = write_token_manifest(
        val_rows,
        val_path,
        tokenizer,
        args.max_seq_len,
        text_field=args.text_field,
        min_chars=args.min_chars,
    )

    print(f"[prepare] dataset={args.dataset} split={args.split}")
    print(f"[prepare] wrote train={train_count} -> {train_path}")
    print(f"[prepare] wrote val={val_count} -> {val_path}")


if __name__ == "__main__":
    main()
