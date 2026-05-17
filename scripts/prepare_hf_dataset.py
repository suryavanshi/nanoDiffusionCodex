#!/usr/bin/env python3
"""Download a small Hugging Face code slice and tokenize it into JSONL manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_diffusion.data.manifest import extract_text, write_token_manifest
from nano_diffusion.data.tokenizer import ByteTokenizer, train_bpe_tokenizer


DATASET_PRESETS = {
    "codeparrot-clean": {
        "dataset": "codeparrot/codeparrot-clean",
        "config": None,
        "split": "train",
        "text_field": "content",
    },
    "codesearchnet-python": {
        "dataset": "code_search_net",
        "config": "python",
        "split": "train",
        "text_field": "whole_func_string",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=sorted(DATASET_PRESETS),
        default="codeparrot-clean",
        help="Known Hugging Face dataset preset. Use explicit flags to override.",
    )
    parser.add_argument("--dataset", default=None, help="Hugging Face dataset name")
    parser.add_argument("--dataset-config", default=None, help="Optional Hugging Face dataset config/subset")
    parser.add_argument("--split", default=None, help="Dataset split")
    parser.add_argument("--text-field", default=None, help="Optional text/code field name")
    parser.add_argument("--max-samples", type=int, default=512, help="Total usable examples to write")
    parser.add_argument("--val-samples", type=int, default=64, help="Examples reserved for validation")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Tokenized sequence length")
    parser.add_argument("--min-chars", type=int, default=32, help="Minimum raw text length")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--tokenizer", choices=["byte", "bpe"], default="bpe", help="Tokenizer type")
    parser.add_argument("--vocab-size", type=int, default=16384, help="BPE vocabulary size")
    parser.add_argument("--tokenizer-path", default=None, help="Path to read/write BPE tokenizer JSON")
    parser.add_argument("--fim-rate", type=float, default=0.5, help="Fraction of samples formatted as fill-in-the-middle")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_dataset_args(args: argparse.Namespace) -> tuple[str, str | None, str, str | None]:
    preset = DATASET_PRESETS[args.preset]
    dataset = args.dataset or preset["dataset"]
    dataset_config = args.dataset_config if args.dataset_config is not None else preset["config"]
    split = args.split or preset["split"]
    text_field = args.text_field if args.text_field is not None else preset["text_field"]
    return dataset, dataset_config, split, text_field


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

    dataset, dataset_config, split, text_field = resolve_dataset_args(args)
    total = args.max_samples + args.val_samples
    load_kwargs = {"split": split, "streaming": args.streaming}
    if dataset_config:
        raw = load_dataset(dataset, dataset_config, **load_kwargs)
    else:
        raw = load_dataset(dataset, **load_kwargs)
    rows = list(limited_rows(iter(raw), total, text_field, args.min_chars))
    if len(rows) < total:
        print(f"[prepare] warning: requested {total} rows, found {len(rows)} usable rows")

    output_dir = Path(args.output_dir)
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else output_dir / "tokenizer.json"
    if args.tokenizer == "bpe":
        tokenizer = train_bpe_tokenizer(
            (extract_text(row, text_field) or "" for row in rows),
            tokenizer_path,
            vocab_size=args.vocab_size,
        )
    else:
        tokenizer = ByteTokenizer()
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    val_rows = rows[: args.val_samples]
    train_rows = rows[args.val_samples :]

    train_count = write_token_manifest(
        train_rows,
        train_path,
        tokenizer,
        args.max_seq_len,
        text_field=text_field,
        min_chars=args.min_chars,
        fim_rate=args.fim_rate,
    )
    val_count = write_token_manifest(
        val_rows,
        val_path,
        tokenizer,
        args.max_seq_len,
        text_field=text_field,
        min_chars=args.min_chars,
        fim_rate=args.fim_rate,
    )

    print(f"[prepare] preset={args.preset} dataset={dataset} config={dataset_config} split={split} text_field={text_field}")
    print(f"[prepare] tokenizer={args.tokenizer} vocab_size={tokenizer.vocab_size} path={tokenizer_path if args.tokenizer == 'bpe' else 'byte'} fim_rate={args.fim_rate}")
    print(f"[prepare] wrote train={train_count} -> {train_path}")
    print(f"[prepare] wrote val={val_count} -> {val_path}")


if __name__ == "__main__":
    main()
