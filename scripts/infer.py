#!/usr/bin/env python3
"""Sample a short completion from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="def add(a, b):\n    ")
    parser.add_argument("--new-tokens", type=int, default=96)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from nano_diffusion.inference.sampler import sample_completion

    print(
        sample_completion(
            args.checkpoint,
            args.prompt,
            new_tokens=args.new_tokens,
            steps=args.steps,
            device_name=args.device,
        )
    )


if __name__ == "__main__":
    main()
