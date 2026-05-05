#!/usr/bin/env python3
"""Train the tiny masked-token diffusion model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/nano_diffusion/configs/default.yaml")
    args = parser.parse_args()

    from nano_diffusion.training.loop import train
    from nano_diffusion.utils.config import load_config

    config = load_config(args.config)
    result = train(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
