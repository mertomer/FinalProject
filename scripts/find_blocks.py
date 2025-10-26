"""CLI to discover the block range for a specific day using Etherscan."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eth_data import load_etherscan_settings  # noqa: E402
from eth_data.services import find_daily_block_range  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a TOML configuration file (defaults to configs/settings.toml).",
    )
    parser.add_argument(
        "--date",
        help="Target date in YYYY-MM-DD format (overrides configuration file).",
    )
    parser.add_argument(
        "--api-key",
        help="Override the Etherscan API key from the configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_etherscan_settings(args.config)

    if args.date:
        settings = replace(
            settings, target_date=datetime.strptime(args.date, "%Y-%m-%d").date()
        )
    if args.api_key:
        settings = replace(settings, api_key=args.api_key)

    if not settings.api_key:
        raise SystemExit(
            "ERROR: An Etherscan API key is required. Update configs/settings.toml or pass --api-key."
        )

    print(f"Searching for block range on {settings.target_date}...")
    print("-" * 30)

    start_block, end_block = find_daily_block_range(settings)

    if start_block and end_block:
        total_blocks = int(end_block) - int(start_block) + 1
        print("Block numbers successfully retrieved. Copy these into your fetch config:\n")
        print(f"[OK] START_BLOCK: {start_block}")
        print(f"[OK] END_BLOCK:   {end_block}")
        print(f"\nTotal blocks in range: {total_blocks}")
    else:
        print("Failed to resolve the full block range. Check your API key and try again.")


if __name__ == "__main__":
    main()
