"""CLI entry-point for downloading Ethereum block data into SQLite."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eth_data import load_fetch_settings, run_block_range  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a TOML configuration file (defaults to configs/settings.toml).",
    )
    parser.add_argument("--start-block", type=int, help="Override start block number.")
    parser.add_argument("--end-block", type=int, help="Override end block number.")
    parser.add_argument(
        "--db-file",
        type=Path,
        help="Override the SQLite database file destination.",
    )
    parser.add_argument(
        "--rpc-url",
        help="Override the RPC URL used for fetching block data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_fetch_settings(args.config)

    overrides = {}
    if args.start_block is not None:
        overrides["start_block"] = args.start_block
    if args.end_block is not None:
        overrides["end_block"] = args.end_block
    if args.rpc_url is not None:
        overrides["rpc_url"] = args.rpc_url
    if args.db_file is not None:
        overrides["db_file"] = args.db_file

    if overrides:
        settings = settings.with_overrides(**overrides)

    run_block_range(settings)


if __name__ == "__main__":
    main()
