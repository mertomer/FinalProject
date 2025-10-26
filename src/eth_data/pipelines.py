"""High-level data ingestion pipelines."""
from __future__ import annotations

import sys
import time
from typing import Iterable

from .config import FetchSettings
from .db import DatabaseError, process_block, setup_database
from .rpc import get_block_data_robust


def iter_block_numbers(settings: FetchSettings) -> Iterable[int]:
    return range(settings.start_block, settings.end_block + 1)


def run_block_range(settings: FetchSettings) -> None:
    """Pull the configured block range and persist it to SQLite."""

    if settings.start_block <= 0 or settings.end_block <= 0:
        print("ERROR: start and end blocks must be greater than zero.")
        sys.exit(1)

    if settings.start_block > settings.end_block:
        print("ERROR: start_block cannot be greater than end_block.")
        sys.exit(1)

    conn = setup_database(settings.db_file)
    total_blocks = settings.end_block - settings.start_block + 1

    print(
        f"Starting ingestion: fetching blocks {settings.start_block} through {settings.end_block}"
    )
    print("This process may take a long time depending on network throughput and API limits.")
    print("-" * 40)

    start_time = time.time()

    try:
        for index, block_number in enumerate(iter_block_numbers(settings), start=1):
            time.sleep(1 / 30)
            print(f"[{index}/{total_blocks}] Fetching block {block_number}...")
            block_data = get_block_data_robust(block_number, settings)
            if not block_data:
                print(f"[{index}/{total_blocks}] Block {block_number} skipped due to errors.")
                continue

            try:
                process_block(conn, block_data)
                tx_count = len(block_data.get("transactions", []))
                print(
                    f"[{index}/{total_blocks}] Block {block_number} saved successfully ({tx_count} transactions)."
                )
            except DatabaseError as exc:
                print(str(exc))
    finally:
        conn.close()

    total_time = time.time() - start_time
    print("-" * 40)
    print("Ingestion finished.")
    print(f"Processed {total_blocks} blocks in {total_time:.2f} seconds.")
    print(f"SQLite file: {settings.db_file}")
