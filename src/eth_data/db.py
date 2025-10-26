"""Database helpers for persisting Ethereum blocks and transactions."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable


BLOCKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS Blocks (
    blockNumber INTEGER PRIMARY KEY,
    timestamp INTEGER,
    hash TEXT NOT NULL,
    miner TEXT,
    gasUsed INTEGER,
    gasLimit INTEGER,
    transactionCount INTEGER
)
"""

TRANSACTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS Transactions (
    txHash TEXT PRIMARY KEY,
    blockNumber INTEGER,
    fromAddress TEXT,
    toAddress TEXT,
    value_wei TEXT,
    gasPrice TEXT,
    gas TEXT
)
"""


class DatabaseError(RuntimeError):
    """Wrap unexpected database errors."""


def setup_database(db_path: Path) -> sqlite3.Connection:
    """Create the SQLite database if needed and return a live connection."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=20)
    cursor = conn.cursor()
    cursor.execute(BLOCKS_TABLE_SQL)
    cursor.execute(TRANSACTIONS_TABLE_SQL)
    conn.commit()
    return conn


def insert_block(cursor: sqlite3.Cursor, block_data: dict[str, Any]) -> None:
    cursor.execute(
        """
        INSERT OR IGNORE INTO Blocks (blockNumber, timestamp, hash, miner, gasUsed, gasLimit, transactionCount)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(block_data["number"], 16),
            int(block_data["timestamp"], 16),
            block_data["hash"],
            block_data["miner"],
            int(block_data["gasUsed"], 16),
            int(block_data["gasLimit"], 16),
            len(block_data.get("transactions", [])),
        ),
    )


def insert_transactions(cursor: sqlite3.Cursor, transactions: Iterable[dict[str, Any]]) -> None:
    payload = []
    for tx in transactions:
        to_address = tx.get("to") or "CONTRACT_CREATION"
        payload.append(
            (
                tx["hash"],
                int(tx["blockNumber"], 16),
                tx["from"],
                to_address,
                tx.get("value", "0x0"),
                tx.get("gasPrice", "0x0"),
                tx.get("gas", "0x0"),
            )
        )

    cursor.executemany(
        """
        INSERT OR IGNORE INTO Transactions (txHash, blockNumber, fromAddress, toAddress, value_wei, gasPrice, gas)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )


def process_block(conn: sqlite3.Connection, block_data: dict[str, Any]) -> bool:
    """Persist block and transaction data within a single transaction."""

    cursor = conn.cursor()
    try:
        insert_block(cursor, block_data)
        insert_transactions(cursor, block_data.get("transactions", []))
        conn.commit()
        return True
    except sqlite3.DatabaseError as exc:  # pragma: no cover - requires database failure
        conn.rollback()
        raise DatabaseError(f"Failed to persist block {int(block_data['number'], 16)}") from exc

