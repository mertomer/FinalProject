"""RPC helpers for interacting with Ethereum nodes."""
from __future__ import annotations

import time
from typing import Any, Dict

import requests

from .config import FetchSettings


class RPCError(RuntimeError):
    """Raised when a block cannot be retrieved after several attempts."""


def get_block_data_robust(
    block_number: int,
    settings: FetchSettings,
    *,
    max_retries: int = 5,
    initial_wait_sec: float = 2.0,
) -> Dict[str, Any] | None:
    """Fetch a block with retries and exponential backoff."""

    hex_block_number = hex(block_number)
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [hex_block_number, True],
        "id": 1,
    }

    wait_time = initial_wait_sec
    for attempt in range(max_retries):
        try:
            response = requests.post(
                settings.rpc_url, json=payload, headers=settings.headers, timeout=15
            )
            response.raise_for_status()
            data = response.json()

            result = data.get("result")
            if result:
                return result

            error_msg = data.get("error", "Unknown RPC error")
            print(
                f"Block {block_number}, attempt {attempt + 1}: RPC error returned: {error_msg}"
            )
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else "unknown"
            if status_code == 429:
                print(
                    f"Block {block_number}: received HTTP 429, waiting {wait_time:.1f}s before retry"
                )
            else:
                print(f"Block {block_number}: HTTP error {status_code}: {exc}")
        except requests.RequestException as exc:
            print(f"Block {block_number}: connection error: {exc}")

        if attempt < max_retries - 1:
            time.sleep(wait_time)
            wait_time *= 2

    print(f"ERROR: block {block_number} could not be retrieved after {max_retries} attempts")
    return None

