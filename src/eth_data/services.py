"""External service integrations (e.g., Etherscan)."""
from __future__ import annotations

import time
from typing import Optional

import requests

from .config import EtherscanSettings


def get_block_number_by_time(
    timestamp: int,
    *,
    closest: str = "after",
    settings: EtherscanSettings,
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    """Retrieve the closest block number for a timestamp using Etherscan."""

    params = {
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": timestamp,
        "closest": closest,
        "chainid": "1",
        "apikey": settings.api_key,
    }

    http = session or requests.Session()
    response = http.get(settings.base_url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()
    status = data.get("status")
    if status == "1":
        return data.get("result")

    message = data.get("message", "Unknown error")
    print(f"API error for closest={closest}: {message} (result: {data.get('result')})")
    return None


def find_daily_block_range(settings: EtherscanSettings) -> tuple[Optional[str], Optional[str]]:
    """Find the block numbers wrapping the configured target date."""

    start_block = get_block_number_by_time(
        settings.start_of_day_timestamp, closest="after", settings=settings
    )
    time.sleep(1)
    end_block = get_block_number_by_time(
        settings.end_of_day_timestamp, closest="before", settings=settings
    )
    return start_block, end_block

