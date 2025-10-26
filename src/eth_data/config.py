"""Configuration helpers for the Ethereum data ingestion project."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import os

try:  
    import tomllib  
except ModuleNotFoundError:  
    import tomli as tomllib  


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "settings.toml"
DEFAULT_EXAMPLE_CONFIG_PATH = REPO_ROOT / "configs" / "settings.example.toml"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "ethereum_data.db"


@dataclass
class FetchSettings:
    """Settings required to download blocks and persist them locally."""

    start_block: int
    end_block: int
    rpc_url: str
    headers: Dict[str, str]
    db_file: Path = DEFAULT_DB_PATH

    def with_overrides(
        self,
        *,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
        rpc_url: Optional[str] = None,
        db_file: Optional[Path] = None,
    ) -> "FetchSettings":
        """Return a copy of the settings with any provided overrides."""

        return replace(
            self,
            start_block=start_block if start_block is not None else self.start_block,
            end_block=end_block if end_block is not None else self.end_block,
            rpc_url=rpc_url if rpc_url is not None else self.rpc_url,
            db_file=db_file if db_file is not None else self.db_file,
        )


@dataclass
class EtherscanSettings:
    """Settings required to interact with the Etherscan block lookup API."""

    api_key: str
    target_date: date
    base_url: str = "https://api.etherscan.io/v2/api"

    @property
    def start_of_day_timestamp(self) -> int:
        return int(datetime.combine(self.target_date, datetime.min.time()).timestamp())

    @property
    def end_of_day_timestamp(self) -> int:
        return int(datetime.combine(self.target_date, datetime.max.time()).timestamp())


def _load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _resolve_config_path(config_path: Optional[Path]) -> Optional[Path]:
    if config_path and config_path.exists():
        return config_path

    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH

    if DEFAULT_EXAMPLE_CONFIG_PATH.exists():
        return DEFAULT_EXAMPLE_CONFIG_PATH

    return None


def load_fetch_settings(config_path: Optional[Path] = None) -> FetchSettings:
    """Load fetch settings from a TOML file or environment variables."""

    resolved_path = _resolve_config_path(config_path)
    data: Dict[str, Any] = {}

    if resolved_path:
        data = _load_toml(resolved_path)

    fetch_data = data.get("fetch", {}) if data else {}
    headers_data = fetch_data.get("headers", {})

    start_block = int(
        os.getenv("FETCH_START_BLOCK", fetch_data.get("start_block", 0))
    )
    end_block = int(os.getenv("FETCH_END_BLOCK", fetch_data.get("end_block", 0)))
    rpc_url = os.getenv("FETCH_RPC_URL", fetch_data.get("rpc_url", ""))
    db_file_value = os.getenv("FETCH_DB_FILE", fetch_data.get("db_file", DEFAULT_DB_PATH))
    db_file = Path(db_file_value)
    if not db_file.is_absolute():
        db_file = (REPO_ROOT / db_file).resolve()

    headers = {str(k): str(v) for k, v in headers_data.items()} or {
        "accept": "application/json",
        "content-type": "application/json",
    }

    return FetchSettings(
        start_block=start_block,
        end_block=end_block,
        rpc_url=rpc_url,
        headers=headers,
        db_file=db_file,
    )


def load_etherscan_settings(config_path: Optional[Path] = None) -> EtherscanSettings:
    """Load Etherscan settings from a TOML file or environment variables."""

    resolved_path = _resolve_config_path(config_path)
    data: Dict[str, Any] = {}

    if resolved_path:
        data = _load_toml(resolved_path)

    etherscan_data = data.get("etherscan", {}) if data else {}

    api_key = os.getenv("ETHERSCAN_API_KEY", etherscan_data.get("api_key", ""))
    target_date_str = os.getenv("ETHERSCAN_TARGET_DATE", etherscan_data.get("target_date", ""))

    if target_date_str:
        try:
            target_date_value = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        except ValueError as exc:  
            raise ValueError(
                "ETHERSCAN_TARGET_DATE must be in YYYY-MM-DD format"
            ) from exc
    else:
        config_target_date = etherscan_data.get("target_date")
        if config_target_date:
            try:
                target_date_value = datetime.strptime(
                    str(config_target_date), "%Y-%m-%d"
                ).date()
            except ValueError as exc:
                raise ValueError(
                    "etherscan.target_date must be in YYYY-MM-DD format"
                ) from exc
        else:
            target_date_value = date.today()

    base_url = os.getenv(
        "ETHERSCAN_BASE_URL",
        etherscan_data.get("base_url", "https://api.etherscan.io/v2/api"),
    )

    return EtherscanSettings(
        api_key=api_key,
        target_date=target_date_value,
        base_url=base_url,
    )
