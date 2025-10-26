# Ethereum Block Data Ingestion Toolkit

This project downloads Ethereum blocks for a specific day, stores full transaction
payloads in SQLite, and helps you discover the block range for a day using the
Etherscan API.

## Project Layout

```
FinalProject/
├── configs/
│   └── settings.example.toml   # Sample configuration (copy to settings.toml)
├── data/
│   └── .gitignore              # Keeps generated SQLite files out of git
├── scripts/
│   ├── fetch_data.py           # CLI wrapper to ingest blocks into SQLite
│   └── find_blocks.py          # CLI wrapper to locate the daily block range
├── src/
│   └── eth_data/
│       ├── __init__.py
│       ├── config.py           # Settings loaders (TOML + env overrides)
│       ├── db.py               # SQLite helpers
│       ├── pipelines.py        # High-level ingestion workflow
│       ├── rpc.py              # Robust Ethereum RPC client
│       └── services.py         # Etherscan integrations
├── tests/                      # (empty) reserved for automated tests
├── README.md
└── Requirements.txt
```

Python modules live under `src/eth_data`, while lightweight CLI entry-points reside in
`scripts/`. Generated data (such as `ethereum_data.db`) should be written into
`data/`.

## Getting Started

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```bash
   pip install -r Requirements.txt
   ```
3. **Configure credentials and block ranges:**
   - Copy `configs/settings.example.toml` to `configs/settings.toml`.
   - Update the file with your Etherscan API key, preferred RPC URL, and the
     desired target date. The `fetch.start_block` and `fetch.end_block` values
     can be left as `0` initially—they will be filled in after discovering the
     daily range.

## Finding the Daily Block Range

Use the `find_blocks.py` CLI to query Etherscan for the block numbers that wrap a
particular day:

```bash
python scripts/find_blocks.py --date 2024-10-01 --api-key YOUR_ETHERSCAN_KEY
```

If you omit `--date` or `--api-key`, the values from `configs/settings.toml` (or
environment variables) are used. The script prints the start and end block numbers
that you should then copy into the fetch configuration.

## Downloading Blocks and Transactions

Once the block range is known, update `fetch.start_block` and `fetch.end_block` in
`configs/settings.toml`, then run the ingestion pipeline:

```bash
python scripts/fetch_data.py
```

The script downloads each block, writes the results into the configured SQLite file
(default: `data/ethereum_data.db`), and reports progress. You can override most
settings directly from the command line:

```bash
python scripts/fetch_data.py --start-block 19998542 --end-block 20005741 --db-file data/custom.db
```

Expect occasional `429` (rate limit) responses from RPC providers; the script handles
these by backing off and retrying automatically.

## Environment Variable Overrides

Both CLIs respect environment variables, allowing non-interactive usage:

- `FETCH_START_BLOCK`, `FETCH_END_BLOCK`, `FETCH_RPC_URL`, `FETCH_DB_FILE`
- `ETHERSCAN_API_KEY`, `ETHERSCAN_TARGET_DATE`, `ETHERSCAN_BASE_URL`

## Next Steps

- Add automated tests under `tests/` to ensure long-term maintainability.
- Consider packaging the project via `pyproject.toml` for easier distribution.
- Extend `scripts/` with analytics or reporting entry-points that reuse the
  shared modules in `src/eth_data`.
