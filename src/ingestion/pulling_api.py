from __future__ import annotations

from IPython.display import display
from pathlib import Path
from typing import Any
import logging
import time

import pandas as pd
import requests


# ---------------------------
# SIMPLE LOGGING (SET ONCE)
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------
# CONFIG
# ---------------------------
BASE_URL = "https://www.fema.gov/api/open"

ENDPOINTS = {
    "declarations": f"{BASE_URL}/v2/DisasterDeclarationsSummaries",
    "public_assistance": f"{BASE_URL}/v2/PublicAssistanceFundedProjectsDetails",
    "disaster_summaries": f"{BASE_URL}/v1/FemaWebDisasterSummaries",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "raw" / "_checkpoints"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "DisasterRecoveryCostPrediction/1.0",
}


# ---------------------------
# HELPERS
# ---------------------------
def _extract_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key, value in payload.items():
        if key not in {"metadata", "count"} and isinstance(value, list):
            return value
    return []


def _safe_request(url: str, params: dict, retries: int = 8, timeout: int = 120):
    last_error: requests.RequestException | None = None

    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers=REQUEST_HEADERS,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else 0
            # On 503/429 honour Retry-After header if present, else back off hard
            if status in (429, 503):
                retry_after = int(e.response.headers.get("Retry-After", 0)) if e.response is not None else 0
                wait = max(retry_after, min(30 * (2 ** attempt), 300))
            else:
                wait = min(5 * (2 ** attempt), 120)
            logger.warning(
                "Attempt %s/%s failed (HTTP %s) for %s ($skip=%s) — retrying in %ss",
                attempt + 1, retries, status,
                url, params.get("$skip", 0), wait,
            )
            time.sleep(wait)
        except requests.RequestException as e:
            last_error = e
            wait = min(5 * (2 ** attempt), 120)
            logger.warning(
                "Attempt %s/%s failed for %s ($skip=%s): %s — retrying in %ss",
                attempt + 1, retries, url, params.get("$skip", 0), e, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Request failed for {url} with params {params}: {last_error}"
    ) from last_error


# ---------------------------
# CHECKPOINT HELPERS
# ---------------------------
def _checkpoint_path(name: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{name}.parquet"


def _load_checkpoint(name: str) -> tuple[pd.DataFrame, int]:
    """Return (records_so_far_df, next_skip). Returns empty df and 0 if none."""
    path = _checkpoint_path(name)
    if path.exists():
        df = pd.read_parquet(path)
        skip = len(df)
        logger.info("Resuming %s from checkpoint at skip=%s (%s rows)", name, skip, len(df))
        return df, skip
    return pd.DataFrame(), 0


def _save_checkpoint(name: str, df: pd.DataFrame) -> None:
    _checkpoint_path(name).write_bytes(
        df.to_parquet(index=False)
    )


def _clear_checkpoint(name: str) -> None:
    path = _checkpoint_path(name)
    if path.exists():
        path.unlink()


# ---------------------------
# DATA FETCHING
# ---------------------------
def fetch_paginated_data(
    endpoint_url: str,
    name: str = "unnamed",
    fields: list[str] | None = None,
    max_records: int | None = None,
    page_size: int = 250,
) -> pd.DataFrame:

    params = {"$format": "json", "$top": page_size}
    if fields:
        params["$select"] = ",".join(fields)

    existing_df, skip = _load_checkpoint(name)
    records = existing_df.to_dict("records") if not existing_df.empty else []

    while True:
        time.sleep(5)  # conservative inter-page delay

        payload = _safe_request(endpoint_url, {**params, "$skip": skip})
        batch = _extract_records(payload)

        if not batch:
            break

        records.extend(batch)
        skip += page_size
        logger.info("Fetched %s records total for %s", len(records), name)

        # Save checkpoint after every page so a 503 mid-run loses nothing
        _save_checkpoint(name, pd.DataFrame(records))

        if max_records and len(records) >= max_records:
            records = records[:max_records]
            break

    _clear_checkpoint(name)
    return pd.DataFrame(records)


# ---------------------------
# SAVE FUNCTION
# ---------------------------
def save_dataframe(df: pd.DataFrame, filename: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{filename}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows → {path}")


# ---------------------------
# FULL INGESTION
# ---------------------------
def run_ingestion():
    logger.info("Starting full data ingestion")
    failures = []

    for name, url in ENDPOINTS.items():
        logger.info(f"Fetching {name}")

        try:
            df = fetch_paginated_data(url, name=name)
            save_dataframe(df, name)
        except RuntimeError as exc:
            logger.error("Failed to fetch %s: %s", name, exc)
            failures.append(name)

    if failures:
        raise RuntimeError(
            f"Data ingestion completed with failures for: {', '.join(failures)}"
        )

    logger.info("Data ingestion complete")


# ===========================
# CONTROL PANEL
# ===========================
MODE = "fetch_all"   # "list_columns", "fetch", "fetch_all"
ENDPOINT_NAME = "public_assistance"
FIELDS = None
MAX_RECORDS = None


# ===========================
# RUN
# ===========================
if MODE == "fetch_all":
    run_ingestion()

elif MODE == "fetch":
    start = time.time()

    df = fetch_paginated_data(
        ENDPOINTS[ENDPOINT_NAME],
        name=ENDPOINT_NAME,
        fields=FIELDS,
        max_records=MAX_RECORDS,
    )

    display(df.head())
    save_dataframe(df, ENDPOINT_NAME)

    logger.info(f"Completed in {time.time() - start:.2f} seconds")

else:
    print("Invalid MODE")