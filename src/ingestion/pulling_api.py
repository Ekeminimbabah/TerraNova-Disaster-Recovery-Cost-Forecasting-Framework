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

RAW_DIR = Path("data/raw")


# ---------------------------
# HELPERS
# ---------------------------
def _extract_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key, value in payload.items():
        if key not in {"metadata", "count"} and isinstance(value, list):
            return value
    return []


def _safe_request(url: str, params: dict, retries: int = 3, timeout: int = 60):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    raise RuntimeError("Max retries exceeded")


# ---------------------------
# COLUMN DISCOVERY
# ---------------------------
def list_available_columns(endpoint_url: str) -> list[str]:
    payload = _safe_request(endpoint_url, {"$format": "json", "$top": 1})
    records = _extract_records(payload)
    return sorted(records[0].keys()) if records else []


# ---------------------------
# DATA FETCHING
# ---------------------------
def fetch_paginated_data(
    endpoint_url: str,
    fields: list[str] | None = None,
    max_records: int | None = None,
    page_size: int = 1000,
) -> pd.DataFrame:

    params = {"$format": "json", "$top": page_size}

    if fields:
        params["$select"] = ",".join(fields)

    skip = 0
    records = []

    while True:
        payload = _safe_request(endpoint_url, {**params, "$skip": skip})
        batch = _extract_records(payload)

        if not batch:
            break

        records.extend(batch)
        logger.info(f"Fetched {len(records)} records")

        if max_records and len(records) >= max_records:
            records = records[:max_records]
            break

        skip += page_size

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

    for name, url in ENDPOINTS.items():
        logger.info(f"Fetching {name}")
        df = fetch_paginated_data(url)
        save_dataframe(df, name)

    logger.info("Data ingestion complete")


# ===========================
# CONTROL PANEL (EDIT HERE)
# ===========================
MODE = "fetch"   # "list_columns", "fetch", "fetch_all"
ENDPOINT_NAME = "public_assistance"
FIELDS = None
MAX_RECORDS = 5000


# ===========================
# RUN
# ===========================
if MODE == "list_columns":
    for name, url in ENDPOINTS.items():
        cols = list_available_columns(url)
        print(f"\n{name} ({len(cols)} columns)")
        for col in cols:
            print(f"- {col}")

elif MODE == "fetch_all":
    run_ingestion()

elif MODE == "fetch":
    if ENDPOINT_NAME not in ENDPOINTS:
        raise ValueError("Invalid endpoint")

    start = time.time()

    df = fetch_paginated_data(
        ENDPOINTS[ENDPOINT_NAME],
        fields=FIELDS,
        max_records=MAX_RECORDS,
    )

    display(df.head())
    save_dataframe(df, ENDPOINT_NAME)

    logger.info(f"Completed in {time.time() - start:.2f} seconds")

else:
    print("Invalid MODE")