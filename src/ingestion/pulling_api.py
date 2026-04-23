from __future__ import annotations

from pathlib import Path
import io
import logging

import pandas as pd
import requests


# ---------------------------
# LOGGING
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------
# CONFIG
# ---------------------------
BASE_URL = "https://www.fema.gov/api/open"

# Bulk CSV URLs — one request downloads the full dataset, no pagination needed
BULK_ENDPOINTS = {
    "declarations":       f"{BASE_URL}/v2/DisasterDeclarationsSummaries.csv",
    "public_assistance":  f"{BASE_URL}/v2/PublicAssistanceFundedProjectsDetails.csv",
    "disaster_summaries": f"{BASE_URL}/v1/FemaWebDisasterSummaries.csv",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

REQUEST_HEADERS = {
    "User-Agent": "DisasterRecoveryCostPrediction/1.0",
}


# ---------------------------
# BULK DOWNLOAD
# ---------------------------
def download_bulk_csv(name: str, url: str) -> pd.DataFrame:
    """Stream the full FEMA bulk CSV in one request — no pagination, no rate limits."""
    logger.info("Downloading %s from %s", name, url)
    with requests.get(url, headers=REQUEST_HEADERS, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        chunks: list[bytes] = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=512 * 1024):  # 512 KB chunks
            chunks.append(chunk)
            downloaded += len(chunk)
            if total:
                logger.info("%s: %.1f%% (%s MB / %s MB)",
                            name, downloaded / total * 100,
                            downloaded // 1_048_576, total // 1_048_576)
            else:
                logger.info("%s: %.1f MB downloaded", name, downloaded / 1_048_576)

    df = pd.read_csv(io.BytesIO(b"".join(chunks)), low_memory=False)
    logger.info("Loaded %s rows, %s columns for %s", len(df), len(df.columns), name)
    return df


# ---------------------------
# SAVE
# ---------------------------
def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{filename}.csv"
    df.to_csv(path, index=False)
    logger.info("Saved %s rows → %s", len(df), path)


# ---------------------------
# FULL INGESTION
# ---------------------------
def run_ingestion() -> None:
    logger.info("Starting full data ingestion")
    failures = []

    for name, url in BULK_ENDPOINTS.items():
        try:
            df = download_bulk_csv(name, url)
            save_dataframe(df, name)
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", name, exc)
            failures.append(name)

    if failures:
        raise RuntimeError(f"Ingestion failed for: {', '.join(failures)}")

    logger.info("Data ingestion complete")


# ===========================
# CONTROL PANEL
# ===========================
MODE = "fetch_all"   # "fetch_all" | "fetch"
ENDPOINT_NAME = "public_assistance"


# ===========================
# RUN
# ===========================
if MODE == "fetch_all":
    run_ingestion()

elif MODE == "fetch":
    if ENDPOINT_NAME not in BULK_ENDPOINTS:
        raise ValueError(f"Unknown endpoint '{ENDPOINT_NAME}'. Choose from: {list(BULK_ENDPOINTS)}")
    df = download_bulk_csv(ENDPOINT_NAME, BULK_ENDPOINTS[ENDPOINT_NAME])
    print(df.head())
    save_dataframe(df, ENDPOINT_NAME)

else:
    print("Invalid MODE. Use 'fetch_all' or 'fetch'.")