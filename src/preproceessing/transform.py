from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR     = PROJECT_ROOT / "data" / "final"


# ── Step 1: Extract features from date columns ───────────────────────────────

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive numeric features from date columns, then drop the raw dates.

    New columns:
      - declaration_month, declaration_quarter, declaration_year
      - incident_duration_days  (how long the incident lasted)
      - days_to_declaration     (lag between incident start and federal declaration)
    """
    if "declarationDate" in df.columns:
        dt = pd.to_datetime(df["declarationDate"], errors="coerce")
        df["declaration_month"]     = dt.dt.month
        df["declaration_quarter"]   = dt.dt.quarter
        df["declaration_year"]      = dt.dt.year
        df["declaration_dayofweek"] = dt.dt.dayofweek

    if {"incidentBeginDate", "incidentEndDate"}.issubset(df.columns):
        begin = pd.to_datetime(df["incidentBeginDate"], errors="coerce")
        end   = pd.to_datetime(df["incidentEndDate"],   errors="coerce")
        df["incident_duration_days"] = (end - begin).dt.days.clip(lower=0)

    if {"incidentBeginDate", "declarationDate"}.issubset(df.columns):
        begin = pd.to_datetime(df["incidentBeginDate"], errors="coerce")
        decl  = pd.to_datetime(df["declarationDate"],   errors="coerce")
        df["days_to_declaration"] = (decl - begin).dt.days.clip(lower=0)

    # Drop raw date strings — the derived numeric columns replace them
    date_cols = [c for c in df.columns if "Date" in c or "date" in c]
    return df.drop(columns=date_cols, errors="ignore")


# ── Step 2: Encode categorical columns ────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert text categories to integers so ML models can use them.

    - incidentType, declarationType, state  →  label encoded (integer per category)
    - boolean program flags                 →  0 / 1
    """
    for col in ["incidentType", "declarationType", "state"]:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.upper()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        logger.info("Encoded %-20s → %s categories", col, len(le.classes_))

    for col in ["paProgramDeclared", "iaProgramDeclared", "hmProgramDeclared", "tribalRequest"]:
        if col in df.columns:
            df[col] = df[col].astype(float).fillna(0).astype(int)

    return df


# ── Step 3: Handle missing values ────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      - Numeric columns  → median (robust to outliers in disaster cost data)
      - Text columns     → 'UNKNOWN'
    """
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].fillna("UNKNOWN")

    return df


# ── Step 4: Drop low-value columns ────────────────────────────────────────

def drop_low_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that would not help a model generalise:
      - Constant columns (same value in every row)
      - Internal FEMA identifiers (region, fyDeclared, numDesignatedAreas)
    """
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        logger.info("Dropping constant columns: %s", constant_cols)
        df = df.drop(columns=constant_cols)

    id_cols = [
        c for c in df.columns
        if c.lower() in {"disasternumber", "region", "fydeclared", "numdesignatedareas"}
    ]
    return df.drop(columns=id_cols, errors="ignore")


# ── Pipeline entry point ───────────────────────────────────────────────────

def run_transform(input_file: str = "merged") -> pd.DataFrame:
    """Run the full transform pipeline: date features → encode → impute → drop → save."""
    path = PROCESSED_DIR / f"{input_file}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/preproceessing/clean_data.py first."
        )

    df = pd.read_csv(path, low_memory=False)
    logger.info("=" * 60)
    logger.info("Starting transform pipeline")
    logger.info("Loaded %s rows, %s cols", len(df), len(df.columns))
    logger.info("=" * 60)

    df = extract_date_features(df)
    df = encode_categoricals(df)
    df = handle_missing(df)
    df = drop_low_value_columns(df)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_DIR / "features.csv"
    df.to_csv(out_path, index=False)
    logger.info("Transform complete — final shape: %s rows, %s cols → %s", len(df), len(df.columns), out_path)

    return df


if __name__ == "__main__":
    df = run_transform()
    print("\nSample rows:")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:", list(df.columns))
