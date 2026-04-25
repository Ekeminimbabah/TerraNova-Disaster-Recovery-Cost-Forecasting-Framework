from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── Constants ─────────────────────────────────────────────────────────────────
# Internal FEMA system columns — no predictive value
META_COLUMNS = {
    "hash", "id", "lastRefresh", "lastIAFilingDate",
    "declarationRequestNumber", "femaDeclarationString",
    "incidentId", "designatedIncidentTypes",
}

# Date columns present in each dataset
DATE_COLUMNS = {
    "declarations":      ["declarationDate", "incidentBeginDate",
                          "incidentEndDate", "disasterCloseoutDate"],
    "public_assistance": ["declarationDate"],
    "disaster_summaries":["declarationDate", "incidentBeginDate",
                          "incidentEndDate", "disasterCloseoutDate"],
}


# Load raw CSVs from data/raw/ into memory as DataFrames, with basic sanity checks.

def load_raw() -> dict[str, pd.DataFrame]:
    """Load the three raw FEMA CSV files from data/raw/."""
    datasets = {}
    for name in ("declarations", "public_assistance", "disaster_summaries"):
        path = RAW_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run src/ingestion/pulling_api.py first."
            )
        df = pd.read_csv(path, low_memory=False)
        logger.info("Loaded %-22s → %s rows, %s cols", name, len(df), len(df.columns))
        datasets[name] = df
    return datasets


# Clean individual datasets 

def _drop_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove internal FEMA system columns that carry no predictive value."""
    return df.drop(columns=[c for c in META_COLUMNS if c in df.columns])


def _parse_date_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert string date columns to timezone-naive datetime."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def clean_declarations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Disaster Declarations dataset.

    Raw data has one row per designated county. We collapse it to one row
    per disaster (disasterNumber) by aggregating key fields.
    """
    df = _drop_meta_columns(df)
    df = _parse_date_columns(df, DATE_COLUMNS["declarations"])

    # Fix boolean columns stored as 'True'/'False' strings
    bool_cols = [
        "ihProgramDeclared", "iaProgramDeclared",
        "paProgramDeclared", "hmProgramDeclared", "tribalRequest",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})

    # Aggregate: one row per disaster
    df = (
        df.sort_values("declarationDate")
        .groupby("disasterNumber", sort=False)
        .agg(
            state             = ("state",             "first"),
            declarationType   = ("declarationType",   "first"),
            declarationDate   = ("declarationDate",   "min"),
            fyDeclared        = ("fyDeclared",         "first"),
            incidentType      = ("incidentType",       "first"),
            incidentBeginDate = ("incidentBeginDate",  "min"),
            incidentEndDate   = ("incidentEndDate",    "max"),
            region            = ("region",             "first"),
            paProgramDeclared = ("paProgramDeclared",  "first"),
            iaProgramDeclared = ("iaProgramDeclared",  "first"),
            hmProgramDeclared = ("hmProgramDeclared",  "first"),
            tribalRequest     = ("tribalRequest",      "first"),
            numDesignatedAreas= ("designatedArea",     "count"),
        )
        .reset_index()
    )

    logger.info("Declarations → %s unique disasters", len(df))
    return df


def clean_public_assistance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Public Assistance Funded Projects dataset.

    Sums all cost/obligation columns per disaster to produce one row
    per disasterNumber with total financial figures.
    """
    df = _drop_meta_columns(df)
    df = _parse_date_columns(df, DATE_COLUMNS["public_assistance"])

    # Convert all monetary columns to numeric (some arrive as strings)
    money_cols = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ("amount", "obligated", "cost"))
    ]
    for col in money_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Build aggregation: sum each monetary column, count total projects
    agg_dict: dict = {col: (col, "sum") for col in money_cols if col in df.columns}
    count_col = "pwNumber" if "pwNumber" in df.columns else df.columns[0]
    agg_dict["numProjects"] = (count_col, "count")

    df = (
        df.groupby("disasterNumber", sort=False)
        .agg(**agg_dict)
        .reset_index()
    )

    logger.info("Public assistance → %s unique disasters", len(df))
    return df


def clean_disaster_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the FEMA Web Disaster Summaries dataset.

    Already one row per disaster. We deduplicate and convert monetary
    columns to numeric.
    """
    df = _drop_meta_columns(df)
    df = _parse_date_columns(df, DATE_COLUMNS["disaster_summaries"])
    df = df.drop_duplicates(subset=["disasterNumber"])

    money_cols = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ("amount", "obligated", "approved"))
    ]
    for col in money_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    logger.info("Disaster summaries → %s rows", len(df))
    return df


# ── Step 3: Merge ─────────────────────────────────────────────────────────────

def merge_datasets(
    declarations: pd.DataFrame,
    public_assistance: pd.DataFrame,
    disaster_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all three datasets on disasterNumber using LEFT joins.

    - Base table  : declarations  (one row per disaster, known at declaration time)
    - Left join 1 : disaster_summaries  (financial totals reported by FEMA)
    - Left join 2 : public_assistance   (project-level cost totals, pre-aggregated)

    LEFT join preserves every declared disaster even if cost data is not yet
    available, avoiding silent data loss from inner joins.
    """
    # Start with the declaration metadata
    merged = declarations.copy()

    # Only bring in columns from disaster_summaries that are not already in declarations
    # to avoid duplicating state, incidentType, etc.
    duplicate_prefixes = (
        "state", "declarationType", "declarationDate",
        "incidentType", "incidentBeginDate", "incidentEndDate",
        "region", "fyDeclared",
    )
    summary_cols = ["disasterNumber"] + [
        c for c in disaster_summaries.columns
        if c != "disasterNumber" and not c.startswith(duplicate_prefixes)
    ]
    merged = merged.merge(disaster_summaries[summary_cols], on="disasterNumber", how="left")
    logger.info("After joining disaster_summaries → %s rows", len(merged))

    # Add aggregated public assistance project costs per disaster
    merged = merged.merge(public_assistance, on="disasterNumber", how="left", suffixes=("", "_pa"))
    logger.info("After joining public_assistance  → %s rows", len(merged))

    return merged


# ── Step 4: Post-merge cleaning ─────────────────────────────────────────────

COST_COLUMNS = [
    "totalObligated", "totalObligatedAmountPa",
    "projectAmount", "federalShareObligated",
    "mitigationAmount", "totalObligatedAmountCatC2g",
    "totalAmountIhpApproved", "totalNumberIaApproved",
    "totalAmountHaApproved", "totalAmountOnaApproved",
    "totalObligatedAmountCatAb", "totalObligatedAmountHmgp",
]

LOAD_DATE_COLUMNS = ["iaLoadDate", "paLoadDate"]


def _post_merge_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning steps that are only meaningful after all datasets are merged.

    - Fill cost/obligation columns with 0 where no PA data exists.
    - Fill numProjects with 0.
    - Fill missing incidentEndDate from incidentBeginDate.
    - Drop administrative load-date columns with no predictive value.
    """
    # Fill cost columns with 0 (NaN means no projects / no reported cost)
    for col in COST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill project count
    if "numProjects" in df.columns:
        df["numProjects"] = df["numProjects"].fillna(0)

    # Impute missing end date from begin date (single-day events)
    if "incidentEndDate" in df.columns and "incidentBeginDate" in df.columns:
        df["incidentEndDate"] = df["incidentEndDate"].fillna(df["incidentBeginDate"])

    # Drop columns with no predictive value
    df = df.drop(columns=[c for c in LOAD_DATE_COLUMNS if c in df.columns])

    logger.info("Post-merge clean complete")
    return df


# ── Step 5: Validate ──────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply final quality filters before saving.

    - Remove rows with no disaster number.
    - Remove duplicate disasters.
    - Keep only disasters where the Public Assistance program was declared,
      since our target variable is PA recovery cost.
    """
    before = len(df)

    df = df.dropna(subset=["disasterNumber"])
    df = df.drop_duplicates(subset=["disasterNumber"])

    if "paProgramDeclared" in df.columns:
        df = df[df["paProgramDeclared"] == True].copy()  # noqa: E712

    logger.info("Validation: %s → %s rows kept", before, len(df))
    return df.reset_index(drop=True)


# ── Step 5: Save ──────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, filename: str = "merged") -> Path:
    """Save the cleaned, merged dataset to data/processed/ as Parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{filename}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    logger.info("Saved %s rows, %s cols → %s", len(df), len(df.columns), path)
    return path


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_cleaning() -> pd.DataFrame:
    """Run the full cleaning pipeline: load → clean → merge → validate → save."""
    logger.info("=" * 60)
    logger.info("Starting data cleaning pipeline")
    logger.info("=" * 60)

    raw = load_raw()

    declarations       = clean_declarations(raw["declarations"])
    public_assistance  = clean_public_assistance(raw["public_assistance"])
    disaster_summaries = clean_disaster_summaries(raw["disaster_summaries"])

    merged = merge_datasets(declarations, public_assistance, disaster_summaries)
    merged = _post_merge_clean(merged)
    merged = validate(merged)
    save(merged)

    logger.info("Cleaning complete — final shape: %s rows, %s cols", *merged.shape)
    return merged


if __name__ == "__main__":
    df = run_cleaning()
    print("\nSample rows:")
    print(df.head())
    print("\nColumns:", list(df.columns))
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\nMissing values:\n", missing)
