"""
etl_seattle.py
────────────────────────────────────────────────
Cleans the raw Seattle crime CSV and outputs a
standardized file ready to load into SQL.

Input:  Data/raw/seattle_data.csv
Output: Data/clean/seattle_data_clean.csv

Run:
    python etl_seattle.py
────────────────────────────────────────────────
Cleaning stages applied:
  1.  Parse and validate offense date/time
  2.  Drop rows with null or out-of-range dates
  3.  Deduplicate on case_id
  4.  Normalize crime type → unified category
  5.  Parse and bounds-check coordinates
  6.  Drop rows outside Seattle's geographic boundary
  7.  Clean optional string fields (fix "nan" strings)
  8.  Normalize whitespace in text fields
  9.  Clean numeric-id field (beat)
  10. Normalize shooting_type field
  11. Drop rows missing any required field
  12. Report unmapped crime types
────────────────────────────────────────────────
"""

import pandas as pd
import os
from crime_categories import CRIME_CATEGORY_MAP

os.makedirs("Data/clean", exist_ok=True)

INPUT  = "Data/raw/seattle_data.csv"
OUTPUT = "Data/clean/seattle_data_clean.csv"

# Seattle's valid geographic bounding box
LAT_MIN, LAT_MAX = 47.30,  47.80
LON_MIN, LON_MAX = -122.55, -122.10

# Earliest plausible record year for this dataset
YEAR_MIN = 2008
YEAR_MAX = 2025   # update as needed


# ── Helpers ─────────────────────────────────────────────────────────

def safe_str(series: pd.Series) -> pd.Series:
    """
    Convert a column to string while preserving real nulls.
    Using plain .astype(str) turns NaN into the literal string "nan",
    which then silently contaminates text fields and filters.
    This helper keeps NaN as NaN.
    """
    return series.where(series.notna(), other=pd.NA).astype(str).where(
        lambda s: s != "nan", other=pd.NA
    ).str.strip()


def normalize_whitespace(series: pd.Series) -> pd.Series:
    """Collapse any run of whitespace characters to a single space."""
    return series.str.replace(r"\s+", " ", regex=True).str.strip()


def clean_numeric_id(series: pd.Series) -> pd.Series:
    """
    Convert a numeric ID column to nullable integer,
    removing trailing '.0' that appears when floats are used to store
    integers with NaN values.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


# ── Main cleaning function ───────────────────────────────────────────

def clean_seattle(input_path: str, output_path: str):
    print("Loading Seattle data...")
    df = pd.read_csv(input_path)
    print(f"  Raw rows: {len(df):,}")

    out = pd.DataFrame()

    # ── Stage 1: Parse offense date/time ────────────────────────────
    # errors='coerce' turns unparseable strings into NaT instead of
    # raising an exception, so the pipeline does not crash on bad rows.
    dates = pd.to_datetime(df["Offense Date"], errors="coerce")

    out["offense_datetime"] = dates.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["year"]             = dates.dt.year.astype("Int64")
    out["month"]            = dates.dt.month.astype("Int64")
    out["day"]              = dates.dt.day.astype("Int64")
    out["hour"]             = dates.dt.hour.astype("Int64")
    out["day_of_week"]      = dates.dt.day_name()

    # ── Stage 2: Drop rows with null or out-of-range dates ──────────
    # Null dates break all time-series analysis.
    # Out-of-range years indicate data entry errors that survived coercion.
    before = len(out)
    out["_date_tmp"] = dates
    out = out[
        out["_date_tmp"].notna() &
        out["year"].between(YEAR_MIN, YEAR_MAX)
    ]
    out = out.drop(columns=["_date_tmp"])
    print(f"  Stage 2 — dropped {before - len(out):,} rows (null or out-of-range date)")

    # ── Stage 3: Deduplicate on case_id ─────────────────────────────
    # A single incident can generate multiple rows if the report was
    # amended or if multiple offences were recorded under one report number.
    # Keep the last occurrence as the most up-to-date version.
    out["case_id"] = df.loc[out.index, "Report Number"].astype(str).str.strip()
    before = len(out)
    out = out.drop_duplicates(subset=["case_id"], keep="last")
    print(f"  Stage 3 — dropped {before - len(out):,} duplicate case_id rows")

    # ── Stage 4: Crime type normalization ───────────────────────────
    # Seattle uses NIBRS Offense Code Descriptions, which differ from
    # Chicago's IUCR-based Primary Type labels. Both are mapped to the
    # shared unified taxonomy defined in crime_categories.py.
    raw = df.loc[out.index, "NIBRS Offense Code Description"].fillna("").str.upper().str.strip()
    out["crime_type_raw"] = raw
    out["crime_category"] = raw.map(CRIME_CATEGORY_MAP).fillna("Other")

    # ── Stage 5: Parse and round coordinates ────────────────────────
    out["latitude"]  = pd.to_numeric(df.loc[out.index, "Latitude"],  errors="coerce").round(6)
    out["longitude"] = pd.to_numeric(df.loc[out.index, "Longitude"], errors="coerce").round(6)

    # ── Stage 6: Drop rows outside Seattle's geographic boundary ────
    # Coordinates of (0, 0) or values outside the city boundary indicate
    # a geocoding failure or data entry error and must be excluded before
    # any spatial or choropleth analysis is run.
    before = len(out)
    out = out[
        out["latitude"].notna() &
        out["longitude"].notna() &
        out["latitude"].between(LAT_MIN, LAT_MAX) &
        out["longitude"].between(LON_MIN, LON_MAX)
    ]
    print(f"  Stage 6 — dropped {before - len(out):,} rows (null or out-of-bounds coordinates)")

    # ── Stage 7: Clean optional string fields ───────────────────────
    # safe_str() prevents NaN from becoming the string "nan",
    # which is a silent bug that corrupts GROUP BY and WHERE queries.
    out["block_address"] = safe_str(df.loc[out.index, "Block Address"])
    out["beat"]          = safe_str(df.loc[out.index, "Beat"])
    out["precinct"]      = safe_str(df.loc[out.index, "Precinct"])
    out["sector"]        = safe_str(df.loc[out.index, "Sector"])
    out["neighborhood"]  = safe_str(df.loc[out.index, "Neighborhood"])
    out["nibrs_group"]   = safe_str(df.loc[out.index, "NIBRS Group AB"])
    out["nibrs_code"]    = safe_str(df.loc[out.index, "NIBRS_offense_code"])
    out["shooting_type"] = safe_str(df.loc[out.index, "Shooting Type Group"])

    # ── Stage 8: Normalize whitespace in text columns ───────────────
    for col in ["block_address", "neighborhood"]:
        out[col] = normalize_whitespace(out[col])

    # ── Stage 9: Normalize shooting_type field ──────────────────────
    # Standardize common variants so that GROUP BY on shooting_type
    # produces clean, consistent groups rather than fragmented counts.
    shooting_map = {
        "NON-SHOOTING":        "Non-Shooting",
        "HANDGUN":             "Handgun",
        "RIFLE":               "Rifle",
        "SHOTGUN":             "Shotgun",
        "OTHER FIREARM":       "Other Firearm",
        "UNKNOWN FIREARM":     "Unknown Firearm",
        "NOT APPLICABLE":      pd.NA,   # not a shooting event — set to null
    }
    out["shooting_type"] = (
        out["shooting_type"]
        .str.upper()
        .str.strip()
        .map(shooting_map)
    )

    # ── Stage 10: Drop rows missing any required field ───────────────
    required = ["case_id", "offense_datetime", "crime_category", "latitude", "longitude"]
    before = len(out)
    out = out.dropna(subset=required)
    print(f"  Stage 10 — dropped {before - len(out):,} rows (null in required field after transforms)")

    # ── Stage 11: Report unmapped crime types ────────────────────────
    unmapped = out.loc[out["crime_category"] == "Other", "crime_type_raw"].unique()
    if len(unmapped) > 0:
        print(f"\n  Note: {len(unmapped)} unmapped crime type(s) labeled 'Other'.")
        print("  Add them to crime_categories.py if needed:")
        for u in sorted(unmapped):
            print(f'    "{u}"')

    # ── Final column order ───────────────────────────────────────────
    col_order = [
        "case_id", "offense_datetime", "year", "month", "day", "hour", "day_of_week",
        "crime_category", "crime_type_raw",
        "block_address", "neighborhood",
        "beat", "precinct", "sector",
        "latitude", "longitude",
        "nibrs_group", "nibrs_code",
        "shooting_type",
    ]
    out = out[[c for c in col_order if c in out.columns]]

    out.to_csv(output_path, index=False)
    print(f"\n  Clean rows: {len(out):,}")
    print(f"  Saved  ->  {output_path}")


if __name__ == "__main__":
    print("=" * 45)
    print("SEATTLE ETL")
    print("=" * 45)
    clean_seattle(INPUT, OUTPUT)
    print("\nDone. Load Data/clean/seattle_data_clean.csv")
    print("into SQL as a table named: seattle_crime")