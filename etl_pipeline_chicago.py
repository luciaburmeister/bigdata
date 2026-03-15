"""
etl_chicago.py
────────────────────────────────────────────────
Cleans the raw Chicago crime CSV and outputs a
standardized file ready to load into SQL.

Input:  Data/raw/chicago_data.csv
Output: Data/clean/chicago_data_clean.csv

Run:
    python etl_chicago.py
────────────────────────────────────────────────
Cleaning stages applied:
  1.  Parse and validate offense date/time
  2.  Drop rows with null or out-of-range dates
  3.  Deduplicate on case_id
  4.  Normalize crime type → unified category
  5.  Parse and bounds-check coordinates
  6.  Drop rows outside Chicago's geographic boundary
  7.  Clean optional string fields (fix "nan" strings)
  8.  Normalize whitespace in text fields
  9.  Clean numeric-id fields (ward, community_area, beat, district)
  10. Zero-pad IUCR code to 4 characters
  11. Normalize boolean flags (arrest_made, domestic) → 1 / 0 / -1
  12. Drop rows missing any required field
  13. Report unmapped crime types
────────────────────────────────────────────────
"""

import pandas as pd
import os
import re
from crime_categories import CRIME_CATEGORY_MAP

os.makedirs("Data/clean", exist_ok=True)

INPUT  = "Data/raw/chicago_data.csv"
OUTPUT = "Data/clean/chicago_data_clean.csv"

# Chicago's valid geographic bounding box
# Any coordinate outside this box is likely a data error
LAT_MIN, LAT_MAX = 41.60,  42.10
LON_MIN, LON_MAX = -87.95, -87.50

# Earliest plausible record year for this dataset
YEAR_MIN = 2001
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
    Police datasets store integer IDs (beat, district, ward, community_area)
    as floats when NaNs are present (e.g., 25.0 instead of 25).
    This converts them to nullable integers, removing the trailing '.0'.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def normalize_bool_flag(series: pd.Series) -> pd.Series:
    """
    Normalize a boolean column that arrives as strings ('true'/'false')
    to integer 1 / 0. Anything that cannot be parsed becomes -1
    (unknown), rather than a silent NaN which would later be
    misread as missing data in SQL aggregations.
    """
    mapped = series.astype(str).str.lower().map({"true": 1, "false": 0})
    return mapped.fillna(-1).astype(int)


# ── Main cleaning function ───────────────────────────────────────────

def clean_chicago(input_path: str, output_path: str):
    print("Loading Chicago data...")
    df = pd.read_csv(input_path)
    print(f"  Raw rows: {len(df):,}")

    out = pd.DataFrame()

    # ── Stage 1: Parse offense date/time ────────────────────────────
    # errors='coerce' turns unparseable strings into NaT instead of
    # raising an exception, so the pipeline does not crash on bad rows.
    dates = pd.to_datetime(df["Date"], errors="coerce")

    out["offense_datetime"] = dates.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["year"]             = dates.dt.year.astype("Int64")
    out["month"]            = dates.dt.month.astype("Int64")
    out["day"]              = dates.dt.day.astype("Int64")
    out["hour"]             = dates.dt.hour.astype("Int64")
    out["day_of_week"]      = dates.dt.day_name()

    # ── Stage 2: Drop rows with null or out-of-range dates ──────────
    # Null dates break all time-series analysis.
    # Out-of-range years (e.g., 1900, 2099) are data entry errors that
    # survived coercion because they are technically valid timestamps.
    before = len(out)
    out["_date_tmp"] = dates
    out = out[
        out["_date_tmp"].notna() &
        out["year"].between(YEAR_MIN, YEAR_MAX)
    ]
    out = out.drop(columns=["_date_tmp"])
    print(f"  Stage 2 — dropped {before - len(out):,} rows (null or out-of-range date)")

    # ── Stage 3: Deduplicate on case_id ─────────────────────────────
    # Chicago's dataset is updated in place; the same incident can appear
    # multiple times if the record was amended. Keep the last occurrence,
    # which reflects the most recent version of the record.
    out["case_id"] = df.loc[out.index, "Case Number"].astype(str).str.strip()
    before = len(out)
    out = out.drop_duplicates(subset=["case_id"], keep="last")
    print(f"  Stage 3 — dropped {before - len(out):,} duplicate case_id rows")

    # ── Stage 4: Crime type normalization ───────────────────────────
    raw = df.loc[out.index, "Primary Type"].fillna("").str.upper().str.strip()
    out["crime_type_raw"] = raw
    out["crime_category"] = raw.map(CRIME_CATEGORY_MAP).fillna("Other")

    # ── Stage 5: Parse and round coordinates ────────────────────────
    out["latitude"]  = pd.to_numeric(df.loc[out.index, "Latitude"],  errors="coerce").round(6)
    out["longitude"] = pd.to_numeric(df.loc[out.index, "Longitude"], errors="coerce").round(6)

    # ── Stage 6: Drop rows outside Chicago's geographic boundary ────
    # Coordinates of (0, 0), (NaN, NaN), or values far outside the city
    # indicate a geocoding failure or data entry error.
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
    out["block_address"]        = safe_str(df.loc[out.index, "Block"])
    out["crime_description"]    = safe_str(df.loc[out.index, "Description"])
    out["location_description"] = safe_str(df.loc[out.index, "Location Description"])
    out["iucr_code"]            = safe_str(df.loc[out.index, "IUCR"])
    out["fbi_code"]             = safe_str(df.loc[out.index, "FBI Code"])

    # ── Stage 8: Normalize whitespace in text columns ───────────────
    # Multiple consecutive spaces appear in address and description fields
    # due to inconsistent data entry. Collapse them to a single space.
    for col in ["block_address", "crime_description", "location_description"]:
        out[col] = normalize_whitespace(out[col])

    # ── Stage 9: Clean numeric ID fields ────────────────────────────
    # These arrive as floats (e.g., 25.0) when NaNs are present.
    # Cast to nullable integer to remove the trailing '.0'.
    out["beat"]           = clean_numeric_id(df.loc[out.index, "Beat"])
    out["district"]       = clean_numeric_id(df.loc[out.index, "District"])
    out["ward"]           = clean_numeric_id(df.loc[out.index, "Ward"])
    out["community_area"] = clean_numeric_id(df.loc[out.index, "Community Area"])

    # ── Stage 10: Zero-pad IUCR code to 4 characters ────────────────
    # IUCR codes are always 4 characters. Codes loaded as integers or
    # stripped of leading zeros (e.g., "31A" instead of "031A") break
    # lookups against the official IUCR reference table.
    out["iucr_code"] = out["iucr_code"].str.zfill(4).where(out["iucr_code"].notna())

    # ── Stage 11: Normalize boolean flags ───────────────────────────
    # Produces 1 (true), 0 (false), or -1 (unknown/unparseable).
    out["arrest_made"] = normalize_bool_flag(df.loc[out.index, "Arrest"])
    out["domestic"]    = normalize_bool_flag(df.loc[out.index, "Domestic"])

    # ── Stage 12: Final required-field null check ────────────────────
    # By this point, date and coordinate nulls have already been dropped.
    # This is a safety check that catches any edge cases introduced by
    # earlier transformations (e.g., a join on index producing NaN).
    required = ["case_id", "offense_datetime", "crime_category", "latitude", "longitude"]
    before = len(out)
    out = out.dropna(subset=required)
    print(f"  Stage 12 — dropped {before - len(out):,} rows (null in required field after transforms)")

    # ── Stage 13: Report unmapped crime types ────────────────────────
    unmapped = out.loc[out["crime_category"] == "Other", "crime_type_raw"].unique()
    if len(unmapped) > 0:
        print(f"\n  Note: {len(unmapped)} unmapped crime type(s) labeled 'Other'.")
        print("  Add them to crime_categories.py if needed:")
        for u in sorted(unmapped):
            print(f'    "{u}"')

    # ── Final column order ───────────────────────────────────────────
    col_order = [
        "case_id", "offense_datetime", "year", "month", "day", "hour", "day_of_week",
        "crime_category", "crime_type_raw", "crime_description",
        "block_address", "location_description",
        "beat", "district", "ward", "community_area",
        "latitude", "longitude",
        "iucr_code", "fbi_code",
        "arrest_made", "domestic",
    ]
    out = out[[c for c in col_order if c in out.columns]]

    out.to_csv(output_path, index=False)
    print(f"\n  Clean rows: {len(out):,}")
    print(f"  Saved  ->  {output_path}")


if __name__ == "__main__":
    print("=" * 45)
    print("CHICAGO ETL")
    print("=" * 45)
    clean_chicago(INPUT, OUTPUT)
    print("\nDone. Load Data/clean/chicago_data_clean.csv")
    print("into SQL as a table named: chicago_crime")