"""
spatial_join_chicago.py
────────────────────────────────────────────────
Adds a 'neighborhood' column to the cleaned
Chicago crime data by spatially joining each
crime's lat/lon against the neighborhood polygons.

Input:  Data/clean/chicago_data_clean.csv
        ../../Downloads/Neighborhoods_chicago.csv  (WKT)
Output: Data/clean/chicago_data_with_neighborhoods.csv

Run:
    python spatial_join_chicago.py
────────────────────────────────────────────────
"""

import pandas as pd
import geopandas as gpd
from shapely import wkt
import os

CRIME_IN   = "Data/clean/chicago_data_clean.csv"
NEIGH_IN   = "/Users/jjaramillo/Downloads/Neighborhoods_chicago.csv"
OUTPUT     = "Data/clean/chicago_data_with_neighborhoods.csv"
CRS        = "EPSG:4326"   # WGS-84 lat/lon

os.makedirs("Data/clean", exist_ok=True)

# ── 1. Load neighborhoods ─────────────────────────────────────────────
print("Loading neighborhood polygons...")
neigh_df = pd.read_csv(NEIGH_IN)
neigh_df["geometry"] = neigh_df["the_geom"].apply(wkt.loads)
neigh_gdf = gpd.GeoDataFrame(neigh_df[["PRI_NEIGH", "SEC_NEIGH", "geometry"]], crs=CRS)
print(f"  {len(neigh_gdf)} neighborhoods loaded")

# ── 2. Load crime data in chunks ──────────────────────────────────────
print("\nLoading crime data and running spatial join...")
CHUNK = 500_000
chunks_out = []
total_in = total_out = 0

for i, chunk in enumerate(pd.read_csv(CRIME_IN, chunksize=CHUNK)):
    total_in += len(chunk)

    # Build points GeoDataFrame from lat/lon
    crime_gdf = gpd.GeoDataFrame(
        chunk,
        geometry=gpd.points_from_xy(chunk["longitude"], chunk["latitude"]),
        crs=CRS,
    )

    # Spatial join: each point → the polygon it falls inside
    joined = gpd.sjoin(
        crime_gdf,
        neigh_gdf[["PRI_NEIGH", "geometry"]],
        how="left",
        predicate="within",
    )

    # sjoin can produce duplicates if a point touches two polygon edges;
    # keep the first match per crime row
    joined = joined[~joined.index.duplicated(keep="first")]

    # Rename and keep only the neighborhood name
    joined = joined.rename(columns={"PRI_NEIGH": "neighborhood"})

    # Drop geometry and index_right before saving
    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")

    total_out += len(joined)
    chunks_out.append(joined)

    rows_done = min(total_in, (i + 1) * CHUNK)
    print(f"  Processed {rows_done:,} rows...", end="\r")

print(f"\n  Spatial join complete — {total_out:,} rows")

# ── 3. Combine and save ───────────────────────────────────────────────
print("\nCombining chunks and saving...")
result = pd.concat(chunks_out, ignore_index=True)

# Summary
mapped   = result["neighborhood"].notna().sum()
unmapped = result["neighborhood"].isna().sum()
print(f"  Crimes mapped to a neighborhood : {mapped:,}  ({mapped/len(result)*100:.1f}%)")
print(f"  Crimes with no neighborhood     : {unmapped:,}  ({unmapped/len(result)*100:.1f}%)")

print("\nTop 10 neighborhoods by crime count:")
print(result["neighborhood"].value_counts().head(10).to_string())

result.to_csv(OUTPUT, index=False)
print(f"\n  Saved -> {OUTPUT}")
print("\nDone.")
