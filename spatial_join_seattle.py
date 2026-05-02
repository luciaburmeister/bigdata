"""
spatial_join_seattle.py
────────────────────────────────────────────────
Adds a 'neighborhood' column to the cleaned
Seattle crime data by spatially joining each
crime's lat/lon against the neighborhood polygons.

Input:  seattle_data_clean.csv
        nma_nhoods_sub.geojson
Output: Data/clean/spatial_join_seattle.csv
────────────────────────────────────────────────
"""

import pandas as pd
import geopandas as gpd
import os

CRIME_IN = "Data/clean/seattle_data_clean.csv"
NEIGH_IN = "Data/shapefiles/nma_nhoods_sub.geojson"
OUTPUT   = "Data/clean/spatial_join_seattle.csv"
CRS      = "EPSG:4326"

os.makedirs("Data/clean", exist_ok=True)

# ── 1. Load neighborhoods ─────────────────────────────────────────────
print("Loading neighborhood polygons...")
neigh_gdf = gpd.read_file(NEIGH_IN)[["S_HOOD", "L_HOOD", "geometry"]].to_crs(CRS)
print(f"  {len(neigh_gdf)} neighborhoods loaded")

# ── 2. Load crime data in chunks and spatial join ─────────────────────
print("\nLoading crime data and running spatial join...")
CHUNK = 200_000
chunks_out = []
total_in = total_out = 0

for i, chunk in enumerate(pd.read_csv(CRIME_IN, chunksize=CHUNK)):
    total_in += len(chunk)

    # Drop the raw neighborhood column from the source data — it will be
    # replaced by the spatially-joined value from the polygon file.
    chunk = chunk.drop(columns=["neighborhood"], errors="ignore")

    crime_gdf = gpd.GeoDataFrame(
        chunk,
        geometry=gpd.points_from_xy(chunk["longitude"], chunk["latitude"]),
        crs=CRS,
    )

    joined = gpd.sjoin(
        crime_gdf,
        neigh_gdf[["S_HOOD", "L_HOOD", "geometry"]],
        how="left",
        predicate="within",
    )

    # Keep first match per row in case a point touches two polygon edges
    joined = joined[~joined.index.duplicated(keep="first")]

    joined = joined.rename(columns={"S_HOOD": "neighborhood", "L_HOOD": "large_neighborhood"})
    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")

    total_out += len(joined)
    chunks_out.append(joined)

    print(f"  Processed {min(total_in, (i+1)*CHUNK):,} rows...", end="\r")

print(f"\n  Spatial join complete — {total_out:,} rows")

# ── 3. Combine and save ───────────────────────────────────────────────
print("\nCombining chunks and saving...")
result = pd.concat(chunks_out, ignore_index=True)

mapped   = int(result["neighborhood"].notna().sum())
unmapped = int(result["neighborhood"].isna().sum())
print(f"  Crimes mapped to a neighborhood : {mapped:,}  ({mapped/len(result)*100:.1f}%)")
print(f"  Crimes with no neighborhood     : {unmapped:,}  ({unmapped/len(result)*100:.1f}%)")

print("\nTop 10 neighborhoods by crime count:")
print(result["neighborhood"].value_counts().head(10).to_string())

result.to_csv(OUTPUT, index=False)
print(f"\n  Saved -> {OUTPUT}")
print("\nDone.")
