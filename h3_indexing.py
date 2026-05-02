"""
h3_indexing.py
────────────────────────────────────────────────
Demonstrates H3 geospatial indexing on crime data.

H3 (developed by Uber) divides the entire world
into hexagonal cells at different resolutions.
This makes spatial queries and aggregations
dramatically faster at scale — instead of checking
if millions of GPS points fall inside complex
neighbourhood polygons, you simply look up which
hex cell each point belongs to.

This script:
  1. Assigns an H3 hex index to every crime record
  2. Aggregates crime counts per hex cell
  3. Generates an interactive hex map for each city

Why this matters for the report:
  - At production scale (millions of rows), spatial
    joins with polygon shapefiles become slow.
  - H3 indexing reduces this to a simple lookup.
  - Reddit posts and crime records in the same hex
    cell can be joined instantly by cell ID.

Input:
    Data/clean/chicago_data_with_neighborhoods.csv
    Data/clean/spatial_join_seattle.csv

Output:
    Data/clean/chicago_h3.csv
    Data/clean/seattle_h3.csv
    Data/outputs/h3_map_chicago.html
    Data/outputs/h3_map_seattle.html

Install dependencies (run once):
    pip install h3 folium pandas

Reference:
    https://h3geo.org/
    https://s2geometry.io/devguide/s2cell_hierarchy.html

Run:
    python h3_indexing.py
────────────────────────────────────────────────
"""

import pandas as pd
import folium
import h3
import os

os.makedirs("Data/outputs", exist_ok=True)
os.makedirs("Data/clean", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────
# H3 resolution — controls hex cell size
# Resolution 8 = ~0.74 km² per cell (good for neighbourhood level)
# Resolution 9 = ~0.10 km² per cell (more granular, street level)
H3_RESOLUTION = 8

CITY_CENTERS = {
    "chicago": [41.8781, -87.6298],
    "seattle":  [47.6062, -122.3321],
}


def assign_h3_index(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """
    Add an H3 hex cell index to each row based on
    its latitude and longitude.
    """
    df = df.copy()
    df = df.dropna(subset=["latitude", "longitude"])

    def lat_lng_to_h3(row):
        try:
            return h3.latlng_to_cell(row["latitude"], row["longitude"], resolution)
        except Exception:
            return None

    df["h3_index"] = df.apply(lat_lng_to_h3, axis=1)
    df = df[df["h3_index"].notna()]
    return df


def aggregate_by_hex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count crimes per H3 hex cell and get the cell centre coordinates.
    """
    hex_counts = (
        df.groupby("h3_index")
        .agg(
            crime_count=("crime_category", "count"),
            top_crime=("crime_category", lambda x: x.value_counts().index[0])
        )
        .reset_index()
    )

    # Get centre coordinates of each hex cell for mapping
    hex_counts["hex_lat"] = hex_counts["h3_index"].apply(
        lambda h: h3.cell_to_latlng(h)[0]
    )
    hex_counts["hex_lng"] = hex_counts["h3_index"].apply(
        lambda h: h3.cell_to_latlng(h)[1]
    )

    return hex_counts.sort_values("crime_count", ascending=False)


def make_h3_map(hex_counts: pd.DataFrame, city: str, output_path: str):
    """
    Generate an interactive folium map with H3 hex cells
    coloured by crime count density.
    """
    print(f"\nGenerating H3 map for {city.title()}...")

    m = folium.Map(
        location=CITY_CENTERS[city],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    max_count = hex_counts["crime_count"].max()

    def get_color(count):
        """Map crime count to a colour gradient."""
        ratio = count / max_count
        if ratio > 0.75:
            return "#e74c3c"   # red — very high crime
        elif ratio > 0.50:
            return "#e67e22"   # orange — high crime
        elif ratio > 0.25:
            return "#f1c40f"   # yellow — medium crime
        else:
            return "#2ecc71"   # green — low crime

    # Plot top 500 hex cells (most active areas)
    top_hexes = hex_counts.head(500)

    for _, row in top_hexes.iterrows():
        try:
            # Get the boundary polygon of the hex cell
            boundary = h3.cell_to_boundary(row["h3_index"])
            # h3 returns (lat, lng) pairs — folium needs [lat, lng]
            polygon_coords = [[lat, lng] for lat, lng in boundary]

            folium.Polygon(
                locations=polygon_coords,
                color=get_color(row["crime_count"]),
                fill=True,
                fill_color=get_color(row["crime_count"]),
                fill_opacity=0.6,
                weight=1,
                tooltip=(
                    f"H3 Cell: {row['h3_index']}<br>"
                    f"Crime Count: {row['crime_count']:,}<br>"
                    f"Top Crime: {row['top_crime']}"
                )
            ).add_to(m)

        except Exception:
            continue

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 15px; border-radius: 8px;
                border: 2px solid #333; font-size: 13px;">
        <b>Crime Density (H3 cells)</b><br>
        <span style="color:#e74c3c;">■</span> Very High<br>
        <span style="color:#e67e22;">■</span> High<br>
        <span style="color:#f1c40f;">■</span> Medium<br>
        <span style="color:#2ecc71;">■</span> Low<br>
        <br><small>H3 Resolution: {}</small>
    </div>
    """.format(H3_RESOLUTION)
    m.get_root().html.add_child(folium.Element(legend_html))

    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                background: rgba(255,255,255,0.9); color: black; padding: 10px 20px;
                border-radius: 8px; font-size: 14px; font-weight: bold;
                border: 2px solid #333; z-index: 1000;">
        {city.title()} — H3 Geospatial Crime Index (Resolution {H3_RESOLUTION})
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(output_path)
    print(f"  Saved -> {output_path}")


def print_h3_stats(hex_counts: pd.DataFrame, city: str):
    """Print summary statistics about the H3 indexing results."""
    print(f"\n  H3 Statistics — {city.title()}:")
    print(f"    Total hex cells with crimes: {len(hex_counts):,}")
    print(f"    Max crimes in one hex cell:  {hex_counts['crime_count'].max():,}")
    print(f"    Avg crimes per hex cell:     {hex_counts['crime_count'].mean():.1f}")
    print(f"\n    Top 5 hex cells by crime count:")
    for _, row in hex_counts.head(5).iterrows():
        print(
            f"      {row['h3_index']} | "
            f"crimes: {row['crime_count']:,} | "
            f"top crime: {row['top_crime']} | "
            f"centre: ({row['hex_lat']:.4f}, {row['hex_lng']:.4f})"
        )


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print(f"H3 GEOSPATIAL INDEXING  (resolution={H3_RESOLUTION})")
    print("=" * 52)
    print("""
  H3 converts GPS coordinates into a hex grid.
  Each crime record gets a hex cell ID.
  Crimes in the same cell share the same ID —
  so joining Reddit posts to crime locations
  becomes a simple ID lookup instead of a slow
  polygon intersection query.
    """)

    for city, input_path, output_csv, output_map in [
        ("chicago",
         "Data/clean/chicago_data_with_neighborhoods.csv",
         "Data/clean/chicago_h3.csv",
         "Data/outputs/h3_map_chicago.html"),
        ("seattle",
         "Data/clean/spatial_join_seattle.csv",
         "Data/clean/seattle_h3.csv",
         "Data/outputs/h3_map_seattle.html"),
    ]:
        print(f"\n{'─'*40}")
        print(city.upper())
        print("─" * 40)

        df = pd.read_csv(input_path)
        print(f"  Loaded {len(df):,} records")

        # Assign H3 index
        df_h3 = assign_h3_index(df, H3_RESOLUTION)
        print(f"  H3 indexed: {len(df_h3):,} records")

        # Save enriched file
        df_h3.to_csv(output_csv, index=False)
        print(f"  Saved -> {output_csv}")

        # Aggregate and map
        hex_counts = aggregate_by_hex(df_h3)
        print_h3_stats(hex_counts, city)
        make_h3_map(hex_counts, city, output_map)

    print("\n" + "=" * 52)
    print("H3 INDEXING COMPLETE")
    print("Open Data/outputs/h3_map_*.html in your browser.")
    print("=" * 52)


if __name__ == "__main__":
    main()