"""
kmeans_clustering.py
────────────────────────────────────────────────
Uses K-Means clustering on crime coordinates to
identify geographic crime hotspots (hubs) in
Chicago and Seattle.

Results are saved as a new column "cluster" in
the output files, and a map is generated for each
city showing the cluster areas.

Input:
    Data/clean/chicago_data_with_neighborhoods.csv
    Data/clean/spatial_join_seattle.csv

Output:
    Data/clean/chicago_clustered.csv
    Data/clean/seattle_clustered.csv
    Data/outputs/clusters_chicago.html
    Data/outputs/clusters_seattle.html

Install dependencies (run once):
    pip install scikit-learn pandas folium

Run:
    python kmeans_clustering.py
────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

os.makedirs("Data/clean", exist_ok=True)
os.makedirs("Data/outputs", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────
N_CLUSTERS = {
    "chicago": 10,   # Chicago is large — more clusters
    "seattle":  7,   # Seattle is smaller
}

CITY_CENTERS = {
    "chicago": [41.8781, -87.6298],
    "seattle":  [47.6062, -122.3321],
}

# Cluster colours for the map
CLUSTER_COLORS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "lightblue", "darkgreen", "cadetblue", "darkpurple"
]


def run_kmeans(df: pd.DataFrame, city: str, n_clusters: int):
    """
    Run K-Means on latitude/longitude coordinates.
    Returns the DataFrame with a new 'cluster' column
    and the fitted KMeans model.
    """
    print(f"\nRunning K-Means for {city.title()} (k={n_clusters})...")

    coords = df[["latitude", "longitude"]].dropna()

    # Sample for speed if dataset is very large
    if len(coords) > 200000:
        coords_sample = coords.sample(200000, random_state=42)
        print(f"  Sampling 200,000 from {len(coords):,} for fitting")
    else:
        coords_sample = coords
        print(f"  Using all {len(coords):,} points")

    # Standardize coordinates before clustering
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords_sample)

    # Fit K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(coords_scaled)

    # Assign cluster labels to the full dataset
    all_coords_scaled = scaler.transform(df[["latitude", "longitude"]].fillna(0))
    df = df.copy()
    df["cluster"] = kmeans.predict(all_coords_scaled)

    # Null out clusters for rows that had missing coordinates
    df.loc[df["latitude"].isna() | df["longitude"].isna(), "cluster"] = -1

    # Print cluster summary
    print(f"\n  Cluster sizes:")
    cluster_counts = df[df["cluster"] >= 0]["cluster"].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"    Cluster {cluster_id}: {count:,} crimes")

    # Get cluster centres in original coordinate space
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    print(f"\n  Cluster centres (lat, lon):")
    for i, center in enumerate(centers):
        print(f"    Cluster {i}: ({center[0]:.4f}, {center[1]:.4f})")

    return df, kmeans, centers


def top_crimes_per_cluster(df: pd.DataFrame, city: str):
    """
    Show the most common crime types in each cluster.
    """
    print(f"\n  Top crime types per cluster ({city.title()}):")
    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id < 0:
            continue
        cluster_df = df[df["cluster"] == cluster_id]
        top = cluster_df["crime_category"].value_counts().head(3)
        crimes_str = ", ".join([f"{c} ({n:,})" for c, n in top.items()])
        print(f"    Cluster {cluster_id}: {crimes_str}")


def make_cluster_map(
    df: pd.DataFrame,
    centers: np.ndarray,
    city: str,
    output_path: str
):
    """
    Generate an interactive map showing K-Means clusters.
    Each cluster gets a different colour. Cluster centres
    are marked with circle markers showing the top crime type.
    """
    print(f"\nGenerating cluster map for {city.title()}...")

    m = folium.Map(
        location=CITY_CENTERS[city],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    # Sample points to plot (too many points slow the browser)
    sample = df[df["cluster"] >= 0].sample(
        min(10000, len(df[df["cluster"] >= 0])),
        random_state=42
    )

    # Plot crime points coloured by cluster
    for _, row in sample.iterrows():
        cluster_id = int(row["cluster"])
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.4,
            weight=0
        ).add_to(m)

    # Plot cluster centres with labels
    for i, center in enumerate(centers):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        cluster_df = df[df["cluster"] == i]
        top_crime = (
            cluster_df["crime_category"].value_counts().index[0]
            if len(cluster_df) > 0 else "Unknown"
        )
        folium.CircleMarker(
            location=[center[0], center[1]],
            radius=18,
            color="black",
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=2,
            tooltip=f"Cluster {i} | Top crime: {top_crime} | {len(cluster_df):,} incidents"
        ).add_to(m)

        folium.Marker(
            location=[center[0], center[1]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px; font-weight:bold; color:white; '
                     f'text-shadow: 1px 1px 2px black;">{i}</div>',
                icon_size=(20, 20),
                icon_anchor=(10, 10)
            )
        ).add_to(m)

    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                background: rgba(255,255,255,0.9); color: black; padding: 10px 20px;
                border-radius: 8px; font-size: 14px; font-weight: bold;
                border: 2px solid #333; z-index: 1000;">
        {city.title()} — K-Means Crime Clusters (hover cluster centres for details)
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(output_path)
    print(f"  Saved -> {output_path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("K-MEANS CRIME CLUSTERING")
    print("=" * 52)

    for city, input_path, output_csv, output_map in [
        ("chicago", "Data/clean/chicago_data_with_neighborhoods.csv",
         "Data/clean/chicago_clustered.csv",
         "Data/outputs/clusters_chicago.html"),
        ("seattle", "Data/clean/spatial_join_seattle.csv",
         "Data/clean/seattle_clustered.csv",
         "Data/outputs/clusters_seattle.html"),
    ]:
        print(f"\n{'─'*40}")
        print(city.upper())
        print("─" * 40)

        df = pd.read_csv(input_path)
        print(f"  Loaded {len(df):,} records")

        n_clusters = N_CLUSTERS[city]
        df_clustered, model, centers = run_kmeans(df, city, n_clusters)
        top_crimes_per_cluster(df_clustered, city)

        df_clustered.to_csv(output_csv, index=False)
        print(f"\n  Clustered data saved -> {output_csv}")

        make_cluster_map(df_clustered, centers, city, output_map)

    print("\n" + "=" * 52)
    print("K-MEANS CLUSTERING COMPLETE")
    print("Open Data/outputs/clusters_*.html in your browser.")
    print("=" * 52)


if __name__ == "__main__":
    main()