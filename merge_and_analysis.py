"""
merge_and_analysis.py
────────────────────────────────────────────────
Merges Reddit NLP indicators with police crime
data to produce:
  1. Perception ratio (Reddit mentions / real crimes)
  2. Time-series comparison charts
  3. Crime type comparison bar charts
  4. Summary statistics CSV

Input:
    Data/clean/chicago_data_with_neighborhoods.csv
    Data/clean/spatial_join_seattle.csv
    Data/clean/reddit_nlp.csv

Output:
    Data/clean/merged_analysis.csv
    Data/outputs/chart_perception_ratio.png
    Data/outputs/chart_crime_trends.png
    Data/outputs/chart_sentiment_by_crime.png
    Data/outputs/summary_stats.csv

Install dependencies (run once):
    pip install pandas matplotlib seaborn

Run:
    python merge_and_analysis.py
────────────────────────────────────────────────
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("Data/outputs", exist_ok=True)
os.makedirs("Data/clean", exist_ok=True)

# ── File paths ────────────────────────────────────────────────────
CHICAGO_PATH = "Data/clean/chicago_data_with_neighborhoods.csv"
SEATTLE_PATH = "Data/clean/spatial_join_seattle.csv"
REDDIT_PATH  = "Data/clean/reddit_nlp.csv"

# ── Style ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "sans-serif"


# ── Step 1: Aggregate police data ────────────────────────────────

def aggregate_police() -> pd.DataFrame:
    """
    Load both city CSVs and aggregate to:
    city + year + month + crime_category → count
    """
    print("Aggregating police data...")

    for path in [CHICAGO_PATH, SEATTLE_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    df_ch = pd.read_csv(CHICAGO_PATH)
    df_se = pd.read_csv(SEATTLE_PATH)

    df_ch["city"] = "chicago"
    df_se["city"] = "seattle"

    print(f"  Chicago records: {len(df_ch):,}")
    print(f"  Seattle records: {len(df_se):,}")

    df = pd.concat([df_ch, df_se], ignore_index=True)

    # Ensure year and month are integers
    df["year"]  = pd.to_numeric(df["year"],  errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "month", "crime_category"])

    police_counts = (
        df.groupby(["city", "year", "month", "crime_category"])
        .size()
        .reset_index(name="real_crime_count")
    )

    print(f"  Police aggregate rows: {len(police_counts):,}")
    return police_counts


# ── Step 2: Aggregate Reddit data ────────────────────────────────

def aggregate_reddit() -> pd.DataFrame:
    """
    Load Reddit NLP CSV and aggregate to:
    city + year + month + crime_category →
        weighted mention count + avg sentiment + post count
    """
    print("\nAggregating Reddit data...")

    if not os.path.exists(REDDIT_PATH):
        raise FileNotFoundError(f"File not found: {REDDIT_PATH}\nRun reddit_nlp.py first.")

    df = pd.read_csv(REDDIT_PATH)
    print(f"  Reddit records loaded: {len(df):,}")

    # Ensure year and month are integers
    df["year"]  = pd.to_numeric(df["year"],  errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "month", "city", "crime_category"])

    # Remove unknown cities
    df = df[df["city"].isin(["chicago", "seattle"])]
    print(f"  Records after filtering: {len(df):,}")

    reddit_agg = (
        df.groupby(["city", "year", "month", "crime_category"])
        .agg(
            reddit_mention_count=("engagement_weight", "sum"),
            avg_sentiment=("sentiment_compound", "mean"),
            raw_post_count=("post_id", "count")
        )
        .reset_index()
    )

    print(f"  Reddit aggregate rows: {len(reddit_agg):,}")
    return reddit_agg


# ── Step 3: Merge ────────────────────────────────────────────────

def merge_datasets(police_df: pd.DataFrame, reddit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Outer join on city + year + month + crime_category.
    Fill missing counts with 0.
    Compute perception ratio.
    """
    print("\nMerging datasets...")

    merged = pd.merge(
        police_df,
        reddit_df,
        on=["city", "year", "month", "crime_category"],
        how="outer"
    )

    # Fill numeric nulls with 0 individually
    merged["real_crime_count"]     = merged["real_crime_count"].fillna(0)
    merged["reddit_mention_count"] = merged["reddit_mention_count"].fillna(0)
    merged["avg_sentiment"]        = merged["avg_sentiment"].fillna(0)
    merged["raw_post_count"]       = merged["raw_post_count"].fillna(0)

    # Perception ratio
    merged["perception_ratio"] = (
        merged["reddit_mention_count"] / (merged["real_crime_count"] + 1)
    ).round(4)

    # Readable year-month string
    merged["year_month"] = (
        merged["year"].astype(int).astype(str) + "-" +
        merged["month"].astype(int).astype(str).str.zfill(2)
    )

    print(f"  Merged rows: {len(merged):,}")
    return merged


# ── Step 4: Charts ───────────────────────────────────────────────

def chart_perception_ratio(merged: pd.DataFrame):
    """
    Bar chart: average perception ratio by crime type for each city.
    Red = over-represented on Reddit, Blue = under-represented.
    """
    print("\nGenerating perception ratio chart...")

    ratio_by_crime = (
        merged.groupby(["city", "crime_category"])["perception_ratio"]
        .mean()
        .reset_index()
    )

    # Top 12 crime categories by total real crime volume
    top_crimes = (
        merged.groupby("crime_category")["real_crime_count"]
        .sum()
        .nlargest(12)
        .index
        .tolist()
    )
    ratio_by_crime = ratio_by_crime[ratio_by_crime["crime_category"].isin(top_crimes)]

    cities = ratio_by_crime["city"].unique().tolist()
    fig, axes = plt.subplots(1, len(cities), figsize=(9 * len(cities), 7))
    if len(cities) == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        city_df = (
            ratio_by_crime[ratio_by_crime["city"] == city]
            .sort_values("perception_ratio", ascending=True)
        )
        if city_df.empty:
            ax.set_title(f"{city.title()} — No data")
            continue

        colors = ["#e74c3c" if x > 1 else "#3498db" for x in city_df["perception_ratio"]]
        bars = ax.barh(city_df["crime_category"], city_df["perception_ratio"], color=colors)
        ax.axvline(x=1, color="black", linestyle="--", linewidth=1.2, label="Ratio = 1 (balanced)")
        ax.set_title(f"{city.title()} — Perception Ratio by Crime Type", fontsize=13, fontweight="bold")
        ax.set_xlabel("Perception Ratio (Reddit mentions / Real crimes)", fontsize=10)
        ax.legend(fontsize=9)

        for bar, val in zip(bars, city_df["perception_ratio"]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)

    plt.suptitle(
        "Crime Perception Ratio: Reddit Discussion vs Actual Police Records\n"
        "Red = over-represented on Reddit  |  Blue = under-represented",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("Data/outputs/chart_perception_ratio.png", bbox_inches="tight")
    plt.close()
    print("  Saved -> Data/outputs/chart_perception_ratio.png")


def chart_crime_trends(merged: pd.DataFrame):
    """
    Line chart: real crime counts vs Reddit mentions over time.
    """
    print("\nGenerating crime trends chart...")

    top_crimes = ["Homicide", "Theft", "Assault", "Robbery"]
    cities     = ["chicago", "seattle"]

    fig, axes = plt.subplots(len(top_crimes), len(cities),
                             figsize=(16, 4 * len(top_crimes)))

    for row_idx, crime in enumerate(top_crimes):
        for col_idx, city in enumerate(cities):
            ax = axes[row_idx][col_idx]

            city_crime = merged[
                (merged["city"] == city) &
                (merged["crime_category"] == crime)
            ].sort_values(["year", "month"])

            if city_crime.empty:
                ax.set_title(f"{city.title()} — {crime} (no data)")
                ax.text(0.5, 0.5, "No data available",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=10, color="grey")
                continue

            ax2 = ax.twinx()
            ax.plot(range(len(city_crime)), city_crime["real_crime_count"],
                    color="#2980b9", linewidth=2, label="Real crimes (police)")
            ax2.plot(range(len(city_crime)), city_crime["reddit_mention_count"],
                     color="#e74c3c", linewidth=1.5, linestyle="--", label="Reddit mentions")

            ax.set_title(f"{city.title()} — {crime}", fontsize=11, fontweight="bold")
            ax.set_ylabel("Real Crime Count", color="#2980b9", fontsize=9)
            ax2.set_ylabel("Reddit Mentions (weighted)", color="#e74c3c", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)

            ticks = list(range(0, len(city_crime), max(1, len(city_crime) // 10)))
            labels = city_crime.iloc[ticks]["year"].astype(int).tolist()
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, fontsize=8)

    plt.suptitle(
        "Real Crime Counts vs Reddit Mentions Over Time\n"
        "Blue = Police Records  |  Red Dashed = Reddit Discussion",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("Data/outputs/chart_crime_trends.png", bbox_inches="tight")
    plt.close()
    print("  Saved -> Data/outputs/chart_crime_trends.png")


def chart_sentiment_by_crime(reddit_df: pd.DataFrame):
    """
    Bar chart: average VADER sentiment score per crime type per city.
    Uses the already-aggregated reddit_df directly.
    """
    print("\nGenerating sentiment chart...")

    # Top 10 crime categories by post volume
    top_crimes = (
        reddit_df.groupby("crime_category")["raw_post_count"]
        .sum()
        .nlargest(10)
        .index
        .tolist()
    )

    sentiment = (
        reddit_df[reddit_df["crime_category"].isin(top_crimes)]
        .groupby(["city", "crime_category"])["avg_sentiment"]
        .mean()
        .reset_index()
    )

    cities = sentiment["city"].unique().tolist()
    if not cities:
        print("  No sentiment data — skipping")
        return

    fig, axes = plt.subplots(1, len(cities), figsize=(9 * len(cities), 6))
    if len(cities) == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        city_df = sentiment[sentiment["city"] == city].sort_values("avg_sentiment")
        if city_df.empty:
            continue

        colors = [
            "#e74c3c" if s < -0.05 else
            "#27ae60" if s > 0.05 else
            "#95a5a6"
            for s in city_df["avg_sentiment"]
        ]
        ax.barh(city_df["crime_category"], city_df["avg_sentiment"], color=colors)
        ax.axvline(x=0, color="black", linewidth=1)
        ax.set_title(f"{city.title()} — Avg Sentiment by Crime Type",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("VADER Compound Score (-1 = very negative, +1 = very positive)", fontsize=9)

    plt.suptitle(
        "Average Reddit Sentiment by Crime Category\n"
        "Red = negative/fearful  |  Grey = neutral  |  Green = positive",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("Data/outputs/chart_sentiment_by_crime.png", bbox_inches="tight")
    plt.close()
    print("  Saved -> Data/outputs/chart_sentiment_by_crime.png")


def save_summary_stats(merged: pd.DataFrame):
    """Save a summary statistics CSV for use in the report."""
    summary = (
        merged.groupby(["city", "crime_category"])
        .agg(
            total_real_crimes    =("real_crime_count",     "sum"),
            total_reddit_mentions=("reddit_mention_count", "sum"),
            avg_perception_ratio =("perception_ratio",     "mean"),
            avg_sentiment        =("avg_sentiment",        "mean")
        )
        .reset_index()
        .sort_values(["city", "avg_perception_ratio"], ascending=[True, False])
    )

    summary.to_csv("Data/outputs/summary_stats.csv", index=False)
    print("  Summary stats saved -> Data/outputs/summary_stats.csv")

    print("\n  TOP 5 MOST OVER-REPRESENTED CRIMES ON REDDIT (by city):")
    for city in ["chicago", "seattle"]:
        city_summary = summary[summary["city"] == city]
        if city_summary.empty:
            continue
        print(f"\n  {city.title()}:")
        for _, row in city_summary.head(5).iterrows():
            print(
                f"    {row['crime_category']}: "
                f"ratio={row['avg_perception_ratio']:.2f} | "
                f"real={int(row['total_real_crimes']):,} | "
                f"reddit={row['total_reddit_mentions']:.0f}"
            )


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("MERGE & ANALYSIS PIPELINE")
    print("=" * 52)

    # Aggregate both sources
    police_df = aggregate_police()
    reddit_df = aggregate_reddit()

    # Merge
    merged = merge_datasets(police_df, reddit_df)
    merged.to_csv("Data/clean/merged_analysis.csv", index=False)
    print(f"\n  Merged dataset saved -> Data/clean/merged_analysis.csv")

    # Charts
    chart_perception_ratio(merged)
    chart_crime_trends(merged)
    chart_sentiment_by_crime(reddit_df)  # passes reddit_df directly

    # Summary stats
    save_summary_stats(merged)

    print("\n" + "=" * 52)
    print("ANALYSIS COMPLETE")
    print("All outputs saved to Data/outputs/")
    print("=" * 52)


if __name__ == "__main__":
    main()