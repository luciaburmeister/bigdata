"""
reddit_nlp.py
────────────────────────────────────────────────
Processes raw Reddit JSON data (Arctic Shift format)
for BOTH Chicago and Seattle at once with three
NLP techniques:
  1. Crime type classification  (keyword matching)
  2. Location extraction        (spaCy Named Entity Recognition)
  3. Sentiment scoring          (VADER)

Works with both posts (selftext) and comments (body).
Handles Arctic Shift format where dates are stored
as created_utc Unix timestamps.

Input:
    Data/raw/reddit_posts_chicago.json
    Data/raw/reddit_posts_seattle.json

Output:
    Data/clean/reddit_nlp.csv   ← both cities combined,
                                   ready for merge_and_analysis.py

Install dependencies (run once):
    pip install vaderSentiment spacy pandas
    python -m spacy download en_core_web_sm

Run:
    python reddit_nlp.py
────────────────────────────────────────────────
"""

import json
import math
import os
import pandas as pd
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

os.makedirs("Data/clean", exist_ok=True)

# ── Input files ───────────────────────────────────────────────────
INPUT_FILES = {
    "chicago": "Data/raw/reddit_posts_chicago.json",
    "seattle": "Data/raw/reddit_posts_seattle.json",
}

OUTPUT = "Data/clean/reddit_nlp.csv"

# ── Crime keyword → unified category map ─────────────────────────
KEYWORD_CATEGORY_MAP = {
    "shooting":      "Homicide",
    "shot":          "Homicide",
    "murder":        "Homicide",
    "homicide":      "Homicide",
    "killed":        "Homicide",
    "dead body":     "Homicide",
    "stabbing":      "Assault",
    "stabbed":       "Assault",
    "assault":       "Assault",
    "attacked":      "Assault",
    "beat up":       "Assault",
    "punched":       "Assault",
    "robbery":       "Robbery",
    "robbed":        "Robbery",
    "mugged":        "Robbery",
    "mugging":       "Robbery",
    "carjacking":    "Vehicle Theft",
    "car stolen":    "Vehicle Theft",
    "vehicle theft": "Vehicle Theft",
    "theft":         "Theft",
    "stolen":        "Theft",
    "shoplifting":   "Theft",
    "pickpocket":    "Theft",
    "burglary":      "Burglary",
    "break-in":      "Burglary",
    "broke in":      "Burglary",
    "breaking in":   "Burglary",
    "drug deal":     "Drugs",
    "drugs":         "Drugs",
    "narcotics":     "Drugs",
    "vandalism":     "Vandalism",
    "graffiti":      "Vandalism",
    "spray paint":   "Vandalism",
    "gun":           "Weapons",
    "firearm":       "Weapons",
    "weapon":        "Weapons",
    "knife":         "Weapons",
    "arson":         "Arson",
    "fire set":      "Arson",
}


# ── Helpers ───────────────────────────────────────────────────────

def parse_utc_timestamp(ts) -> dict:
    """
    Convert a Unix UTC timestamp (integer or float)
    into year, month, day, hour, day_of_week fields.
    """
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return {
            "year":        dt.year,
            "month":       dt.month,
            "day":         dt.day,
            "hour":        dt.hour,
            "day_of_week": dt.strftime("%A"),
        }
    except Exception:
        return {
            "year": None, "month": None,
            "day":  None, "hour":  None,
            "day_of_week": None,
        }


def get_text(record: dict) -> str:
    """
    Extract main text from a record.
    Posts use 'selftext', comments use 'body'.
    Combines with title if available.
    """
    title    = str(record.get("title",    "") or "")
    selftext = str(record.get("selftext", "") or "")
    body     = str(record.get("body",     "") or "")

    content = body if body and body not in ("[deleted]", "[removed]") \
              else selftext

    return (title + " " + content).strip()


def classify_crime(text: str) -> str:
    """
    Scan text for crime keywords.
    Returns the first matching unified category or 'Other'.
    """
    text_lower = str(text).lower()
    for keyword, category in KEYWORD_CATEGORY_MAP.items():
        if keyword in text_lower:
            return category
    return "Other"


def extract_locations(text: str, nlp) -> str:
    """
    Use spaCy NER to find geographic references.
    Returns comma-separated string of locations found.
    """
    doc = nlp(str(text)[:1000])
    locations = [
        ent.text for ent in doc.ents
        if ent.label_ in ("GPE", "LOC")
    ]
    return ", ".join(set(locations)) if locations else ""


def compute_engagement_weight(score, num_comments) -> float:
    """
    log(upvotes + comments + 1) so high-visibility
    posts carry more weight in the analysis.
    """
    try:
        s = max(float(score or 0), 0)
        c = max(float(num_comments or 0), 0)
        return math.log(s + c + 1)
    except Exception:
        return 0.0


def load_records(filepath: str) -> list:
    """Load JSON Lines file — one object per line."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def process_city(city: str, filepath: str, nlp, analyzer) -> pd.DataFrame:
    """
    Run the full NLP pipeline for one city.
    Returns a DataFrame with all processed records.
    """
    print(f"\n{'─'*40}")
    print(f"Processing {city.upper()}...")
    print(f"{'─'*40}")

    if not os.path.exists(filepath):
        print(f"  WARNING: File not found — {filepath}")
        print(f"  Skipping {city}. Run merge_{city}_json.py first.")
        return pd.DataFrame()

    # Load
    print(f"  Loading {filepath}...")
    records = load_records(filepath)
    print(f"  Records loaded: {len(records):,}")

    if not records:
        print(f"  No records found — skipping {city}")
        return pd.DataFrame()

    # Build rows
    rows = []
    for rec in records:
        temporal = parse_utc_timestamp(rec.get("created_utc"))
        text     = get_text(rec)

        rows.append({
            "post_id":      rec.get("id", ""),
            "city":         city,
            "subreddit":    rec.get("subreddit", ""),
            "record_type":  "comment" if (rec.get("body") and not rec.get("title")) else "post",
            "year":         temporal["year"],
            "month":        temporal["month"],
            "day":          temporal["day"],
            "hour":         temporal["hour"],
            "day_of_week":  temporal["day_of_week"],
            "full_text":    text,
            "title":        rec.get("title", ""),
            "score":        rec.get("score", 0),
            "num_comments": rec.get("num_comments", 0),
        })

    df = pd.DataFrame(rows)

    # Drop empty text
    before = len(df)
    df = df[df["full_text"].str.len() > 10]
    print(f"  Dropped {before - len(df):,} empty records")
    print(f"  Posts:    {(df['record_type'] == 'post').sum():,}")
    print(f"  Comments: {(df['record_type'] == 'comment').sum():,}")

    # Step 1 — Crime classification
    print(f"  Classifying crime types...")
    df["crime_category"] = df["full_text"].apply(classify_crime)

    # Step 2 — Location extraction
    print(f"  Extracting locations with spaCy...")
    df["locations_mentioned"] = df["full_text"].apply(
        lambda t: extract_locations(t, nlp)
    )

    # Step 3 — Sentiment
    print(f"  Running VADER sentiment...")
    df["sentiment_compound"] = df["full_text"].apply(
        lambda t: analyzer.polarity_scores(str(t))["compound"]
    )
    df["sentiment_label"] = df["sentiment_compound"].apply(
        lambda s: "positive" if s >= 0.05 else "negative" if s <= -0.05 else "neutral"
    )

    # Step 4 — Engagement weight
    df["engagement_weight"] = df.apply(
        lambda row: compute_engagement_weight(row["score"], row["num_comments"]),
        axis=1
    )

    # Print quick summary
    print(f"\n  Crime category distribution:")
    print(df["crime_category"].value_counts().to_string())
    print(f"\n  Sentiment distribution:")
    print(df["sentiment_label"].value_counts().to_string())
    print(f"  Avg sentiment: {df['sentiment_compound'].mean():.3f}")

    return df


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("REDDIT NLP PIPELINE — BOTH CITIES")
    print("=" * 52)

    # Load NLP models once — reused for both cities
    print("\nLoading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    print("Loading VADER...")
    analyzer = SentimentIntensityAnalyzer()

    # Process each city
    all_dfs = []
    for city, filepath in INPUT_FILES.items():
        df = process_city(city, filepath, nlp, analyzer)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("\n  ERROR: No data processed. Check your input files.")
        return

    # Combine both cities
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n{'─'*40}")
    print(f"COMBINED TOTAL: {len(df_combined):,} records")
    print(f"{'─'*40}")
    print(df_combined["city"].value_counts().to_string())

    # Select output columns
    output_cols = [
        "post_id", "city", "subreddit", "record_type",
        "year", "month", "day", "hour", "day_of_week",
        "crime_category", "sentiment_compound", "sentiment_label",
        "locations_mentioned", "engagement_weight",
        "score", "num_comments", "title"
    ]
    output_cols = [c for c in output_cols if c in df_combined.columns]
    df_out = df_combined[output_cols]

    df_out.to_csv(OUTPUT, index=False)
    print(f"\n  Saved -> {OUTPUT}")

    # Final summary
    print("\n" + "=" * 52)
    print("SUMMARY")
    print("=" * 52)
    print(f"  Total records: {len(df_out):,}")
    print(f"\n  Per city:")
    print(df_out["city"].value_counts().to_string())
    print(f"\n  Crime-related records (not 'Other'):")
    crime = df_out[df_out["crime_category"] != "Other"]
    print(f"  {len(crime):,} records")
    print(f"\n  Avg sentiment by crime type:")
    print(
        df_out.groupby("crime_category")["sentiment_compound"]
        .mean().sort_values().to_string()
    )
    print(f"\n  Next step: run merge_and_analysis.py")


if __name__ == "__main__":
    main()