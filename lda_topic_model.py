"""
lda_topic_model.py
────────────────────────────────────────────────
Runs LDA (Latent Dirichlet Allocation) topic
modeling on Reddit posts within each K-Means
crime cluster to answer:

"What are people in each high-crime area
 actually talking about on Reddit?"

Input:
    Data/clean/chicago_clustered.csv
    Data/clean/seattle_clustered.csv
    Data/clean/reddit_nlp.csv

Output:
    Data/outputs/lda_topics_chicago.csv
    Data/outputs/lda_topics_seattle.csv
    Data/outputs/lda_summary.txt

Install dependencies (run once):
    pip install scikit-learn pandas nltk
    python -m nltk.downloader stopwords

Run:
    python lda_topic_model.py
────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords

os.makedirs("Data/outputs", exist_ok=True)

# Download NLTK stopwords if not already downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

# Add custom stopwords relevant to Reddit and our context
CUSTOM_STOP_WORDS = {
    "https", "http", "www", "com", "reddit", "post", "comment",
    "just", "like", "people", "really", "think", "know", "get",
    "going", "one", "would", "also", "even", "said", "says",
    "deleted", "removed", "gt", "amp", "chicago", "seattle",
    "city", "area", "neighborhood", "street"
}
STOP_WORDS = STOP_WORDS.union(CUSTOM_STOP_WORDS)

# ── Config ────────────────────────────────────────────────────────
N_TOPICS     = 5    # number of topics per cluster
N_TOP_WORDS  = 10   # words to show per topic
MIN_POSTS    = 20   # minimum posts in a cluster to run LDA


def clean_text(text: str) -> str:
    """
    Basic text cleaning for LDA:
    - lowercase
    - remove URLs, punctuation, numbers
    - remove short words
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)         # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters
    text = re.sub(r"\b\w{1,2}\b", "", text)     # remove 1-2 char words
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_lda(texts: list, n_topics: int = N_TOPICS):
    """
    Fit LDA model on a list of text documents.
    Returns the model and vectorizer.
    """
    vectorizer = CountVectorizer(
        max_features=500,
        stop_words=list(STOP_WORDS),
        min_df=2,           # word must appear in at least 2 docs
        max_df=0.95         # ignore words in >95% of docs
    )

    dtm = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method="online"
    )
    lda.fit(dtm)

    return lda, vectorizer


def get_top_words(lda_model, vectorizer, n_words: int = N_TOP_WORDS):
    """
    Extract the top N words for each LDA topic.
    Returns a list of lists: [[word1, word2, ...], ...]
    """
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)
    return topics


def analyze_city(city: str, clustered_path: str, reddit_df: pd.DataFrame):
    """
    For each K-Means cluster in a city, run LDA on Reddit posts
    that mention that city and summarize what people are discussing.
    """
    print(f"\n{'─'*40}")
    print(f"LDA TOPIC MODELING — {city.upper()}")
    print("─" * 40)

    df_clustered = pd.read_csv(clustered_path)
    city_reddit  = reddit_df[reddit_df["city"] == city].copy()

    print(f"  Crime records: {len(df_clustered):,}")
    print(f"  Reddit posts:  {len(city_reddit):,}")

    if "title" not in city_reddit.columns:
        print("  No title column in Reddit data — skipping")
        return pd.DataFrame()

    # Combine title and any body text for LDA
    title_col   = city_reddit["title"].fillna("") if "title" in city_reddit.columns else ""
    body_col    = city_reddit["body"].fillna("") if "body" in city_reddit.columns else ""
    selftext_col= city_reddit["selftext"].fillna("") if "selftext" in city_reddit.columns else ""
    content_col = body_col if body_col is not "" else selftext_col
    city_reddit["text_clean"] = (title_col + " " + content_col).apply(clean_text)

    # Get unique clusters from crime data
    clusters = sorted([c for c in df_clustered["cluster"].unique() if c >= 0])

    results = []

    # Also run LDA on ALL city Reddit posts (not per cluster)
    # since Reddit posts don't have cluster assignments yet
    print(f"\n  Running LDA on all {city.title()} Reddit posts...")
    all_texts = city_reddit["text_clean"].tolist()
    all_texts = [t for t in all_texts if len(t.split()) >= 5]

    if len(all_texts) >= MIN_POSTS:
        lda_model, vectorizer = run_lda(all_texts, n_topics=N_TOPICS)
        topics = get_top_words(lda_model, vectorizer)

        print(f"\n  Top {N_TOPICS} topics found in {city.title()} Reddit discussions:")
        for topic_idx, words in enumerate(topics):
            print(f"    Topic {topic_idx + 1}: {', '.join(words)}")
            results.append({
                "city":       city,
                "cluster":    "all",
                "topic":      topic_idx + 1,
                "top_words":  ", ".join(words),
                "n_posts":    len(all_texts)
            })
    else:
        print(f"  Not enough posts for LDA (need {MIN_POSTS}, have {len(all_texts)})")

    # Run per crime category if enough posts
    print(f"\n  Running LDA per crime category...")
    for category in city_reddit["crime_category"].unique():
        cat_df    = city_reddit[city_reddit["crime_category"] == category]
        cat_texts = [t for t in cat_df["text_clean"].tolist() if len(t.split()) >= 5]

        if len(cat_texts) < MIN_POSTS:
            continue

        lda_model, vectorizer = run_lda(cat_texts, n_topics=min(3, N_TOPICS))
        topics = get_top_words(lda_model, vectorizer, n_words=8)

        print(f"\n    Crime category: {category} ({len(cat_texts)} posts)")
        for topic_idx, words in enumerate(topics):
            print(f"      Topic {topic_idx + 1}: {', '.join(words)}")
            results.append({
                "city":       city,
                "cluster":    f"category_{category}",
                "topic":      topic_idx + 1,
                "top_words":  ", ".join(words),
                "n_posts":    len(cat_texts)
            })

    return pd.DataFrame(results)


def save_summary(results: dict, output_path: str):
    """
    Save a plain text summary of all LDA results
    that can be pasted directly into the report.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("LDA TOPIC MODELING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("This file summarizes the main topics found in Reddit\n")
        f.write("crime discussions for Chicago and Seattle.\n\n")

        for city, df in results.items():
            if df.empty:
                continue
            f.write(f"\n{'─'*40}\n")
            f.write(f"{city.upper()}\n")
            f.write("─" * 40 + "\n")

            for cluster in df["cluster"].unique():
                cluster_df = df[df["cluster"] == cluster]
                f.write(f"\n  Scope: {cluster}\n")
                for _, row in cluster_df.iterrows():
                    f.write(f"    Topic {row['topic']}: {row['top_words']}\n")

    print(f"\nSummary saved -> {output_path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("LDA TOPIC MODELING")
    print("=" * 52)

    # Load Reddit NLP output
    print("\nLoading Reddit NLP data...")
    reddit_df = pd.read_csv("Data/clean/reddit_nlp.csv")
    print(f"  Reddit posts loaded: {len(reddit_df):,}")

    all_results = {}

    # Chicago
    df_chicago = analyze_city(
        "chicago",
        "Data/clean/chicago_clustered.csv",
        reddit_df
    )
    if not df_chicago.empty:
        df_chicago.to_csv("Data/outputs/lda_topics_chicago.csv", index=False)
        print(f"\n  Chicago topics saved -> Data/outputs/lda_topics_chicago.csv")
    all_results["chicago"] = df_chicago

    # Seattle
    df_seattle = analyze_city(
        "seattle",
        "Data/clean/seattle_clustered.csv",
        reddit_df
    )
    if not df_seattle.empty:
        df_seattle.to_csv("Data/outputs/lda_topics_seattle.csv", index=False)
        print(f"\n  Seattle topics saved -> Data/outputs/lda_topics_seattle.csv")
    all_results["seattle"] = df_seattle

    # Save plain text summary
    save_summary(all_results, "Data/outputs/lda_summary.txt")

    print("\n" + "=" * 52)
    print("LDA TOPIC MODELING COMPLETE")
    print("=" * 52)


if __name__ == "__main__":
    main()