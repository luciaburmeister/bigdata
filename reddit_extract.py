"""
reddit_extraction.py
────────────────────────────────────────────────────────────────────
Collects crime-related posts and comments from Chicago and Seattle
subreddits using the Reddit API (via PRAW).

Output:
    Data/raw/reddit_posts.json
    Data/raw/reddit_comments.json

SETUP (run once before using this script)
────────────────────────────────────────────────────────────────────
1. Install dependencies:
       pip install praw python-dotenv pandas

2. Create a file called .env in the same folder as this script.
   Paste these three lines into it:

       REDDIT_CLIENT_ID=paste_your_client_id_here
       REDDIT_CLIENT_SECRET=paste_your_client_secret_here
       REDDIT_USER_AGENT=crime_project_scraper_v1

3. Run:
       python reddit_extraction.py
────────────────────────────────────────────────────────────────────
"""

import os
import json
import time
import praw
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ── Load credentials from .env ────────────────────────────────────
load_dotenv()

CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "crime_project_scraper_v1")

if not CLIENT_ID or not CLIENT_SECRET:
    raise EnvironmentError(
        "\nReddit credentials not found.\n"
        "Make sure your .env file exists and contains:\n"
        "  REDDIT_CLIENT_ID=...\n"
        "  REDDIT_CLIENT_SECRET=...\n"
    )

# ── Output paths ──────────────────────────────────────────────────
os.makedirs("Data/raw", exist_ok=True)
POSTS_OUTPUT    = "Data/raw/reddit_posts.json"
COMMENTS_OUTPUT = "Data/raw/reddit_comments.json"

# ── Subreddits to collect from ────────────────────────────────────
SUBREDDITS = {
    "chicago": ["chicago", "ChicagoCrime", "ChicagoSuburbs"],
    "seattle": ["Seattle", "SeattleWA", "seattlecrime"],
}

# ── Keywords to search for within each subreddit ──────────────────
KEYWORDS = [
    "shooting",
    "robbery",
    "assault",
    "burglary",
    "theft",
    "murder",
    "homicide",
    "stabbing",
    "carjacking",
    "break-in",
    "gun",
    "crime",
    "unsafe",
    "police",
    "arrested",
    "neighborhood watch",
    "mugged",
    "drug deal",
    "vandalism",
    "heard gunshots",
]

POSTS_PER_KEYWORD = 50
COMMENT_DEPTH     = 1
REQUEST_PAUSE     = 0.5


# ── Connect to Reddit ─────────────────────────────────────────────

def get_reddit_client() -> praw.Reddit:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )
    print(f"  Connected to Reddit API  |  read-only: {reddit.read_only}")
    return reddit


# ── Helpers ───────────────────────────────────────────────────────

def utc_to_str(ts: float) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def save_json(data: list, path: str):
    """Save list of dicts as JSON Lines — one record per line."""
    with open(path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} records  ->  {path}")


# ── Post collection ───────────────────────────────────────────────

def collect_posts(reddit: praw.Reddit) -> list:
    records  = []
    seen_ids = set()

    for city, subreddit_list in SUBREDDITS.items():
        for subreddit_name in subreddit_list:
            subreddit = reddit.subreddit(subreddit_name)
            print(f"\n  Subreddit: r/{subreddit_name}  ({city})")

            for keyword in KEYWORDS:
                try:
                    results = subreddit.search(
                        query=keyword,
                        sort="relevance",
                        time_filter="all",
                        limit=POSTS_PER_KEYWORD,
                    )

                    new_this_keyword = 0

                    for post in results:

                        if post.id in seen_ids:
                            continue
                        seen_ids.add(post.id)

                        if post.selftext in ["[removed]", "[deleted]", ""]:
                            continue

                        records.append({
                            "post_id":         post.id,
                            "city":            city,
                            "subreddit":       subreddit_name,
                            "keyword_matched": keyword,
                            "title":           post.title,
                            "body":            post.selftext,
                            "post_flair":      post.link_flair_text,
                            "score":           post.score,
                            "upvote_ratio":    post.upvote_ratio,
                            "num_comments":    post.num_comments,
                            "total_awards":    post.total_awards_received,
                            "created_utc":     utc_to_str(post.created_utc),
                            "year":            datetime.utcfromtimestamp(post.created_utc).year,
                            "month":           datetime.utcfromtimestamp(post.created_utc).month,
                            "day":             datetime.utcfromtimestamp(post.created_utc).day,
                            "hour":            datetime.utcfromtimestamp(post.created_utc).hour,
                            "day_of_week":     datetime.utcfromtimestamp(post.created_utc).strftime("%A"),
                            "author":          str(post.author) if post.author else "[deleted]",
                            "url":             post.url,
                            "permalink":       f"https://reddit.com{post.permalink}",
                        })
                        new_this_keyword += 1
                        time.sleep(REQUEST_PAUSE)

                    print(f"    keyword='{keyword}'  ->  {new_this_keyword} new posts")

                except Exception as e:
                    print(f"    ERROR — keyword='{keyword}' in r/{subreddit_name}: {e}")
                    continue

    print(f"\n  Total posts collected: {len(records):,}")
    return records


# ── Comment collection ────────────────────────────────────────────

def collect_comments(reddit: praw.Reddit, posts: list) -> list:
    records     = []
    total_posts = len(posts)

    for i, post_record in enumerate(posts, start=1):
        print(f"  Fetching comments: {i}/{total_posts}  (post_id={post_record['post_id']})", end="\r")

        try:
            submission = reddit.submission(id=post_record["post_id"])
            submission.comments.replace_more(limit=0)

            for comment in submission.comments.list():

                if COMMENT_DEPTH == 0 and comment.depth > 0:
                    continue
                if comment.depth > COMMENT_DEPTH:
                    continue
                if not hasattr(comment, "body"):
                    continue
                if comment.body in ["[removed]", "[deleted]", ""]:
                    continue

                records.append({
                    "comment_id":  comment.id,
                    "post_id":     post_record["post_id"],
                    "city":        post_record["city"],
                    "subreddit":   post_record["subreddit"],
                    "parent_id":   comment.parent_id,
                    "depth":       comment.depth,
                    "body":        comment.body,
                    "score":       comment.score,
                    "created_utc": utc_to_str(comment.created_utc),
                    "year":        datetime.utcfromtimestamp(comment.created_utc).year,
                    "month":       datetime.utcfromtimestamp(comment.created_utc).month,
                    "day":         datetime.utcfromtimestamp(comment.created_utc).day,
                    "hour":        datetime.utcfromtimestamp(comment.created_utc).hour,
                    "day_of_week": datetime.utcfromtimestamp(comment.created_utc).strftime("%A"),
                    "author":      str(comment.author) if comment.author else "[deleted]",
                })

            time.sleep(REQUEST_PAUSE)

        except Exception as e:
            print(f"\n  ERROR fetching comments for post {post_record['post_id']}: {e}")
            continue

    print(f"\n  Total comments collected: {len(records):,}")
    return records


# ── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 52)
    print("REDDIT EXTRACTION — CRIME PERCEPTION PROJECT")
    print("=" * 52)

    reddit = get_reddit_client()

    print("\n[1/3] Collecting posts...")
    posts = collect_posts(reddit)
    save_json(posts, POSTS_OUTPUT)

    print("\n[2/3] Collecting comments...")
    comments = collect_comments(reddit, posts)
    save_json(comments, COMMENTS_OUTPUT)

    print("\n[3/3] Summary")
    print(f"  Posts collected:    {len(posts):,}")
    print(f"  Comments collected: {len(comments):,}")
    print(f"\n  Posts file:    {POSTS_OUTPUT}")
    print(f"  Comments file: {COMMENTS_OUTPUT}")

    cities = {}
    for p in posts:
        cities[p["city"]] = cities.get(p["city"], 0) + 1
    print("\n  Posts per city:")
    for city, count in cities.items():
        print(f"    {city}: {count:,}")

    print("\nDone.")