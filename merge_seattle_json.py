"""
merge_seattle_json.py
────────────────────────────────────────────────
Reads all JSON files from Data/json_seattle/
and merges them into one clean file ready for
reddit_nlp.py to process.

Input:
    Data/json_seattle/   ← put all your Seattle
                           JSON files in here

Output:
    Data/raw/reddit_posts_seattle.json

Run:
    python merge_seattle_json.py
────────────────────────────────────────────────
"""

import json
import os

INPUT_FOLDER = "/Users/lucia/Desktop/Big Data Management/Project/Data/json_seattle/"
OUTPUT_FILE  = "/Users/lucia/Desktop/Big Data Management/Project/Data/raw/reddit_posts_seattle.json"

os.makedirs("Data/raw", exist_ok=True)

# ── Subreddits that belong to Seattle ────────────────────────────
SEATTLE_SUBREDDITS = {
    "seattle", "seattlewa", "askseattle", "seattlenews",
    "pacificnorthwest", "washington", "seattleorca"
}


def load_json_file(filepath: str) -> list:
    """
    Load a JSON file that could be in any of these formats:
      - JSON Lines (one object per line)  ← Arctic Shift format
      - A JSON array  [ {...}, {...} ]
      - A single JSON object  { ... }
      - A dict with a "data" key  { "data": [ {...} ] }
    """
    records = []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    # Try JSON Lines first (Arctic Shift default)
    if content.startswith("{"):
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if records:
            return records

    # Try as a full JSON array or object
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if "data" in data:
                return data["data"]
            else:
                return [data]
    except json.JSONDecodeError:
        pass

    return records


def clean_record(record: dict) -> dict:
    """
    Standardise each record so reddit_nlp.py can
    work with it regardless of which subreddit it
    came from. Adds city tag and normalises key fields.
    """
    return {
        "id":           record.get("id", ""),
        "city":         "seattle",
        "subreddit":    record.get("subreddit", ""),
        "title":        record.get("title", ""),
        # posts use selftext, comments use body
        "selftext":     record.get("selftext", ""),
        "body":         record.get("body", ""),
        "score":        record.get("score", 0),
        "num_comments": record.get("num_comments", 0),
        "created_utc":  record.get("created_utc", None),
        "author":       record.get("author", ""),
        "permalink":    record.get("permalink", ""),
        # keep link_id so comments can be traced back to posts
        "link_id":      record.get("link_id", ""),
        "parent_id":    record.get("parent_id", ""),
    }


def main():
    print("=" * 52)
    print("MERGE SEATTLE JSON FILES")
    print("=" * 52)

    if not os.path.exists(INPUT_FOLDER):
        print(f"\n  ERROR: Folder not found: {INPUT_FOLDER}")
        print("  Create the folder and put your Seattle JSON files inside it.")
        return

    json_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.endswith(".json")
    ]

    if not json_files:
        print(f"\n  ERROR: No JSON files found in {INPUT_FOLDER}")
        return

    print(f"\n  Found {len(json_files)} JSON file(s) in {INPUT_FOLDER}:")
    for f in json_files:
        print(f"    - {f}")

    all_records = []
    seen_ids    = set()

    for filename in json_files:
        filepath = os.path.join(INPUT_FOLDER, filename)
        print(f"\n  Reading {filename}...")

        records = load_json_file(filepath)
        print(f"    Raw records found: {len(records):,}")

        new_count     = 0
        skipped_count = 0
        wrong_city    = 0

        for rec in records:
            # Skip if subreddit clearly doesn't belong to Seattle
            subreddit = str(rec.get("subreddit", "")).lower()
            if subreddit and subreddit not in SEATTLE_SUBREDDITS:
                wrong_city += 1
                continue

            # Deduplicate by record ID
            rec_id = str(rec.get("id", ""))
            if rec_id and rec_id in seen_ids:
                skipped_count += 1
                continue
            if rec_id:
                seen_ids.add(rec_id)

            all_records.append(clean_record(rec))
            new_count += 1

        print(f"    Added:       {new_count:,}")
        print(f"    Duplicates:  {skipped_count:,}")
        if wrong_city > 0:
            print(f"    Wrong city:  {wrong_city:,} (skipped)")

    print(f"\n  Total Seattle records merged: {len(all_records):,}")

    # Save as JSON Lines — one record per line
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Saved -> {OUTPUT_FILE}")

    # Quick summary
    posts    = sum(1 for r in all_records if r.get("title"))
    comments = sum(1 for r in all_records if r.get("body") and not r.get("title"))
    print(f"\n  Posts:    {posts:,}")
    print(f"  Comments: {comments:,}")
    print("\n  Done. Now run reddit_nlp.py")


if __name__ == "__main__":
    main()
