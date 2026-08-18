from dotenv import load_dotenv
import os
import sqlite3
from datetime import datetime, timedelta, timezone
import googleapiclient.discovery
from googleapiclient.errors import HttpError

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

DB_PATH = "youtube_trends.db"
GAP_START = datetime(2026, 7, 27, 9, tzinfo=timezone.utc)
GAP_END = datetime(2026, 8, 18, 9, tzinfo=timezone.utc)
MAX_CALLS = 75

def rfc3339_format(dt):
    """Format a datetime object to RFC 3339 format."""
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

def main():
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=API_KEY
    )

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS backfill_log (hour TEXT PRIMARY KEY, n_videos INTEGER, done_at TEXT)")
    conn.commit()

    done = {row[0] for row in cur.execute("SELECT hour FROM backfill_log")}

    calls = 0
    inserted_total = 0
    hour = GAP_START
    while hour < GAP_END:
        key = rfc3339_format(hour)
        if key in done:
            hour += timedelta(hours=1)
            continue
        if calls >= MAX_CALLS:
            print(f"Reached max calls ({MAX_CALLS}). Stopping backfill.")
            break

        try:
            resp = youtube.search().list(
                part="snippet",
                q="cooking recipe",
                type="video",
                maxResults=50,
                order="date",
                publishedAfter=rfc3339_format(hour),
                publishedBefore=rfc3339_format(hour + timedelta(hours=1))
            ).execute()
        except HttpError as e:
            print(f"HTTP error occurred: {e}")
            break
        calls += 1

        video_ids = [item["id"]["videoId"] for item in resp.get("items", [])]

        video_data = []
        if video_ids:
            vresp = youtube.videos().list(
                part="snippet",
                id=",".join(video_ids)
            ).execute()

            for item in vresp.get("items", []):
                video_data.append({
                    "video_id": item["id"],
                    "title": item["snippet"]["title"],
                    "published_at": item["snippet"]["publishedAt"],
                    "description": item["snippet"]["description"],
                    "channel_title": item["snippet"]["channelTitle"],
                    "channel_id": item["snippet"]["channelId"],
                    "category_id": item["snippet"].get("categoryId"),
                })

        for video in video_data:
            cur.execute('''
                INSERT OR IGNORE INTO videos (video_id, title, published_at, description, channel_title, channel_id, category_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (video["video_id"], video["title"], video["published_at"], video["description"], video["channel_title"], video["channel_id"], video["category_id"]))
        cur.execute('''
                INSERT INTO backfill_log (hour, n_videos, done_at) VALUES (?, ?, ?)
            ''', (key, len(video_data), datetime.now(timezone.utc).isoformat()))
        conn.commit()
        inserted_total += len(video_data)
        print(f"Backfilled hour {key}: {len(video_data)} videos inserted.")
        hour += timedelta(hours=1)

    total_hours = int((GAP_END - GAP_START).total_seconds() // 3600)
    logged = cur.execute("SELECT COUNT(*) FROM backfill_log").fetchone()[0]
    remaining = total_hours - logged
    print(f"Total hours: {total_hours}, Logged hours: {logged}, Remaining hours: {remaining}, Total videos inserted: {inserted_total}.")
    if remaining == 0:
        print("All hours have been backfilled.")
    conn.close()

if __name__ == "__main__":
    main()