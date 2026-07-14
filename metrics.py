import pandas as pd
from dotenv import load_dotenv
import os
import googleapiclient.discovery
import sqlite3

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")


def main():
    api_service_name = "youtube"
    api_version = "v3"

    # Read from SQLite database to get video IDs
    conn = sqlite3.connect("youtube_trends.db")
    df = pd.read_sql_query(
        "SELECT video_id, published_at FROM videos ORDER BY rowid DESC LIMIT 50", conn)
    video_ids = df["video_id"].tolist()

    # Build client using API key
    # https://developers.google.com/youtube/v3/docs
    youtube = googleapiclient.discovery.build(
        api_service_name,
        api_version,
        developerKey=API_KEY
    )

    # Get video details from YouTube API
    # https://developers.google.com/youtube/v3/docs/videos/list
    request = youtube.videos().list(
        part="statistics",
        id=",".join(video_ids)
    )
    response = request.execute()

    # Only keep relevant fields for analysis
    video_data = []
    for item in response["items"]:
        video_data.append({
            "video_id": item["id"],
            "view_count": int(item["statistics"].get("viewCount", 0)),
            "like_count": int(item["statistics"].get("likeCount", 0)),
            "comment_count": int(item["statistics"].get("commentCount", 0)),
        })

    # Add published_at to video_data
    published_at_map = dict(zip(df["video_id"], df["published_at"]))
    for item in video_data:
        item["published_at"] = published_at_map.get(item["video_id"], None)

    # Calculate view velocity and engagement ratio

    # View Velocity = view count / hours since published
    # View Velocity measures how quickly a video is gaining views, which can indicate how viral it is. A high view velocity suggests that the video is rapidly gaining popularity, while a low view velocity may indicate slower growth.
    view_velocity = []
    for item in video_data:
        published_at = pd.to_datetime(item["published_at"])
        hours_since_published = (pd.Timestamp.now(
            tz='UTC') - published_at).total_seconds() / 3600
        if hours_since_published > 0:
            velocity = item["view_count"] / hours_since_published
        else:
            velocity = 0
        view_velocity.append(velocity)

    # Engagement Ratio = (like count + comment count) / view count
    # Engagement Ratio measures how engaged viewers are with the content. A high engagement ratio indicates that viewers are actively liking and commenting on the video, which can be a sign of strong audience interest and interaction. Conversely, a low engagement ratio may suggest that viewers are passively consuming the content without much interaction.
    engagement_ratio = []
    for item in video_data:
        if item["view_count"] > 0:
            ratio = (item["like_count"] + item["comment_count"]
                     ) / item["view_count"]
        else:
            ratio = 0
        engagement_ratio.append(ratio)

    # Save metrics to SQLite database
    cursor = conn.cursor()
    for i, item in enumerate(video_data):
        cursor.execute('''
            INSERT INTO metrics (video_id, view_velocity, collected_at, engagement_ratio, view_count, like_count, comment_count)
            VALUES (?, ?, datetime('now'), ?, ?, ?, ?)
        ''', (item["video_id"], view_velocity[i], engagement_ratio[i], item["view_count"], item["like_count"], item["comment_count"]))

    print("Data saved to metrics table in youtube_trends.db")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
