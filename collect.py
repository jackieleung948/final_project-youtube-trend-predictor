from dotenv import load_dotenv
import os
import googleapiclient.discovery
import sqlite3

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

def main():
    api_service_name = "youtube"
    api_version = "v3"

    # Build client using API key
    # https://developers.google.com/youtube/v3/docs
    youtube = googleapiclient.discovery.build(
        api_service_name, 
        api_version, 
        developerKey=API_KEY
    )

    # Step 1: Search for cooking videos to get video IDs
    # https://developers.google.com/youtube/v3/docs/search/list
    request = youtube.search().list(
        part="snippet",
        q="cooking recipe",
        type="video",
        maxResults=50,
        order="date"
    )
    response = request.execute()
    video_ids = [item["id"]["videoId"] for item in response["items"]]


    # Step 2: Get video details 
    # https://developers.google.com/youtube/v3/docs/videos/list
    request = youtube.videos().list(
        part="snippet",
        id=",".join(video_ids)
    )
    response = request.execute()

    # Only keep relevant fields for analysis
    video_data = []
    for item in response["items"]:
        video_data.append({
            "video_id": item["id"],
            "title": item["snippet"]["title"],
            "published_at": item["snippet"]["publishedAt"],
            "description": item["snippet"]["description"],
            "channel_title": item["snippet"]["channelTitle"],
            "channel_id": item["snippet"]["channelId"],
            "category_id": item["snippet"]["categoryId"],
        })

    # Save to SQL database
    conn = sqlite3.connect("youtube_trends.db")
    cursor = conn.cursor()
    for video in video_data:
        cursor.execute('''
            INSERT OR IGNORE INTO videos (video_id, title, published_at, description, channel_title, channel_id, category_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (video["video_id"], video["title"], video["published_at"], video["description"], video["channel_title"], video["channel_id"], video["category_id"]))
        
    conn.commit()
    conn.close()

    print("Data saved to videos table in youtube_trends.db")

if __name__ == "__main__":
    main()