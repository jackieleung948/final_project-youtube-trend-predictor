from dotenv import load_dotenv
import os
import googleapiclient.discovery
import pandas as pd


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
        part="snippet,statistics",
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
            "view_count": int(item["statistics"].get("viewCount", 0)),
            "like_count": int(item["statistics"].get("likeCount", 0)),
            "comment_count": int(item["statistics"].get("commentCount", 0))
        })
    print(video_data)

    # Save to CSV
    df = pd.DataFrame(video_data)
    df.to_csv("youtube_videos.csv", index=False)
    print("Data saved to youtube_videos.csv")

if __name__ == "__main__":
    main()