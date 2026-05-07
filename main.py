from dotenv import load_dotenv
import os
import googleapiclient.discovery

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

    # Search for cooking videos
    # https://developers.google.com/youtube/v3/docs/search/list
    request = youtube.search().list(
        part="snippet",
        q="smash burger recipe",
        type="video",
        maxResults=5,
        order="date"
    )
    response = request.execute()

    print(response)

if __name__ == "__main__":
    main()