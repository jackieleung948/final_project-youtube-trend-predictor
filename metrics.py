import pandas as pd

# Read CSV file
df = pd.read_csv("youtube_videos.csv")

# Calculate view velocity and engagement ratio

# View Velocity = view count / hours since published
# View Velocity measures how quickly a video is gaining views, which can indicate how viral it is. A high view velocity suggests that the video is rapidly gaining popularity, while a low view velocity may indicate slower growth.
view_velocity = df["view_count"] / ((pd.to_datetime("now", utc=True) -
                                    pd.to_datetime(df["published_at"], utc=True)).dt.total_seconds() / 3600)
view_velocity = view_velocity.replace(
    [float('inf'), float('-inf')], 0).fillna(0)


# Engagement Ratio = (like count + comment count) / view count
# Engagement Ratio measures how engaged viewers are with the content. A high engagement ratio indicates that viewers are actively liking and commenting on the video, which can be a sign of strong audience interest and interaction. Conversely, a low engagement ratio may suggest that viewers are passively consuming the content without much interaction.
engagement_ratio = (df["like_count"] + df["comment_count"]) / df["view_count"]
engagement_ratio = engagement_ratio.fillna(0)

# Add new columns to DataFrame
df["view_velocity"] = view_velocity
df["engagement_ratio"] = engagement_ratio

# Save updated DataFrame to new CSV
df.to_csv("youtube_videos_with_metrics.csv", index=False)
print("Metrics calculated and saved to youtube_videos_with_metrics.csv")
