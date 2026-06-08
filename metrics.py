import pandas as pd

# Read CSV file
df = pd.read_csv("youtube_videos.csv")

# Calculate view velocity and engagement ratio

# View Velocity = view count / hours since published
view_velocity = df["view_count"] / ((pd.to_datetime("now", utc=True) - pd.to_datetime(df["published_at"], utc=True)).dt.total_seconds() / 3600)
voew_velocity = view_velocity.fillna(0)

# Engagement Ratio = (like count + comment count) / view count
engagement_ratio = (df["like_count"] + df["comment_count"]) / df["view_count"]
engagement_ratio = engagement_ratio.fillna(0)

# Add new columns to DataFrame
df["view_velocity"] = view_velocity
df["engagement_ratio"] = engagement_ratio

# Save updated DataFrame to new CSV
df.to_csv("youtube_videos_with_metrics.csv", index=False)
print("Metrics calculated and saved to youtube_videos_with_metrics.csv")
