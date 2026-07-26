import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

def main():

    # Read from SQLite database to get keywords and their associated metrics and calculate trend scores
    conn = sqlite3.connect("youtube_trends.db")
    df = pd.read_sql_query("SELECT "
                           "keywords.keyword, "
                           "STRFTIME('%Y-%m-%d %H', keywords.collected_at) as hour_bucket, "
                           "COUNT(keywords.keyword) AS keyword_count, "
                           "AVG(metrics.view_velocity) as avg_view_velocity, "
                           "AVG(metrics.engagement_ratio) as avg_engagement_ratio, "
                           "(COUNT(keywords.keyword) * AVG(metrics.view_velocity) * AVG(metrics.engagement_ratio)) AS trend_score "
                           "FROM keywords JOIN metrics ON keywords.video_id=metrics.video_id " 
                           "WHERE collected_at >= '2026-07-26T14:22:00' "
                           "GROUP BY keywords.keyword, hour_bucket ORDER BY trend_score DESC", conn)

    # Calculate growth rate between hours for each keyword
    df = df.sort_values(["keyword", "hour_bucket"])
    df["growth_rate"] = df.groupby("keyword")["keyword_count"].pct_change()
    df["growth_rate"] = df["growth_rate"].fillna(0)
    
    # Output ranked list of keywords with their trend scores to database
    df.to_sql("keyword_trends", conn, if_exists="replace", index=False)

    # Visualize the top 10 trending keywords over time as a line chart
    top_keywords = df.groupby("keyword")["trend_score"].sum().nlargest(10).index
    pivot_df = df[df["keyword"].isin(top_keywords)].pivot(index="hour_bucket", columns="keyword", values="trend_score")
    pivot_df.plot(kind="line", title="Top 10 Trending Keywords Over Time", figsize=(12, 6))
    plt.xlabel("Hour Bucket")
    plt.ylabel("Trend Score")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("keyword_trends.png")

    # Visualize the top 10 trending keywords with their trend scores as a bar chart
    top10 = df.groupby("keyword")["trend_score"].sum().nlargest(10)
    top10.plot(kind="barh", title="Top 10 Trending Keywords Now", figsize=(12, 6), legend=False)
    plt.xlabel("Trend Score")
    plt.ylabel("Keyword")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig("top10_trends.png")

    conn.close()

if __name__ == "__main__":
    main()