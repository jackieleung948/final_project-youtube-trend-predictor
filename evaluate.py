import argparse
import sqlite3
import pandas as pd
import numpy as np
from config import (DB_PATH, DATA_START, TRAIN_END, TEST_END, MIN_HOUR_BUCKETS, BUCKET_HOURS, MIN_VIDEO_COUNT, TOP_N)


# Get the data from the database and filter it based on the configuration
def get_filtered_data():
    conn = sqlite3.connect(DB_PATH)

    # Find keywords appearing in at least MIN_VIDEO_COUNT unique videos
    video_keyword_df = pd.read_sql_query(
        """
        SELECT k.keyword,
        STRFTIME('%Y-%m-%d %H', v.published_at) as hour_bucket,
        COUNT(DISTINCT k.video_id) AS frequency
        FROM (SELECT DISTINCT keyword, video_id FROM keywords) k
        JOIN videos v ON k.video_id = v.video_id
        WHERE v.published_at >= ?
        GROUP BY keyword, hour_bucket
        ORDER BY keyword, hour_bucket
        """, conn, params=(DATA_START,)
    )

    train_df = video_keyword_df[video_keyword_df["hour_bucket"] < TRAIN_END]

    hours_per_kw = train_df.groupby("keyword")["hour_bucket"].nunique()
    valid_hours = set(hours_per_kw[hours_per_kw >= MIN_HOUR_BUCKETS].index)

    valid_video_df = pd.read_sql_query(
        """
        SELECT k.keyword, 
        COUNT(DISTINCT k.video_id) AS video_count
        FROM keywords k
        JOIN videos v ON k.video_id = v.video_id
        WHERE v.published_at >= ? AND STRFTIME('%Y-%m-%d %H', v.published_at) < ?
        GROUP BY keyword
        """, conn, params=(DATA_START, TRAIN_END)
    )
    valid_videos = set(valid_video_df[valid_video_df["video_count"] >= MIN_VIDEO_COUNT]["keyword"])

    keep = valid_hours & valid_videos
    train_df = train_df[train_df["keyword"].isin(keep)]
    top = train_df.groupby("keyword")["frequency"].sum().nlargest(TOP_N).index

    # Hourly series for each keyword, filling missing hours with 0 frequency
    series = {}
    full_range = pd.date_range(
        pd.to_datetime(DATA_START).floor("h"),
        pd.to_datetime(TEST_END, format='%Y-%m-%d %H'),
        freq ="h", 
        inclusive="left"
    )
    keep_df = video_keyword_df[video_keyword_df["keyword"].isin(top)]
    for keyword, group in keep_df.groupby("keyword"):
        index = pd.to_datetime(group["hour_bucket"], format='%Y-%m-%d %H')
        hourly = pd.Series(group["frequency"].values, index=index)
        hourly = hourly.reindex(full_range, fill_value=0).astype(float)  # Fill missing hours with 0
        series[keyword] = hourly.resample(f"{BUCKET_HOURS}h", origin="start").sum()
        print(keyword, len(index), index.is_unique, len(full_range))

    conn.close()
    return series

if __name__ == "__main__":
    s = get_filtered_data()
    print(f"Filtered data contains {len(s)} keywords.")
    for keyword, ser in list(s.items())[:10]:  # Print stats for the first 10 keywords
        print(f"{keyword:25s} mean={ser.mean():.2f}, max={ser.max()}, nonzero={int((ser > 0).sum())}/{len(ser)}")