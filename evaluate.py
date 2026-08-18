import argparse
import sqlite3
import pandas as pd
import numpy as np

# Configuration
DB_PATH = "youtube_trends.db"
DATA_START = '2026-07-26T14:22:00' 
TRAIN_END = '2026-08-16T14:22:00'  # End of training data for backtesting
TEST_END = '2026-08-23T14:22:00'  # End of testing data for backtesting
HORIZON = 6 # Number of hours to forecast ahead for backtesting
STEP = 6
MIN_HOUR_BUCKETS = 10  # Minimum number of hour buckets required for a keyword to be considered for backtesting
MIN_VIDEO_COUNT = 4  # Minimum number of distinct videos required for a keyword to be considered for backtesting
TOP_N = 50  # Limit to top N keywords by total frequency for backtesting
LABELS_CSV = "backtest_labels.csv"  # CSV file to save the backtesting labels (actual frequencies)

# Get the data from the database and filter it based on the configuration
def get_filtered_data():
    conn = sqlite3.connect(DB_PATH)

    # Find keywords appearing in at least MIN_VIDEO_COUNT unique videos
    video_keyword_df = pd.read_sql_query(
        f"""
        SELECT keyword
        STRFTIME('%Y-%m-%d %H', collected_at) as hour_bucket,
        COUNT(DISTINCT video_id) AS frequency
        FROM keywords
        WHERE collected_at >= '{DATA_START}'
        GROUP BY keyword, hour_bucket
        ORDER BY keyword, hour_bucket
        """, conn, params={"DATA_START": DATA_START}
    )

    train_df = video_keyword_df[video_keyword_df["hour_bucket"] < TRAIN_END]

    hours_per_kw = train_df.groupby("keyword")["hour_bucket"].nunique()
    valid_hours = set(hours_per_kw[hours_per_kw >= MIN_HOUR_BUCKETS].index)

    valid_video_df = pd.read_sql_query(
        f"""
        SELECT keyword, 
        COUNT(DISTINCT video_id) AS video_count
        FROM keywords
        WHERE collected_at >= '{DATA_START}' AND STRFTIME('%Y-%m-%d %H', collected_at) < '{TRAIN_END}'
        GROUP BY keyword
        """, conn, params={"DATA_START": DATA_START, "TRAIN_END": TRAIN_END}
    )
    valid_videos = set(valid_video_df[valid_video_df["video_count"] >= MIN_VIDEO_COUNT]["keyword"])

    keep = valid_hours & valid_videos
    train_df = train_df[train_df["keyword"].isin(keep)]
    top = train_df.groupby("keyword")["frequency"].sum().nlargest(TOP_N).index
    train_df = train_df[train_df["keyword"].isin(top)]

    # Hourly series for each keyword, filling missing hours with 0 frequency
    series = {}
    full_range = pd.date_range(
        pd.to_datetime(DATA_START).floor("H"),
        pd.to_datetime(TEST_END, format='%Y-%m-%d %H'),
        freq ="H", 
        inclusive="left"
    )

    for keyword, group in train_df.groupby("keyword"):
        group = group.set_index("hour_bucket").reindex(full_range, fill_value=0)
        series[keyword] = group["frequency"]