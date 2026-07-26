import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

def main():
    conn = sqlite3.connect("youtube_trends.db")

    # Read ARIMA model predictions from SQLite database
    df = pd.read_sql_query("SELECT "
                           "keyword, "
                           "hour_bucket, "
                           "predicted_frequency, "
                           "is_trending "
                           "FROM arima_predictions "
                           "ORDER BY keyword, hour_bucket", conn)

    # Read LSTM model predictions from SQLite database
    lstm_df = pd.read_sql_query("SELECT "
                                "keyword, "
                                "hour_bucket, "
                                "predicted_frequency, "
                                "is_trending "
                                "FROM lstm_predictions "
                                "ORDER BY keyword, hour_bucket", conn)

    # Merge ARIMA and LSTM predictions on keyword and hour_bucket
    merged_df = pd.merge(df, lstm_df, on=['keyword', 'hour_bucket'], suffixes=('_arima', '_lstm'))

    # Weighted average of ARIMA and LSTM predictions (weights can be adjusted based on model performance)
    weight_arima = 0.5
    weight_lstm = 1 - weight_arima
    merged_df['predicted_frequency'] = (merged_df['predicted_frequency_arima'] * weight_arima) + (merged_df['predicted_frequency_lstm'] * weight_lstm)

    # Flag keywords as trending if both models predict them as trending
    merged_df['is_trending'] = merged_df['is_trending_arima'].astype(bool) & merged_df['is_trending_lstm'].astype(bool)

    # Save the combined predictions to a new table in the SQLite database
    merged_df.to_sql("hybrid_predictions", conn, if_exists="replace", index=False)

    # Visualize the top 10 trending keywords over time as a line chart
    top_keywords = merged_df[merged_df['is_trending']].groupby('keyword')['predicted_frequency'].sum().nlargest(10).index
    pivot_df = merged_df[merged_df['keyword'].isin(top_keywords)].pivot(index='hour_bucket', columns='keyword', values='predicted_frequency')
    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind='line', title='Top 10 Trending Keywords Over Time (Hybrid Model)', figsize=(12, 6))
    plt.xlabel('Hour Bucket')
    plt.ylabel('Predicted Frequency')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("hybrid_predictions.png")
    plt.close()
    print("Hybrid predictions chart saved to hybrid_predictions.png")

    # Visualize the top 10 trending keywords as a bar chart
    plt.figure(figsize=(12, 6))
    top10 = merged_df[merged_df['is_trending']].groupby('keyword')['predicted_frequency'].sum().nlargest(10)
    top10.plot(kind='barh', title='Top 10 Trending Keywords (Hybrid Model)', figsize=(12, 6), legend=False)
    plt.xlabel('Predicted Frequency')
    plt.ylabel('Keyword')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig("hybrid_top10_predictions.png")
    plt.close()
    print("Hybrid top 10 predictions chart saved to hybrid_top10_predictions.png")

    conn.close()

if __name__ == "__main__":
    main()