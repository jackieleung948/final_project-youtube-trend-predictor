import sqlite3
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(42)  # For reproducibility
np.random.seed(42)  # For reproducibility

# Code from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def main():
    conn = sqlite3.connect("youtube_trends.db")

    # Find keywords appearing in at least 4 unique videos
    video_filter_df = pd.read_sql_query(
        """
        SELECT keyword
        FROM keywords
        WHERE collected_at >= '2026-07-26T14:22:00'
        GROUP BY keyword
        HAVING COUNT(DISTINCT video_id) > 3
        """, conn
    )
    valid_by_video_count = video_filter_df["keyword"].tolist()

    # Read from SQLite database to get keywords and their associated frequency
    # Temporary: Filtered to keywords collected after 2026-07-17 18:00:00 to validate keyword filtering approach for ARIMA modeling
    # namely additional custom stopwords and low relevanace score (<0.5)
    # Temporary: Updated date filter to 2026-07-26T14:22:00 for keyword deduplication and normalization testing
    df = pd.read_sql_query("SELECT "
                           "keyword,"
                           "STRFTIME('%Y-%m-%d %H', collected_at) as hour_bucket, "
                           "COUNT(*) AS frequency "
                           "FROM keywords "
                           "WHERE collected_at >= '2026-07-26T14:22:00' "
                           "GROUP BY keyword, hour_bucket "
                           "ORDER BY keyword, hour_bucket", conn)

    # Filter to keywords with at least 10 hour buckets
    keyword_counts = df.groupby("keyword")["hour_bucket"].nunique()
    valid_keywords = keyword_counts[keyword_counts >= 10].index
    df = df[df["keyword"].isin(valid_keywords)]
    print(
        f"Filtered to {len(valid_keywords)} keywords with at least 10 hour buckets for hybrid modeling.")
    print(valid_keywords[:10])

    # Filter to keywords appearing in >3 distinct videos
    df = df[df["keyword"].isin(valid_by_video_count)]
    print(
        f"Filtered to {len(valid_by_video_count)} keywords appearing in at least 3 distinct videos.")

    # Limit to top 50 keywords by total frequency for hybrid modeling
    top_keywords = df.groupby("keyword")["frequency"].sum().nlargest(50).index
    df = df[df["keyword"].isin(top_keywords)]
    print("Filtered to top 50 keywords by total frequency for hybrid modeling:")

    # Following approach from https://doi.org/10.1016/S0925-2312(01)00702-0

    predictions = []
    for keyword in top_keywords:
        # Fit ARIMA on the frequency series for the keyword
        keyword_df = df[df["keyword"] == keyword].copy()
        keyword_df["hour_bucket"] = pd.to_datetime(
            keyword_df["hour_bucket"], format='%Y-%m-%d %H')
        keyword_df = keyword_df.set_index("hour_bucket").asfreq('h').fillna(0)

        # Get ARIMA's fitted values
        try:
            arima_model = ARIMA(keyword_df["frequency"], order=(1, 1, 1))
            arima_fitted = arima_model.fit()

            # Calculate residuals (actual - fitted)
            residuals = (keyword_df["frequency"] -
                         arima_fitted.fittedvalues).values

            # Train LSTM on the residuals
            n_steps = 6
            X, y = split_sequence(residuals, n_steps)
            if len(X) == 0:
                print(f"Not enough data to train LSTM for keyword: {keyword}")
                continue
            X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM
            model = Sequential()
            model.add(Input(shape=(n_steps, 1)))
            model.add(LSTM(50, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=200, verbose=0)

            # ARIMA forecast for the next 6 hours
            arima_forecast = arima_fitted.forecast(steps=6).values

            # Train LSTM on the residuals and forecast the next 6 hours
            input_seq = residuals[-n_steps:].tolist()
            residual_forecast = []
            for _ in range(6):
                x_input = np.array(
                    input_seq[-n_steps:]).reshape((1, n_steps, 1))
                next_residual = model.predict(x_input, verbose=0)[0][0]
                residual_forecast.append(next_residual)
                input_seq.append(next_residual)

            # Final forecast = ARIMA forecast + LSTM residual forecast
            for i in range(6):
                final_forecast = max(0.0, float(arima_forecast[i] + residual_forecast[i]))
                predictions.append({
                    "keyword": keyword,
                    "hour_bucket": (pd.to_datetime(keyword_df.index[-1]) + pd.Timedelta(hours=i+1)).strftime('%Y-%m-%d %H'),
                    "predicted_frequency": final_forecast
                })

        except Exception as e:
            print(f"Hybrid model failed for keyword '{keyword}': {e}")
            continue

    # Flag keywords where forecast > current average frequency
    for prediction in predictions:
        keyword = prediction["keyword"]
        current_avg_frequency = df[df["keyword"] == keyword]["frequency"].mean()
        prediction["is_trending"] = prediction["predicted_frequency"] > current_avg_frequency

    # Save to database
    predictions_df = pd.DataFrame(predictions)
    if predictions_df.empty:
        print("No predictions generated.")
        conn.close()
        return

    predictions_df.to_sql("residual_hybrid_predictions", conn, if_exists="replace", index=False)
    print("Residual hybrid predictions saved to residual_hybrid_predictions table")

    top10 = predictions_df[predictions_df["is_trending"]].groupby(
        "keyword")["predicted_frequency"].sum().nlargest(10)
    print("\n=== TOP 10 RESIDUAL HYBRID PREDICTED TRENDING KEYWORDS ===")
    print(top10)

    # Visualize the top 10 predicted trending keywords over time as a line chart
    plt.figure(figsize=(12, 6))
    top_kw = predictions_df[predictions_df['is_trending']].groupby('keyword')['predicted_frequency'].sum().nlargest(10).index
    pivot_df = predictions_df[predictions_df['keyword'].isin(top_kw)].pivot(
        index='hour_bucket', columns='keyword', values='predicted_frequency')
    pivot_df.plot(kind='line', title='Top 10 Trending Keywords Over Time (Residual Hybrid)', ax=plt.gca())
    plt.xlabel('Hour Bucket')
    plt.ylabel('Predicted Frequency')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("residual_hybrid_line.png")
    plt.close()

    # Visualize the top 10 predicted trending keywords as a bar chart
    plt.figure(figsize=(12, 6))
    top10.plot(kind='barh', title='Top 10 Trending Keywords (Residual Hybrid)', legend=False, ax=plt.gca())
    plt.xlabel('Predicted Frequency')
    plt.ylabel('Keyword')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig("residual_hybrid_bar.png")
    plt.close()
    print("Charts saved")

    conn.close()

if __name__ == "__main__":
    main()