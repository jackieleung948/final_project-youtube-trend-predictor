import pandas as pd
import sqlite3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import numpy as np
import matplotlib.pyplot as plt

# Followed steps from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# https://www.tensorflow.org/tutorials/structured_data/time_series


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
    # Read from SQLite database to get keywords and their associated frequency
    conn = sqlite3.connect("youtube_trends.db")
    df = pd.read_sql_query("SELECT "
                           "keyword,"
                           "STRFTIME('%Y-%m-%d %H', collected_at) as hour_bucket, "
                           "COUNT(*) AS frequency "
                           "FROM keywords "
                           "WHERE collected_at >= '2026-07-26T14:22:00' "
                           "GROUP BY keyword, hour_bucket ORDER BY keyword, hour_bucket", conn)

    # Filter to keywords with at least 10 hour buckets
    keyword_counts = df.groupby("keyword")["hour_bucket"].nunique()
    valid_keywords = keyword_counts[keyword_counts >= 10].index
    df = df[df["keyword"].isin(valid_keywords)]
    print(
        f"Filtered to {len(valid_keywords)} keywords with at least 10 hour buckets for LSTM modeling.")
    print(valid_keywords[:10])

    # Limit to top 50 keywords by total frequency for LSTM modeling
    top_keywords = df.groupby("keyword")["frequency"].sum().nlargest(50).index
    df = df[df["keyword"].isin(top_keywords)]
    print("Filtered to top 50 keywords by total frequency for LSTM modeling:")

    # For each keyword, fit an LSTM model to predict the next 6 hours' frequency
    # Followed steps from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    predictions = []
    for keyword in df["keyword"].unique():
        keyword_df = df[df["keyword"] == keyword].copy()
        keyword_df["hour_bucket"] = pd.to_datetime(
            keyword_df["hour_bucket"], format='%Y-%m-%d %H')
        keyword_df = keyword_df.set_index("hour_bucket").asfreq('h').fillna(0)

        # Choose a number of time steps for the LSTM model. Using 6 time steps to match ARIMA's 6-hour forecast horizon
        n_steps = 6

        # Split the sequence into input/output pairs
        X, y = split_sequence(keyword_df["frequency"].values, n_steps)

        # Reshape input to be [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Define the LSTM model
        model = Sequential()
        model.add(Input(shape=(n_steps, 1)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Fit the model
        model.fit(X, y, epochs=200, verbose=0)

        # Make predictions for the next 6 hours
        # Followed steps from https://machinelearningmastery.com/multi-step-time-series-forecasting/
        # https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
        input_seq = keyword_df["frequency"].values[-n_steps:].tolist()
        forecast_horizons = 6

        for i in range(forecast_horizons):
            x_input = np.array(input_seq[-n_steps:]).reshape((1, n_steps, 1))
            next_pred = model.predict(x_input, verbose=0)[0][0]

            predictions.append({
                "keyword": keyword,
                "hour_bucket": (pd.to_datetime(keyword_df.index[-1]) + pd.Timedelta(hours=i+1)).strftime('%Y-%m-%d %H'),
                "predicted_frequency": next_pred
            })

            input_seq.append(next_pred)

    # Flag keywords where forecast > current average frequency
    for prediction in predictions:
        keyword = prediction["keyword"]
        current_avg_frequency = df[df["keyword"]
                                   == keyword]["frequency"].mean()
        prediction["is_trending"] = prediction["predicted_frequency"] > current_avg_frequency

    # Save to database
    predictions_df = pd.DataFrame(predictions)
    if predictions_df.empty:
        print("No predictions generated.")
        conn.close()
        return

    predictions_df.to_sql("lstm_predictions", conn,
                          if_exists="replace", index=False)
    print("LSTM predictions saved to lstm_predictions table")

    top10 = predictions_df[predictions_df["is_trending"]].groupby(
        "keyword")["predicted_frequency"].sum().nlargest(10)
    print("\n=== TOP 10 LSTM PREDICTED TRENDING KEYWORDS ===")
    print(top10)

    # Visualize the top 10 predicted trending keywords as a bar chart
    top10 = predictions_df[predictions_df["is_trending"]].groupby("keyword")["predicted_frequency"].sum().nlargest(10)
    top10.plot(kind="barh", title="LSTM Predicted Trending Keywords", figsize=(12, 6), legend=False)
    plt.xlabel("Predicted Frequency")
    plt.ylabel("Keyword")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig("lstm_predictions.png")
    print("LSTM predictions chart saved to lstm_predictions.png")

    conn.close()


if __name__ == "__main__":
    main()
