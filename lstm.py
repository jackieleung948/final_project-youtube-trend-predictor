import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from config import DB_PATH, HORIZON_BUCKETS, BUCKET_HOURS
from data import get_filtered_data
from models import fit_lstm, forecast_lstm

def main():
    series = get_filtered_data()
    predictions = []
    for keyword, ser in series.items():
        fitted = fit_lstm(ser.values)
        fc = forecast_lstm(fitted, ser.values, HORIZON_BUCKETS)
        if fc is None:
            continue
        mean_freq = ser.mean()
        for i, value in enumerate(fc, start=1):
            predictions.append({
                "keyword": keyword,
                "bucket": (ser.index[-1] + i * pd.Timedelta(hours=BUCKET_HOURS)).strftime("%Y-%m-%d %H"),
                "predicted_frequency": float(value),
                # Determine if the keyword is trending based on whether the predicted frequency exceeds the mean frequency
                "is_trending": bool(fc.mean() > mean_freq),
            })

    predictions_df = pd.DataFrame(predictions)
    if predictions_df.empty:
        print("No LSTM predictions generated")
        return
    conn = sqlite3.connect(DB_PATH)
    predictions_df.to_sql("lstm_predictions", conn, if_exists="replace", index=False)
    conn.close()

    top10 = predictions_df[predictions_df["is_trending"]].groupby(
        "keyword")["predicted_frequency"].sum().nlargest(10)
    print("\n=== TOP 10 LSTM PREDICTED TRENDING KEYWORDS ===")
    print(top10)

    # Visualize the top 10 predicted trending keywords as a bar chart
    plt.figure(figsize=(12, 6))
    top10.plot(kind="barh", title="LSTM Predicted Trending Keywords",
               figsize=(12, 6), legend=False)
    plt.xlabel("Predicted Frequency")
    plt.ylabel("Keyword")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig("lstm_predictions.png")
    print("LSTM predictions chart saved to lstm_predictions.png")
    plt.close()


if __name__ == "__main__":
    main()
