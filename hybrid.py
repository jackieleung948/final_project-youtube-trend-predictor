import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from data import get_filtered_data
from config import DB_PATH, HORIZON_BUCKETS
from models import forecast_arima, fit_lstm, forecast_lstm


def main():

    series = get_filtered_data()
    predictions = []
    for keyword, ser in series.items():
        arima_fc = forecast_arima(ser, HORIZON_BUCKETS)
        fitted_lstm = fit_lstm(ser.values)
        lstm_fc = forecast_lstm(fitted_lstm, ser.values, HORIZON_BUCKETS)
        if arima_fc is None or lstm_fc is None:
            continue
        hybrid_fc = 0.5 * arima_fc + 0.5 * lstm_fc
        mean_freq = ser.mean()
        for i, value in enumerate(hybrid_fc, start=1):
            predictions.append({
                "keyword": keyword,
                "bucket": (ser.index[-1] + i * (ser.index[1] - ser.index[0])).strftime("%Y-%m-%d %H"),
                "predicted_frequency": float(value),
                "is_trending": bool(hybrid_fc.mean() > mean_freq),
            })

    merged_df = pd.DataFrame(predictions)
    if merged_df.empty:
        print("No predictions available for visualization.")
        return
    conn = sqlite3.connect(DB_PATH)
    merged_df.to_sql("hybrid_predictions", conn,
                     if_exists="replace", index=False)
    conn.close()

    # Visualize the top 10 trending keywords as a bar chart
    plt.figure(figsize=(12, 6))
    top10 = merged_df[merged_df['is_trending']].groupby(
        'keyword')['predicted_frequency'].sum().nlargest(10)
    top10.plot(kind='barh', title='Top 10 Trending Keywords (Hybrid Model)',
               figsize=(12, 6), legend=False)
    plt.xlabel('Predicted Frequency')
    plt.ylabel('Keyword')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig("hybrid_bar.png")
    plt.close()
    print("Hybrid top 10 predictions chart saved to hybrid_bar.png")


if __name__ == "__main__":
    main()
