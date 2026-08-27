import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from config import DB_PATH, HORIZON_BUCKETS, BUCKET_HOURS
from data import get_filtered_data
from models import forecast_arima

def main():

    series = get_filtered_data()
    predictions = []
    for keyword, ser in series.items():
        fc = forecast_arima(ser, HORIZON_BUCKETS)
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
    conn = sqlite3.connect(DB_PATH)
    predictions_df.to_sql("arima_predictions", conn, if_exists="replace", index=False)
    conn.close()
    
    # Visualize the top 10 predicted trending keywords as a bar chart
    top10 = predictions_df[predictions_df["is_trending"]].groupby(
        "keyword")["predicted_frequency"].sum().nlargest(10)
    plt.figure(figsize=(12, 6)) 
    top10.plot(kind="barh", title="ARIMA Predicted Trending Keywords",
               figsize=(12, 6), legend=False)
    plt.xlabel("Predicted Frequency")
    plt.ylabel("Keyword")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig("arima_predictions.png")
    print("ARIMA predictions chart saved to arima_predictions.png")
    plt.close()

if __name__ == "__main__":
    main()
