import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():

    conn = sqlite3.connect("youtube_trends.db")

    DATA_START = '2026-07-26T14:22:00'


    # Find keywords appearing in at least 4 unique videos
    video_filter_df = pd.read_sql_query(
        """
        SELECT keyword
        FROM keywords
        WHERE collected_at >= '{DATA_START}'
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
                           "COUNT(DISTINCT video_id) AS frequency "
                           "FROM keywords "
                           f"WHERE collected_at >= '{DATA_START}' "
                           "GROUP BY keyword, hour_bucket "
                           "ORDER BY keyword, hour_bucket", conn)

    # Filter to keywords with at least 10 hour buckets
    keyword_counts = df.groupby("keyword")["hour_bucket"].nunique()
    valid_keywords = keyword_counts[keyword_counts >= 10].index
    df = df[df["keyword"].isin(valid_keywords)]
    print(
        f"Filtered to {len(valid_keywords)} keywords with at least 10 hour buckets for ARIMA modeling.")
    print(valid_keywords[:10])

    # Filter to keywords appearing in >3 distinct videos
    df = df[df["keyword"].isin(valid_by_video_count)]
    print(
        f"Filtered to {len(valid_by_video_count)} keywords appearing in at least 4 distinct videos.")

    # Limit to top 50 keywords by total frequency for ARIMA modeling
    top_keywords = df.groupby("keyword")["frequency"].sum().nlargest(50).index
    df = df[df["keyword"].isin(top_keywords)]
    print("Filtered to top 50 keywords by total frequency for ARIMA modeling:")

    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
    # https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html

    # For each keyword, fit an ARIMA model to predict the next 6 hours' frequency
    predictions = []
    avg_frequency = {}
    for keyword in df["keyword"].unique():
        keyword_df = df[df["keyword"] == keyword].copy()
        keyword_df["hour_bucket"] = pd.to_datetime(
            keyword_df["hour_bucket"], format='%Y-%m-%d %H')
        keyword_df = keyword_df.set_index("hour_bucket").asfreq('h').fillna(0)
        avg_frequency[keyword] = keyword_df["frequency"].mean()

        try:
            model = ARIMA(keyword_df["frequency"], order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=6)
            for i, value in enumerate(forecast):
                predictions.append({
                    "keyword": keyword,
                    "hour_bucket": (pd.to_datetime(keyword_df.index[-1]) + pd.Timedelta(hours=i+1)).strftime('%Y-%m-%d %H'),
                    "predicted_frequency": value
                })
        except Exception as e:
            print(f"ARIMA model failed for keyword '{keyword}': {e}")
            continue

    # Flag keywords where forecast > current average frequency
    for prediction in predictions:
        keyword = prediction["keyword"]
        current_avg_frequency = avg_frequency[prediction["keyword"]]
        prediction["is_trending"] = prediction["predicted_frequency"] > current_avg_frequency

    # Save predictions to SQLite database
    predictions_df = pd.DataFrame(predictions)

    # Check if predictions list is empty before saving to database
    if predictions_df.empty:
        print("No predictions were generated. Exiting without saving to database.")
        conn.close()
        return

    predictions_df.to_sql("arima_predictions", conn,
                          if_exists="replace", index=False)
    print("ARIMA predictions saved to arima_predictions table in youtube_trends.db")

    # Visualize the top 10 predicted trending keywords as a bar chart
    top10 = predictions_df[predictions_df["is_trending"]].groupby(
        "keyword")["predicted_frequency"].sum().nlargest(10)
    top10.plot(kind="barh", title="ARIMA Predicted Trending Keywords",
               figsize=(12, 6), legend=False)
    plt.xlabel("Predicted Frequency")
    plt.ylabel("Keyword")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig("arima_predictions.png")
    print("ARIMA predictions chart saved to arima_predictions.png")

    conn.close()


if __name__ == "__main__":
    main()
