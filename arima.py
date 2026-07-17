import pandas as pd
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():

    # Read from SQLite database to get keywords and their associated frequency
    conn = sqlite3.connect("youtube_trends.db")
    df = pd.read_sql_query("SELECT "
                           "keyword,"
                           "STRFTIME('%Y-%m-%d %H', collected_at) as hour_bucket, "
                           "COUNT(*) AS frequency "
                           "FROM keywords GROUP BY keyword, hour_bucket ORDER BY keyword, hour_bucket", conn)
    
    # Filter to keywords with at least 10 hour buckets
    keyword_counts = df.groupby("keyword")["hour_bucket"].nunique()
    valid_keywords = keyword_counts[keyword_counts >= 10].index
    df = df[df["keyword"].isin(valid_keywords)]
    print(f"Filtered to {len(valid_keywords)} keywords with at least 10 hour buckets for ARIMA modeling.")
    print(valid_keywords[:10])

    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
    # https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html

    # For each keyword, fit an ARIMA model to predict the next 6 hours' frequency
    predictions = []
    for keyword in df["keyword"].unique():
        keyword_df = df[df["keyword"] == keyword].copy()
        keyword_df["hour_bucket"] = pd.to_datetime(keyword_df["hour_bucket"], format='%Y-%m-%d %H')
        keyword_df = keyword_df.set_index("hour_bucket").asfreq('h').fillna(0)
        
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
        current_avg_frequency = df[df["keyword"] == keyword]["frequency"].mean()
        prediction["is_trending"] = prediction["predicted_frequency"] > current_avg_frequency

    
    # Save predictions to SQLite database
    predictions_df = pd.DataFrame(predictions)

    # Check if predictions list is empty before saving to database
    if predictions_df.empty:
        print("No predictions were generated. Exiting without saving to database.")
        conn.close()
        return
    
    predictions_df.to_sql("keyword_predictions", conn, if_exists="replace", index=False)
    print("ARIMA predictions saved to keyword_predictions table in youtube_trends.db")

    # Print top 10 predicted trending keywords for validation
    top10 = predictions_df[predictions_df["is_trending"]].groupby("keyword")["predicted_frequency"].sum().nlargest(10)
    print("\n=== TOP 10 PREDICTED TRENDING KEYWORDS ===")
    print(top10)

    conn.close()

if __name__ == "__main__":
    main()