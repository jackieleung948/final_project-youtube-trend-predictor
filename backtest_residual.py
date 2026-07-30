
import sqlite3
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
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
    # Temporary: Filtered to keywords collected after 2026-07-17 18:00:00 to validate keyword filtering approach for backtesting (Residual)
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
            f"Filtered to {len(valid_keywords)} keywords with at least 10 hour buckets for backtesting (Residual).")
    print(valid_keywords[:10])
    
    # Filter to keywords appearing in >3 distinct videos
    df = df[df["keyword"].isin(valid_by_video_count)]
    print(
            f"Filtered to {len(valid_by_video_count)} keywords appearing in at least 3 distinct videos.")
    
    # Limit to top 50 keywords by total frequency for backtesting (Residual)
    top_keywords = df.groupby("keyword")["frequency"].sum().nlargest(50).index
    df = df[df["keyword"].isin(top_keywords)]
    print("Filtered to top 50 keywords by total frequency for backtesting (Residual):")

    # For each keyword, split the data into training and testing sets for backtesting Residual model
    backtest_results = []
    for keyword in df["keyword"].unique():
        keyword_df = df[df["keyword"] == keyword].copy()
        keyword_df["hour_bucket"] = pd.to_datetime(keyword_df["hour_bucket"], format='%Y-%m-%d %H')
        keyword_df = keyword_df.set_index("hour_bucket").asfreq('h').fillna(0)

        # Split the data into training and testing sets (last 6 hours for testing)
        train = keyword_df.iloc[:-6]
        test = keyword_df.iloc[-6:]

         # Normalize the training data for Residual model
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train[["frequency"]]).flatten()
        test_scaled = scaler.transform(test[["frequency"]]).flatten()

        # Fit Residual model on the training set and forecast the next 6 hours
        try:
            n_steps = 6
            
            # Fit Arima model on the training set and forecast the next 6 hours
            arima_model = ARIMA(train_scaled, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=6)

            # Calculate residuals on training data
            residuals = train_scaled - arima_fit.fittedvalues
            
            # Train LSTM on residuals
            X, y = split_sequence(residuals, n_steps)
            if len(X) == 0:
                continue
            X = X.reshape((X.shape[0], X.shape[1], 1))
            model = Sequential()
            model.add(Input(shape=(n_steps, 1)))
            model.add(LSTM(50, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=200, verbose=0)

            input_seq = residuals[-n_steps:].tolist()
            residual_forecast = []
            for _ in range(6):
                x_input = np.array(input_seq[-n_steps:]).reshape((1, n_steps, 1))
                next_pred = model.predict(x_input, verbose=0)[0][0]
                residual_forecast.append(next_pred)
                input_seq.append(next_pred)
            residual_forecast = np.array(residual_forecast)
            
            # Combine
            final_forecast = np.maximum(0, arima_forecast + residual_forecast)
            
            mae = abs(final_forecast - test_scaled).mean()
            backtest_results.append({"keyword": keyword, "mae": mae})
            print(f"Keyword: {keyword}, MAE: {mae:.4f}")
        except Exception as e:
            print(f"Residual hybrid failed for keyword '{keyword}': {e}")
            continue

    # Calculate overall MAE across all keywords and compare to Dubey benchmark of 0.23
    if backtest_results:
        overall_mae = sum(r["mae"] for r in backtest_results) / len(backtest_results)
        print(f"\n=== OVERALL Residual Hybrid BACKTESTING ===")
        print(f"Overall MAE across {len(backtest_results)} keywords: {overall_mae:.4f}")
        if overall_mae < 0.23:
            print(f"MAE {overall_mae:.4f} is BELOW Dubey benchmark of 0.23 - good performance")
        else:
            print(f"MAE {overall_mae:.4f} is ABOVE Dubey benchmark of 0.23")

    conn.close()

if __name__ == "__main__":
    main()