import argparse
import sqlite3
import pandas as pd
import numpy as np
from config import (DB_PATH, DATA_START, TRAIN_END, TEST_END, MIN_HOUR_BUCKETS, BUCKET_HOURS, MIN_VIDEO_COUNT, TOP_N, HORIZON_HOURS, HORIZON_BUCKETS, STEP, LABELS_CSV)
from models import forecast_arima, naive_last, naive_mean

# Get the data from the database and filter it based on the configuration
def get_filtered_data():
    conn = sqlite3.connect(DB_PATH)

    # Find keywords appearing in at least MIN_VIDEO_COUNT unique videos
    video_keyword_df = pd.read_sql_query(
        """
        SELECT k.keyword,
        STRFTIME('%Y-%m-%d %H', v.published_at) as hour_bucket,
        COUNT(DISTINCT k.video_id) AS frequency
        FROM (SELECT DISTINCT keyword, video_id FROM keywords) k
        JOIN videos v ON k.video_id = v.video_id
        WHERE v.published_at >= ?
        GROUP BY keyword, hour_bucket
        ORDER BY keyword, hour_bucket
        """, conn, params=(DATA_START,)
    )

    train_df = video_keyword_df[video_keyword_df["hour_bucket"] < TRAIN_END]

    hours_per_kw = train_df.groupby("keyword")["hour_bucket"].nunique()
    valid_hours = set(hours_per_kw[hours_per_kw >= MIN_HOUR_BUCKETS].index)

    valid_video_df = pd.read_sql_query(
        """
        SELECT k.keyword, 
        COUNT(DISTINCT k.video_id) AS video_count
        FROM keywords k
        JOIN videos v ON k.video_id = v.video_id
        WHERE v.published_at >= ? AND STRFTIME('%Y-%m-%d %H', v.published_at) < ?
        GROUP BY keyword
        """, conn, params=(DATA_START, TRAIN_END)
    )
    valid_videos = set(valid_video_df[valid_video_df["video_count"] >= MIN_VIDEO_COUNT]["keyword"])

    keep = valid_hours & valid_videos
    train_df = train_df[train_df["keyword"].isin(keep)]
    top = train_df.groupby("keyword")["frequency"].sum().nlargest(TOP_N).index

    # Hourly series for each keyword, filling missing hours with 0 frequency
    series = {}
    full_range = pd.date_range(
        pd.to_datetime(DATA_START).floor("h"),
        pd.to_datetime(TEST_END, format='%Y-%m-%d %H'),
        freq ="h", 
        inclusive="left"
    )
    keep_df = video_keyword_df[video_keyword_df["keyword"].isin(top)]
    for keyword, group in keep_df.groupby("keyword"):
        index = pd.to_datetime(group["hour_bucket"], format='%Y-%m-%d %H')
        hourly = pd.Series(group["frequency"].values, index=index)
        hourly = hourly.reindex(full_range, fill_value=0).astype(float)  # Fill missing hours with 0
        series[keyword] = hourly.resample(f"{BUCKET_HOURS}h", origin="start").sum()

    conn.close()
    return series

def mae_raw(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.
    
    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values
    
    Returns:
    - mae: float, the mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred))

def mae_norm(y_true, y_pred, low, high):
    """
    Calculate the normalized Mean Absolute Error (MAE) between true and predicted values.
    
    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values
    - low: float, lower bound for normalization
    - high: float, upper bound for normalization
    
    Returns:
    - mae_norm: float, the normalized mean absolute error
    """
    range = high - low or 1.0
    return np.mean(np.abs(y_true - y_pred)) / range

def rolling_windows(n_train, n_total, h, step):
    """
    Generate rolling window indices for backtesting.
    
    Parameters:
    - n_train: int, number of training samples
    - n_total: int, total number of samples
    - h: int, forecast horizon
    - step: int, step size for rolling windows
    
    Returns:
    - windows: list of tuples, each containing (train_start, train_end, test_start, test_end)
    """
    windows, origin = [], n_train
    while origin + h <= n_total:
        windows.append((0, origin, origin, origin + h))
        origin += step
    return windows

MODEL_FUNCTIONS = {
    "arima": forecast_arima,
    "naive_last": naive_last,
    "naive_mean": naive_mean
}

def run(models):
    """
    Run the evaluation for the specified models.
    
    Parameters:
    - models: list of str, names of the models to evaluate
    """
    series = get_filtered_data()
    for keyword, ser in list(series.items())[:10]:  # Print stats for the first 10 keywords
        print(f"{keyword:25s} mean={ser.mean():.2f}, max={ser.max()}, nonzero={int((ser > 0).sum())}/{len(ser)}")
    boundary = pd.to_datetime(TRAIN_END, format='%Y-%m-%d %H')
    records, failures = [], 0
    for keyword, ser in series.items():
        train_ser = ser[ser.index < boundary]
        train_min, train_max, train_mean = train_ser.min(), train_ser.max(), train_ser.mean()
        n_train, n_total = len(train_ser), len(ser)
        windows = rolling_windows(n_train, n_total, HORIZON_BUCKETS, max(1, STEP // BUCKET_HOURS))
        for model_name in models:
            model_func = MODEL_FUNCTIONS.get(model_name)
            if model_func is None:
                print(f"Model {model_name} not recognized. Skipping.")
                continue
            for train_start, train_end, test_start, test_end in windows:
                train_data = ser.iloc[train_start:train_end]
                test_data = ser.iloc[test_start:test_end]
                try:
                    forecast = model_func(train_data, HORIZON_BUCKETS)
                    if forecast is None or len(forecast) != len(test_data):
                        raise ValueError("Forecast length mismatch or None returned.")
                    records.append({
                        "keyword": keyword,
                        "model": model_name,
                        "train_start": train_data.index[0],
                        "train_end": train_data.index[-1],
                        "test_start": test_data.index[0],
                        "test_end": test_data.index[-1],
                        "mae_raw": mae_raw(test_data.values, forecast),
                        "mae_norm": mae_norm(test_data.values, forecast, train_min, train_max),
                        "pred_trending": bool(forecast.mean() > train_mean)

                    })
                except Exception as e:
                    print(f"Error evaluating {model_name} for keyword '{keyword}': {e}")
                    failures += 1
    results_df = pd.DataFrame(records)
    print(f"Evaluation completed with {failures} failures.")
    return results_df

def summarize(res):
    """
    Summarize the evaluation results by calculating the mean MAE for each model.
    
    Parameters:
    - res: pandas DataFrame, containing evaluation results with columns ['keyword', 'model', 'mae']
    
    Returns:
    - summary_df: pandas DataFrame, containing mean MAE for each model
    """
    summary_df = res.groupby("model").agg(
        mean_mae_raw=("mae_raw", "mean"),
        mean_mae_norm=("mae_norm", "mean"),
        windows=("mae_raw", "size"),
    ).round(3).sort_values(by="mean_mae_raw").reset_index()
    return summary_df

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate forecasting models on YouTube trend data.")
    ap.add_argument("--models", nargs="+", required=True, help="List of models to evaluate (e.g., arima naive_last naive_mean)")
    args = ap.parse_args()
    results = run(args.models)
    summary = summarize(results)
    print("\nEvaluation Summary:")
    print(summary)
    if not results.empty:
        results.to_csv("evaluation_results.csv", index=False)
        print("\nDetailed results saved to 'evaluation_results.csv'.")