import argparse
import sqlite3
import pandas as pd
import numpy as np
import os
from config import (TRAIN_END, BUCKET_HOURS, HORIZON_BUCKETS, STEP, LABELS_CSV)
from models import forecast_arima, naive_last, naive_mean, fit_lstm, forecast_lstm, forecast_residual_hybrid, fit_residual_lstm
from data import get_filtered_data

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
    rng = high - low or 1.0
    return np.mean(np.abs(y_true - y_pred)) / rng

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
    boundary = pd.to_datetime(TRAIN_END, format='%Y-%m-%d %H')
    records, failures = [], 0
    for keyword, ser in series.items():
        train_ser = ser[ser.index < boundary]
        fitted_lstm = fit_lstm(train_ser.values) if ("lstm" in models or "hybrid_avg" in models) else None
        fitted_residual = fit_residual_lstm(train_ser.values) if "hybrid_residual" in models else None
        train_min, train_max, train_mean = train_ser.min(), train_ser.max(), train_ser.mean()
        n_train, n_total = len(train_ser), len(ser)
        windows = rolling_windows(n_train, n_total, HORIZON_BUCKETS, max(1, STEP // BUCKET_HOURS))
        for train_start, train_end, test_start, test_end in windows:
            train_data = ser.iloc[train_start:train_end]
            test_data = ser.iloc[test_start:test_end]
            arima_fc = lstm_fc = None
            for model_name in models:
                if model_name == "arima":
                    forecast = arima_fc = forecast_arima(train_data, HORIZON_BUCKETS)
                elif model_name == "lstm":
                    forecast = lstm_fc = forecast_lstm(fitted_lstm, train_data.values, HORIZON_BUCKETS)
                elif model_name == "hybrid_avg":
                    forecast = None if (arima_fc is None or lstm_fc is None) else 0.5 * arima_fc + 0.5 * lstm_fc
                elif model_name == "hybrid_residual":
                    forecast = forecast_residual_hybrid(fitted_residual, train_data.values, HORIZON_BUCKETS)
                elif model_name in MODEL_FUNCTIONS:
                    forecast = MODEL_FUNCTIONS[model_name](train_data, HORIZON_BUCKETS)
                else:
                    print(f"Model {model_name} not recognized. Skipping")
                    continue
                if forecast is None or len(forecast) != len(test_data):
                    failures += 1
                    continue
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

def evaluate_labels(res):
    """
    Precision and Recall of pred_trending against Google Trends labels.
    """
    if not os.path.exists(LABELS_CSV):
        print(f"\n(no labels file at {LABELS_CSV} - skipping precision/recall)")
        return
    labels = pd.read_csv(LABELS_CSV)
    pred = (res.groupby(["model", "keyword"])["pred_trending"].any().reset_index().rename(columns={"pred_trending": "pred"}))
    merged = pred.merge(labels[["keyword", "trending_25", "trending_50", "insufficient_data"]], on="keyword", how="inner")
    merged = merged[merged["insufficient_data"] == 0]
    if merged.empty:
        print("\n(no labels after filtering insufficient data - skipping precision/recall)")
        return
    for threshold in ["trending_25", "trending_50"]:
        print(f"\nEvaluating against {threshold} labels:")
        for model, g in merged.groupby("model"):
            tp = ((g["pred"] == 1) & (g[threshold] == 1)).sum()
            fp = ((g["pred"] == 1) & (g[threshold] == 0)).sum()
            fn = ((g["pred"] == 0) & (g[threshold] == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan
            print(f"Model: {model}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, TP: {tp}, FP: {fp}, FN: {fn}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate forecasting models on YouTube trend data.")
    ap.add_argument("--models", nargs="+", required=True, help="List of models to evaluate (e.g., arima naive_last naive_mean)")
    args = ap.parse_args()
    results = run(args.models)
    summary = summarize(results)
    print("\nEvaluation Summary:")
    print(summary)
    evaluate_labels(results)
    if not results.empty:
        results.to_csv("evaluation_results.csv", index=False)
        print("\nDetailed results saved to 'evaluation_results.csv'.")