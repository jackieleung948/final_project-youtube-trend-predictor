# YouTube Trend Predictor

Final project for CM3070.
Template: Predictive Modeling of Social Media Trend Emergence

Predicts which food/cooking keywords are about to trend on YouTube by collecting newly published cooking videos hourly via the YouTube Data API, extracting keywords with KeyBERT, and forecasting each keyword's videos-by-bucket frequency with ARIMA, LSTM, and hybrid of ARIMA and LSTM. Forecasts are evaluated against Google Trends as ground-truth labels.

## Pipeline
```
collect.py -> videos -> keywords.py -> keywords -> models -> predictions
    |                                                |
metrics.py -> metrics (trend score)                evaluate.py <- labels
```

- collect.py - hourly scheduled task to fetch the 50 most recent cooking videos published in the last hour using search.list. The output is stored in videos table.
- keywords.py - extracts KeyBERT keywords for unprocessed videos into keywords. The words are sorted alphabetically for simple deduplication.
- metrics.py - hourly task that refreshes view and engagement statistics, which is used to calculate trend score
- backfill.py - a daily quota-capped recovery job due to a collection gap from Jul 27 to Aug 18. This only fetches data from search.list to recover video metadata. Views and engagement metrics are not recoverable as the API only return current statistics
- Frequency is defined as COUNT(DISTINCT video_id) per keyword, bucketed by the video's published_at timestamp

## Production scripts
- arima.py: uses ARIMA time-series forecasting, outputs arima_predictions table and arima_predictions.png
- lstm.py: uses LSTM time-series forecasting, outputs lstm_predictions table and lstm_predictions.png
- residual_hybrid.py: uses ARIMA and LSTM which trains on the residuals, outputs residual_hybrid_predictions table and residual_hybrid_bar.png
- hybrid.py: uses a weighted average (50% each) of ARIMA and LSTM forecasts, outputs hybrid_predictions table and hybrid_bar.png
- analyze.py: analyzes top 10 keywords over time with their distinct video counts, outputs top10_trends_over_time.png and top10_trends_bar_chart.png

## Evaluation
```
python evaluate.py --models naive_last naive_mean arima lstm hybrid_avg hybrid_residual
```

- Expanding-window backtests - models are refitted and reapplied at each forecast origin. 
- Metrics: MAE (raw and min-max normalized on the training range), precision/recall/F1 of trending flags against Google Trends labels at +25% and +50% threshold
- data.py - shared data module
- models.py - all models functions shared by production and evaluation
- config.py - all the configurations shared
- Ground-truth labels - see labels/README.md for more details

## Setup
```
pip install -r requirements.txt
```
- .env: includes the YouTube Data API, which is never commited for privacy reasons
- Deployed on PythonAnywhere for hourly fetching of data and backfill