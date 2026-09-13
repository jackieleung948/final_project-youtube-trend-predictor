# YouTube Trend Predictor

Final project for CM3070.
Template 1.2: Predictive Modeling of Social Media Trend Emergence

Predicts which food/cooking keywords are about to trend on YouTube by collecting newly published cooking videos hourly via the YouTube Data API, extracting keywords with KeyBERT, and forecasting each keyword's videos-by-bucket frequency with ARIMA, LSTM, and two ARIMA-LSTM hybrids (weighted-average and residual). Forecasts are evaluated against Google Trends as ground-truth labels. The evaluation found detection did not succeed in the tested scale, see report for more details.

## Pipeline
```
collect.py -> videos -> keywords.py -> keywords -> models -> predictions
    |                                                |
metrics.py -> metrics (retired trend score)        evaluate.py <- labels
```

- collect.py - hourly scheduled task to fetch the 50 most recent cooking videos published in the last hour using search.list. The output is stored in videos table.
- keywords.py - extracts KeyBERT keywords for unprocessed videos into keywords. The words are sorted alphabetically for simple deduplication.
- metrics.py - hourly task that refreshes view and engagement statistics, which is used to calculate trend score, but this is retired
- backfill.py - a daily quota-capped recovery job due to a collection gap from Jul 27 to Aug 18. This only fetches data from search.list to recover video metadata. Views and engagement metrics are not recoverable as the API only returns current statistics
- window_diagnostic.py - per model and keyword, the fraction of backtest windows flagged trending. Produces window_flag_fractions.csv
- cap_incidence.py - per keyword, the share of nonzero training buckets containing a collection hour. Produces cap_incidence.csv
- labels_skeleton.py - dumps the top 50 keywords worklist (labels/backtest_labels_todo.csv) for manual query used / insufficient data entry
- labels_compute.py - computes trending_25 and trending_25 labels from the manually downloaded Google Trends CSVs into labels/backtest_labels.csv
- make_panel.py - combines all the predictions of the models into a single panel
- plot_rotimaker.py - plots the rotmakingmachine series with one-step forecasts
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

- Expanding-window backtests - ARIMA is refitted at each origin. LSTMs are fitted once on the training window and reused 
- Metrics: MAE (raw and min-max normalized on the training range), precision/recall/F1 of trending flags against Google Trends labels at +25% and +50% threshold
- data.py - shared data module
- models.py - all models functions shared by production and evaluation
- config.py - all the configurations shared
- Ground-truth labels - see labels/README.md for more details

## Setup
```
pip install -r requirements.txt
```
- Python 3.11
- Create .env containing 'YOUTUBE_API_KEY=your_key_here': never committed for privacy reasons
- Production collection scripts (collect.py, keywords.py, metrics.py, backfill.py) are deployed on PythonAnywhere for hourly fetching of data and backfill
- models.py imports TensorFlow at the module level, so all model scripts need the full requirements.txt
- Runtimes: evaluate.py with all 6 models takes about 25m, LSTM-based scripts take a few minute each, ARIMA runs in seconds

## Reproducing the Results
Data: evaluation_results.csv (Aug 27 backtest output). Full DB is attached as youtube_trends.db gz (contains data up until Sept 13), but data.py truncates at TEST_END and filters on the training window, so the backtest reproduces identically. 
cap_incidence.py and plot_rotimaker requires DB
```
python evaluate.py --models naive_last naive_mean arima lstm hybrid_avg hybrid_residual
python labels_compute.py # optional, only if re-label is needed (labels/backtest_labels.csv is committed)
python window_diagnostic.py
python cap_incidence.py
python make_panel.py && python plot_rotimaker.py
```
Expected numbers: naive_last raw MAE 0.295. 42 labeled keywords, 1 positive, all models TP = 1 at 12-37 FP