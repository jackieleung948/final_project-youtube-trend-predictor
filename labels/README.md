# Google Trends labels

Ground truth for the backtest - whether the evaluated keyword trended during the test week, according to Google Trends (YouTube Search).

## Layout
- 'raw/' - one CSV per keyword downloaded manually from Google Trends as there is no publicly available official API
- 'backtest_labels.csv' - produced by 'make_label.py', which 'evaluate.py' reads

## Query Settings for Google Trends
- Region: Worldwide
- Search type: YouTube Search
- Category: all categories
- Date Range: 2026-07-26 to end date
- One keyword per query

## Steps
1. Navigate to trends.google.com/trends/explore and enter the keyword
2. Apply the settings above
3. "Interest over time" graph and download, which produces "multiTimeline.csv"
4. Save and rename the file to keyword with underscores.csv

## Label Rules from 'make_labels.py'
- 'train_mean" = mean daily interest before 'TRAIN_END'
- 'test_mean" = mean on or after 'TRAIN_END'
- 'pct_change' = (test_mean - train_mean) / train_mean
- 'trending_25' = 1 if pct_change > 0.25
- 'trending_50' = 1 if pct_change > 0.50
- Values reported by Google as '<1' is treated as 0.5
- If train_mean or test_mean is 0, 'insufficent_date' = 1 and keyword is labeled as not trending

## Keyword matching
- 'keywords.py' stores keywords with words sorted alphabetically
- 'make_labels.py' applies the same sort so labels can be joined to the db
- Original search string is kept in 'keyword_search'