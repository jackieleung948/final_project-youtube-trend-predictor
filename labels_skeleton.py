import pandas as pd
from data import get_filtered_data

kws = sorted(get_filtered_data().keys())

pd.DataFrame({
    "keyword": kws,
    "query_used": "",
    "trending_25": "",
    "trending_50": "",
    "insufficient_data": ""
}).to_csv("labels/backtest_labels_todo.csv", index=False)

print(len(kws), "keywords written to labels/backtest_labels_todo.csv")