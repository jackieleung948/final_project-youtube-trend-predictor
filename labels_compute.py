import pandas as pd
import os
from config import TRAIN_END

SPLIT = pd.to_datetime(TRAIN_END, format="%Y-%m-%d %H").normalize()

todo = pd.read_csv("labels/backtest_labels_todo.csv")
results = []

for _, row in todo.iterrows():
    out = row.to_dict()
    if row["insufficient_data"] == 1:
        results.append(out)
        continue
    path = f"labels/trends_csvs/{row['keyword']}.csv"
    if not os.path.exists(path):
        path = f"labels/trends_csv/{row['query_used']}.csv"
    if not os.path.exists(path):
        out["note"] = "Missing backtest label file"
        results.append(out)
        continue
    ts = pd.read_csv(path)
    ts["Time"] = pd.to_datetime(ts["Time"])
    ts = ts[ts["Time"] >= "2026-07-26"]
    values = pd.to_numeric(ts.iloc[:, 1], errors="coerce").fillna(0)
    train = values[ts["Time"] < SPLIT]
    test = values[ts["Time"] >= SPLIT]
    train_mean = train.mean()
    test_mean = test.mean()
    trending_25 = int(test_mean > 1.25 * train_mean)
    trending_50 = int(test_mean > 1.5 * train_mean)
    if train_mean == 0 and test_mean == 0:
        out["note"] = "Both training and test means are zero"
        out["insufficient_data"] = 1
        results.append(out)
        continue
    if train_mean == 0:
        out["note"] = "Zero train mean, eyeball manually"
    out["insufficient_data"] = 0
    out["train_mean"] = train_mean
    out["test_mean"] = test_mean
    out["trending_25"] = trending_25
    out["trending_50"] = trending_50
    results.append(out)

final = pd.DataFrame(results)
final.to_csv("labels/backtest_labels.csv", index=False)

insufficient_data = (final["insufficient_data"] == 1).sum()
missing = (final["note"] == "Missing backtest label file").sum() if "note" in final.columns else 0
labeled = final["trending_25"].notna().sum()
print(f"Labeled: {labeled}, "
      f"trending_25: {final['trending_25'].sum()}, "
      f"trending_50: {final['trending_50'].sum()}, "
      f"Insufficient data : {insufficient_data},"
      f"Missing files: {missing}")