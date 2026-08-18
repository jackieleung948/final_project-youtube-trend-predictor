import glob, pandas as pd
from config import TRAIN_END

rows = []
for path in glob.glob("labels/raw/*.csv"):
    df = pd.read_csv(path, skiprows=2)
    keyword = df.columns[1].split(":")[0].strip().lower()
    df.columns = ["date", "interest"]
    df["interest"] = pd.to_numeric(df["interest"].replace("<1", 0.5), errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] < pd.to_datetime(TRAIN_END[:10])]["interest"]
    test = df[df["date"] >= pd.to_datetime(TRAIN_END[:10])]["interest"]
    train_mean, test_mean = train.mean(), test.mean()
    pct = (test_mean - train_mean) / train_mean if train_mean > 0 else float("nan")
    rows.append({"keyword_search": keyword, 
                 "keyword": " ".join(sorted(keyword.split())),
                    "train_mean": train_mean,
                    "test_mean": test_mean,
                    "pct_change": pct,
                    "insufficient_data": int(train_mean == 0 or test_mean == 0),
                    "trending_25": int(pct >= 0.25) if train_mean > 0 else 0,
                    "trending_50": int(pct > 0.5) if train_mean > 0 else 0
                    })

pd.DataFrame(rows).round(2).to_csv("labels/backtest_labels.csv", index=False)  