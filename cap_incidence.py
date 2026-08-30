import sqlite3
import pandas as pd
from config import DATA_START, DB_PATH, BUCKET_HOURS, TRAIN_END
from data import get_filtered_data

# Hour-level video counts to capped hours (>=49, the observed ceiling)
conn = sqlite3.connect(DB_PATH)

hours = pd.read_sql_query(
    "SELECT SUBSTR(published_at, 1, 13) AS hour, COUNT(*) AS video_count "
    "FROM videos "
    "GROUP BY hour",
    conn
)
conn.close()
hours['hour'] = pd.to_datetime(hours['hour'], format='%Y-%m-%dT%H')
capped_hours = set(hours.loc[hours['video_count'] >= 49]['hour'])

# Map each capped hour to its 6h bucket

origin = pd.to_datetime(DATA_START, format="%Y-%m-%d %H")

def bucket_of(ts):
    return origin + ((ts - origin) // pd.Timedelta(hours=BUCKET_HOURS)) * pd.Timedelta(hours=BUCKET_HOURS)


capped_bucket = {bucket_of(hour) for hour in capped_hours}

# For each top 50 keyword, what fraction are cap-touched?
split = pd.to_datetime(TRAIN_END, format="%Y-%m-%d %H")
rows = []
for kw, ser in get_filtered_data().items():
    train_nonzero = ser[(ser.index < split) & (ser > 0)]
    if len(train_nonzero) == 0:
        continue
    touched = sum(1 for ts in train_nonzero.index if bucket_of(
        ts) in capped_bucket)
    rows.append({
        "keyword": kw,
        "total_buckets": len(train_nonzero),
        "capped_buckets": touched,
        "cap_fraction": touched / len(train_nonzero)
    })

output = pd.DataFrame(rows).sort_values("cap_fraction", ascending=False)
print(f"Overall: {output['capped_buckets'].sum()} capped buckets out of {output['total_buckets'].sum()} total buckets ({output['capped_buckets'].sum() / output['total_buckets'].sum():.2%})")
output.to_csv("cap_incidence.csv", index=False)
