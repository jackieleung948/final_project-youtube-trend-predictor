import numpy as np, pandas as pd, matplotlib.pyplot as plt
from config import TRAIN_END, HORIZON_BUCKETS
from data import get_filtered_data
from models import forecast_arima, naive_last, fit_lstm, forecast_lstm

kw = "rotimakingmachine"
ser = get_filtered_data()[kw]
split = pd.to_datetime(TRAIN_END, format="%Y-%m-%d %H")
train = ser[ser.index < split]; test = ser[ser.index >= split]
train_mean = train.mean()
fitted = fit_lstm(train.values)

rows = []
for i in range(len(train), len(ser)):          # one origin per test bucket
    hist = ser.iloc[:i]
    rows.append({"bucket": ser.index[i],
                 "naive_last": naive_last(hist, HORIZON_BUCKETS)[0],
                 "ARIMA":      forecast_arima(hist, HORIZON_BUCKETS)[0],
                 "LSTM":       forecast_lstm(fitted, hist.values, HORIZON_BUCKETS)[0]})
fc = pd.DataFrame(rows).set_index("bucket")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ser.index, ser.values, color="black", lw=1.5, label="observed (distinct videos / 6h)")
for col, style in [("naive_last", "--"), ("ARIMA", "-."), ("LSTM", ":")]:
    ax.plot(fc.index, fc[col], style, lw=1.5, label=f"{col} one-step forecast")
ax.axhline(train_mean, color="grey", lw=1, label=f"training mean = {train_mean:.2f} (trending reference)")
ax.axvline(split, color="red", lw=1, alpha=.6, label="TRAIN_END")
ax.set_title(f'"{kw}": observed series and one-step forecasts over the test window')
ax.set_ylabel("distinct videos per 6-hour bucket"); ax.set_xlabel("bucket start (UTC)")
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figure_rotimaker.png", dpi=160)
print("saved figure_rotimaker.png")