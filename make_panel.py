import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from config import DB_PATH

panels = [
    ("arima_predictions",           "ARIMA"),
    ("lstm_predictions",            "LSTM"),
    ("hybrid_predictions",          "Weighted-average hybrid"),
    ("residual_hybrid_predictions", "Residual hybrid"),
]
conn = sqlite3.connect(DB_PATH)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for ax, (table, title) in zip(axes.flat, panels):
    df = pd.read_sql_query(f"SELECT keyword, predicted_frequency, is_trending FROM {table}", conn)
    flagged = df[df["is_trending"] == 1]["keyword"].nunique()
    top = (df[df["is_trending"] == 1].groupby("keyword")["predicted_frequency"]
             .sum().nlargest(10).sort_values())
    ax.set_title(f"{title} ({flagged} flagged, top 10 shown)", fontsize=12)
    ax.barh(top.index, top.values)
    ax.set_xlabel("Summed predicted frequency (6-hour buckets)", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="x", alpha=0.3)
conn.close()
fig.suptitle("Top predicted trending keywords per model script", fontsize=13)
plt.savefig("figure_model_panel.png", dpi=160)
print("saved figure_model_panel.png")