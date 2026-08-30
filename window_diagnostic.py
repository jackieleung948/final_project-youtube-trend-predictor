import pandas as pd

df = pd.read_csv("evaluation_results.csv")

# Compute per keyword flag fraction
frac = df.groupby(["model", "keyword"])["pred_trending"].mean().reset_index(name="flag_fraction")


# Compute per-model summary
summary = frac.groupby("model")["flag_fraction"].agg(mean="mean", 
                                                     flagged=lambda s: (s > 0).sum(),
                                                     persistent=lambda s: (s >= 0.5).sum(), 
                                                     fluke=lambda s: ((s >0) & (s < 0.1)).sum(), 
                                                     never=lambda s: (s == 0).sum())
print(summary)

# Print rotimakingmachine row
print(frac[frac["keyword"] == "rotimakingmachine"])

# Save frac to CSV
frac.to_csv("window_flag_fractions.csv", index=False)

