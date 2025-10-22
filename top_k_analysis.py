import ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_name = "yelp_GraphSAGE_random"
FILE_PATH = Path(f"{path_name}.txt")

# Load JSON-lines-> list of {k: acc}
results = []
with FILE_PATH.open("r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = ast.literal_eval(line)           # parse Python dict literal
        d = {float(k): float(v) for k, v in d.items()}  # ensure float keys/vals
        results.append(d)

# Aggregate mean & std per k
all_ks = sorted({k for d in results for k in d})
rows = []
for k in all_ks:
    vals = np.array([d[k] for d in results if k in d], dtype=float)
    rows.append({
        "k": k,
        "mean_acc": vals.mean(),
        "std_acc": vals.std(ddof=1) if len(vals) > 1 else 0.0,
        "n": len(vals),
    })
df = pd.DataFrame(rows)

# Semi-log scatter with error bars (std)
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(df["k"], df["mean_acc"], yerr=df["std_acc"], fmt="o", capsize=3, linestyle="none")
ax.set_xscale("log")
ax.set_xlabel("k (% of test set)")
ax.set_ylabel("Accuracy at top-k")
ax.set_title(f"Top-k Accuracy (mean ± std) — {FILE_PATH.stem}")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.set_ylim(-0.3, 1.2)
plt.show()

# Save
fig.savefig(FILE_PATH.with_suffix(".topk_semilog_errorbar.png"), dpi=300, bbox_inches="tight")

import json

# Save mean values per k into JSON
mean_dict = {str(k): float(m) for k, m in zip(df["k"], df["mean_acc"])}

json_path = FILE_PATH.with_suffix(".topk_means.json")
with open(json_path, "w") as f:
    json.dump(mean_dict, f, indent=2)

json_path
