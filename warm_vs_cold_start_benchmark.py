import os
import time
import pickle
import pandas as pd

# ================= CONFIG =================
RESULTS_CSV = "results_all_datasets.csv"
PKL_DIR = "serialized_indexes"
OUT_CSV = "startup_time_comparison.csv"
# ==========================================


def measure_load_time(pkl_path):
    start = time.perf_counter()
    with open(pkl_path, "rb") as f:
        _ = pickle.load(f)
    end = time.perf_counter()
    return (end - start) * 1000  # ms


# ---------- Load previously computed cold-start results ----------
results_df = pd.read_csv(RESULTS_CSV)

# Keep only Hybrid results (cold start)
hybrid_df = results_df[results_df["Method"] == "Hybrid"].copy()

rows = []

for _, row in hybrid_df.iterrows():
    dataset = row["Dataset"]
    cold_start_ms = row["BuildTime(ms)"]

    pkl_path = os.path.join(PKL_DIR, f"{dataset}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Serialized file not found: {pkl_path}")

    print(f"[INFO] Measuring warm start for dataset: {dataset}")

    warm_start_ms = measure_load_time(pkl_path)

    rows.append({
        "Dataset": dataset,
        "Cold_Start_Build_ms": round(cold_start_ms, 2),
        "Warm_Start_Load_ms": round(warm_start_ms, 2),
        "Reduction_Factor": round(cold_start_ms / warm_start_ms, 2)
    })


# ---------- Save & display table ----------
out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)

print("\n=== Hybrid Initialization Time: Cold Start vs Warm Start ===")
print(out_df.to_string(index=False))
print(f"\n[OK] Results written to {OUT_CSV}")


# ================= OPTIONAL PLOT (COMMENTED OUT) =================
"""
import matplotlib.pyplot as plt

plt.figure()

plt.plot(out_df["Dataset"], out_df["Cold_Start_Build_ms"],
         marker="o", label="Cold Start (Hybrid Build)")

plt.plot(out_df["Dataset"], out_df["Warm_Start_Load_ms"],
         marker="o", label="Warm Start (Serialized Load)")

plt.xticks(rotation=30)
plt.ylabel("Initialization Time (ms)")
plt.title("Hybrid Initialization Time: Cold Start vs Warm Start")
plt.legend()
plt.tight_layout()
plt.show()
"""
