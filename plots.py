import pandas as pd
import matplotlib.pyplot as plt

# Load unified results
df = pd.read_csv("results_all_datasets.csv")

# Filter baseline and hybrid only
baseline = df[df["Method"] == "Baseline_L3"]
hybrid   = df[df["Method"] == "Hybrid"]

datasets = baseline["Dataset"].tolist()

# ======================================================
# FIGURE 1: Max Bucket Size
# ======================================================
plt.figure()

plt.bar(datasets, baseline["MaxBucket"], label="Baseline", alpha=0.9)
plt.bar(datasets, hybrid["MaxBucket"], label="Hybrid", alpha=0.8)

plt.xticks(rotation=30)
plt.ylabel("Max Bucket Size")
plt.title("Max Bucket Size Across Datasets")
plt.legend()
plt.tight_layout()
plt.show()


# ======================================================
# FIGURE 2: False Positive Rate (LOG SCALE)
# ======================================================
plt.figure()

plt.plot(datasets, baseline["FPR"], marker="o", label="Baseline")
plt.plot(datasets, hybrid["FPR"], marker="o", label="Hybrid")

plt.yscale("log")
plt.xticks(rotation=30)
plt.ylabel("False Positive Rate (log scale)")
plt.title("False Positive Rate Across Datasets")
plt.legend()
plt.tight_layout()
plt.show()


# ======================================================
# FIGURE 3: Memory Usage
# ======================================================
plt.figure()

plt.plot(datasets, baseline["Memory(MiB)"], marker="o", label="Baseline")
plt.plot(datasets, hybrid["Memory(MiB)"], marker="o", label="Hybrid")

plt.xticks(rotation=30)
plt.ylabel("Memory Usage (MiB)")
plt.title("Memory Usage Across Datasets")
plt.legend()
plt.tight_layout()
plt.show()
