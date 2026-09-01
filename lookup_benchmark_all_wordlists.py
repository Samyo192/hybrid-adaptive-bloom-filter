import os
import time
import random
import string
import pandas as pd
import matplotlib.pyplot as plt

# ==== IMPORT YOUR EXISTING LOGIC ====
from bloom_prefix_optimization_experiments import read_words, run_baseline
from hybrid_adaptive_prefix_bloom import run_hybrid


# ================= CONFIG =================
WORDLIST_DIR = "wordlists"   # <-- CHANGE THIS
N_QUERIES = 10000            # total queries per dataset
PRESENT_RATIO = 0.5          # 50% present, 50% absent
SEED = 42
# ==========================================

random.seed(SEED)


# ============ LOOKUP HELPERS (INLINE LOGIC) ============

def baseline_lookup(structure, word, L=3):
    if len(word) < L:
        prefix, suffix = word, ""
    else:
        prefix, suffix = word[:L], word[L:]

    bucket = structure.get(prefix)
    if bucket is None:
        return False

    # bucket is Bloom filter
    return suffix in bucket


def hybrid_lookup(structure, word, L=3):
    if len(word) < L:
        prefix, suffix = word, ""
    else:
        prefix, suffix = word[:L], word[L:]

    entry = structure.get(prefix)
    if entry is None:
        return False

    # raw list
    if isinstance(entry, list):
        return suffix in entry

    # Bloom filter
    return suffix in entry


# ================= QUERY GENERATION =================

def generate_queries(words, n_queries):
    n_present = int(n_queries * PRESENT_RATIO)
    n_absent = n_queries - n_present

    present = random.sample(words, min(n_present, len(words)))

    absent = []
    while len(absent) < n_absent:
        w = "".join(random.choice(string.ascii_lowercase)
                    for _ in range(random.randint(5, 10)))
        if w not in words:
            absent.append(w)

    queries = present + absent
    random.shuffle(queries)
    return queries


# ================= BENCHMARK =================

def benchmark_lookup(lookup_fn, structure, queries):
    start = time.perf_counter()
    for q in queries:
        lookup_fn(structure, q)
    end = time.perf_counter()

    avg_time_us = (end - start) / len(queries) * 1e6
    return avg_time_us


# ================= MAIN DRIVER =================

results = []

for fname in sorted(os.listdir(WORDLIST_DIR)):
    if not fname.endswith(".txt"):
        continue

    dataset = fname.replace(".txt", "")
    path = os.path.join(WORDLIST_DIR, fname)

    print(f"[INFO] Processing dataset: {dataset}")

    words = read_words(path)
    queries = generate_queries(words, N_QUERIES)

    # -------- Baseline --------
    baseline_struct = run_baseline(words)[3]["Structure"]
    t_base = benchmark_lookup(baseline_lookup, baseline_struct, queries)

    # -------- Hybrid --------
    hybrid_struct = run_hybrid(words)["Structure"]
    t_hybrid = benchmark_lookup(hybrid_lookup, hybrid_struct, queries)

    results.append({
        "Dataset": dataset,
        "Baseline_Avg_Lookup_us": t_base,
        "Hybrid_Avg_Lookup_us": t_hybrid
    })


# ================= SAVE CSV =================

df = pd.DataFrame(results)
df.to_csv("lookup_time_summary.csv", index=False)
print("[OK] lookup_time_summary.csv written")


# ================= PLOT =================

plt.figure()

plt.plot(df["Dataset"], df["Baseline_Avg_Lookup_us"],
         marker="o", label="Baseline")

plt.plot(df["Dataset"], df["Hybrid_Avg_Lookup_us"],
         marker="o", label="Hybrid")

plt.xticks(rotation=30)
plt.ylabel("Average Lookup Time (µs)")
plt.title("Average Word Lookup Time Across Wordlists")
plt.legend()
plt.tight_layout()
plt.show()
