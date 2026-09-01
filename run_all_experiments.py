import csv
from bloom_prefix_optimization_experiments import (
    read_words,
    run_baseline,
    run_adaptive_prefixing,
    run_skip_small,
    run_adaptive_sizing
)
from hybrid_adaptive_prefix_bloom import run_hybrid

WORDLIST_PATH = "synthetic_skewed.txt"
OUT_CSV = "results_summary_synthetic.csv"
import pandas as pd

def run_all_experiments(words):
    rows = []

    # ---- Baseline (L=3 only)
    baseline = run_baseline(words)[3]
    rows.append({
        "Method": "Baseline_L3",
        **baseline
    })

    # ---- Individual ideas
    rows.append({
        "Method": "AdaptivePrefix",
        **run_adaptive_prefixing(words)
    })

    rows.append({
        "Method": "SkipSmall",
        **run_skip_small(words)
    })

    rows.append({
        "Method": "AdaptiveP",
        **run_adaptive_sizing(words)
    })

    # ---- Hybrid
    rows.append({
        "Method": "Hybrid",
        **run_hybrid(words)
    })

    return pd.DataFrame(rows)

def main():
    words = read_words(WORDLIST_PATH)

    df = run_all_experiments(words)

    df.to_csv(OUT_CSV, index=False)

    print(f"[OK] Results written to {OUT_CSV}")


if __name__ == "__main__":
    main()
