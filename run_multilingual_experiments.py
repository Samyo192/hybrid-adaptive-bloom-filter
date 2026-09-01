import os
import pandas as pd

# CHANGE THIS import to your actual experiment file
from run_all_experiments import read_words, run_all_experiments

WORDLIST_DIR = "wordlists"
OUTPUT_CSV = "results_all_datasets.csv"

all_results = []

for fname in sorted(os.listdir(WORDLIST_DIR)):
    if not fname.endswith(".txt"):
        continue

    dataset_name = fname.replace(".txt", "")
    path = os.path.join(WORDLIST_DIR, fname)

    print(f"Running experiments on: {dataset_name}")

    words = read_words(path)

    df = run_all_experiments(words)

    # ensure DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # add dataset column
    df["Dataset"] = dataset_name

    all_results.append(df)

# merge everything
final_df = pd.concat(all_results, ignore_index=True)

# save ONE csv
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone. Unified results written to: {OUTPUT_CSV}")
