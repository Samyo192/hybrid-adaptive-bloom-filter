import os
import pandas as pd

# ================= CONFIG =================
WORDLIST_DIR = "wordlists"
PKL_DIR = "serialized_indexes"
OUT_CSV = "file_size_comparison.csv"
# ==========================================


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


rows = []

for fname in sorted(os.listdir(WORDLIST_DIR)):
    if not fname.endswith(".txt"):
        continue

    dataset = fname.replace(".txt", "")
    txt_path = os.path.join(WORDLIST_DIR, fname)
    pkl_path = os.path.join(PKL_DIR, f"{dataset}.pkl")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing serialized file: {pkl_path}")

    txt_size = get_file_size_mb(txt_path)
    pkl_size = get_file_size_mb(pkl_path)

    rows.append({
        "Dataset": dataset,
        "TXT_Size_MB": round(txt_size, 3),
        "PKL_Size_MB": round(pkl_size, 3),
        "Compression_Ratio": round(txt_size / pkl_size, 2)
    })


df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("\n=== File Size Comparison: Raw Wordlist vs Serialized Index ===")
print(df.to_string(index=False))
print(f"\n[OK] Results written to {OUT_CSV}")
        