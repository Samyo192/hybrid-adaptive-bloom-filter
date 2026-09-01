import os
import pickle

from bloom_prefix_optimization_experiments import read_words
from hybrid_adaptive_prefix_bloom import run_hybrid


WORDLIST_DIR = "wordlists"
OUT_DIR = "serialized_indexes"

os.makedirs(OUT_DIR, exist_ok=True)


def build_and_serialize(wordlist_path):
    name = os.path.basename(wordlist_path).replace(".txt", "")
    print(f"[BUILD] {name}")

    words = read_words(wordlist_path)

    # Build hybrid structure ONCE
    result = run_hybrid(words)
    structure = result["Structure"]

    out_path = os.path.join(OUT_DIR, f"{name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(structure, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Serialized index written to {out_path}")


if __name__ == "__main__":
    for fname in sorted(os.listdir(WORDLIST_DIR)):
        if fname.endswith(".txt"):
            build_and_serialize(os.path.join(WORDLIST_DIR, fname))
