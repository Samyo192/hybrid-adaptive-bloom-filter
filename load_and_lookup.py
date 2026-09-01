import pickle


def load_index(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def hybrid_lookup(structure, word, L=3):
    if len(word) < L:
        prefix, suffix = word, ""
    else:
        prefix, suffix = word[:L], word[L:]

    entry = structure.get(prefix)
    if entry is None:
        return False

    if isinstance(entry, list):
        return suffix in entry

    return suffix in entry  # Bloom filter


if __name__ == "__main__":
    index = load_index("serialized_indexes/twl.pkl")

    while True:
        q = input("Query word (or exit): ").strip().lower()
        if q == "exit":
            break
        print("Present?", hybrid_lookup(index, q))
