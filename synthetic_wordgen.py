import random
import string

def generate_skewed_words(
    n_words=150_000,
    dominant_prefix="aaa",
    dominant_frac=0.5,
    min_len=6,
    max_len=10,
    seed=42
):
    random.seed(seed)
    words = set()

    def rand_suffix(k):
        return "".join(random.choice(string.ascii_lowercase) for _ in range(k))

    # 1) Dominant prefix words
    n_dom = int(n_words * dominant_frac)
    while len(words) < n_dom:
        L = random.randint(min_len, max_len)
        words.add(dominant_prefix + rand_suffix(L - len(dominant_prefix)))

    # 2) Remaining words (random prefixes)
    while len(words) < n_words:
        L = random.randint(min_len, max_len)
        prefix = rand_suffix(3)
        words.add(prefix + rand_suffix(L - 3))

    return sorted(words)


if __name__ == "__main__":
    words = generate_skewed_words()
    with open("synthetic_skewed.txt", "w") as f:
        for w in words:
            f.write(w + "\n")

    print(f"Generated {len(words)} synthetic skewed words")
