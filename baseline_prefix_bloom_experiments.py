import math, os, random, time
from collections import defaultdict, Counter

# ======= CONFIG =======
WORDLIST_PATH = r"twl06.txt"  # <-- CHANGE THIS
PREFIX_LENGTHS = [2, 3, 4]
TARGET_P = 0.01      # target false-positive rate per bucket
NEGATIVE_PROBES = 50000  # how many non-words to test FPR
SEED = 42
# ======================

random.seed(SEED)

def read_words(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        words = [w.strip() for w in f if w.strip()]
    # normalize to lowercase a-z only for this experiment
    cleaned = []
    for w in words:
        w2 = "".join(ch for ch in w.lower() if "a" <= ch <= "z")
        if w2:
            cleaned.append(w2)
    return cleaned

# Simple fast hash mix (avoid Python's randomized hash)
def djb2(s):
    h = 5381
    for c in s:
        h = ((h << 5) + h) + ord(c)
    return h & 0xFFFFFFFFFFFFFFFF

def sdbm(s):
    h = 0
    for c in s:
        h = ord(c) + (h << 6) + (h << 16) - h
    return h & 0xFFFFFFFFFFFFFFFF

def k_hash_positions(s, m_bits, k):
    # double hashing scheme
    h1 = djb2(s)
    h2 = sdbm(s) | 1  # ensure odd
    for i in range(k):
        yield (h1 + i * h2) % m_bits

class Bloom:
    __slots__ = ("m_bits", "k", "bitarr")
    def __init__(self, m_bits, k):
        self.m_bits = max(1, int(m_bits))
        self.k = max(1, int(k))
        self.bitarr = 0  # store bits as integer

    def add(self, s):
        for pos in k_hash_positions(s, self.m_bits, self.k):
            self.bitarr |= (1 << pos)

    def maybe(self, s):
        for pos in k_hash_positions(s, self.m_bits, self.k):
            if (self.bitarr >> pos) & 1 == 0:
                return False
        return True

    def mem_bytes(self):
        # m_bits rounded up to whole bytes
        return (self.m_bits + 7) // 8

def optimal_m_k(n, p):
    if n <= 0:
        return 8, 1  # tiny placeholder filter
    m = - (n * math.log(p)) / (math.log(2)**2)
    k = (m / n) * math.log(2)
    return int(max(8, math.ceil(m))), int(max(1, round(k)))

def make_buckets(words, L):
    buckets = defaultdict(list)
    for w in words:
        if len(w) <= L:
            # we can treat suffix="" for very short words
            pref = w
            suf = ""
        else:
            pref = w[:L]
            suf = w[L:]
        buckets[pref].append((w, suf))
    return buckets

def mutate_word_to_nonword(w):
    # flip one letter to a random different letter
    if not w:
        return "z"
    i = random.randrange(len(w))
    c = w[i]
    letters = "abcdefghijklmnopqrstuvwxyz"
    alt = random.choice([x for x in letters if x != c])
    return w[:i] + alt + w[i+1:]

def empirical_fpr(bloom_by_bucket, words_by_bucket, L, total_neg=10000):
    # Build negatives by mutating existing words (aim to miss set)
    negs = []
    keys = list(words_by_bucket.keys())
    while len(negs) < total_neg:
        pref = random.choice(keys)
        words = words_by_bucket[pref]
        if not words:
            continue
        w, suf = random.choice(words)
        nw = mutate_word_to_nonword(w)
        # compute its prefix/suffix at same L
        if len(nw) <= L:
            p2, s2 = nw, ""
        else:
            p2, s2 = nw[:L], nw[L:]
        negs.append((p2, s2))

    # probe
    N = 0
    FP = 0
    for p2, s2 in negs:
        N += 1
        b = bloom_by_bucket.get(p2)
        if b is None:
            continue  # definite miss; not a FP for BF
        if b.maybe(s2):
            # it might be a false positive OR the mutated word coincidentally exists;
            # we check against actual list to confirm.
            full_list = set(w for (w, _) in words_by_bucket[p2])
            guessed = (p2 + s2)
            if guessed not in full_list:
                FP += 1
    return FP / max(1, N)

def run_experiment(words):
    words = list(set(words))  # unique for cleanliness
    print(f"Loaded {len(words)} unique words.")
    for L in PREFIX_LENGTHS:
        print("\n=== Prefix length L =", L, "===")
        buckets = make_buckets(words, L)
        sizes = [len(v) for v in buckets.values()]
        non_empty = sum(1 for s in sizes if s > 0)
        total_buckets = len(buckets)
        sizes_sorted = sorted(sizes, reverse=True)
        def pct(p):
            idx = int(len(sizes_sorted) * p / 100)
            idx = min(max(idx, 0), len(sizes_sorted)-1)
            return sizes_sorted[idx]
        print(f"Total buckets: {total_buckets} (non-empty: {non_empty})")
        print(f"Avg bucket size: {sum(sizes)/max(1,len(sizes)):.2f}")
        print(f"Median: {pct(50)}, p90: {pct(10)}, p99: {pct(1)}, Max: {sizes_sorted[0] if sizes_sorted else 0}")

        # Build per-bucket Bloom filters
        bloom_by_bucket = {}
        mem_bytes_total = 0
        t0 = time.time()
        for pref, items in buckets.items():
            n = len(items)
            m_bits, k = optimal_m_k(n, TARGET_P)
            bf = Bloom(m_bits, k)
            for (_, suf) in items:
                bf.add(suf)
            bloom_by_bucket[pref] = bf
            mem_bytes_total += bf.mem_bytes()
        build_ms = (time.time() - t0) * 1000.0

        # Empirical FPR
        fpr = empirical_fpr(bloom_by_bucket, buckets, L, total_neg=NEGATIVE_PROBES)

        # Hot prefixes
        topN = 10
        hottest = sorted(((pref, len(items)) for pref, items in buckets.items()),
                         key=lambda x: x[1], reverse=True)[:topN]

        print(f"Per-bucket BFs target p={TARGET_P:.4f}")
        print(f"Estimated BF memory: {mem_bytes_total/1024/1024:.3f} MiB")
        print(f"Build time: {build_ms:.1f} ms")
        print(f"Empirical false-positive rate: {fpr:.5f}")
        print("Top-10 hottest prefixes (prefix → bucket size):")
        for pref, n in hottest:
            print(f"  {pref: <6} -> {n}")
    print("\nDone.")

if __name__ == "__main__":
    if not os.path.exists(WORDLIST_PATH):
        raise SystemExit(f"File not found: {WORDLIST_PATH}")
    words = read_words(WORDLIST_PATH)
    run_experiment(words)
