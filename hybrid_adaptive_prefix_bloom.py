import os, math, time, random, hashlib
from collections import defaultdict, Counter

WORDLIST_PATH = r"twl06.txt"   # adjust if needed

# ---------- Bloom Filter ----------
class BloomFilter:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.m = self.optimal_m(n, p)
        self.k = self.optimal_k(self.m, n)
        self.bits = bytearray(self.m // 8 + 1)

    def optimal_m(self, n, p):
        return max(8, int(-n * math.log(p) / (math.log(2) ** 2)))

    def optimal_k(self, m, n):
        return max(1, int((m / n) * math.log(2))) if n > 0 else 1

    def _hashes(self, item):
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, item):
        for pos in self._hashes(item):
            self.bits[pos // 8] |= 1 << (pos % 8)

    def __contains__(self, item):
        return all(
            self.bits[pos // 8] & (1 << (pos % 8))
            for pos in self._hashes(item)
        )

# ---------- IO ----------
def read_words(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sorted(set(w.strip().lower() for w in f if w.strip()))

# ---------- Helpers ----------
def build_prefix_buckets(words, L):
    buckets = defaultdict(list)
    for w in words:
        if len(w) < L:
            continue
        buckets[w[:L]].append(w[L:])
    return buckets

def build_bloom_filters(buckets, p=0.01, skip_small=False):
    filters = {}
    memory_bits = 0

    for prefix, suffixes in buckets.items():
        n = len(suffixes)
        if n == 0:
            continue

        # Skip Bloom filter for small buckets
        if skip_small and n < 10:
            filters[prefix] = suffixes
            continue

        bf = BloomFilter(n, p)
        for suf in suffixes:
            bf.add(suf)

        filters[prefix] = bf
        memory_bits += bf.m

    return filters, memory_bits / 8 / (1024 * 1024)  # MiB

def measure_fpr(filters, L, n_trials=5000):
    false_positives = 0

    for _ in range(n_trials):
        fake = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(L + 5))
        prefix, suffix = fake[:L], fake[L:]
        f = filters.get(prefix)

        if f:
            if isinstance(f, list):
                if suffix in f:
                    false_positives += 1
            else:
                if suffix in f:
                    false_positives += 1

    return false_positives / n_trials

# ---------- Experiments ----------
def run_baseline(words, L=3):
    t0 = time.time()

    buckets = build_prefix_buckets(words, L)
    filters, mem = build_bloom_filters(buckets, p=0.01)
    fpr = measure_fpr(filters, L)

    return {
        "Buckets": len(buckets),
        "Memory(MiB)": round(mem, 3),
        "FPR": round(fpr, 5),
        "MaxBucket": max(len(v) for v in buckets.values()),
        "BuildTime(ms)": round((time.time() - t0) * 1000, 1)
    }

def run_hybrid(words, heavy_thresh=1000):
    """
    Hybrid = Adaptive Prefixing (L=3 -> L=4 for heavy buckets)
             + Skip Bloom filters for small buckets
    """
    t0 = time.time()

    # Step 1: Base L=3 buckets
    buckets = build_prefix_buckets(words, 3)
    new_buckets = {}

    for prefix, suffixes in buckets.items():
        if len(suffixes) > heavy_thresh:
            # Split heavy buckets into L=4
            sub = build_prefix_buckets([prefix + s for s in suffixes], 4)
            for sp, subsuf in sub.items():
                new_buckets[sp] = subsuf
        else:
            new_buckets[prefix] = suffixes

    # Step 2: Build Bloom filters, skipping small buckets
    filters, mem = build_bloom_filters(new_buckets, p=0.01, skip_small=True)
    fpr = measure_fpr(filters, 3)

    return {
        "Buckets": len(new_buckets),
        "Memory(MiB)": round(mem, 3),
        "FPR": round(fpr, 5),
        "MaxBucket": max(len(v) for v in new_buckets.values()),
        "BuildTime(ms)": round((time.time() - t0) * 1000, 1),
        "Structure": filters
    }

# ---------- Main (safe to import) ----------
if __name__ == "__main__":
    words = read_words(WORDLIST_PATH)
    print(f"Loaded {len(words)} unique words.\n")

    baseline = run_baseline(words, L=3)
    hybrid = run_hybrid(words)

    print("=== Baseline (L=3) ===")
    print(baseline)

    print("\n=== Hybrid (Adaptive Prefixing + Skip Small Buckets) ===")
    print(hybrid)
