import os, math, time, random, hashlib
from collections import defaultdict, Counter

WORDLIST_PATH = r"twl06.txt"   # <-- change if needed

# ---------- Bloom Filter Helpers ----------
class BloomFilter:
    def __init__(self, n, p):
        self.n = n  # items
        self.p = p  # target FP rate
        self.m = self.optimal_m(n, p)  # bit array size
        self.k = self.optimal_k(self.m, n)  # number of hashes
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
        return all(self.bits[pos // 8] & (1 << (pos % 8))
                   for pos in self._hashes(item))

def read_words(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sorted(set(w.strip().lower() for w in f if w.strip()))

# ---------- Core Experiments ----------
def build_prefix_buckets(words, L):
    buckets = defaultdict(list)
    for w in words:
        if len(w) < L: continue
        buckets[w[:L]].append(w[L:])
    return buckets

def build_bloom_filters(buckets, p=0.01, skip_small=False, adaptive_p=False):
    filters = {}
    memory_bits = 0
    for prefix, suffixes in buckets.items():
        n = len(suffixes)
        if n == 0:
            continue

        # Idea 2: skip small buckets
        if skip_small and n < 10:
            filters[prefix] = suffixes  # raw list
            continue

        # Idea 3: adaptive p
        target_p = p
        if adaptive_p:
            if n >= 500: target_p = 0.005
            elif n <= 20: target_p = 0.05

        bf = BloomFilter(n, target_p)
        for suf in suffixes:
            bf.add(suf)
        filters[prefix] = bf
        memory_bits += bf.m

    return filters, memory_bits / 8 / (1024 * 1024)  # MiB

def measure_fpr(filters, L, n_trials=5000):
    false_positives = 0
    for _ in range(n_trials):
        fake = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(L+5))
        prefix, suffix = fake[:L], fake[L:]
        f = filters.get(prefix)
        if f:
            if isinstance(f, list):  # raw list
                if suffix in f:
                    false_positives += 1
            else:
                if suffix in f:
                    false_positives += 1
    return false_positives / n_trials

def run_baseline(words):
    results = {}
    for L in [2, 3, 4]:
        t0 = time.time()
        buckets = build_prefix_buckets(words, L)
        filters, mem = build_bloom_filters(buckets, p=0.01)
        fpr = measure_fpr(filters, L)
        hot = Counter({k: len(v) if isinstance(v, list) else 0 for k,v in buckets.items()}).most_common(10)
        results[L] = {
            "Buckets": len(buckets),
            "Memory(MiB)": round(mem,3),
            "FPR": round(fpr,5),
            "MaxBucket": max(len(v) for v in buckets.values()),
            "BuildTime(ms)": round((time.time()-t0)*1000,1),
            "Structure": filters
        }
    return results

def run_adaptive_prefixing(words, heavy_thresh=1000):
    t0 = time.time()
    buckets = build_prefix_buckets(words, 3)
    new_buckets = {}
    for prefix, suffixes in buckets.items():
        if len(suffixes) > heavy_thresh:
            sub_buckets = build_prefix_buckets([prefix+s for s in suffixes], 4)
            for sp, subsuf in sub_buckets.items():
                new_buckets[sp] = subsuf
        else:
            new_buckets[prefix] = suffixes
    filters, mem = build_bloom_filters(new_buckets, p=0.01)
    fpr = measure_fpr(filters, 3)
    return {
        "Buckets": len(new_buckets),
        "Memory(MiB)": round(mem,3),
        "FPR": round(fpr,5),
        "MaxBucket": max(len(v) for v in new_buckets.values()),
        "BuildTime(ms)": round((time.time()-t0)*1000,1)
    }

def run_skip_small(words):
    t0 = time.time()
    buckets = build_prefix_buckets(words, 3)
    filters, mem = build_bloom_filters(buckets, p=0.01, skip_small=True)
    fpr = measure_fpr(filters, 3)
    return {
        "Buckets": len(buckets),
        "Memory(MiB)": round(mem,3),
        "FPR": round(fpr,5),
        "MaxBucket": max(len(v) for v in buckets.values()),
        "BuildTime(ms)": round((time.time()-t0)*1000,1)
    }

def run_adaptive_sizing(words):
    t0 = time.time()
    buckets = build_prefix_buckets(words, 3)
    filters, mem = build_bloom_filters(buckets, p=0.01, adaptive_p=True)
    fpr = measure_fpr(filters, 3)
    return {
        "Buckets": len(buckets),
        "Memory(MiB)": round(mem,3),
        "FPR": round(fpr,5),
        "MaxBucket": max(len(v) for v in buckets.values()),
        "BuildTime(ms)": round((time.time()-t0)*1000,1)
    }

# ---------- Main ----------
if __name__ == "__main__":
    words = read_words(WORDLIST_PATH)
    print(f"Loaded {len(words)} unique words.\n")

    baseline = run_baseline(words)
    adaptive_prefixing = run_adaptive_prefixing(words)
    skip_small = run_skip_small(words)
    adaptive_sizing = run_adaptive_sizing(words)

    print("=== Baseline Results (L=2,3,4) ===")
    for L,res in baseline.items():
        print(f"L={L}: {res}")
    print("\n=== Adaptive Prefixing (Idea 1) ===")
    print(adaptive_prefixing)
    print("\n=== Skip Small Buckets (Idea 2) ===")
    print(skip_small)
    print("\n=== Adaptive Sizing (Idea 3) ===")
    print(adaptive_sizing)

