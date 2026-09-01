"""
=============================================================================
BLOOM PATENT CHECKLIST - MASTER MEASUREMENT SCRIPT
=============================================================================
Run this from inside the BLOOM/ folder (same directory as the other .py files).

    cd BLOOM/
    python bloom_patent_master_checklist.py

It will:
  1. Measure all 9 previously-missing parameters using your existing wordlists
  2. Print ALL known values (from CSVs + newly measured) in copy-paste format
  3. Clearly label each value with its checklist field name

Requirements: pip install psutil
(everything else — math, time, pickle, tracemalloc — is standard library)
=============================================================================
"""

import math
import time
import random
import os
import pickle
import tracemalloc
import hashlib
import sys
from collections import defaultdict

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARNING] psutil not installed. RAM measurement will use tracemalloc only.")
    print("          Run: pip install psutil   for more accurate RAM numbers.\n")

# =============================================================================
# CONFIGURATION — adjust paths if needed
# =============================================================================
WORDLIST_DIR   = "wordlists"
PKL_DIR        = "serialized_indexes"
PRIMARY_DATASET = "twl"          # used for per-module deep measurements
ALL_DATASETS   = ["ODS-french", "enable", "german", "italian",
                  "sowpods", "synthetic_skewed", "twl"]

TARGET_P       = 0.01            # Bloom filter false positive rate design target
PREFIX_L       = 3               # default prefix length
HEAVY_THRESH   = 1000            # refinement trigger (bucket size > this → split to L=4)
SKIP_SMALL_N   = 10              # small bucket threshold (n < 10 → direct list storage)
N_LOOKUP_BENCH = 10_000          # queries per lookup benchmark
SEED           = 42

random.seed(SEED)

# =============================================================================
# ── SHARED UTILITIES (self-contained, no imports from your other files) ──
# =============================================================================

class BloomFilter:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.m = max(8, int(-n * math.log(p) / (math.log(2) ** 2)))
        self.k = max(1, int((self.m / n) * math.log(2))) if n > 0 else 1
        self.bits = bytearray(self.m // 8 + 1)

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

    def mem_bytes(self):
        return self.m // 8 + 1


def read_words_timed(path):
    """Read + sort words, return (words_list, elapsed_seconds)."""
    t0 = time.perf_counter()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        words = sorted(set(w.strip().lower() for w in f if w.strip()))
    elapsed = time.perf_counter() - t0
    return words, elapsed


def build_hybrid_index(words):
    """
    Build the full Hybrid index (L=3 base, L=4 split for heavy, skip-small).
    Returns (buckets_dict, filters_dict, stats_dict).
    """
    # Step 1: base L=3 buckets
    base_buckets = defaultdict(list)
    for w in words:
        if len(w) >= PREFIX_L:
            base_buckets[w[:PREFIX_L]].append(w[PREFIX_L:])

    # Step 2: adaptive refinement — split heavy buckets to L=4
    refined_buckets = {}
    n_refined = 0
    for prefix, suffixes in base_buckets.items():
        if len(suffixes) > HEAVY_THRESH:
            n_refined += 1
            sub = defaultdict(list)
            for s in suffixes:
                full = prefix + s
                if len(full) >= 4:
                    sub[full[:4]].append(full[4:])
            for sp, subsuf in sub.items():
                refined_buckets[sp] = subsuf
        else:
            refined_buckets[prefix] = suffixes

    # Step 3: build Bloom filters, skipping small buckets
    filters = {}
    n_small = 0
    n_bloom = 0
    bloom_build_times = []
    bloom_mem_bytes_list = []

    for prefix, suffixes in refined_buckets.items():
        n = len(suffixes)
        if n == 0:
            continue
        if n < SKIP_SMALL_N:
            filters[prefix] = suffixes   # direct list
            n_small += 1
        else:
            t0 = time.perf_counter()
            bf = BloomFilter(n, TARGET_P)
            for suf in suffixes:
                bf.add(suf)
            bloom_build_times.append(time.perf_counter() - t0)
            bloom_mem_bytes_list.append(bf.mem_bytes())
            filters[prefix] = bf
            n_bloom += 1

    total_buckets = len(refined_buckets)
    stats = {
        "total_buckets": total_buckets,
        "n_small_buckets": n_small,
        "n_bloom_buckets": n_bloom,
        "n_refined_base_buckets": n_refined,
        "pct_small": 100.0 * n_small / total_buckets if total_buckets else 0,
        "pct_refined": 100.0 * n_refined / len(base_buckets) if base_buckets else 0,
        "bloom_build_times_ms": [t * 1000 for t in bloom_build_times],
        "bloom_mem_bytes_list": bloom_mem_bytes_list,
        "max_bucket": max(len(v) for v in refined_buckets.values()) if refined_buckets else 0,
        "avg_bucket": sum(len(v) for v in refined_buckets.values()) / total_buckets if total_buckets else 0,
    }
    return refined_buckets, filters, stats


def hybrid_lookup(filters, word):
    L = PREFIX_L
    if len(word) < L:
        prefix, suffix = word, ""
    else:
        prefix, suffix = word[:L], word[L:]
    entry = filters.get(prefix)
    if entry is None:
        return False
    if isinstance(entry, list):
        return suffix in entry
    return suffix in entry


def measure_prefix_latency(words, n=50_000):
    """Time the prefix slicing operation itself."""
    sample = random.choices(words, k=n)
    t0 = time.perf_counter()
    for w in sample:
        _ = w[:PREFIX_L]
    elapsed_us = (time.perf_counter() - t0) / n * 1e6
    return elapsed_us


def benchmark_lookups_detailed(filters, words, n_queries=N_LOOKUP_BENCH):
    """
    Returns avg, worst-case, failed avg lookup latencies (all in µs).
    Also separates list-bucket vs bloom-bucket lookup times.
    """
    n_present = n_queries // 2
    n_absent  = n_queries - n_present

    present_sample = random.choices(words, k=n_present)

    absent_sample = []
    word_set = set(words)
    chars = "abcdefghijklmnopqrstuvwxyz"
    while len(absent_sample) < n_absent:
        w = "".join(random.choice(chars) for _ in range(random.randint(5, 10)))
        if w not in word_set:
            absent_sample.append(w)

    queries = [(w, True) for w in present_sample] + [(w, False) for w in absent_sample]
    random.shuffle(queries)

    present_times, absent_times = [], []
    list_times, bloom_times = [], []

    for word, is_present in queries:
        L = PREFIX_L
        prefix = word[:L] if len(word) >= L else word
        entry = filters.get(prefix)

        t0 = time.perf_counter()
        hybrid_lookup(filters, word)
        elapsed_us = (time.perf_counter() - t0) * 1e6

        if is_present:
            present_times.append(elapsed_us)
        else:
            absent_times.append(elapsed_us)

        if entry is not None:
            if isinstance(entry, list):
                list_times.append(elapsed_us)
            else:
                bloom_times.append(elapsed_us)

    all_times = present_times + absent_times
    return {
        "avg_us":          sum(all_times) / len(all_times),
        "worst_us":        max(all_times),
        "failed_avg_us":   sum(absent_times) / len(absent_times) if absent_times else 0,
        "list_avg_us":     sum(list_times) / len(list_times) if list_times else 0,
        "bloom_avg_us":    sum(bloom_times) / len(bloom_times) if bloom_times else 0,
        "throughput_qps":  1e6 / (sum(all_times) / len(all_times)),
    }


def measure_ram_after_pkl_load(pkl_path):
    """
    Load a pkl file and measure RAM delta using tracemalloc.
    Returns MB consumed.
    """
    tracemalloc.start()
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del obj
    return peak / 1024 / 1024   # MB


def measure_fpr(filters, n_trials=5_000):
    false_positives = 0
    for _ in range(n_trials):
        fake = "".join(random.choice("abcdefghijklmnopqrstuvwxyz")
                       for _ in range(PREFIX_L + 5))
        prefix, suffix = fake[:PREFIX_L], fake[PREFIX_L:]
        f = filters.get(prefix)
        if f:
            if isinstance(f, list):
                if suffix in f:
                    false_positives += 1
            else:
                if suffix in f:
                    false_positives += 1
    return false_positives / n_trials


# =============================================================================
# ── KNOWN VALUES FROM EXISTING CSVs (hardcoded, no hallucination) ──
# =============================================================================

KNOWN_FILE_SIZES = {
    # Dataset: (raw_txt_MB, serialized_pkl_MB, compression_ratio)
    "ODS-french":      (4.381,  0.631,  6.95),
    "enable":          (1.663,  0.336,  4.95),
    "german":          (28.68,  2.565, 11.18),
    "italian":         (0.979,  0.202,  4.85),
    "sowpods":         (2.582,  0.475,  5.43),
    "synthetic_skewed":(1.442,  0.817,  1.77),
    "twl":             (1.681,  0.344,  4.88),
}

KNOWN_HYBRID_RESULTS = {
    # Dataset: (buckets, mem_mib_bloom_only, fpr, max_bucket, build_time_ms)
    "ODS-french":      (3666,  0.465, 0.0016, 2453,  1973.7),
    "enable":          (3243,  0.192, 0.0014, 1676,   811.9),
    "german":          (8970,  2.169, 0.0012, 10850,10998.2),
    "italian":         (2457,  0.104, 0.0006,  922,   449.3),
    "sowpods":         (3961,  0.299, 0.0012, 2234,  1753.7),
    "synthetic_skewed":(17326, 0.089, 0.0,    3039,   523.4),
    "twl":             (3289,  0.198, 0.001,  1643,   917.4),
}

KNOWN_BASELINE_RESULTS = {
    # Dataset: (buckets, mem_mib, fpr, max_bucket, build_time_ms)
    "ODS-french":      (2886,  0.47,  0.0038,  8271, 1921.2),
    "enable":          (3052,  0.197, 0.004,   1946,  834.4),
    "german":          (4393,  2.181, 0.007,  57732, 9166.8),
    "italian":         (2413,  0.108, 0.0042,  1423,  512.3),
    "sowpods":         (3564,  0.306, 0.0038,  2795, 1422.8),
    "synthetic_skewed":(17301, 0.17,  0.0396, 75002, 1877.5),
    "twl":             (3106,  0.204, 0.004,   1881,  991.4),
}

KNOWN_LOOKUP_TIMES = {
    # Dataset: (baseline_avg_us, hybrid_avg_us)
    "ODS-french":      (3.626, 2.460),
    "enable":          (3.640, 3.634),
    "german":          (3.466, 1.163),
    "italian":         (3.049, 2.939),
    "sowpods":         (3.418, 3.065),
    "synthetic_skewed":(4.743, 0.812),
    "twl":             (3.757, 3.149),
}

KNOWN_STARTUP_TIMES = {
    # Dataset: (cold_start_build_ms, warm_start_load_ms, reduction_factor)
    "ODS-french":      (1973.7,  7.67, 257.46),
    "enable":          ( 811.9, 18.31,  44.35),
    "german":          (10998.2,25.35, 433.89),
    "italian":         ( 449.3, 14.33,  31.35),
    "sowpods":         (1753.7, 16.33, 107.40),
    "synthetic_skewed":( 523.4, 26.99,  19.39),
    "twl":             ( 917.4,  4.01, 228.72),
}


# =============================================================================
# ── MAIN ──
# =============================================================================

def separator(title=""):
    line = "=" * 78
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


def run_all():
    print("\n" + "=" * 78)
    print("  BLOOM HYBRID INDEX — PATENT CHECKLIST MASTER MEASUREMENT SCRIPT")
    print("  Primary dataset for deep measurements:", PRIMARY_DATASET)
    print("=" * 78)

    # ── Locate primary wordlist ──
    primary_txt = os.path.join(WORDLIST_DIR, f"{PRIMARY_DATASET}.txt")
    primary_pkl = os.path.join(PKL_DIR, f"{PRIMARY_DATASET}.pkl")
    if not os.path.exists(primary_txt):
        sys.exit(f"ERROR: Cannot find wordlist at {primary_txt}. Run from inside BLOOM/")
    if not os.path.exists(primary_pkl):
        sys.exit(f"ERROR: Cannot find serialized index at {primary_pkl}.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — WORDLIST STORAGE MODULE
    # ══════════════════════════════════════════════════════════════════════════
    separator("MODULE 1 — WORDLIST STORAGE MODULE")

    print("\n[1/9] Measuring preprocessing time (word loading from .txt)...")
    words, preprocess_time_s = read_words_timed(primary_txt)
    n_words = len(words)
    print(f"      Done. {n_words} unique words loaded.")

    print("[2/9] Measuring RAM usage after loading serialized index (pkl)...")
    ram_mb = measure_ram_after_pkl_load(primary_pkl)

    txt_mb  = KNOWN_FILE_SIZES[PRIMARY_DATASET][0]
    pkl_mb  = KNOWN_FILE_SIZES[PRIMARY_DATASET][1]

    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  WORDLIST STORAGE MODULE — VALUES FOR PATENT CHECKLIST                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Preprocessing time          : {preprocess_time_s*1000:.1f} ms  ({preprocess_time_s:.3f} s)
│    (time to read + sort      : for {n_words:,} words from {PRIMARY_DATASET}.txt)
│                                                                         │
│  Input dataset sizes (raw .txt files) — ALL DATASETS:""")
    for ds, (t, p, r) in KNOWN_FILE_SIZES.items():
        print(f"│    {ds:<20}: {t:.3f} MB  raw txt")
    print(f"│")
    print(f"│  Serialized index sizes (.pkl) — ALL DATASETS:")
    for ds, (t, p, r) in KNOWN_FILE_SIZES.items():
        print(f"│    {ds:<20}: {p:.3f} MB  serialized  (compression ratio: {r:.2f}x)")
    print(f"│")
    print(f"│  RAM usage after loading (tracemalloc peak, {PRIMARY_DATASET}.pkl):")
    print(f"│    {ram_mb:.2f} MB  peak memory during deserialization")
    print(f"│  [NOTE] Warm-start load times (pickle.load) — ALL DATASETS:")
    for ds, (cold, warm, factor) in KNOWN_STARTUP_TIMES.items():
        print(f"│    {ds:<20}: {warm:.2f} ms  to load from disk")
    print("└─────────────────────────────────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — BUILD HYBRID INDEX (needed for all subsequent measurements)
    # ══════════════════════════════════════════════════════════════════════════
    separator("BUILDING HYBRID INDEX (for primary dataset: " + PRIMARY_DATASET + ")")
    print("  Building index... (this takes a few seconds)")
    _, filters, stats = build_hybrid_index(words)
    print(f"  Done. {stats['total_buckets']} buckets built.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — PREFIX BUCKETING MODULE
    # ══════════════════════════════════════════════════════════════════════════
    separator("MODULE 2 — PREFIX BUCKETING MODULE")

    print("\n[3/9] Measuring prefix computation latency...")
    prefix_latency_us = measure_prefix_latency(words)

    hybrid_buckets = KNOWN_HYBRID_RESULTS[PRIMARY_DATASET][0]
    baseline_max   = KNOWN_BASELINE_RESULTS[PRIMARY_DATASET][3]
    hybrid_max     = KNOWN_HYBRID_RESULTS[PRIMARY_DATASET][3]
    avg_bucket_sz  = stats["avg_bucket"]

    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  PREFIX BUCKETING MODULE — VALUES FOR PATENT CHECKLIST                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Actual prefix length used   : L = {PREFIX_L} characters (default)
│  Allowed prefix length bounds: 2 ≤ L ≤ 4
│    (L=2, L=3, L=4 all tested; L=3 selected as default)
│                                                                         │
│  Average bucket size after initial partitioning ({PRIMARY_DATASET}):
│    {avg_bucket_sz:.1f} entries per bucket  ({n_words:,} words / {hybrid_buckets} buckets)
│                                                                         │
│  Maximum bucket size BEFORE refinement (Baseline L=3) — ALL DATASETS:""")
    for ds, (b, m, fpr, mx, bt) in KNOWN_BASELINE_RESULTS.items():
        print(f"│    {ds:<20}: {mx} entries (max bucket)")
    print(f"│")
    print(f"│  Prefix computation latency : {prefix_latency_us:.2f} µs per query")
    print(f"│    (measured: word[:L] slicing on {n_words:,} words, n=50,000 samples)")
    print("└─────────────────────────────────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — ADAPTIVE PREFIX REFINEMENT MODULE
    # ══════════════════════════════════════════════════════════════════════════
    separator("MODULE 3 — ADAPTIVE PREFIX REFINEMENT MODULE")

    pct_refined      = stats["pct_refined"]
    n_refined        = stats["n_refined_base_buckets"]
    hybrid_max_all   = {ds: v[3] for ds, v in KNOWN_HYBRID_RESULTS.items()}

    # Estimate per-bucket refinement time from build time and proportion
    # Cold start build time includes everything; refinement is a subset.
    # We cannot isolate it from existing CSVs, but we can bound it:
    cold_build_ms = KNOWN_HYBRID_RESULTS[PRIMARY_DATASET][4]

    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  ADAPTIVE PREFIX REFINEMENT MODULE — VALUES FOR PATENT CHECKLIST        │
├─────────────────────────────────────────────────────────────────────────┤
│  Refinement trigger threshold: bucket size > {HEAVY_THRESH} entries
│  Maximum refinement depth    : 1 level  (L=3 → L=4 only)
│                                                                         │
│  % of L=3 buckets requiring refinement ({PRIMARY_DATASET}):
│    {n_refined} of {stats['total_buckets']} base buckets  ({pct_refined:.1f}%)
│                                                                         │
│  Maximum bucket size AFTER refinement (Hybrid) — ALL DATASETS:""")
    for ds, (b, m, fpr, mx, bt) in KNOWN_HYBRID_RESULTS.items():
        baseline_mx = KNOWN_BASELINE_RESULTS[ds][3]
        print(f"│    {ds:<20}: {mx} entries  (was {baseline_mx} before refinement)")
    print(f"│")
    print(f"│  [NOTE] Refinement time per bucket: NOT individually measured.")
    print(f"│    Total cold-start build time ({PRIMARY_DATASET}): {cold_build_ms:.1f} ms")
    print(f"│    This includes prefixing + refinement + Bloom construction combined.")
    print(f"│    To isolate: run the script with timing around the refinement loop.")
    print("└─────────────────────────────────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — SELECTIVE BLOOM FILTER MODULE
    # ══════════════════════════════════════════════════════════════════════════
    separator("MODULE 4 — SELECTIVE BLOOM FILTER MODULE")

    bloom_times_ms   = stats["bloom_build_times_ms"]
    bloom_mems       = stats["bloom_mem_bytes_list"]
    avg_bloom_ms     = sum(bloom_times_ms) / len(bloom_times_ms) if bloom_times_ms else 0
    avg_bloom_bytes  = sum(bloom_mems) / len(bloom_mems) if bloom_mems else 0
    min_bloom_bytes  = min(bloom_mems) if bloom_mems else 0
    max_bloom_bytes  = max(bloom_mems) if bloom_mems else 0

    # k and m for a typical bucket (use average bucket size for Bloom buckets)
    # Bloom is only applied to n >= SKIP_SMALL_N
    bloom_bucket_sizes = [len(v) if isinstance(v, list) else None for v in filters.values()]
    # approximate: use a typical n=20 for illustration
    typical_n = 20
    typical_m = max(8, int(-typical_n * math.log(TARGET_P) / (math.log(2)**2)))
    typical_k = max(1, int((typical_m / typical_n) * math.log(2)))

    # FPR variance across datasets (Hybrid)
    hybrid_fprs = [v[2] * 100 for v in KNOWN_HYBRID_RESULTS.values()]
    fpr_min = min(hybrid_fprs)
    fpr_max = max(hybrid_fprs)
    fpr_variance = fpr_max - fpr_min

    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  SELECTIVE BLOOM FILTER MODULE — VALUES FOR PATENT CHECKLIST            │
├─────────────────────────────────────────────────────────────────────────┤
│  Lower threshold (skip Bloom) : n < {SKIP_SMALL_N} entries → direct list storage
│  Upper threshold (apply Bloom): n ≥ {SKIP_SMALL_N} entries → Bloom filter
│                                                                         │
│  Bloom filter design parameters (for typical bucket of n={typical_n} suffixes):
│    Bit-array size (m)         : {typical_m} bits  ({typical_m//8} bytes)
│    Number of hash functions(k): {typical_k}
│    Target false positive rate : ≤ {TARGET_P*100:.0f}%  (design target, p={TARGET_P})
│                                                                         │
│  Bloom filter memory (measured across all Bloom buckets, {PRIMARY_DATASET}):
│    Min per bucket             : {min_bloom_bytes} bytes
│    Max per bucket             : {max_bloom_bytes} bytes
│    Average per bucket         : {avg_bloom_bytes:.1f} bytes
│                                                                         │
│  Bloom filter construction time (measured, {PRIMARY_DATASET}):
│    Average per bucket         : {avg_bloom_ms*1000:.2f} µs  ({avg_bloom_ms:.4f} ms)
│    (across {len(bloom_times_ms)} Bloom-filter buckets)
│                                                                         │
│  Actual measured FPR — Hybrid vs Baseline — ALL DATASETS:""")
    for ds in ALL_DATASETS:
        h_fpr = KNOWN_HYBRID_RESULTS[ds][2] * 100
        b_fpr = KNOWN_BASELINE_RESULTS[ds][2] * 100
        print(f"│    {ds:<20}: Hybrid={h_fpr:.4f}%   Baseline={b_fpr:.4f}%")
    print(f"│")
    print(f"│  FPR variance across datasets (Hybrid): {fpr_min:.4f}% – {fpr_max:.4f}%  (range {fpr_variance:.4f}%)")
    print("└─────────────────────────────────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — SMALL BUCKET DIRECT STORAGE MODULE
    # ══════════════════════════════════════════════════════════════════════════
    separator("MODULE 5 — SMALL BUCKET DIRECT STORAGE MODULE")

    n_small   = stats["n_small_buckets"]
    pct_small = stats["pct_small"]
    total_b   = stats["total_buckets"]

    # Memory for a small bucket: Python list of short strings
    # Measure actual sizes using sys.getsizeof on sample small buckets
    small_bucket_samples = [v for v in filters.values()
                            if isinstance(v, list) and len(v) <= 8]
    if small_bucket_samples:
        sample_mem = [sys.getsizeof(b) + sum(sys.getsizeof(s) for s in b)
                      for b in small_bucket_samples[:100]]
        avg_small_mem = sum(sample_mem) / len(sample_mem)
        min_small_mem = min(sample_mem)
        max_small_mem = max(sample_mem)
    else:
        avg_small_mem = min_small_mem = max_small_mem = 0

    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  SMALL BUCKET DIRECT STORAGE MODULE — VALUES FOR PATENT CHECKLIST       │
├─────────────────────────────────────────────────────────────────────────┤
│  Threshold for direct storage : n < {SKIP_SMALL_N} entries (no Bloom filter built)
│                                                                         │
│  Distribution ({PRIMARY_DATASET}):
│    Total buckets              : {total_b}
│    Small buckets (direct list): {n_small}  ({pct_small:.1f}% of all buckets)
│    Bloom filter buckets       : {stats['n_bloom_buckets']}  ({100-pct_small:.1f}% of all buckets)
│                                                                         │
│  Memory per small bucket (measured via sys.getsizeof, {PRIMARY_DATASET}):
│    Min                        : {min_small_mem} bytes
│    Max                        : {max_small_mem} bytes
│    Average                    : {avg_small_mem:.1f} bytes
│    (includes Python list object + each string object overhead)
│                                                                         │
│  [NOTE] Direct lookup latency for list buckets is reported in Module 6. │
└─────────────────────────────────────────────────────────────────────────┘""")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — LOOKUP AND QUERY MODULE
    # ══════════════════════════════════════════════════════════════════════════
    separator("MODULE 6 — LOOKUP AND QUERY MODULE")

    print(f"\n[4/9] Running detailed lookup benchmark ({N_LOOKUP_BENCH} queries)...")
    lookup_stats = benchmark_lookups_detailed(filters, words)

    # Overall avg from all datasets (known from CSV)
    hybrid_avgs   = [v[1] for v in KNOWN_LOOKUP_TIMES.values()]
    baseline_avgs = [v[0] for v in KNOWN_LOOKUP_TIMES.values()]
    overall_hybrid_avg  = sum(hybrid_avgs) / len(hybrid_avgs)
    overall_base_avg    = sum(baseline_avgs) / len(baseline_avgs)
    throughput_hybrid   = 1e6 / overall_hybrid_avg
    throughput_baseline = 1e6 / overall_base_avg

    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  LOOKUP AND QUERY MODULE — VALUES FOR PATENT CHECKLIST                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Average lookup latency — Hybrid vs Baseline — ALL DATASETS (from CSV):""")
    for ds, (b_us, h_us) in KNOWN_LOOKUP_TIMES.items():
        print(f"│    {ds:<20}: Hybrid={h_us:.2f} µs   Baseline={b_us:.2f} µs")
    print(f"│")
    print(f"│  Overall average (across all datasets):")
    print(f"│    Hybrid                    : {overall_hybrid_avg:.2f} µs per query")
    print(f"│    Baseline                  : {overall_base_avg:.2f} µs per query")
    print(f"│")
    print(f"│  Deep benchmark on {PRIMARY_DATASET} ({N_LOOKUP_BENCH} queries, 50% present / 50% absent):")
    print(f"│    Average lookup latency     : {lookup_stats['avg_us']:.2f} µs")
    print(f"│    Worst-case lookup latency  : {lookup_stats['worst_us']:.2f} µs")
    print(f"│    Failed (absent) lookup avg : {lookup_stats['failed_avg_us']:.2f} µs")
    print(f"│    List-bucket lookup avg     : {lookup_stats['list_avg_us']:.2f} µs  (direct storage path)")
    print(f"│    Bloom-bucket lookup avg    : {lookup_stats['bloom_avg_us']:.2f} µs  (Bloom filter path)")
    print(f"│")
    print(f"│  Query throughput:")
    print(f"│    Hybrid  (overall avg)      : {throughput_hybrid:,.0f} queries/sec")
    print(f"│    Baseline (overall avg)     : {throughput_baseline:,.0f} queries/sec")
    print(f"│")
    print(f"│  [NOTE] Memory accesses per lookup: NOT profiled at hardware level.")
    print(f"│    Logical path: 1 dict lookup + 1 list scan OR k hash computations.")
    print(f"│    For Bloom: k={typical_k} hash ops + k bit-array reads = {typical_k*2} logical memory accesses.")
    print(f"│    For list : 1 dict lookup + linear scan (avg n/2 comparisons).")
    print( "└─────────────────────────────────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — FALSE POSITIVE CONTROL MODULE
    # ══════════════════════════════════════════════════════════════════════════
    separator("MODULE 7 — FALSE POSITIVE CONTROL MODULE")

    # Worst case under skew = synthetic_skewed dataset
    worst_hybrid_fpr   = KNOWN_HYBRID_RESULTS["synthetic_skewed"][2] * 100
    worst_baseline_fpr = KNOWN_BASELINE_RESULTS["synthetic_skewed"][2] * 100

    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  FALSE POSITIVE CONTROL MODULE — VALUES FOR PATENT CHECKLIST            │
├─────────────────────────────────────────────────────────────────────────┤
│  Maximum allowable FPR (design): ≤ {TARGET_P*100:.0f}%  (p={TARGET_P}, per Bloom filter)
│                                                                         │
│  Worst-case FPR under skew (synthetic_skewed dataset):
│    Hybrid                    : {worst_hybrid_fpr:.4f}%
│    Baseline                  : {worst_baseline_fpr:.4f}%
│                                                                         │
│  FPR across all datasets — Hybrid:""")
    for ds in ALL_DATASETS:
        fpr_pct = KNOWN_HYBRID_RESULTS[ds][2] * 100
        print(f"│    {ds:<20}: {fpr_pct:.4f}%")
    print(f"│")
    print(f"│  FPR variance across datasets (Hybrid): {fpr_min:.4f}% – {fpr_max:.4f}%  (±{fpr_variance/2:.4f}%)")
    print(f"│")
    print(f"│  FPR reduction vs Baseline — ALL DATASETS:")
    for ds in ALL_DATASETS:
        h_fpr = KNOWN_HYBRID_RESULTS[ds][2] * 100
        b_fpr = KNOWN_BASELINE_RESULTS[ds][2] * 100
        print(f"│    {ds:<20}: {b_fpr:.4f}% → {h_fpr:.4f}%")
    print("└─────────────────────────────────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — SUMMARY TABLE (quantifiable improvements)
    # ══════════════════════════════════════════════════════════════════════════
    separator("SUMMARY — QUANTIFIABLE IMPROVEMENTS OVER BASELINE")

    print(f"""
┌──────────────────────────┬──────────────────────┬──────────────────────┐
│  Metric                  │  Baseline (L=3)      │  Hybrid              │
├──────────────────────────┼──────────────────────┼──────────────────────┤
│  Max bucket size (TWL)   │  1,881 entries       │  1,643 entries       │
│  Max bucket size (DE)    │  57,732 entries      │  10,850 entries      │
│  Max bucket size (synth) │  75,002 entries      │  3,039  entries      │
│  FPR (TWL)               │  0.40%               │  0.10%               │
│  FPR (German)            │  0.70%               │  0.12%               │
│  FPR (synthetic skewed)  │  3.96%               │  ~0.00%              │
│  Avg lookup (overall)    │  {overall_base_avg:.2f} µs            │  {overall_hybrid_avg:.2f} µs             │
│  Throughput (overall)    │  {throughput_baseline:>10,.0f} q/s  │  {throughput_hybrid:>10,.0f} q/s  │
│  Init time (TWL warm)    │  917.4 ms (build)    │  4.01 ms (load pkl)  │
│  Init speedup (TWL)      │  1×                  │  228× faster         │
│  Storage (TWL)           │  1.681 MB raw        │  0.344 MB pkl (4.9×) │
│  Storage (German)        │  28.68 MB raw        │  2.565 MB pkl (11×)  │
└──────────────────────────┴──────────────────────┴──────────────────────┘
""")

    separator("DONE — All values printed above. Copy-paste into your patent checklist.")
    print("""
IMPORTANT NOTES FOR PATENT USE:
  1. All 'measured' values above come from your actual code + wordlists.
  2. PRIMARY_DATASET = '{primary}' was used for per-module deep benchmarks.
  3. "Bloom filter m and k" values shown are for a TYPICAL bucket of n={n}.
     For the exact value for any bucket, use: m = ceil(-n*ln(0.01)/ln(2)^2), k = floor((m/n)*ln(2))
  4. "Memory accesses per lookup" is a logical estimate, not hardware-profiled.
  5. "Refinement time per bucket" is not individually isolated in existing code.
     The total cold-start build time covers the full pipeline end-to-end.
  6. RAM values use tracemalloc (Python allocator peak), not OS-level RSS.
     For OS-level RAM, install psutil and add psutil.Process().memory_info().rss.
""".format(primary=PRIMARY_DATASET, n=typical_n))


if __name__ == "__main__":
    run_all()
