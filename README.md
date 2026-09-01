# Hybrid Adaptive Prefix Bloom Filter

A Python implementation of a hybrid adaptive prefix bloom filter optimization scheme for efficient dictionary and wordlist lookup, featuring prefix bucketing, adaptive prefix length scaling for heavy buckets, and raw list fallbacks for small buckets.

## Tech Stack
- **Python 3**
- Standard libraries: `os`, `math`, `time`, `random`, `hashlib`, `collections`, `pickle`
- Optional data processing/plotting dependencies referenced in experiments: `pandas`, `matplotlib` (or similar for plotting)

## Features & Architecture
1. **Core Bloom Filter**: Implements a standard Bloom filter with optimal bit array size ($m$) and hash function count ($k$) calculations using MD5 and SHA-1 based double hashing.
2. **Prefix Bucketing**: Groups words by a fixed prefix length ($L$), dividing the dictionary into manageable sub-buckets.
3. **Adaptive Prefixing**: Dynamically increases prefix length (e.g., from $L=3$ to $L=4$) for heavy buckets that exceed a specified frequency threshold.
4. **Small Bucket Optimization**: Skips Bloom filter allocation for small buckets, storing suffixes as raw Python lists for space/time efficiency.
5. **Serialization & Benchmarking**: Includes utilities to serialize optimized indexes using `pickle`, run multilingual/synthetic benchmarks, and measure False Positive Rates (FPR), memory usage, and build/lookup times.

## Setup & Running
1. Ensure Python 3 is installed.
2. Place your target wordlist files (e.g., `twl06.txt` or files in `wordlists/`) in the working directory.
3. Run the hybrid benchmark or experiments:
   ```bash
   python hybrid_adaptive_prefix_bloom.py
   python run_all_experiments.py
   ```
4. Build and serialize indexes:
   ```bash
   python serialize_hybrid_index.py
   ```
5. Perform interactive lookups:
   ```bash
   python load_and_lookup.py
   ```
