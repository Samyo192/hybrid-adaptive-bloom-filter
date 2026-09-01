# Hybrid Adaptive Prefix Bloom Filter

A Python implementation of an optimized prefix-bucketed Bloom filter scheme designed for memory-efficient dictionary lookups.

## Overview

This project explores and implements optimizations for dictionary lookups using Bloom filters partitioned by word prefixes. Rather than applying a uniform Bloom filter across all partitions, the system introduces:

- **Adaptive Prefixing**: Automatically increases prefix length (e.g., from $L=3$ to $L=4$) for high-frequency "heavy" buckets to control maximum bucket size.
- **Bucket-Size Thresholding**: Bypasses Bloom filter construction for small buckets, storing suffixes as direct lists to optimize memory and lookup efficiency.

## Repository Structure

- `hybrid_adaptive_prefix_bloom.py`: Core hybrid adaptive prefix Bloom filter implementation and evaluation logic.
- `bloom_prefix_optimization_experiments.py`: Individual optimization experiments (baseline, adaptive prefixing, skip-small, adaptive sizing).
- `run_all_experiments.py` & `run_multilingual_experiments.py`: Benchmarking scripts across various wordlists and synthetic datasets.
- `serialize_hybrid_index.py`: Script to build and serialize optimized indexes using `pickle`.
- `load_and_lookup.py`: Utility to load serialized indexes and perform queries.
- `wordlists/`: Directory containing dictionary files (TWL06, SOWPODS, ENABLE, ODS, etc.).

## Usage

### Prerequisites
- Python 3.8+
- `pandas`

### Running the Hybrid Model
```bash
python hybrid_adaptive_prefix_bloom.py
```

### Running Benchmarks
```bash
python run_all_experiments.py
```

### Serializing and Querying Indices
Build and serialize indices:
```bash
python serialize_hybrid_index.py
```

Interactive lookup:
```bash
python load_and_lookup.py
```
