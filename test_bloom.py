import os
import unittest
import tempfile
from hybrid_adaptive_prefix_bloom import (
    BloomFilter,
    read_words,
    build_prefix_buckets,
    build_bloom_filters,
    measure_fpr,
    run_hybrid
)
from bloom_prefix_optimization_experiments import (
    run_adaptive_prefixing,
    run_skip_small,
    run_adaptive_sizing
)


class TestBloomPrefixProject(unittest.TestCase):

    def test_bloom_filter_basic(self):
        bf = BloomFilter(n=100, p=0.01)
        self.assertGreater(bf.m, 0)
        self.assertGreater(bf.k, 0)

        items = ["apple", "banana", "cherry", "date"]
        for item in items:
            bf.add(item)

        for item in items:
            self.assertIn(item, bf)

        self.assertEqual(bf.n, 100)
        self.assertEqual(bf.p, 0.01)

    def test_bloom_filter_empty_and_edges(self):
        bf = BloomFilter(n=1, p=0.05)
        self.assertGreaterEqual(bf.m, 8)
        self.assertGreaterEqual(bf.k, 1)
        bf.add("test")
        self.assertIn("test", bf)

    def test_read_words(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = os.path.join(tmpdir, "test_words.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("Apple\nBANANA\n  apple  \n\ncherry\n")

            words = read_words(p)
            self.assertEqual(words, ["apple", "banana", "cherry"])

    def test_build_prefix_buckets(self):
        words = ["cat", "car", "dog", "dot", "a"]
        buckets = build_prefix_buckets(words, L=3)
        self.assertIn("cat", buckets)
        self.assertIn("car", buckets)
        self.assertIn("dog", buckets)
        self.assertIn("dot", buckets)
        self.assertEqual(len(buckets["cat"]), 1)
        self.assertEqual(buckets["cat"][0], "")

    def test_build_bloom_filters_skip_small(self):
        buckets = {
            "abc": ["1", "2", "3"],
            "def": [str(i) for i in range(15)]
        }

        filters, mem = build_bloom_filters(buckets, p=0.01, skip_small=True)

        self.assertIsInstance(filters["abc"], list)
        self.assertEqual(filters["abc"], ["1", "2", "3"])

        self.assertIsInstance(filters["def"], BloomFilter)
        self.assertGreater(mem, 0)

    def test_adaptive_prefixing_heavy_bucket(self):
        words = [f"abc{i:04d}" for i in range(50)]
        words.extend(["xyz1", "xyz2"])

        result = run_hybrid(words, heavy_thresh=10)
        self.assertGreater(result["Buckets"], 0)

    def test_fpr_bound(self):
        words = [f"word{i}" for i in range(200)]
        buckets = build_prefix_buckets(words, L=2)
        filters, _ = build_bloom_filters(buckets, p=0.01, skip_small=False)

        fpr = measure_fpr(filters, L=2, n_trials=500)
        self.assertLess(fpr, 0.20)


if __name__ == "__main__":
    unittest.main()
