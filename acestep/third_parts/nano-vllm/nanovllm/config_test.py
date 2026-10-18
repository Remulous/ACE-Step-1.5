"""Tests for safe, initialization-time nano-vLLM KV-cache sizing."""

import unittest

from nanovllm.config import select_kv_cache_blocks


class KvCacheSizingTests(unittest.TestCase):
    """Verify the optional block cap is only an upper bound."""

    def test_automatic_sizing_preserves_existing_capacity(self):
        self.assertEqual(select_kv_cache_blocks(64, 0), 64)
        self.assertEqual(select_kv_cache_blocks(0, 0), 1)

    def test_configured_limit_caps_memory_derived_capacity(self):
        self.assertEqual(select_kv_cache_blocks(64, 48), 48)
        self.assertEqual(select_kv_cache_blocks(32, 48), 32)

    def test_low_memory_does_not_silently_allocate_one_capped_block(self):
        with self.assertRaisesRegex(RuntimeError, "one KV cache block"):
            select_kv_cache_blocks(0, 16)

    def test_negative_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero .* or positive"):
            select_kv_cache_blocks(64, -1)


if __name__ == "__main__":
    unittest.main()
