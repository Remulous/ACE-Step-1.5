"""Tests for opt-in request-level cache occupancy instrumentation."""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.engine.llm_engine import LLMEngine


class CacheProfileTests(unittest.TestCase):
    """Exercise profiler bookkeeping without CUDA initialization."""

    def test_schedule_records_peak_blocks_and_sequence_lengths(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine._active_cache_profile = {
            "peak_used_blocks": 0,
            "sequence_peak_token_lengths": {},
        }
        engine.scheduler = SimpleNamespace(
            block_manager=SimpleNamespace(used_block_ids={0, 1})
        )
        class SizedSequence:
            seq_id = 7

            def __len__(self):
                return 300

        engine._record_cache_profile([SizedSequence()])
        self.assertEqual(engine._active_cache_profile["peak_used_blocks"], 2)
        self.assertEqual(
            engine._active_cache_profile["sequence_peak_token_lengths"], {"7": 300}
        )

    def test_generation_profile_is_disabled_by_default(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine._generate_lock = threading.Lock()
        engine._closed = False
        engine.scheduler = SimpleNamespace(
            block_manager=SimpleNamespace(
                used_block_ids=set(), free_block_ids=set(range(8)), blocks=list(range(8))
            )
        )
        engine._last_cache_profile = None
        engine._active_cache_profile = None
        engine._generate_impl = lambda *_args: "ok"
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(engine.generate([], []), "ok")
        self.assertIsNone(engine._last_cache_profile)
        self.assertIsNone(engine._active_cache_profile)


if __name__ == "__main__":
    unittest.main()
