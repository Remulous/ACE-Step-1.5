"""Tests for nano-vLLM engine close ordering and idempotence."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.engine.llm_engine import LLMEngine


class EngineLifecycleTests(unittest.TestCase):
    """Exercise lifecycle coordination without initializing CUDA."""

    def engine(self):
        """Create a minimal engine shell with observable cleanup operations."""
        calls = []
        engine = LLMEngine.__new__(LLMEngine)
        engine._generate_lock = threading.Lock()
        engine._closed = False
        engine._atexit_callback = None
        engine.ps = []
        engine.events = []
        engine.scheduler = SimpleNamespace(is_finished=lambda: True)
        engine.model_runner = SimpleNamespace(call=lambda name: calls.append(name))
        engine.tokenizer = object()
        engine.calls = calls
        engine.reset = lambda: calls.append("reset")
        return engine

    def test_close_releases_runner_once(self):
        """A second close is harmless and the CUDA barrier precedes deletion."""
        engine = self.engine()
        with patch("torch.cuda.synchronize") as synchronize:
            engine.close()
            engine.close()
        self.assertEqual(engine.calls, ["reset", "exit"])
        self.assertIsNone(engine.scheduler)
        self.assertIsNone(engine.tokenizer)
        self.assertTrue(engine._closed)
        synchronize.assert_any_call()

    def test_close_waits_for_generation_lock(self):
        """Close cannot enter runner teardown while generate owns its lock."""
        engine = self.engine()
        started = threading.Event()
        completed = threading.Event()

        def close_engine():
            started.set()
            engine.close()
            completed.set()

        with patch("torch.cuda.synchronize"):
            engine._generate_lock.acquire()
            worker = threading.Thread(target=close_engine)
            worker.start()
            self.assertTrue(started.wait(1))
            time.sleep(0.05)
            self.assertFalse(completed.is_set())
            engine._generate_lock.release()
            worker.join(2)
        self.assertTrue(completed.is_set())
        self.assertEqual(engine.calls, ["reset", "exit"])

    def test_close_propagates_runner_failure(self):
        """A failed runner teardown must not be reported as a closed engine."""
        engine = self.engine()
        engine.model_runner.call = lambda _name: (_ for _ in ()).throw(RuntimeError("close"))
        with patch("torch.cuda.synchronize"), self.assertRaisesRegex(RuntimeError, "close"):
            engine.close()
        self.assertFalse(engine._closed)
        self.assertIsNotNone(engine.model_runner)


if __name__ == "__main__":
    unittest.main()
