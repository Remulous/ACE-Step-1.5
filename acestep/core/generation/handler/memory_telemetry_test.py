"""Focused tests for opt-in peak accounting and resource lifetime correctness."""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from . import memory_telemetry as mt
from .memory_inventory import storage_bytes
from .residency_request import gate, request_scope, worker_target


class MemoryTelemetryTests(unittest.TestCase):
    """Check disabled path, peak reset nesting and worker-owned finalization."""

    def setUp(self):
        """Mock CUDA accounting only; real GPU validity is tested in staging."""
        self.host = SimpleNamespace(device="cuda", offload_to_cpu=False)
        self.events = []
        self.patches = [
            patch.dict(os.environ, {"ACESTEP_MEMORY_PROFILE": "attribution"}),
            patch.object(mt, "emit", side_effect=lambda event, **kw: self.events.append((event, kw))),
            patch.object(mt, "inventory", return_value={}),
            patch.object(mt.torch.cuda, "synchronize"),
            patch.object(mt.torch.cuda, "reset_peak_memory_stats"),
            patch.object(mt, "sample", return_value={"allocated": 10, "reserved": 20,
                         "peak_allocated": 10, "peak_reserved": 20}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_nested_peaks_survive_resets(self):
        """An inner phase cannot erase the parent/request high-water mark."""
        profile = mt.Profile(self.host, "test")
        with profile.phase("outer"):
            mt.sample.return_value = {"allocated": 30, "reserved": 40,
                                      "peak_allocated": 100, "peak_reserved": 120}
            with profile.phase("inner"):
                mt.sample.return_value = {"allocated": 30, "reserved": 40,
                                          "peak_allocated": 50, "peak_reserved": 60}
        profile.finish()
        ends = [value for name, value in self.events if name == "phase_end"]
        self.assertEqual(ends[-1]["interval_aggregate_peak_allocated"], 100)
        self.assertEqual(self.events[-1][1]["request_peak_allocated"], 100)

    def test_disabled_does_not_touch_cuda(self):
        """Normal inference disabled path calls original function with no CUDA work."""
        with patch.dict(os.environ, {"ACESTEP_MEMORY_PROFILE": "off"}):
            wrapped = mt.measured("test")(lambda host: 42)
            self.assertEqual(wrapped(self.host), 42)
        mt.torch.cuda.synchronize.assert_not_called()
        mt.torch.cuda.reset_peak_memory_stats.assert_not_called()

    def test_exception_records_and_removes_probe(self):
        """Exceptional operation exits do not retain spans/hooks."""
        self.host.model = torch.nn.Linear(1, 1)
        profile = mt.Profile(self.host, "exception")
        with self.assertRaisesRegex(ValueError, "expected"):
            with profile.phase("bad"):
                raise ValueError("expected")
        self.assertEqual(profile.spans, [])
        profile.finish()
        record = next(v for k, v in self.events if k == "phase_end")
        self.assertEqual(record["error"], "ValueError")

    def test_timeout_caller_does_not_finish_live_worker(self):
        """Worker reference owns final peak read and admission until it really exits."""
        started, finish = threading.Event(), threading.Event()
        with request_scope(self.host) as request:
            target, _ = worker_target(lambda: (started.set(), finish.wait(2)))
            worker = threading.Thread(target=target)
            worker.start()
            self.assertTrue(started.wait(1))
        self.assertTrue(gate(self.host).locked())
        self.assertFalse(any(k == "request_end" for k, _ in self.events))
        finish.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(gate(self.host).locked())
        self.assertEqual(sum(k == "request_end" for k, _ in self.events), 1)

    def test_different_handlers_cannot_reset_counters_concurrently(self):
        """Profiling serializes CUDA peak ownership across independent handlers."""
        other = SimpleNamespace(device="cuda", offload_to_cpu=False)
        attempting, entered = threading.Event(), threading.Event()

        def second():
            attempting.set()
            with request_scope(other):
                entered.set()

        with request_scope(self.host):
            worker = threading.Thread(target=second)
            worker.start()
            self.assertTrue(attempting.wait(1))
            self.assertFalse(entered.wait(0.02))
        worker.join(2)
        self.assertTrue(entered.is_set())
        self.assertFalse(worker.is_alive())

    def test_forward_probes_removed_after_worker_finishes(self):
        """Temporary hooks observe forwards without retaining a module after completion."""
        decoder = torch.nn.Linear(2, 2)
        self.host.model = SimpleNamespace(decoder=decoder)
        profile = mt.Profile(self.host, "hooks")
        decoder(torch.zeros(1, 2))
        self.assertEqual(len(decoder._forward_pre_hooks), 1)
        self.assertEqual(len(decoder._forward_hooks), 1)
        profile.finish()
        self.assertEqual(len(decoder._forward_pre_hooks), 0)
        self.assertEqual(len(decoder._forward_hooks), 0)
        self.assertEqual(profile.spans, [])

    def test_shared_storage_counted_once(self):
        """Tied weights/views are not summed as independent allocations."""
        tensor = torch.zeros(10)
        totals = storage_bytes([tensor, tensor[:5], tensor])
        self.assertEqual(sum(totals.values()), 40)

    def test_timing_mode_does_not_reset_subphase(self):
        """Representative timing mode does not add a sync for every nested helper."""
        with patch.dict(os.environ, {"ACESTEP_MEMORY_PROFILE": "timing"}):
            profile = mt.Profile(self.host, "timing")
            before = mt.torch.cuda.synchronize.call_count
            with profile.phase("outer"):
                with profile.phase("inner"):
                    pass
            self.assertEqual(mt.torch.cuda.synchronize.call_count, before)
            self.assertEqual(mt.torch.cuda.reset_peak_memory_stats.call_count, 1)


if __name__ == "__main__":
    unittest.main()
