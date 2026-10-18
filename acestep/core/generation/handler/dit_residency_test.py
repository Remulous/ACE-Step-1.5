"""CPU-only regression tests for DiT ownership, nesting, and phase boundaries."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from acestep.core.generation.handler.dit_residency import DiTResidency


class Host:
    """Represent device placement without allocating CUDA tensors."""

    def __init__(self):
        self.device = "cuda:0"
        self.dtype = torch.bfloat16
        self.current_offload_cost = 0
        self.moves = []
        self.tensor = SimpleNamespace(device=torch.device("cpu"))
        self.model = SimpleNamespace(parameters=lambda: iter([self.tensor]), buffers=lambda: [])

    def _recursive_to_device(self, model, device, dtype):
        self.moves.append(str(device))
        self.tensor.device = torch.device(device)

    def _synchronize(self):
        pass

    def _release_system_memory(self):
        pass


class ResidencyTests(unittest.TestCase):
    """Validate actual transitions, exclusion, and all exit paths."""

    def setUp(self):
        self.host = Host()
        self.residency = DiTResidency(self.host)

    def test_nested_lease_moves_only_at_outer_boundaries(self):
        with self.residency.lease():
            with self.residency.lease():
                self.assertEqual(self.residency.depth, 2)
            self.assertEqual(self.host.moves, ["cuda:0"])
        self.assertEqual(self.host.moves, ["cuda:0", "cpu"])
        self.assertEqual(self.residency.depth, 0)

    def test_sequential_phase_leases_share_residency(self):
        with self.residency.phase():
            for _ in range(3):
                with self.residency.lease():
                    pass
            self.assertEqual(self.host.moves, ["cuda:0"])
        self.assertEqual(self.host.moves, ["cuda:0", "cpu"])

    def test_exception_and_cancellation_restore_cpu(self):
        for error in (ValueError, KeyboardInterrupt):
            with self.assertRaises(error), self.residency.phase():
                with self.residency.lease():
                    raise error()
            self.assertEqual(self.host.tensor.device.type, "cpu")
            self.assertEqual(self.residency.depth, 0)
            self.assertEqual(self.residency.phases, 0)

    def test_already_resident_skips_cuda_transfer(self):
        self.host.tensor.device = torch.device("cuda:0")
        with self.residency.lease():
            pass
        self.assertEqual(self.host.moves, ["cpu"])

    def test_vae_boundary_offloads_sticky_model(self):
        with self.residency.phase():
            with self.residency.lease():
                with self.assertRaises(RuntimeError):
                    self.residency.before_vae()
            self.residency.before_vae()
            self.assertEqual(self.host.tensor.device.type, "cpu")

    def test_concurrent_caller_cannot_move_active_model(self):
        entered = threading.Event()
        attempted = threading.Event()
        def other():
            attempted.set()
            with self.residency.lease():
                entered.set()
        with self.residency.lease():
            thread = threading.Thread(target=other)
            thread.start()
            self.assertTrue(attempted.wait(1))
            self.assertFalse(entered.wait(0.05))
            self.assertEqual(self.host.moves, ["cuda:0"])
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(entered.is_set())
        self.assertEqual(self.host.moves, ["cuda:0", "cpu", "cuda:0", "cpu"])

    def test_transfer_failure_blocks_further_use(self):
        with patch.object(self.host, "_recursive_to_device", side_effect=RuntimeError("failure")):
            with self.assertRaises(RuntimeError), self.residency.lease():
                pass
        self.assertTrue(self.residency.poisoned)
        with self.assertRaisesRegex(RuntimeError, "restart required"):
            with self.residency.lease():
                pass


if __name__ == "__main__":
    unittest.main()
