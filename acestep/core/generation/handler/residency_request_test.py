"""Request admission and detached-worker ownership regression tests."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.core.generation.handler.generate_music_execute_test import _Host as ExecuteHost

from acestep.core.generation.handler.residency_request import (
    enabled, gate, guarded, request_scope, worker_target,
)


class RequestTests(unittest.TestCase):
    """Ensure a caller timeout cannot admit another request before worker completion."""

    def test_real_execute_timeout_keeps_admission_until_worker_exits(self):
        host = ExecuteHost()
        host.device = "cpu"
        started = threading.Event()
        finish = threading.Event()
        def service(**kwargs):
            started.set()
            if not finish.wait(2):
                raise RuntimeError("test worker did not complete")
            return {"target_latents": "ok"}
        host.service_generate = service
        inputs = dict(captions_batch=["caption"], lyrics_batch=["lyrics"],
                      metas_batch=["meta"], vocal_languages_batch=["en"],
                      target_wavs_tensor=None, repainting_start_batch=[0],
                      repainting_end_batch=[1], instructions_batch=["instruction"],
                      audio_code_hints_batch=None, should_return_intermediate=False)
        with patch("acestep.core.generation.handler.generate_music_execute."
                   "_DEFAULT_GENERATION_TIMEOUT", 0.02):
            try:
                with request_scope(host), self.assertRaises(TimeoutError):
                    host._run_generate_music_service_with_progress(
                        progress=lambda *a, **kw: None, actual_batch_size=1,
                        audio_duration=10, inference_steps=8, timesteps=None,
                        service_inputs=inputs, refer_audios=None, guidance_scale=1,
                        actual_seed_list=[1729], audio_cover_strength=1,
                        cover_noise_strength=0, use_adg=False, cfg_interval_start=0,
                        cfg_interval_end=1, shift=1, infer_method="ode",
                    )
                self.assertTrue(started.is_set())
                self.assertFalse(gate(host).acquire(blocking=False))
            finally:
                finish.set()
        self.assertTrue(gate(host).acquire(timeout=2))
        gate(host).release()

    def test_cancel_during_thread_start_does_not_release_running_worker(self):
        host = ExecuteHost()
        host.device = "cpu"
        started, finish = threading.Event(), threading.Event()
        def service(**kwargs):
            started.set()
            finish.wait(2)
            return {"target_latents": "ok"}
        host.service_generate = service
        original_thread = threading.Thread
        def interrupted_thread(*args, **kwargs):
            thread = original_thread(*args, **kwargs)
            original_start = thread.start
            def start():
                original_start()
                started.wait(1)
                raise KeyboardInterrupt()
            thread.start = start
            return thread
        inputs = dict(captions_batch=["c"], lyrics_batch=["l"], metas_batch=["m"],
                      vocal_languages_batch=["en"], target_wavs_tensor=None,
                      repainting_start_batch=[0], repainting_end_batch=[1],
                      instructions_batch=["i"], audio_code_hints_batch=None,
                      should_return_intermediate=False)
        try:
            with request_scope(host), self.assertRaises(KeyboardInterrupt), patch(
                "acestep.core.generation.handler.generate_music_execute.threading.Thread",
                side_effect=interrupted_thread,
            ):
                host._run_generate_music_service_with_progress(
                    progress=lambda *a, **kw: None, actual_batch_size=1, audio_duration=10,
                    inference_steps=8, timesteps=None, service_inputs=inputs,
                    refer_audios=None, guidance_scale=1, actual_seed_list=[1729],
                    audio_cover_strength=1, cover_noise_strength=0, use_adg=False,
                    cfg_interval_start=0, cfg_interval_end=1, shift=1, infer_method="ode",
                )
            self.assertFalse(gate(host).acquire(blocking=False))
        finally:
            finish.set()
        self.assertTrue(gate(host).acquire(timeout=2))
        gate(host).release()

    def test_worker_retains_gate_after_caller_exits(self):
        host = SimpleNamespace(device="cpu")
        started = threading.Event()
        finish = threading.Event()
        def work():
            started.set()
            self.assertTrue(finish.wait(2))
        with request_scope(host):
            target, _ = worker_target(work)
            thread = threading.Thread(target=target)
            thread.start()
            self.assertTrue(started.wait(1))
        self.assertFalse(gate(host).acquire(blocking=False))
        finish.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(gate(host).acquire(blocking=False))
        gate(host).release()

    def test_worker_exception_releases_ownership(self):
        host = SimpleNamespace(device="cpu")
        def fail():
            raise ValueError("worker failure")
        with request_scope(host):
            target, _ = worker_target(fail)
            with self.assertRaises(ValueError):
                target()
        self.assertTrue(gate(host).acquire(blocking=False))
        gate(host).release()

    def test_failed_thread_start_releases_retained_ownership(self):
        host = SimpleNamespace(device="cpu")
        with request_scope(host):
            _, start_failed = worker_target(lambda: None)
            start_failed()
        self.assertTrue(gate(host).acquire(blocking=False))
        gate(host).release()

    def test_nested_request_uses_same_admission(self):
        host = SimpleNamespace(device="cpu")
        with request_scope(host) as first:
            with request_scope(host) as second:
                self.assertIs(first, second)
        self.assertTrue(gate(host).acquire(blocking=False))
        gate(host).release()

    def test_disabled_policy_preserves_unprotected_legacy_call(self):
        host = SimpleNamespace(device="cpu")
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(enabled(host))
            self.assertEqual(guarded(lambda self: 7)(host), 7)
        self.assertNotIn("_residency_admission", vars(host))

    def test_non_cuda_policy_remains_disabled(self):
        host = SimpleNamespace(device="mps", offload_to_cpu=True, offload_dit_to_cpu=True)
        with patch.dict("os.environ", {"ACESTEP_DIT_RESIDENCY": "phase"}):
            self.assertFalse(enabled(host))


if __name__ == "__main__":
    unittest.main()
