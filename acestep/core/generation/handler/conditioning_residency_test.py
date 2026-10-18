"""Check phase scheduling without loading model weights or requiring CUDA."""

import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import torch

from acestep.core.generation.handler.conditioning_target_test import _Host
from acestep.core.generation.handler.conditioning_batch_test import _Host as BatchHost
from acestep.core.generation.handler.conditioning_embed_test import _Host as EmbedHost


class ConditioningResidencyTests(unittest.TestCase):
    """Exercise code targets, source encoding, and reused reference embeddings."""

    def test_early_embeddings_move_cpu_token_ids_to_encoder_device(self):
        host = EmbedHost()
        host.device = "cuda:0"
        tokens = [MagicMock() for _ in range(3)]
        for token in tokens:
            token.to.return_value = torch.ones(1, 8, dtype=torch.long)
        result = host._infer_prompt_embeddings(*tokens)
        for token in tokens:
            token.to.assert_called_once_with("cuda:0")
        self.assertEqual(len(result), 3)

    def test_reference_preparation_precedes_code_target_and_is_reused(self):
        batch_host = BatchHost()
        reference = (torch.ones(1, 128, 6), torch.tensor([0]))
        order = []
        batch_host._load_model_context = lambda _: nullcontext()
        batch_host._infer_prompt_embeddings = lambda *args: order.append("text") or (
            torch.zeros(1, 8, 6), torch.zeros(1, 12, 6), None
        )
        batch_host.infer_refer_latent = lambda _: order.append("reference") or reference
        original = batch_host._prepare_target_latents_and_wavs
        def prepare_target(*args, **kwargs):
            order.append("target")
            return original(*args, **kwargs)
        batch_host._prepare_target_latents_and_wavs = prepare_target
        with patch("acestep.core.generation.handler.conditioning_batch.enabled", return_value=True):
            batch = batch_host._prepare_batch(
                captions=["caption"], target_wavs=torch.zeros(1, 2, 7680),
                audio_code_hints=["codes"],
            )
        self.assertEqual(order, ["reference", "text", "target"])
        self.assertIs(batch["precomputed_reference"], reference)
        batch["target_latents"] = torch.zeros(1, 128, 6)
        batch["src_latents"] = torch.zeros(1, 128, 6)
        host = EmbedHost()
        with patch.object(host, "infer_refer_latent", side_effect=AssertionError("duplicate encode")), \
                patch.object(host, "_infer_prompt_embeddings",
                             side_effect=AssertionError("duplicate text encode")):
            result = host.preprocess_batch(batch)
        self.assertTrue(torch.equal(result[9], reference[0]))

    def test_code_only_target_does_not_load_vae(self):
        host = _Host()
        host._decode_audio_codes_to_latents = lambda _: torch.ones(1, 4, 3)
        with patch("acestep.core.generation.handler.conditioning_target.enabled", return_value=True):
            with patch.object(host, "_load_model_context", wraps=host._load_model_context) as load:
                host._prepare_target_latents_and_wavs(1, torch.ones(1, 2, 7680), ["codes"])
        self.assertEqual(host.encode_calls, 0)
        load.assert_not_called()

    def test_source_encoding_still_uses_vae_context(self):
        host = _Host()
        with patch("acestep.core.generation.handler.conditioning_target.enabled", return_value=True):
            with patch.object(host, "_load_model_context", wraps=host._load_model_context) as load:
                host._prepare_target_latents_and_wavs(1, torch.ones(1, 2, 7680), [None])
        self.assertEqual(host.encode_calls, 1)
        load.assert_called_once_with("vae")

    def test_silent_target_does_not_load_vae(self):
        host = _Host()
        with patch("acestep.core.generation.handler.conditioning_target.enabled", return_value=True):
            with patch.object(host, "_load_model_context", wraps=host._load_model_context) as load:
                host._prepare_target_latents_and_wavs(1, torch.zeros(1, 2, 7680), [None])
        self.assertEqual(host.encode_calls, 0)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
