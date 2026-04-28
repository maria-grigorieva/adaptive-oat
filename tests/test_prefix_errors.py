import types
import unittest
import sys

import torch
import torch.nn as nn

zarr_stub = types.ModuleType("zarr")
zarr_stub.Array = type("Array", (), {})
sys.modules.setdefault("zarr", zarr_stub)

dill_stub = types.ModuleType("dill")
sys.modules.setdefault("dill", dill_stub)

hydra_stub = types.ModuleType("hydra")
hydra_stub.utils = types.SimpleNamespace(get_class=lambda _: None)
sys.modules.setdefault("hydra", hydra_stub)

encoder_stub = types.ModuleType("oat.tokenizer.oat.encoder.register_encoder")
encoder_stub.RegisterEncoder = object
sys.modules.setdefault("oat.tokenizer.oat.encoder.register_encoder", encoder_stub)

decoder_stub = types.ModuleType("oat.tokenizer.oat.decoder.single_pass_decoder")
decoder_stub.SinglePassDecoder = object
sys.modules.setdefault("oat.tokenizer.oat.decoder.single_pass_decoder", decoder_stub)

quantizer_stub = types.ModuleType("oat.tokenizer.oat.quantizer.fsq")
quantizer_stub.FSQ = object
sys.modules.setdefault("oat.tokenizer.oat.quantizer.fsq", quantizer_stub)

from oat.tokenizer.oat.tokenizer import OATTok


class DummyEncoder(nn.Module):
    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return sample


class DummyDecoder(nn.Module):
    def __init__(self, latent_horizon: int):
        super().__init__()
        self.latent_horizon = latent_horizon

    def forward(self, latents: torch.Tensor, eval_keep_k=None) -> torch.Tensor:
        return latents


class DummyQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebook_size = 16

    def forward(self, latents: torch.Tensor):
        batch_size, seq_len = latents.shape[:2]
        tokens = torch.zeros((batch_size, seq_len), dtype=torch.long, device=latents.device)
        return latents, tokens

    def indices_to_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        return indices.to(torch.float32).unsqueeze(-1)


class IdentityActionNormalizer:
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyNormalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def __getitem__(self, key: str) -> IdentityActionNormalizer:
        if key != "action":
            raise KeyError(key)
        return IdentityActionNormalizer()

    def state_dict(self):
        return {}


def make_tokenizer(latent_horizon: int = 4) -> OATTok:
    tokenizer = OATTok(
        encoder=DummyEncoder(),
        decoder=DummyDecoder(latent_horizon=latent_horizon),
        quantizer=DummyQuantizer(),
    )
    tokenizer.normalizer = DummyNormalizer()
    return tokenizer


class PrefixErrorTests(unittest.TestCase):
    def test_compute_prefix_reconstruction_errors_shape_and_monotonicity(self):
        tokenizer = make_tokenizer(latent_horizon=4)
        samples = torch.tensor(
            [
                [[1.0], [2.0], [3.0]],
                [[2.0], [4.0], [6.0]],
            ],
            dtype=torch.float32,
        )
        keep_ks = [1, 2, 4]

        def fake_encode(self, batch_samples: torch.Tensor):
            batch_size = batch_samples.shape[0]
            latents = torch.zeros((batch_size, self.latent_horizon, 1), dtype=batch_samples.dtype)
            tokens = torch.zeros((batch_size, self.latent_horizon), dtype=torch.long)
            return latents, tokens

        def fake_decode(self, latents: torch.Tensor, eval_keep_k=None):
            scales = torch.tensor(eval_keep_k, dtype=self._test_samples.dtype).view(-1, 1, 1)
            scales = scales / float(self.latent_horizon)
            return self._test_samples * scales

        tokenizer._test_samples = samples
        original_encode = tokenizer.encode
        original_decode = tokenizer.decode
        try:
            tokenizer.encode = types.MethodType(fake_encode, tokenizer)
            tokenizer.decode = types.MethodType(fake_decode, tokenizer)
            prefix_errors = tokenizer.compute_prefix_reconstruction_errors(samples, keep_ks)
        finally:
            tokenizer.encode = original_encode
            tokenizer.decode = original_decode

        self.assertEqual(prefix_errors.shape, (samples.shape[0], len(keep_ks)))
        self.assertTrue(torch.all(prefix_errors[:, 1:] <= prefix_errors[:, :-1] + 1e-8))

    def test_derive_oracle_keep_k_returns_values_from_keep_ks(self):
        tokenizer = make_tokenizer(latent_horizon=4)
        keep_ks = [1, 2, 4]
        prefix_errors = torch.tensor(
            [
                [0.50, 0.02, 0.01],
                [0.08, 0.05, 0.05],
                [0.30, 0.20, 0.10],
            ],
            dtype=torch.float32,
        )

        oracle_keep_k = tokenizer.derive_oracle_keep_k(
            prefix_errors=prefix_errors,
            keep_ks=keep_ks,
            tolerance=1e-3,
        )

        self.assertEqual(oracle_keep_k.shape, (prefix_errors.shape[0],))
        self.assertEqual(oracle_keep_k.tolist(), [4, 2, 4])
        self.assertTrue(set(oracle_keep_k.tolist()).issubset(set(keep_ks)))

    def test_detokenize_respects_token_lens(self):
        tokenizer = make_tokenizer(latent_horizon=4)
        tokens = torch.tensor(
            [
                [3, 5],
                [7, 9],
            ],
            dtype=torch.long,
        )
        token_lens = [1, 2]

        def fake_decode(self, latents: torch.Tensor, eval_keep_k=None):
            keep = torch.tensor(eval_keep_k, dtype=torch.float32).view(-1, 1, 1)
            return keep.expand(latents.shape[0], 2, 1)

        original_decode = tokenizer.decode
        try:
            tokenizer.decode = types.MethodType(fake_decode, tokenizer)
            samples = tokenizer.detokenize(tokens, token_lens=token_lens)
        finally:
            tokenizer.decode = original_decode

        self.assertEqual(samples.shape, (2, 2, 1))
        self.assertTrue(torch.allclose(samples[0], torch.ones((2, 1))))
        self.assertTrue(torch.allclose(samples[1], torch.full((2, 1), 2.0)))


if __name__ == "__main__":
    unittest.main()
