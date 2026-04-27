import unittest

import torch

from oat.model.common.normalizer import LinearNormalizer
from oat.tokenizer.oat.adaptive_prefix import (
    compute_action_complexity,
    select_prefix_depth,
)
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.quantizer.fsq import FSQ
from oat.tokenizer.oat.tokenizer import OATTok


def make_test_tokenizer() -> OATTok:
    tokenizer = OATTok(
        encoder=RegisterEncoder(
            sample_dim=3,
            sample_horizon=8,
            emb_dim=32,
            head_dim=8,
            depth=1,
            pdropout=0.0,
            latent_dim=4,
            num_registers=4,
        ),
        decoder=SinglePassDecoder(
            sample_dim=3,
            sample_horizon=8,
            emb_dim=32,
            head_dim=8,
            depth=1,
            pdropout=0.0,
            token_dropout_mode="pow2",
            use_causal_decoder=True,
            latent_dim=4,
            latent_horizon=4,
        ),
        quantizer=FSQ(levels=[8, 5, 5, 5]),
    )
    tokenizer.eval()
    return tokenizer


def make_actions(batch_size: int = 6, horizon: int = 8, action_dim: int = 3) -> torch.Tensor:
    time_axis = torch.linspace(0.0, 1.0, steps=horizon)
    actions = []
    for idx in range(batch_size):
        smooth = torch.stack(
            [0.5 * torch.sin((dim + 1) * torch.pi * time_axis) for dim in range(action_dim)],
            dim=1,
        )
        if idx % 2 == 0:
            action = smooth
        else:
            noise = 0.35 * torch.randn(horizon, action_dim)
            action = smooth + noise
        actions.append(action)
    return torch.stack(actions, dim=0).to(torch.float32)


class AdaptivePrefixOATTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.samples = make_actions()
        self.tokenizer = make_test_tokenizer()
        normalizer = LinearNormalizer()
        normalizer.fit({"action": self.samples}, last_n_dims=1)
        self.tokenizer.set_normalizer(normalizer)

    def test_complexity_prefers_larger_prefix_for_noisy_actions(self):
        smooth = self.samples[0:1]
        noisy = self.samples[1:2]
        stacked = torch.cat([smooth, noisy], dim=0)

        complexity = compute_action_complexity(stacked)
        selected_k = select_prefix_depth(
            complexity=complexity,
            available_depths=[1, 2, 4, 8],
            thresholds=[0.02, 0.08, 0.20],
        )

        self.assertLess(complexity[0].item(), complexity[1].item())
        self.assertLess(selected_k[0].item(), selected_k[1].item())

    def test_fixed_prefix_baseline_is_unchanged(self):
        full_keep_k = [self.tokenizer.latent_horizon] * self.samples.shape[0]
        recon_default = self.tokenizer.autoencode(self.samples)
        recon_explicit = self.tokenizer.autoencode(self.samples, eval_keep_k=full_keep_k)

        self.assertTrue(torch.allclose(recon_default, recon_explicit))

    def test_adaptive_reconstruction_runs_on_cpu(self):
        metrics = self.tokenizer.evaluate_reconstruction(
            samples=self.samples,
            adaptive_prefix=True,
            adaptive_available_depths=[1, 2, 4],
            adaptive_thresholds=[0.02, 0.08],
        )

        self.assertEqual(len(metrics["selected_k"]), self.samples.shape[0])
        self.assertTrue(all(k in {1, 2, 4} for k in metrics["selected_k"]))
        self.assertIn("runtime_sec", metrics)
        self.assertIn("complexity", metrics)


if __name__ == "__main__":
    unittest.main()
