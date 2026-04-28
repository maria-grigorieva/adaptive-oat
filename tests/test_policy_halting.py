import sys
import types
import unittest

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

from oat.policy.oatpolicy import OATPolicy


class DummyObsEncoder(nn.Module):
    def __init__(self, feature_dim: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Linear(2, feature_dim)

    def forward(self, obs_dict):
        return self.proj(obs_dict["state"].to(torch.float32))

    def modalities(self):
        return ["state"]

    def output_feature_dim(self):
        return self.feature_dim

    def set_normalizer(self, normalizer):
        return None


class DummyTokenizer(nn.Module):
    def __init__(self, codebook_size: int = 8, latent_horizon: int = 4):
        super().__init__()
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.quantizer = types.SimpleNamespace(codebook_size=codebook_size)
        self.latent_horizon = latent_horizon
        self.last_detokenize_tokens = None
        self.last_detokenize_token_lens = None

    def tokenize(self, samples: torch.Tensor) -> torch.Tensor:
        batch_size = samples.shape[0]
        base = torch.arange(1, self.latent_horizon + 1, device=samples.device, dtype=torch.long)
        return base.unsqueeze(0).expand(batch_size, -1).clone()

    def detokenize(self, tokens, token_lens=None):
        self.last_detokenize_tokens = tokens.clone()
        if isinstance(token_lens, torch.Tensor):
            token_lens = token_lens.clone()
        else:
            token_lens = torch.tensor(token_lens, dtype=torch.long, device=tokens.device)
        self.last_detokenize_token_lens = token_lens
        return token_lens.to(torch.float32).view(-1, 1, 1).expand(-1, 3, 1)

    def compute_oracle_keep_k_from_samples(self, samples, keep_ks, tolerance=1e-3):
        oracle = torch.tensor([keep_ks[0]] * samples.shape[0], device=samples.device, dtype=torch.long)
        errors = torch.zeros((samples.shape[0], len(keep_ks)), device=samples.device)
        return oracle, errors


class DummyGenerateModel(nn.Module):
    def __init__(self, generated_tokens: torch.Tensor):
        super().__init__()
        self.generated_tokens = generated_tokens

    def generate(self, prefix, cond, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        return self.generated_tokens.to(prefix.device)


def make_policy(use_adaptive_halting: bool = True) -> OATPolicy:
    shape_meta = {
        "obs": {
            "state": {
                "shape": [2],
                "type": "state",
            }
        },
        "action": {
            "shape": [1],
        },
    }
    return OATPolicy(
        shape_meta=shape_meta,
        obs_encoder=DummyObsEncoder(feature_dim=4),
        action_tokenizer=DummyTokenizer(codebook_size=8, latent_horizon=4),
        n_action_steps=2,
        n_obs_steps=1,
        embed_dim=8,
        n_layers=1,
        n_heads=1,
        dropout=0.0,
        temperature=0.0,
        topk=1,
        use_adaptive_halting=use_adaptive_halting,
        halt_keep_ks=[1, 2, 4],
    )


class PolicyHaltingTests(unittest.TestCase):
    def test_adaptive_training_batch_appends_eos_and_masks_padding(self):
        policy = make_policy(use_adaptive_halting=True)
        action_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
        oracle_keep_k = torch.tensor([2, 4], dtype=torch.long)

        model_tokens, target_tokens, target_mask = policy.build_adaptive_training_batch(
            action_tokens=action_tokens,
            oracle_keep_k=oracle_keep_k,
        )

        self.assertEqual(model_tokens.tolist()[0], [policy.bos_id, 1, 2, policy.eos_id, policy.eos_id])
        self.assertEqual(target_tokens.tolist()[0], [1, 2, policy.eos_id, policy.loss_ignore_index, policy.loss_ignore_index])
        self.assertEqual(target_mask.tolist()[0], [True, True, True, False, False])
        self.assertEqual(target_tokens.tolist()[1], [5, 6, 7, 8, policy.eos_id])
        self.assertEqual(target_mask.tolist()[1], [True, True, True, True, True])

    def test_adaptive_training_batch_truncates_to_oracle_keep_k(self):
        policy = make_policy(use_adaptive_halting=True)
        action_tokens = torch.tensor([[4, 3, 2, 1]], dtype=torch.long)
        oracle_keep_k = torch.tensor([1], dtype=torch.long)

        model_tokens, target_tokens, target_mask = policy.build_adaptive_training_batch(
            action_tokens=action_tokens,
            oracle_keep_k=oracle_keep_k,
        )

        self.assertEqual(model_tokens.shape[1], 2)
        self.assertEqual(model_tokens.tolist()[0], [policy.bos_id, 4])
        self.assertEqual(target_tokens.tolist()[0], [4, policy.eos_id])
        self.assertEqual(target_mask.tolist()[0], [True, True])

    def test_predict_action_stops_at_eos_and_passes_token_lens(self):
        policy = make_policy(use_adaptive_halting=True)
        generated = torch.tensor(
            [
                [policy.bos_id, 1, 2, policy.eos_id],
                [policy.bos_id, 3, policy.eos_id, policy.eos_id],
            ],
            dtype=torch.long,
        )
        policy.model = DummyGenerateModel(generated)

        obs_dict = {"state": torch.zeros((2, 1, 2), dtype=torch.float32)}
        result = policy.predict_action(obs_dict, adaptive_halting=True)

        self.assertEqual(result["token_lens"].tolist(), [2, 1])
        self.assertEqual(result["eos_generated"].tolist(), [True, True])
        self.assertEqual(policy.action_tokenizer.last_detokenize_token_lens.tolist(), [2, 1])
        self.assertEqual(policy.action_tokenizer.last_detokenize_tokens.tolist(), [[1, 2, 0, 0], [3, 0, 0, 0]])

    def test_predict_action_falls_back_to_max_k_without_eos(self):
        policy = make_policy(use_adaptive_halting=True)
        generated = torch.tensor(
            [
                [policy.bos_id, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        )
        policy.model = DummyGenerateModel(generated)

        obs_dict = {"state": torch.zeros((1, 1, 2), dtype=torch.float32)}
        result = policy.predict_action(obs_dict, use_k_tokens=4, adaptive_halting=True)

        self.assertEqual(result["token_lens"].tolist(), [4])
        self.assertEqual(result["eos_generated"].tolist(), [False])
        self.assertEqual(policy.action_tokenizer.last_detokenize_tokens.tolist(), [[1, 2, 3, 4]])


if __name__ == "__main__":
    unittest.main()
