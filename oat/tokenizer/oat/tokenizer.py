import torch
import torch.nn.functional as F
from time import perf_counter
from typing import Tuple, Union, List, Optional

from oat.model.common.normalizer import LinearNormalizer 
from oat.tokenizer.base_tokenizer import BaseTokenizer 
from oat.tokenizer.oat.adaptive_prefix import (
    build_adaptive_thresholds,
    compute_action_complexity,
    select_prefix_depth,
)
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.fsq import FSQ

def pad_token_seq(token_ids: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """
    Pad the token sequence to the maximum length.

    Args:
        token_ids: The token id sequence of shape [B, l].
        max_seq_len: The maximum sequence length L.

    Returns:
        Padded token id sequence.
    """
    device, dtype = token_ids.device, token_ids.dtype
    pad_len = max_seq_len - token_ids.shape[1]
    pad_seq = torch.zeros((token_ids.shape[0], pad_len), device=device, dtype=dtype)
    return torch.cat([token_ids, pad_seq], dim=1)  # [B, L]


class OATTok(BaseTokenizer):
    def __init__(self,
        encoder: RegisterEncoder,
        decoder: SinglePassDecoder,
        quantizer: FSQ,
        adaptive_prefix: bool = False,
        adaptive_available_depths: Tuple[int, ...] = (1, 2, 4, 8),
        adaptive_low_threshold: float = 0.01,
        adaptive_high_threshold: float = 0.10,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.normalizer = LinearNormalizer()
        self.latent_horizon = self.decoder.latent_horizon
        self.adaptive_prefix = adaptive_prefix
        self.adaptive_available_depths = tuple(int(depth) for depth in adaptive_available_depths)
        self.adaptive_low_threshold = float(adaptive_low_threshold)
        self.adaptive_high_threshold = float(adaptive_high_threshold)

    def get_optimizer(
        self, 
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with weight decay for 2D parameters only."""
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _resolve_available_depths(
        self,
        adaptive_available_depths: Optional[List[int]] = None,
    ) -> List[int]:
        depths = adaptive_available_depths
        if depths is None:
            depths = list(self.adaptive_available_depths)

        valid_depths = sorted({int(depth) for depth in depths if int(depth) <= self.latent_horizon})
        if len(valid_depths) == 0:
            raise ValueError(
                "No adaptive prefix depths are valid for this tokenizer. "
                f"Requested={depths}, latent_horizon={self.latent_horizon}."
            )
        return valid_depths

    def _resolve_adaptive_thresholds(
        self,
        available_depths: List[int],
        adaptive_thresholds: Optional[List[float]] = None,
        adaptive_low_threshold: Optional[float] = None,
        adaptive_high_threshold: Optional[float] = None,
    ) -> Optional[List[float]]:
        if adaptive_thresholds is not None:
            return [float(threshold) for threshold in adaptive_thresholds]

        low_threshold = self.adaptive_low_threshold if adaptive_low_threshold is None else adaptive_low_threshold
        high_threshold = self.adaptive_high_threshold if adaptive_high_threshold is None else adaptive_high_threshold
        return build_adaptive_thresholds(
            available_depths=available_depths,
            low_threshold=float(low_threshold),
            high_threshold=float(high_threshold),
        )

    def select_eval_keep_k(
        self,
        samples: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
        adaptive_prefix: Optional[bool] = None,
        adaptive_available_depths: Optional[List[int]] = None,
        adaptive_thresholds: Optional[List[float]] = None,
        adaptive_low_threshold: Optional[float] = None,
        adaptive_high_threshold: Optional[float] = None,
    ) -> Tuple[List[int], Optional[torch.Tensor]]:
        if eval_keep_k is not None:
            return [int(k) for k in eval_keep_k], None

        if adaptive_prefix is None:
            adaptive_prefix = self.adaptive_prefix

        if not adaptive_prefix:
            return [self.latent_horizon] * samples.shape[0], None

        normalized_samples = self.normalizer["action"].normalize(samples)
        complexity = compute_action_complexity(normalized_samples)
        available_depths = self._resolve_available_depths(adaptive_available_depths)
        thresholds = self._resolve_adaptive_thresholds(
            available_depths=available_depths,
            adaptive_thresholds=adaptive_thresholds,
            adaptive_low_threshold=adaptive_low_threshold,
            adaptive_high_threshold=adaptive_high_threshold,
        )
        selected_k = select_prefix_depth(
            complexity=complexity,
            available_depths=available_depths,
            thresholds=thresholds,
        )
        return selected_k.tolist(), complexity

    def forward(self, batch) -> torch.Tensor:
        samples = batch['action']
        
        # normalize
        nsamples = self.normalizer['action'].normalize(samples)
        
        # encode & quantize
        latents = self.encoder(nsamples)
        latents, _ = self.quantizer(latents)

        # decode
        recons = self.decoder(latents)
        loss = F.mse_loss(recons, nsamples)

        return loss

    def encode(self, samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # samples: (B, T, sample_dim)

        # normalize
        nsamples = self.normalizer['action'].normalize(samples)
        # encode & quantize
        latents = self.encoder(nsamples)
        latents, tokens = self.quantizer(latents)
        return latents, tokens

    def decode(self, 
        latents: torch.Tensor, 
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # latents: (B, T', encoder_emb_dim)
        if eval_keep_k is None:
            eval_keep_k = [latents.shape[1]] * latents.shape[0]

        assert all([k <= self.latent_horizon for k in eval_keep_k]), \
            f"All eval_keep_k must be <= {self.latent_horizon}"

        # decode - returns normalized samples
        nsamples = self.decoder(latents, eval_keep_k=eval_keep_k)

        # unnormalize
        samples = self.normalizer['action'].unnormalize(nsamples)
        return samples

    def autoencode(self, 
        samples: torch.Tensor, 
        eval_keep_k: Optional[List[int]] = None,
        adaptive_prefix: Optional[bool] = None,
        adaptive_available_depths: Optional[List[int]] = None,
        adaptive_thresholds: Optional[List[float]] = None,
        adaptive_low_threshold: Optional[float] = None,
        adaptive_high_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        # samples: (B, T, sample_dim)
        eval_keep_k, _ = self.select_eval_keep_k(
            samples=samples,
            eval_keep_k=eval_keep_k,
            adaptive_prefix=adaptive_prefix,
            adaptive_available_depths=adaptive_available_depths,
            adaptive_thresholds=adaptive_thresholds,
            adaptive_low_threshold=adaptive_low_threshold,
            adaptive_high_threshold=adaptive_high_threshold,
        )
        latents, _ = self.encode(samples)
        recons = self.decode(latents, eval_keep_k=eval_keep_k)
        return recons

    def evaluate_reconstruction(
        self,
        samples: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
        adaptive_prefix: Optional[bool] = None,
        adaptive_available_depths: Optional[List[int]] = None,
        adaptive_thresholds: Optional[List[float]] = None,
        adaptive_low_threshold: Optional[float] = None,
        adaptive_high_threshold: Optional[float] = None,
        return_recons: bool = False,
    ):
        selected_k, complexity = self.select_eval_keep_k(
            samples=samples,
            eval_keep_k=eval_keep_k,
            adaptive_prefix=adaptive_prefix,
            adaptive_available_depths=adaptive_available_depths,
            adaptive_thresholds=adaptive_thresholds,
            adaptive_low_threshold=adaptive_low_threshold,
            adaptive_high_threshold=adaptive_high_threshold,
        )

        start_time = perf_counter()
        recons = self.autoencode(samples=samples, eval_keep_k=selected_k)
        runtime_sec = perf_counter() - start_time

        per_sample_mse = (recons - samples).square().flatten(start_dim=1).mean(dim=1)
        metrics = {
            "selected_k": selected_k,
            "avg_k": float(sum(selected_k) / len(selected_k)),
            "avg_token_count": float(sum(selected_k) / len(selected_k)),
            "reconstruction_mse": float(per_sample_mse.mean().item()),
            "reconstruction_mse_per_sample": per_sample_mse.detach().cpu().tolist(),
            "runtime_sec": float(runtime_sec),
            "runtime_per_sample_sec": float(runtime_sec / max(1, samples.shape[0])),
        }
        if complexity is not None:
            metrics["complexity"] = complexity.detach().cpu().tolist()

        if return_recons:
            return recons, metrics
        return metrics

    def tokenize(self, samples: torch.Tensor) -> torch.Tensor:
        # samples: (B, T, sample_dim)
        _, tokens = self.encode(samples)
        return tokens

    def detokenize(self, 
        tokens: Union[torch.Tensor, List[List[int]]],
    ) -> torch.Tensor:
        # tokens: (B, T') or list of list of int
        
        # standardize
        if isinstance(tokens, list):
            token_lens = [t.shape[1] for t in tokens]
            tokens = torch.cat([
                pad_token_seq(t, self.latent_horizon)
                for t in tokens
            ], dim=0)
        elif isinstance(tokens, torch.Tensor):
            token_lens = [tokens.shape[1]] * tokens.shape[0]
            if tokens.shape[-1] < self.latent_horizon:
                tokens = pad_token_seq(tokens, self.latent_horizon)
        else:
            raise ValueError(f'Unknown token type {type(tokens)}')

        # codebook lookup & decode
        latents = self.quantizer.indices_to_embedding(tokens)
        samples = self.decode(latents, eval_keep_k=token_lens)
        return samples
