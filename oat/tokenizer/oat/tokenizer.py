import torch
import torch.nn.functional as F
from typing import Tuple, Union, List, Optional

from oat.model.common.normalizer import LinearNormalizer 
from oat.tokenizer.base_tokenizer import BaseTokenizer 
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
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.normalizer = LinearNormalizer()
        self.latent_horizon = self.decoder.latent_horizon

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
    ) -> torch.Tensor:
        # samples: (B, T, sample_dim)
        latents, _ = self.encode(samples)
        recons = self.decode(latents, eval_keep_k=eval_keep_k)
        return recons

    def compute_prefix_reconstruction_errors(
        self,
        samples: torch.Tensor,
        keep_ks: List[int],
    ) -> torch.Tensor:
        # samples: (B, T, sample_dim)
        if len(keep_ks) == 0:
            raise ValueError("keep_ks must be non-empty")

        nsamples = self.normalizer['action'].normalize(samples)
        latents, _ = self.encode(samples)

        prefix_errors = []
        for keep_k in keep_ks:
            recons = self.decode(latents, eval_keep_k=[keep_k] * samples.shape[0])
            nrecons = self.normalizer['action'].normalize(recons)
            mse = (nrecons - nsamples).pow(2).flatten(start_dim=1).mean(dim=1)
            prefix_errors.append(mse)

        return torch.stack(prefix_errors, dim=1)

    def derive_oracle_keep_k(
        self,
        prefix_errors: torch.Tensor,
        keep_ks: List[int],
        tolerance: float = 1e-3,
    ) -> torch.Tensor:
        if prefix_errors.ndim != 2:
            raise ValueError(f"prefix_errors must have shape [B, K], got {prefix_errors.shape}")
        if len(keep_ks) == 0:
            raise ValueError("keep_ks must be non-empty")
        if prefix_errors.shape[1] != len(keep_ks):
            raise ValueError(
                f"prefix_errors second dim ({prefix_errors.shape[1]}) must match len(keep_ks) ({len(keep_ks)})"
            )

        keep_ks_tensor = torch.tensor(keep_ks, device=prefix_errors.device, dtype=torch.long)
        sorted_keep_ks, sort_idx = torch.sort(keep_ks_tensor)
        sorted_errors = prefix_errors[:, sort_idx]

        full_error = sorted_errors[:, -1:]
        valid_mask = sorted_errors <= (full_error + tolerance)
        first_valid_idx = valid_mask.to(torch.int64).argmax(dim=1)
        return sorted_keep_ks[first_valid_idx]

    def compute_oracle_keep_k_from_samples(
        self,
        samples: torch.Tensor,
        keep_ks: List[int],
        tolerance: float = 1e-3,
    ):
        prefix_errors = self.compute_prefix_reconstruction_errors(samples, keep_ks)
        oracle_keep_k = self.derive_oracle_keep_k(
            prefix_errors=prefix_errors,
            keep_ks=keep_ks,
            tolerance=tolerance,
        )
        return oracle_keep_k, prefix_errors

    def tokenize(self, samples: torch.Tensor) -> torch.Tensor:
        # samples: (B, T, sample_dim)
        _, tokens = self.encode(samples)
        return tokens

    def detokenize(self, 
        tokens: Union[torch.Tensor, List[List[int]]],
        token_lens: Optional[Union[torch.Tensor, List[int]]] = None,
    ) -> torch.Tensor:
        # tokens: (B, T') or list of list of int

        if token_lens is None:
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
        else:
            if isinstance(token_lens, torch.Tensor):
                token_lens = token_lens.tolist()
            else:
                token_lens = list(token_lens)

            if isinstance(tokens, list):
                if len(tokens) != len(token_lens):
                    raise ValueError(
                        f"Expected {len(token_lens)} token sequences, got {len(tokens)}"
                    )
                tokens = torch.cat([
                    pad_token_seq(t, self.latent_horizon)
                    for t in tokens
                ], dim=0)
            elif isinstance(tokens, torch.Tensor):
                if tokens.shape[0] != len(token_lens):
                    raise ValueError(
                        f"Expected token_lens for {tokens.shape[0]} samples, got {len(token_lens)}"
                    )
                if tokens.shape[-1] < self.latent_horizon:
                    tokens = pad_token_seq(tokens, self.latent_horizon)
            else:
                raise ValueError(f'Unknown token type {type(tokens)}')

        # codebook lookup & decode
        latents = self.quantizer.indices_to_embedding(tokens)
        samples = self.decode(latents, eval_keep_k=token_lens)
        return samples
