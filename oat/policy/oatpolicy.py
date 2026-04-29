import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
# from oat.model.autoregressive.transformer import AutoregressiveModel
from oat.model.autoregressive.transformer_cache import AutoregressiveModel


class OATPolicy(BasePolicy):
    loss_ignore_index = -100
    stop_reason_eos = 0
    stop_reason_entropy = 1
    stop_reason_max_k = 2

    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer: OATTok,
        n_action_steps: int,
        n_obs_steps: int,
        # policy model params
        embed_dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        # policy inference params
        temperature: float = 1.0,
        topk: int = 10,
        use_adaptive_halting: bool = False,
        entropy_threshold: float = 0.5,
        halt_tolerance: float = 1e-3,
        halt_keep_ks: Optional[List[int]] = None,
    ):
        super().__init__()
        
        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in shape_meta['obs'].items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr['type']
            if type in modalities:
                obs_ports.append(key)

        # freeze action tokenizer
        for param in action_tokenizer.parameters():
            param.requires_grad_(False)
        action_tokenizer.eval()

        # create AR model
        codebook_size = action_tokenizer.quantizer.codebook_size
        latent_horizon = action_tokenizer.latent_horizon
        if halt_keep_ks is None:
            halt_keep_ks = list(range(1, latent_horizon + 1))
        if len(halt_keep_ks) == 0:
            raise ValueError("halt_keep_ks must be non-empty")
        if max(halt_keep_ks) > latent_horizon:
            raise ValueError(
                f"halt_keep_ks must be <= latent horizon ({latent_horizon}), got {halt_keep_ks}"
            )
        vocab_size = codebook_size + 2 if use_adaptive_halting else codebook_size + 1
        model = AutoregressiveModel(
            vocab_size=vocab_size,
            max_seq_len=latent_horizon + 1,
            max_cond_len=n_obs_steps,
            cond_dim=obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
        )
        bos_id = codebook_size  # last token id for <BOS>
        eos_id = codebook_size + 1 if use_adaptive_halting else None

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.action_tokenizer = action_tokenizer
        self.model = model
        self.max_seq_len = latent_horizon
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.use_adaptive_halting = use_adaptive_halting
        self.entropy_threshold = float(entropy_threshold)
        self.halt_tolerance = halt_tolerance
        self.halt_keep_ks = list(halt_keep_ks)
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.topk = topk
        self._last_forward_info = dict()

        # report
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs_params = sum(p.numel() for p in obs_encoder.parameters() if p.requires_grad)
        obs_trainable_ratio = num_trainable_obs_params / num_obs_params
        num_tok_params = sum(p.numel() for p in action_tokenizer.parameters())
        num_trainable_tok_params = sum(p.numel() for p in action_tokenizer.parameters() if p.requires_grad)
        tok_trainable_ratio = num_trainable_tok_params / num_tok_params
        num_model_params = sum(p.numel() for p in model.parameters())
        num_trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_trainable_ratio = num_trainable_model_params / num_model_params
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc: {num_obs_params/1e6:.1f}M ({obs_trainable_ratio:.5%} trainable)\n"
            f"  act tok: {num_tok_params/1e6:.1f}M ({tok_trainable_ratio:.5%} trainable)\n"
            f"  policy : {num_model_params/1e6:.1f}M ({model_trainable_ratio:.5%} trainable)\n"
        )

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities
    
    def get_observation_ports(self):
        return self.obs_ports
    
    def get_policy_name(self):
        base_name = 'oatpolicy_'
        for modality in self.modalities:
            if modality != 'state':
                base_name += modality + '|'
        return base_name[:-1]

    def create_dummy_observation(self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device
        )

    def set_normalizer(self, normalizer):
        self.obs_encoder.set_normalizer(normalizer)
        # self.action_tokenizer.set_normalizer(normalizer)

    def get_last_forward_info(self) -> Dict[str, torch.Tensor]:
        return self._last_forward_info

    def build_adaptive_training_batch(
        self,
        action_tokens: torch.Tensor,
        oracle_keep_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_adaptive_halting:
            raise RuntimeError("Adaptive halting batch construction requires use_adaptive_halting=True")

        B, _ = action_tokens.shape
        oracle_keep_k = oracle_keep_k.to(device=action_tokens.device, dtype=torch.long)
        max_keep_k = int(oracle_keep_k.max().item())
        seq_len = max_keep_k + 2  # <BOS> + tokens + <EOS>

        full_sequences = torch.full(
            (B, seq_len),
            self.eos_id,
            dtype=action_tokens.dtype,
            device=action_tokens.device,
        )
        valid_mask = torch.zeros((B, seq_len), dtype=torch.bool, device=action_tokens.device)
        full_sequences[:, 0] = self.bos_id
        valid_mask[:, 0] = True

        for batch_idx, keep_k in enumerate(oracle_keep_k.tolist()):
            if keep_k > 0:
                full_sequences[batch_idx, 1:1 + keep_k] = action_tokens[batch_idx, :keep_k]
            full_sequences[batch_idx, 1 + keep_k] = self.eos_id
            valid_mask[batch_idx, :keep_k + 2] = True

        model_tokens = full_sequences[:, :-1]
        target_tokens = full_sequences[:, 1:].clone()
        target_mask = valid_mask[:, 1:]
        target_tokens[~target_mask] = self.loss_ignore_index
        return model_tokens, target_tokens, target_mask

    def extract_action_prefix(
        self,
        generated_tokens: torch.Tensor,
        max_action_tokens: int,
        adaptive_min_k: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, generated_len = generated_tokens.shape
        device = generated_tokens.device

        decode_tokens = torch.zeros(
            (B, max_action_tokens),
            dtype=generated_tokens.dtype,
            device=device,
        )
        copy_len = min(generated_len, max_action_tokens)
        if copy_len > 0:
            decode_tokens[:, :copy_len] = generated_tokens[:, :copy_len]

        if not self.use_adaptive_halting:
            token_lens = torch.full((B,), copy_len, dtype=torch.long, device=device)
            eos_generated = torch.zeros(B, dtype=torch.bool, device=device)
            return decode_tokens, token_lens, eos_generated

        adaptive_min_k = int(adaptive_min_k)
        adaptive_min_k = max(1, min(adaptive_min_k, max_action_tokens))
        eos_mask = generated_tokens == self.eos_id
        token_positions = torch.arange(generated_len, device=device).unsqueeze(0)
        valid_eos_mask = eos_mask & (token_positions >= adaptive_min_k)
        has_eos = valid_eos_mask.any(dim=1)
        first_eos_idx = valid_eos_mask.to(torch.int64).argmax(dim=1)
        token_lens = torch.where(
            has_eos,
            first_eos_idx,
            torch.full((B,), copy_len, dtype=torch.long, device=device),
        )
        token_lens = token_lens.clamp(min=0, max=max_action_tokens)
        eos_generated = has_eos & (first_eos_idx <= max_action_tokens)

        action_token_positions = torch.arange(max_action_tokens, device=device).unsqueeze(0)
        decode_tokens[action_token_positions >= token_lens.unsqueeze(1)] = 0
        return decode_tokens, token_lens, eos_generated

    def _compute_sampling_distribution(
        self,
        logits: torch.Tensor,
        temperature: float,
        topk: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sampling_logits = logits.clone()
        if temperature > 0:
            sampling_logits = sampling_logits / temperature

        if topk is not None:
            v, _ = torch.topk(sampling_logits, min(topk, sampling_logits.size(-1)))
            sampling_logits[sampling_logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(sampling_logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        return probs, entropy

    def _sample_tokens_from_distribution(
        self,
        probs: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if temperature > 0:
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(probs, dim=-1, keepdim=True)

    def _generate_action_tokens_with_entropy_halting(
        self,
        features: torch.Tensor,
        max_action_tokens: int,
        temperature: float,
        topk: Optional[int],
        adaptive_min_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = features.shape[0]
        device = features.device
        adaptive_min_k = max(1, min(int(adaptive_min_k), max_action_tokens))

        generated_action_tokens = torch.zeros(
            (B, max_action_tokens),
            dtype=torch.long,
            device=device,
        )
        token_lens = torch.full(
            (B,),
            max_action_tokens,
            dtype=torch.long,
            device=device,
        )
        eos_generated = torch.zeros(B, dtype=torch.bool, device=device)
        stop_reasons = torch.full(
            (B,),
            self.stop_reason_max_k,
            dtype=torch.long,
            device=device,
        )
        entropy_at_stop = torch.full(
            (B,),
            float("nan"),
            dtype=features.dtype,
            device=device,
        )
        last_entropy = torch.zeros(B, dtype=features.dtype, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        bos_tokens = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)

        for step in range(max_action_tokens):
            prefix_tokens = torch.cat([bos_tokens, generated_action_tokens[:, :step]], dim=1)
            logits = self.model(prefix_tokens, cond=features)[:, -1, :]

            sampling_logits = logits.clone()
            if self.eos_id is not None and step < adaptive_min_k:
                sampling_logits[:, self.eos_id] = -float("inf")

            probs, entropy = self._compute_sampling_distribution(
                sampling_logits,
                temperature=temperature,
                topk=topk,
            )
            last_entropy = entropy

            can_stop = (~finished) & (step >= adaptive_min_k)
            entropy_stop = can_stop & (entropy < self.entropy_threshold)
            if entropy_stop.any():
                finished[entropy_stop] = True
                token_lens[entropy_stop] = step
                stop_reasons[entropy_stop] = self.stop_reason_entropy
                entropy_at_stop[entropy_stop] = entropy[entropy_stop]

            active = ~finished
            if not active.any():
                break

            next_tokens = self._sample_tokens_from_distribution(
                probs=probs,
                temperature=temperature,
            )
            next_token_ids = next_tokens.squeeze(-1)

            eos_stop = active & (step >= adaptive_min_k) & (next_token_ids == self.eos_id)
            if eos_stop.any():
                finished[eos_stop] = True
                eos_generated[eos_stop] = True
                token_lens[eos_stop] = step
                stop_reasons[eos_stop] = self.stop_reason_eos
                entropy_at_stop[eos_stop] = entropy[eos_stop]

            append_mask = active & ~eos_stop
            if append_mask.any():
                generated_action_tokens[append_mask, step] = next_token_ids[append_mask]

            if finished.all():
                break

        max_k_mask = ~finished
        if max_k_mask.any():
            token_lens[max_k_mask] = max_action_tokens
            stop_reasons[max_k_mask] = self.stop_reason_max_k
            entropy_at_stop[max_k_mask] = last_entropy[max_k_mask]

        return generated_action_tokens, token_lens, eos_generated, entropy_at_stop, stop_reasons

    def _oracle_keep_k_from_batch(self, batch) -> torch.Tensor:
        oracle_keep_k, _ = self.action_tokenizer.compute_oracle_keep_k_from_samples(
            samples=batch['action'],
            keep_ks=self.halt_keep_ks,
            tolerance=self.halt_tolerance,
        )
        return oracle_keep_k.to(device=batch['action'].device, dtype=torch.long)

    def get_optimizer(
        self, 
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with weight decay for 2D parameters only."""
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.

        encoder_decay_params = []
        encoder_nodecay_params = []
        for name, param in self.obs_encoder.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                encoder_decay_params.append(param)
            else:
                encoder_nodecay_params.append(param)

        policy_decay_params = []
        policy_nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                policy_decay_params.append(param)
            else:
                policy_nodecay_params.append(param)
        
        optim_groups = [
            {'params': policy_decay_params, 'lr': policy_lr, 'weight_decay': weight_decay},
            {'params': policy_nodecay_params, 'lr': policy_lr, 'weight_decay': 0.0},
            {'params': encoder_decay_params, 'lr': obs_enc_lr, 'weight_decay': weight_decay},
            {'params': encoder_nodecay_params, 'lr': obs_enc_lr, 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, betas=betas)
        return optimizer

    def predict_action(self, 
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        adaptive_halting: Optional[bool] = None,
        adaptive_min_k: int = 1,
    ) -> Dict[str, torch.Tensor]:
        if use_k_tokens is None:
            use_k_tokens = self.max_seq_len
        else:
            use_k_tokens = min(use_k_tokens, self.max_seq_len)
        if adaptive_halting is None:
            adaptive_halting = self.use_adaptive_halting
        elif adaptive_halting and not self.use_adaptive_halting:
            raise ValueError("Adaptive halting is not enabled for this policy checkpoint")
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk

        # encode observation
        features = self.obs_encoder(obs_dict)   # [B, To, d]
        B = features.shape[0]

        # autoregressive generation
        if adaptive_halting:
            decode_tokens, token_lens, eos_generated, entropy_at_stop, stop_reasons = (
                self._generate_action_tokens_with_entropy_halting(
                    features=features,
                    max_action_tokens=use_k_tokens,
                    temperature=temperature,
                    topk=topk,
                    adaptive_min_k=adaptive_min_k,
                )
            )
        else:
            action_tokens = torch.full( # [B, 1] seq: [<BOS>,]
                (B, 1), self.bos_id, 
                dtype=torch.long, device=self.device
            )
            generated_tokens = self.model.generate(
                action_tokens,
                cond=features,
                max_new_tokens=use_k_tokens,
                temperature=temperature,
                top_k=topk,
                eos_id=None,
            )[:, 1:]    # drop <BOS>

            decode_tokens, token_lens, eos_generated = self.extract_action_prefix(
                generated_tokens=generated_tokens,
                max_action_tokens=use_k_tokens,
                adaptive_min_k=adaptive_min_k,
            )
            entropy_at_stop = torch.full(
                (B,),
                float("nan"),
                dtype=features.dtype,
                device=features.device,
            )
            stop_reasons = torch.full(
                (B,),
                self.stop_reason_max_k,
                dtype=torch.long,
                device=features.device,
            )

        # decode action tokens
        with torch.inference_mode():
            action_pred = self.action_tokenizer.detokenize(
                tokens=decode_tokens,
                token_lens=token_lens,
            )

        # receeding horizon
        action = action_pred[:,:self.n_action_steps]

        result = {
            'action': action,
            'action_pred': action_pred,
            'action_tokens': decode_tokens,
            'token_lens': token_lens,
            'eos_generated': eos_generated,
            'entropy_at_stop': entropy_at_stop,
            'stop_reasons': stop_reasons,
            'adaptive_min_k': torch.tensor(int(adaptive_min_k), device=token_lens.device),
        }
        return result


    def forward(self, batch) -> torch.Tensor:
        # tokenize trajectory
        with torch.inference_mode():
            action_tokens = self.action_tokenizer.tokenize(batch['action'])
            oracle_keep_k = None
            if self.use_adaptive_halting:
                oracle_keep_k = self._oracle_keep_k_from_batch(batch)

        B = batch['action'].shape[0]
        device = batch['action'].device

        # encode observation
        features = self.obs_encoder(batch['obs'])   # [B, To, d]

        if self.use_adaptive_halting:
            model_tokens, target_tokens, target_mask = self.build_adaptive_training_batch(
                action_tokens=action_tokens,
                oracle_keep_k=oracle_keep_k,
            )
            logits = self.model(model_tokens, cond=features)
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_tokens.reshape(-1),
                ignore_index=self.loss_ignore_index,
            )
            mean_keep_k = oracle_keep_k.to(torch.float32).mean()
            token_ratio = mean_keep_k / float(self.max_seq_len)
            self._last_forward_info = {
                'mean_keep_k': mean_keep_k.detach(),
                'token_ratio': token_ratio.detach(),
                'batch_size': torch.tensor(float(B), device=device),
                'target_mask_tokens': target_mask.sum().to(torch.float32).detach(),
            }
        else:
            # prepend <BOS> token
            action_tokens = torch.cat([
                torch.full(
                    (B, 1), self.bos_id, 
                    dtype=torch.long, device=device
                ),
                action_tokens
            ], dim=1)

            # forward model
            logits = self.model(action_tokens[:, :-1], cond=features)

            # compute loss
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),     # (B*T, vocab_size)
                action_tokens[:, 1:].reshape(-1)    # (B*T,)
            )
            self._last_forward_info = {
                'mean_keep_k': torch.tensor(float(self.max_seq_len), device=device),
                'token_ratio': torch.tensor(1.0, device=device),
                'batch_size': torch.tensor(float(B), device=device),
            }
        return loss
