import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from oat.model.common.normalizer import LinearNormalizer
from oat.tokenizer.base_tokenizer import BaseTokenizer
from oat.tokenizer.oat.adaptive_prefix import build_adaptive_thresholds
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.tokenizer import OATTok
from oat.tokenizer.oat.quantizer.fsq import FSQ


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU-only experiment for entropy-gated adaptive OAT prefix length."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional OAT tokenizer checkpoint.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--available-depths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--adaptive-low-threshold", type=float, default=0.01)
    parser.add_argument("--adaptive-high-threshold", type=float, default=0.08)
    parser.add_argument("--quick-fit-steps", type=int, default=200)
    parser.add_argument("--quick-fit-batch-size", type=int, default=32)
    parser.add_argument("--quick-fit-lr", type=float, default=3e-4)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def generate_synthetic_actions(
    num_samples: int,
    horizon: int,
    action_dim: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    time_axis = torch.linspace(0.0, 1.0, steps=horizon)
    actions = []

    for sample_idx in range(num_samples):
        base_freq = 1.0 + (sample_idx % 3)
        phase = torch.rand(action_dim, generator=generator) * torch.pi
        amp = 0.3 + 0.4 * torch.rand(action_dim, generator=generator)
        smooth_wave = amp.unsqueeze(0) * torch.sin(
            (2.0 * torch.pi * base_freq * time_axis).unsqueeze(1) + phase.unsqueeze(0)
        )
        drift = torch.linspace(-0.25, 0.25, steps=horizon).unsqueeze(1) * (
            0.5 - torch.rand(action_dim, generator=generator)
        ).unsqueeze(0)

        if sample_idx % 2 == 0:
            action = smooth_wave + drift
        else:
            high_freq = 0.2 * torch.sin(
                (12.0 * torch.pi * time_axis).unsqueeze(1) + phase.unsqueeze(0)
            )
            noise = 0.18 * torch.randn(horizon, action_dim, generator=generator)
            bursts = torch.zeros(horizon, action_dim)
            burst_positions = torch.randint(0, horizon, (3,), generator=generator)
            bursts[burst_positions] = 0.35 * torch.randn(3, action_dim, generator=generator)
            action = smooth_wave + drift + high_freq + noise + bursts

        actions.append(action)

    return torch.stack(actions, dim=0).to(torch.float32)


def create_tiny_tokenizer(action_dim: int, horizon: int) -> OATTok:
    latent_dim = 4
    latent_horizon = 8
    tokenizer = OATTok(
        encoder=RegisterEncoder(
            sample_dim=action_dim,
            sample_horizon=horizon,
            emb_dim=64,
            head_dim=16,
            depth=2,
            pdropout=0.0,
            latent_dim=latent_dim,
            num_registers=latent_horizon,
        ),
        decoder=SinglePassDecoder(
            sample_dim=action_dim,
            sample_horizon=horizon,
            emb_dim=64,
            head_dim=16,
            depth=2,
            pdropout=0.0,
            token_dropout_mode="pow2",
            use_causal_decoder=True,
            latent_dim=latent_dim,
            latent_horizon=latent_horizon,
        ),
        quantizer=FSQ(levels=[8, 5, 5, 5]),
        adaptive_prefix=False,
    )
    return tokenizer


def fit_normalizer(tokenizer: OATTok, actions: torch.Tensor):
    normalizer = LinearNormalizer()
    normalizer.fit({"action": actions}, last_n_dims=1)
    tokenizer.set_normalizer(normalizer)


def quick_fit_tokenizer(
    tokenizer: OATTok,
    train_actions: torch.Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    if steps <= 0:
        return

    optimizer = tokenizer.get_optimizer(
        learning_rate=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    tokenizer.train()
    for _ in range(steps):
        batch_indices = torch.randint(
            low=0,
            high=train_actions.shape[0],
            size=(min(batch_size, train_actions.shape[0]),),
            generator=generator,
        )
        batch = {"action": train_actions[batch_indices]}
        loss = tokenizer(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    tokenizer.eval()


def load_tokenizer_from_checkpoint(checkpoint: str, device: torch.device):
    tokenizer, cfg = BaseTokenizer.from_checkpoint(
        checkpoint,
        return_configuration=True,
        map_location=device,
    )
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    return tokenizer, cfg


def evaluate_setting(
    tokenizer: OATTok,
    actions: torch.Tensor,
    name: str,
    eval_keep_k: Optional[List[int]] = None,
    adaptive_prefix: bool = False,
    available_depths: Optional[Sequence[int]] = None,
    adaptive_low_threshold: Optional[float] = None,
    adaptive_high_threshold: Optional[float] = None,
) -> Dict:
    thresholds = None
    if adaptive_prefix and available_depths is not None:
        thresholds = build_adaptive_thresholds(
            available_depths=available_depths,
            low_threshold=adaptive_low_threshold,
            high_threshold=adaptive_high_threshold,
        )

    with torch.inference_mode():
        metrics = tokenizer.evaluate_reconstruction(
            samples=actions,
            eval_keep_k=eval_keep_k,
            adaptive_prefix=adaptive_prefix,
            adaptive_available_depths=list(available_depths) if available_depths is not None else None,
            adaptive_thresholds=thresholds,
            return_recons=False,
        )

    metrics["name"] = name
    metrics["mode"] = "adaptive" if adaptive_prefix else "fixed"
    if thresholds is not None:
        metrics["adaptive_thresholds"] = thresholds
    return metrics


def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.checkpoint is not None:
        tokenizer, cfg = load_tokenizer_from_checkpoint(args.checkpoint, device=device)
        horizon = int(tokenizer.decoder.sample_horizon)
        action_dim = int(tokenizer.decoder.sample_dim)
        metadata = {
            "tokenizer_source": "checkpoint",
            "checkpoint": args.checkpoint,
            "latent_horizon": int(tokenizer.latent_horizon),
            "config_name": getattr(cfg, "name", None),
        }
    else:
        horizon = args.horizon
        action_dim = args.action_dim
        tokenizer = create_tiny_tokenizer(action_dim=action_dim, horizon=horizon).to(device)
        train_actions = generate_synthetic_actions(
            num_samples=args.train_samples,
            horizon=horizon,
            action_dim=action_dim,
            seed=args.seed,
        ).to(device)
        fit_normalizer(tokenizer, train_actions)
        quick_fit_tokenizer(
            tokenizer=tokenizer,
            train_actions=train_actions,
            steps=args.quick_fit_steps,
            batch_size=args.quick_fit_batch_size,
            learning_rate=args.quick_fit_lr,
            seed=args.seed,
        )
        metadata = {
            "tokenizer_source": "synthetic_tiny_oat",
            "checkpoint": None,
            "latent_horizon": int(tokenizer.latent_horizon),
            "quick_fit_steps": args.quick_fit_steps,
        }

    eval_actions = generate_synthetic_actions(
        num_samples=args.eval_samples,
        horizon=horizon,
        action_dim=action_dim,
        seed=args.seed + 1,
    ).to(device)

    available_depths = sorted(
        depth for depth in args.available_depths if depth <= tokenizer.latent_horizon
    )
    if len(available_depths) == 0:
        raise ValueError(
            f"No requested depths fit inside latent horizon {tokenizer.latent_horizon}: "
            f"{args.available_depths}"
        )

    results = []
    for depth in available_depths:
        results.append(
            evaluate_setting(
                tokenizer=tokenizer,
                actions=eval_actions,
                name=f"fixed_k_{depth}",
                eval_keep_k=[depth] * eval_actions.shape[0],
            )
        )

    results.append(
        evaluate_setting(
            tokenizer=tokenizer,
            actions=eval_actions,
            name="adaptive_k",
            adaptive_prefix=True,
            available_depths=available_depths,
            adaptive_low_threshold=args.adaptive_low_threshold,
            adaptive_high_threshold=args.adaptive_high_threshold,
        )
    )

    summary = {
        "metadata": {
            **metadata,
            "device": str(device),
            "eval_samples": int(eval_actions.shape[0]),
            "horizon": int(horizon),
            "action_dim": int(action_dim),
            "available_depths": available_depths,
            "adaptive_low_threshold": args.adaptive_low_threshold,
            "adaptive_high_threshold": args.adaptive_high_threshold,
        },
        "results": results,
    }

    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
