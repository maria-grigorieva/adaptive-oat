from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


DEFAULT_KEEP_KS: Tuple[int, ...] = (1, 2, 4, 8)
DEFAULT_THRESHOLD_QUANTILES: Tuple[float, ...] = (0.25, 0.5, 0.75)
DEFAULT_ALT_THRESHOLD_SETS: Tuple[Tuple[float, ...], ...] = (
    (0.15, 0.4, 0.7),
    (0.3, 0.6, 0.85),
)


@dataclass(frozen=True)
class TrainConfig:
    action_dim: int = 7
    horizon: int = 32
    train_samples: int = 512
    test_samples: int = 256
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    emb_dim: int = 64
    head_dim: int = 16
    encoder_depth: int = 2
    decoder_depth: int = 2
    latent_levels: Tuple[int, ...] = (8, 5, 5, 5)
    pdropout: float = 0.1
    train_token_dropout_mode: str = "pow2"
    synthetic_noise_scale: float = 0.03


@dataclass(frozen=True)
class MethodSpec:
    name: str
    method_type: str
    complexity_metric: Optional[str] = None
    mapping_strategy: Optional[str] = None
    thresholds: Optional[Tuple[float, ...]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistically rigorous CPU experiment for adaptive OAT prefix depth."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/adaptive_prefix"))
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--mse-tolerance", type=float, default=1e-3)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--synthetic-noise-scale", type=float, default=0.03)
    parser.add_argument("--complexity-metric", choices=("delta_l2_mean", "temporal_variance"), default="delta_l2_mean")
    parser.add_argument(
        "--keep-ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_KEEP_KS),
        help="Allowed prefix lengths. Keep these power-of-two for OAT nested decoding.",
    )
    parser.add_argument(
        "--threshold-quantiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLD_QUANTILES),
        help="Quantiles for the primary adaptive threshold mapping.",
    )
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help="Add no-threshold, alternate-threshold, and alternate-metric adaptive baselines.",
    )
    parser.add_argument("--disable-plots", action="store_true")
    return parser.parse_args()


def validate_keep_ks(keep_ks: Sequence[int]) -> Tuple[int, ...]:
    ordered = tuple(sorted(int(k) for k in keep_ks))
    if not ordered:
        raise ValueError("keep_ks must not be empty.")
    if ordered[0] < 1:
        raise ValueError(f"keep_ks must be positive, got {ordered}.")
    if ordered[0] & (ordered[0] - 1):
        raise ValueError(f"keep_ks must be powers of two, got {ordered}.")
    for prev, curr in zip(ordered, ordered[1:]):
        if curr <= prev:
            raise ValueError(f"keep_ks must be strictly increasing, got {ordered}.")
        if curr & (curr - 1):
            raise ValueError(f"keep_ks must be powers of two, got {ordered}.")
    return ordered


def validate_thresholds(thresholds: Sequence[float], keep_ks: Sequence[int]) -> Tuple[float, ...]:
    thresholds = tuple(float(x) for x in thresholds)
    if len(thresholds) != len(keep_ks) - 1:
        raise ValueError(
            f"Expected {len(keep_ks) - 1} thresholds for keep_ks={tuple(keep_ks)}, got {thresholds}."
        )
    if not all(0.0 < x < 1.0 for x in thresholds):
        raise ValueError(f"Threshold quantiles must lie in (0, 1), got {thresholds}.")
    if any(b <= a for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError(f"Threshold quantiles must be strictly increasing, got {thresholds}.")
    return thresholds


def set_global_seed(seed: int, *, enable_torch: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if enable_torch:
        import torch

        torch.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _smooth_component(time_axis: np.ndarray, rng: np.random.Generator, action_dim: int) -> np.ndarray:
    base_freq = rng.uniform(0.5, 2.0, size=(action_dim, 1))
    high_freq = rng.uniform(2.0, 5.0, size=(action_dim, 1))
    phase = rng.uniform(0.0, 2.0 * math.pi, size=(action_dim, 1))
    harmonic_phase = rng.uniform(0.0, 2.0 * math.pi, size=(action_dim, 1))
    amplitude = rng.uniform(0.2, 0.7, size=(action_dim, 1))
    harmonic = rng.uniform(0.05, 0.2, size=(action_dim, 1))
    signal = amplitude * np.sin(2.0 * math.pi * base_freq * time_axis + phase)
    signal += harmonic * np.sin(2.0 * math.pi * high_freq * time_axis + harmonic_phase)
    return signal.T


def generate_synthetic_actions(
    num_samples: int,
    horizon: int,
    action_dim: int,
    seed: int,
    noise_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[None, :]
    samples = np.zeros((num_samples, horizon, action_dim), dtype=np.float32)
    latent_complexity = rng.uniform(0.0, 1.0, size=num_samples).astype(np.float32)

    for idx, complexity in enumerate(latent_complexity):
        base = _smooth_component(time_axis, rng, action_dim)
        local_noise = rng.normal(
            loc=0.0,
            scale=noise_scale * (0.4 + 2.6 * complexity),
            size=(horizon, action_dim),
        )
        drift = rng.normal(loc=0.0, scale=0.03 + 0.08 * complexity, size=(horizon, action_dim))
        drift = np.cumsum(drift, axis=0)
        drift -= drift.mean(axis=0, keepdims=True)

        jump_mask = (rng.random((horizon, action_dim)) < (0.02 + 0.18 * complexity)).astype(np.float32)
        jump_values = rng.normal(loc=0.0, scale=0.12 + 0.8 * complexity, size=(horizon, action_dim))
        jumps = np.cumsum(jump_mask * jump_values, axis=0)

        high_freq = np.sin(
            2.0 * math.pi * rng.uniform(4.0, 9.0, size=(action_dim, 1)) * time_axis
            + rng.uniform(0.0, 2.0 * math.pi, size=(action_dim, 1))
        ).T

        trajectory = base
        trajectory += (0.15 + 0.65 * complexity) * drift
        trajectory += (0.02 + 0.55 * complexity) * high_freq
        trajectory += 0.1 * jumps
        trajectory += local_noise
        trajectory = np.tanh(trajectory).astype(np.float32)
        samples[idx] = trajectory

    return samples, latent_complexity


def compute_complexity_scores(actions: np.ndarray, metric: str) -> np.ndarray:
    if actions.ndim != 3:
        raise ValueError(f"Expected actions with shape [N, T, D], got {actions.shape}.")
    if metric == "delta_l2_mean":
        deltas = np.diff(actions, axis=1)
        return np.linalg.norm(deltas, axis=2).mean(axis=1)
    if metric == "temporal_variance":
        return actions.var(axis=1).mean(axis=1)
    raise ValueError(f"Unsupported complexity metric: {metric}")


def calibrate_quantile_thresholds(scores: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    quantiles = np.asarray(tuple(float(q) for q in quantiles), dtype=np.float64)
    return np.quantile(scores, quantiles)


def map_scores_to_keep_k(
    scores: np.ndarray,
    keep_ks: Sequence[int],
    strategy: str,
    calibration_scores: np.ndarray,
    quantiles: Optional[Sequence[float]] = None,
) -> np.ndarray:
    keep_ks_array = np.asarray(tuple(int(k) for k in keep_ks), dtype=np.int64)
    if strategy == "quantile":
        if quantiles is None:
            raise ValueError("quantiles are required for quantile mapping.")
        thresholds = calibrate_quantile_thresholds(calibration_scores, quantiles)
        bins = np.digitize(scores, thresholds, right=True)
        return keep_ks_array[bins]

    if strategy == "linear":
        min_score = float(calibration_scores.min())
        max_score = float(calibration_scores.max())
        denom = max(max_score - min_score, 1e-8)
        normalized = np.clip((scores - min_score) / denom, 0.0, 1.0)
        scaled = normalized * (len(keep_ks_array) - 1)
        return keep_ks_array[np.rint(scaled).astype(np.int64)]

    raise ValueError(f"Unsupported mapping strategy: {strategy}")


def sample_random_keep_k(
    num_samples: int,
    keep_ks: Sequence[int],
    seed: int,
    probs: Optional[Sequence[float]] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep_ks_array = np.asarray(tuple(int(k) for k in keep_ks), dtype=np.int64)
    if probs is None:
        probs = np.full(len(keep_ks_array), 1.0 / len(keep_ks_array), dtype=np.float64)
    return rng.choice(keep_ks_array, size=num_samples, replace=True, p=np.asarray(probs, dtype=np.float64))


def allocation_distribution(keep_k: np.ndarray, keep_ks: Sequence[int]) -> Dict[str, float]:
    total = max(int(keep_k.size), 1)
    return {str(k): float(np.sum(keep_k == k) / total) for k in keep_ks}


def find_failure_modes(keep_k: np.ndarray, keep_ks: Sequence[int]) -> Dict[str, bool]:
    max_fraction = max(allocation_distribution(keep_k, keep_ks).values())
    return {
        "always_min_k": bool(np.all(keep_k == min(keep_ks))),
        "always_max_k": bool(np.all(keep_k == max(keep_ks))),
        "near_degenerate": bool(max_fraction >= 0.95),
    }


def bootstrap_mean_ci(values: Sequence[float], seed: int, num_bootstrap: int = 2000) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(num_bootstrap, dtype=np.float64)
    for idx in range(num_bootstrap):
        sample = rng.choice(array, size=array.size, replace=True)
        means[idx] = sample.mean()
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def exact_sign_flip_pvalue(diffs: Sequence[float], alternative: str = "less") -> float:
    array = np.asarray(diffs, dtype=np.float64)
    if array.size == 0:
        raise ValueError("diffs must not be empty.")
    observed = float(array.mean())
    total = 1 << array.size
    signs = np.ones(array.size, dtype=np.float64)
    extreme = 0
    for mask in range(total):
        for bit in range(array.size):
            signs[bit] = -1.0 if (mask >> bit) & 1 else 1.0
        candidate = float((array * signs).mean())
        if alternative == "less":
            extreme += candidate <= observed + 1e-12
        elif alternative == "greater":
            extreme += candidate >= observed - 1e-12
        else:
            raise ValueError(f"Unsupported alternative: {alternative}")
    return float(extreme / total)


def paired_t_test_if_available(
    lhs: Sequence[float],
    rhs: Sequence[float],
    *,
    popmean_shift: float,
    alternative: str,
) -> Optional[Dict[str, float]]:
    if len(lhs) < 2 or len(rhs) < 2:
        return None
    try:
        from scipy import stats  # type: ignore
    except ImportError:
        return None

    diffs = np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)
    result = stats.ttest_1samp(diffs, popmean=popmean_shift, alternative=alternative)
    statistic = float(getattr(result, "statistic", np.nan))
    pvalue = float(getattr(result, "pvalue", np.nan))
    if math.isnan(statistic) or math.isnan(pvalue):
        return None
    return {"statistic": statistic, "pvalue": pvalue}


def summarise_runs(records: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row["method"]), []).append(row)

    summary: Dict[str, Dict[str, Any]] = {}
    for method, rows in grouped.items():
        mse_values = [float(row["mse"]) for row in rows]
        expected_k_values = [float(row["expected_k"]) for row in rows]
        runtime_values = [float(row["runtime_sec"]) for row in rows]
        summary[method] = {
            "num_runs": len(rows),
            "mse_mean": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values, ddof=1)) if len(rows) > 1 else 0.0,
            "mse_ci95": bootstrap_mean_ci(mse_values, seed + 11 * len(rows)),
            "expected_k_mean": float(np.mean(expected_k_values)),
            "expected_k_std": float(np.std(expected_k_values, ddof=1)) if len(rows) > 1 else 0.0,
            "expected_k_ci95": bootstrap_mean_ci(expected_k_values, seed + 19 * len(rows)),
            "runtime_mean_sec": float(np.mean(runtime_values)),
            "runtime_std_sec": float(np.std(runtime_values, ddof=1)) if len(rows) > 1 else 0.0,
            "allocation_mean": {
                key: float(np.mean([float(row[key]) for row in rows]))
                for key in rows[0]
                if key.startswith("alloc_k_")
            },
        }
    return summary


def run_pairwise_tests(
    records: Sequence[Mapping[str, Any]],
    adaptive_method: str,
    fixed_methods: Sequence[str],
    mse_tolerance: float,
) -> Dict[str, Dict[str, Any]]:
    by_method: Dict[str, List[Mapping[str, Any]]] = {}
    for row in records:
        by_method.setdefault(str(row["method"]), []).append(row)

    adaptive_rows = sorted(by_method[adaptive_method], key=lambda row: int(row["seed"]))
    results: Dict[str, Dict[str, Any]] = {}
    adaptive_mse = [float(row["mse"]) for row in adaptive_rows]
    adaptive_k = [float(row["expected_k"]) for row in adaptive_rows]

    for baseline in fixed_methods:
        baseline_rows = sorted(by_method[baseline], key=lambda row: int(row["seed"]))
        baseline_mse = [float(row["mse"]) for row in baseline_rows]
        baseline_k = [float(row["expected_k"]) for row in baseline_rows]

        mse_diffs = np.asarray(adaptive_mse, dtype=np.float64) - np.asarray(baseline_mse, dtype=np.float64)
        k_diffs = np.asarray(adaptive_k, dtype=np.float64) - np.asarray(baseline_k, dtype=np.float64)

        mse_test = paired_t_test_if_available(
            adaptive_mse,
            baseline_mse,
            popmean_shift=mse_tolerance,
            alternative="less",
        )
        if mse_test is None:
            mse_test = {
                "statistic": float("nan"),
                "pvalue": exact_sign_flip_pvalue(mse_diffs - mse_tolerance, alternative="less"),
            }
            mse_test["test_name"] = "exact_sign_flip_noninferiority"
        else:
            mse_test["test_name"] = "paired_t_test_noninferiority"

        budget_test = paired_t_test_if_available(
            adaptive_k,
            baseline_k,
            popmean_shift=0.0,
            alternative="less",
        )
        if budget_test is None:
            budget_test = {
                "statistic": float("nan"),
                "pvalue": exact_sign_flip_pvalue(k_diffs, alternative="less"),
            }
            budget_test["test_name"] = "exact_sign_flip_budget"
        else:
            budget_test["test_name"] = "paired_t_test_budget"

        results[baseline] = {
            "mse_diff_mean": float(mse_diffs.mean()),
            "mse_diff_ci95": bootstrap_mean_ci(mse_diffs, seed=101 + len(mse_diffs)),
            "budget_diff_mean": float(k_diffs.mean()),
            "budget_diff_ci95": bootstrap_mean_ci(k_diffs, seed=211 + len(k_diffs)),
            "mse_tolerance": mse_tolerance,
            "mse_test": mse_test,
            "budget_test": budget_test,
            "noninferior_at_alpha_0_05": bool(mse_test["pvalue"] < 0.05 and mse_diffs.mean() <= mse_tolerance),
            "lower_budget_at_alpha_0_05": bool(budget_test["pvalue"] < 0.05 and k_diffs.mean() < 0.0),
        }

    return results


def build_method_specs(
    keep_ks: Sequence[int],
    complexity_metric: str,
    threshold_quantiles: Sequence[float],
    run_ablations: bool,
) -> List[MethodSpec]:
    specs: List[MethodSpec] = [
        MethodSpec(name=f"fixed_k_{k}", method_type="fixed") for k in keep_ks
    ]
    specs.append(MethodSpec(name="random_k_uniform", method_type="random"))
    specs.append(
        MethodSpec(
            name=f"adaptive_{complexity_metric}_quantile",
            method_type="adaptive",
            complexity_metric=complexity_metric,
            mapping_strategy="quantile",
            thresholds=tuple(float(x) for x in threshold_quantiles),
        )
    )

    if run_ablations:
        specs.append(
            MethodSpec(
                name=f"adaptive_{complexity_metric}_linear",
                method_type="adaptive",
                complexity_metric=complexity_metric,
                mapping_strategy="linear",
            )
        )
        other_metric = "temporal_variance" if complexity_metric == "delta_l2_mean" else "delta_l2_mean"
        specs.append(
            MethodSpec(
                name=f"adaptive_{other_metric}_quantile",
                method_type="adaptive",
                complexity_metric=other_metric,
                mapping_strategy="quantile",
                thresholds=tuple(float(x) for x in threshold_quantiles),
            )
        )
        for idx, thresholds in enumerate(DEFAULT_ALT_THRESHOLD_SETS, start=1):
            if len(thresholds) == len(keep_ks) - 1:
                specs.append(
                    MethodSpec(
                        name=f"adaptive_{complexity_metric}_quantile_alt{idx}",
                        method_type="adaptive",
                        complexity_metric=complexity_metric,
                        mapping_strategy="quantile",
                        thresholds=thresholds,
                    )
                )
    return specs


def train_tokenizer(
    train_actions: np.ndarray,
    keep_ks: Sequence[int],
    config: TrainConfig,
    seed: int,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from oat.model.common.normalizer import LinearNormalizer
    from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
    from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
    from oat.tokenizer.oat.quantizer.fsq import FSQ
    from oat.tokenizer.oat.tokenizer import OATTok

    set_global_seed(seed, enable_torch=True)

    tokenizer = OATTok(
        encoder=RegisterEncoder(
            sample_dim=config.action_dim,
            sample_horizon=config.horizon,
            emb_dim=config.emb_dim,
            head_dim=config.head_dim,
            depth=config.encoder_depth,
            pdropout=config.pdropout,
            latent_dim=len(config.latent_levels),
            num_registers=max(keep_ks),
        ),
        decoder=SinglePassDecoder(
            sample_dim=config.action_dim,
            sample_horizon=config.horizon,
            emb_dim=config.emb_dim,
            head_dim=config.head_dim,
            depth=config.decoder_depth,
            pdropout=config.pdropout,
            token_dropout_mode=config.train_token_dropout_mode,
            use_causal_decoder=True,
            latent_dim=len(config.latent_levels),
            latent_horizon=max(keep_ks),
        ),
        quantizer=FSQ(levels=list(config.latent_levels)),
    )

    normalizer = LinearNormalizer()
    normalizer.fit({"action": train_actions}, last_n_dims=1)
    tokenizer.set_normalizer(normalizer)

    optimizer = tokenizer.get_optimizer(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    data_tensor = torch.from_numpy(train_actions).float()
    dataset = TensorDataset(data_tensor)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    tokenizer.train()
    for _ in range(config.epochs):
        for (batch_actions,) in dataloader:
            optimizer.zero_grad(set_to_none=True)
            loss = tokenizer({"action": batch_actions})
            loss.backward()
            optimizer.step()

    tokenizer.eval()
    return tokenizer


def evaluate_method(
    tokenizer: Any,
    actions: np.ndarray,
    keep_k: np.ndarray,
    batch_size: int,
    seed: int,
    method_name: str,
    complexity_scores: np.ndarray,
    latent_complexity: np.ndarray,
    keep_ks: Sequence[int],
) -> Dict[str, Any]:
    import torch

    set_global_seed(seed, enable_torch=True)

    actions_tensor = torch.from_numpy(actions).float()
    total_squared_error = 0.0
    total_values = 0

    start_time = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(actions), batch_size):
            end = min(start + batch_size, len(actions))
            batch_actions = actions_tensor[start:end]
            batch_keep_k = keep_k[start:end].tolist()
            recons = tokenizer.autoencode(batch_actions, eval_keep_k=batch_keep_k)
            total_squared_error += float((recons - batch_actions).pow(2).sum().item())
            total_values += int(batch_actions.numel())
    runtime_sec = time.perf_counter() - start_time

    allocation = allocation_distribution(keep_k, keep_ks)
    failures = find_failure_modes(keep_k, keep_ks)
    record: Dict[str, Any] = {
        "method": method_name,
        "mse": total_squared_error / max(total_values, 1),
        "expected_k": float(np.mean(keep_k)),
        "runtime_sec": runtime_sec,
        "complexity_score_mean": float(np.mean(complexity_scores)),
        "complexity_score_std": float(np.std(complexity_scores)),
        "latent_complexity_mean": float(np.mean(latent_complexity)),
        "latent_complexity_std": float(np.std(latent_complexity)),
    }
    for key, value in allocation.items():
        record[f"alloc_k_{key}"] = value
    record.update(failures)
    return record


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def maybe_make_plots(
    output_dir: Path,
    detail_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    adaptive_method: str,
    keep_ks: Sequence[int],
) -> Optional[List[str]]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plot_paths: List[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    adaptive_rows = [row for row in detail_rows if row["method"] == adaptive_method]
    if adaptive_rows:
        x_values = [float(row["complexity_score"]) for row in adaptive_rows]
        y_values = [int(row["keep_k"]) for row in adaptive_rows]
        plt.figure(figsize=(7, 4))
        plt.scatter(x_values, y_values, s=18, alpha=0.55)
        plt.yticks(list(keep_ks))
        plt.xlabel("Complexity score")
        plt.ylabel("Allocated prefix length")
        plt.title("Adaptive prefix allocation vs complexity")
        plt.tight_layout()
        scatter_path = output_dir / "adaptive_allocation_vs_complexity.png"
        plt.savefig(scatter_path, dpi=180)
        plt.close()
        plot_paths.append(str(scatter_path))

    methods = list(summary.keys())
    budget_values = [float(summary[method]["expected_k_mean"]) for method in methods]
    mse_values = [float(summary[method]["mse_mean"]) for method in methods]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(methods)), budget_values)
    plt.xticks(range(len(methods)), methods, rotation=45, ha="right")
    plt.ylabel("Expected prefix length")
    plt.title("Token budget by method")

    plt.subplot(1, 2, 2)
    plt.bar(range(len(methods)), mse_values)
    plt.xticks(range(len(methods)), methods, rotation=45, ha="right")
    plt.ylabel("Reconstruction MSE")
    plt.title("Reconstruction error by method")
    plt.tight_layout()
    summary_path = output_dir / "method_summary.png"
    plt.savefig(summary_path, dpi=180)
    plt.close()
    plot_paths.append(str(summary_path))

    return plot_paths


def run_single_seed(
    seed: int,
    config: TrainConfig,
    keep_ks: Sequence[int],
    method_specs: Sequence[MethodSpec],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_actions, _ = generate_synthetic_actions(
        num_samples=config.train_samples,
        horizon=config.horizon,
        action_dim=config.action_dim,
        seed=seed,
        noise_scale=config.synthetic_noise_scale,
    )
    test_actions, latent_complexity = generate_synthetic_actions(
        num_samples=config.test_samples,
        horizon=config.horizon,
        action_dim=config.action_dim,
        seed=seed + 10_000,
        noise_scale=config.synthetic_noise_scale,
    )

    tokenizer = train_tokenizer(train_actions, keep_ks, config=config, seed=seed)

    per_method_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    cached_scores: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for spec in method_specs:
        if spec.method_type == "fixed":
            fixed_k = int(spec.name.rsplit("_", maxsplit=1)[-1])
            keep_k = np.full(config.test_samples, fixed_k, dtype=np.int64)
            scores = np.zeros(config.test_samples, dtype=np.float32)

        elif spec.method_type == "random":
            keep_k = sample_random_keep_k(config.test_samples, keep_ks, seed=seed + 77)
            scores = np.zeros(config.test_samples, dtype=np.float32)

        elif spec.method_type == "adaptive":
            if spec.complexity_metric not in cached_scores:
                train_scores = compute_complexity_scores(train_actions, spec.complexity_metric or "")
                test_scores = compute_complexity_scores(test_actions, spec.complexity_metric or "")
                cached_scores[spec.complexity_metric or ""] = (train_scores, test_scores)
            train_scores, test_scores = cached_scores[spec.complexity_metric or ""]
            keep_k = map_scores_to_keep_k(
                scores=test_scores,
                keep_ks=keep_ks,
                strategy=spec.mapping_strategy or "quantile",
                calibration_scores=train_scores,
                quantiles=spec.thresholds,
            )
            scores = test_scores.astype(np.float32)

        else:
            raise ValueError(f"Unknown method type: {spec.method_type}")

        record = evaluate_method(
            tokenizer=tokenizer,
            actions=test_actions,
            keep_k=keep_k,
            batch_size=config.batch_size,
            seed=seed,
            method_name=spec.name,
            complexity_scores=scores,
            latent_complexity=latent_complexity,
            keep_ks=keep_ks,
        )
        record["seed"] = seed
        per_method_rows.append(record)

        if spec.method_type == "adaptive":
            for idx in range(len(test_actions)):
                detail_rows.append(
                    {
                        "seed": seed,
                        "method": spec.name,
                        "sample_index": idx,
                        "complexity_score": float(scores[idx]),
                        "latent_complexity": float(latent_complexity[idx]),
                        "keep_k": int(keep_k[idx]),
                    }
                )

    return per_method_rows, detail_rows


def format_summary_table(summary: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "method,mse_mean,mse_std,expected_k_mean,expected_k_std,runtime_mean_sec"
    ]
    for method, stats in summary.items():
        lines.append(
            ",".join(
                [
                    method,
                    f"{stats['mse_mean']:.6f}",
                    f"{stats['mse_std']:.6f}",
                    f"{stats['expected_k_mean']:.3f}",
                    f"{stats['expected_k_std']:.3f}",
                    f"{stats['runtime_mean_sec']:.4f}",
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    keep_ks = validate_keep_ks(args.keep_ks)
    threshold_quantiles = validate_thresholds(args.threshold_quantiles, keep_ks)
    config = TrainConfig(
        action_dim=args.action_dim,
        horizon=args.horizon,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        emb_dim=args.emb_dim,
        head_dim=args.head_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        synthetic_noise_scale=args.synthetic_noise_scale,
    )

    method_specs = build_method_specs(
        keep_ks=keep_ks,
        complexity_metric=args.complexity_metric,
        threshold_quantiles=threshold_quantiles,
        run_ablations=args.run_ablations,
    )
    adaptive_method = f"adaptive_{args.complexity_metric}_quantile"
    fixed_methods = [f"fixed_k_{k}" for k in keep_ks]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    for run_idx in range(args.num_runs):
        seed = args.base_seed + run_idx
        rows, details = run_single_seed(seed=seed, config=config, keep_ks=keep_ks, method_specs=method_specs)
        all_rows.extend(rows)
        detail_rows.extend(details)

    summary = summarise_runs(all_rows, seed=args.base_seed)
    pairwise = run_pairwise_tests(
        records=all_rows,
        adaptive_method=adaptive_method,
        fixed_methods=fixed_methods,
        mse_tolerance=args.mse_tolerance,
    )
    plot_paths = None if args.disable_plots else maybe_make_plots(
        output_dir=output_dir,
        detail_rows=detail_rows,
        summary=summary,
        adaptive_method=adaptive_method,
        keep_ks=keep_ks,
    )

    save_csv(output_dir / "per_run_metrics.csv", all_rows)
    save_csv(output_dir / "adaptive_detail.csv", detail_rows)
    save_json(
        output_dir / "summary.json",
        {
            "config": asdict(config),
            "keep_ks": keep_ks,
            "threshold_quantiles": threshold_quantiles,
            "adaptive_method": adaptive_method,
            "summary": summary,
            "pairwise_tests": pairwise,
            "plot_paths": plot_paths,
        },
    )

    readme_lines = [
        f"Adaptive method: {adaptive_method}",
        f"MSE tolerance: {args.mse_tolerance}",
        "",
        format_summary_table(summary),
        "",
        json.dumps(pairwise, indent=2, sort_keys=True),
    ]
    (output_dir / "report.txt").write_text("\n".join(readme_lines), encoding="utf-8")
    print("\n".join(readme_lines))


if __name__ == "__main__":
    main()
