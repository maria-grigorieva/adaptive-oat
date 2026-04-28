#!/usr/bin/env python
"""
Run adaptive halting vs fixed-K evaluation.

The default path is a synthetic, CPU-compatible benchmark with no required
robotics dependencies. A lightweight, optional LIBERO smoke-evaluation path is
also provided and exits gracefully when dependencies or checkpoints are missing.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


NUMERIC_METRIC_SPECS = [
    ("avg_prefix_depth", "avg_K"),
    ("recon_mse", "recon_mse"),
    ("token_ratio", "token_ratio"),
    ("runtime_sec", "runtime_sec"),
    ("eos_rate", "eos_rate"),
    ("halting_accuracy", "halting_accuracy"),
]

LIBERO_DEPENDENCY_MODULES = ("libero", "robosuite", "mujoco")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adaptive halting vs fixed-K baselines.")
    parser.add_argument("--mode", choices=["fixed", "adaptive", "all"], default="all")
    parser.add_argument("--keep-ks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num-samples", type=int, default=768)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-checkpoint", type=str, default=None)
    parser.add_argument("--libero-eval", action="store_true", help="Optional full eval path via scripts/eval_policy_sim.py")
    parser.add_argument("--libero-smoke-eval", action="store_true", help="Optional lightweight LIBERO smoke test")
    parser.add_argument("--libero-suite", type=str, default="libero_spatial")
    parser.add_argument("--libero-num-tasks", type=int, default=1)
    parser.add_argument("--libero-num-episodes", type=int, default=1)
    parser.add_argument("--libero-headless", action="store_true", default=True)
    parser.add_argument("--include-random-baseline", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--halt-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("adaptive_halting_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def distribution_from_values(values: np.ndarray, keep_ks: Sequence[int]) -> Dict[str, float]:
    values = np.asarray(values)
    if values.size == 0:
        return {str(k): 0.0 for k in keep_ks}
    return {str(k): float(np.mean(values == k)) for k in keep_ks}


def mean_std_ci(values: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    ci95 = float(1.96 * std / math.sqrt(len(arr))) if len(arr) > 0 else None
    return mean, std, ci95


@dataclass
class SyntheticDataset:
    obs: np.ndarray
    actions: np.ndarray
    trajectory_types: np.ndarray


class OrderedPrefixTokenizer:
    def __init__(self, seq_len: int, action_dim: int, kmax: int):
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.kmax = kmax
        self.basis = self._build_basis(seq_len, kmax)

    @staticmethod
    def _build_basis(seq_len: int, kmax: int) -> np.ndarray:
        t = np.arange(seq_len, dtype=np.float64)
        basis = []
        for k in range(kmax):
            vec = np.cos(np.pi * (t + 0.5) * k / seq_len)
            vec = vec / np.linalg.norm(vec)
            basis.append(vec.astype(np.float32))
        return np.stack(basis, axis=0)

    def encode(self, actions: np.ndarray) -> np.ndarray:
        return np.einsum("ntd,kt->nkd", actions, self.basis, optimize=True)

    def reconstruct_from_codes(self, codes: np.ndarray, keep_k: int) -> np.ndarray:
        clipped_k = max(0, min(int(keep_k), self.kmax))
        if clipped_k == 0:
            return np.zeros((codes.shape[0], self.seq_len, self.action_dim), dtype=np.float32)
        return np.einsum(
            "nkd,kt->ntd",
            codes[:, :clipped_k, :],
            self.basis[:clipped_k],
            optimize=True,
        ).astype(np.float32)

    def prefix_errors(self, actions: np.ndarray, keep_ks: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        codes = self.encode(actions)
        errors = []
        for keep_k in keep_ks:
            recon = self.reconstruct_from_codes(codes, keep_k)
            mse = np.mean((recon - actions) ** 2, axis=(1, 2))
            errors.append(mse.astype(np.float32))
        return np.stack(errors, axis=1), codes

    def derive_oracle_keep_k(
        self,
        prefix_errors: np.ndarray,
        keep_ks: Sequence[int],
        tolerance: float,
    ) -> np.ndarray:
        keep_ks_arr = np.asarray(sorted(keep_ks), dtype=np.int64)
        sorted_errors = prefix_errors[:, np.argsort(keep_ks_arr)]
        full_error = sorted_errors[:, -1:]
        mask = sorted_errors <= (full_error + tolerance)
        first_valid_idx = mask.argmax(axis=1)
        return keep_ks_arr[first_valid_idx]


class SoftmaxHaltingPolicy:
    def __init__(self, keep_ks: Sequence[int], feature_dim: int, seed: int):
        self.keep_ks = np.asarray(list(keep_ks), dtype=np.int64)
        self.feature_dim = feature_dim
        self.num_classes = len(self.keep_ks)
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0.0, 0.05, size=(feature_dim, self.num_classes)).astype(np.float32)
        self.bias = np.zeros(self.num_classes, dtype=np.float32)
        self.mean = np.zeros(feature_dim, dtype=np.float32)
        self.std = np.ones(feature_dim, dtype=np.float32)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        num_steps: int = 300,
        learning_rate: float = 0.15,
        weight_decay: float = 1e-3,
    ) -> List[float]:
        self.mean = x_train.mean(axis=0, keepdims=False).astype(np.float32)
        self.std = x_train.std(axis=0, keepdims=False).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

        x_norm = self._normalize(x_train)
        y_one_hot = np.eye(self.num_classes, dtype=np.float32)[y_train]
        history = []
        for _ in range(num_steps):
            logits = x_norm @ self.weights + self.bias
            probs = softmax(logits)
            loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
            loss += 0.5 * weight_decay * np.sum(self.weights ** 2)
            history.append(float(loss))

            grad_logits = (probs - y_one_hot) / x_norm.shape[0]
            grad_w = x_norm.T @ grad_logits + weight_decay * self.weights
            grad_b = grad_logits.sum(axis=0)

            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b
        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self._normalize(x) @ self.weights + self.bias
        return softmax(logits)

    def predict_keep_k(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return self.keep_ks[np.argmax(probs, axis=1)]

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        probs = self.predict_proba(x)
        y_one_hot = np.eye(self.num_classes, dtype=np.float32)[y]
        return float(-np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1)))


def generate_synthetic_dataset(
    num_samples: int,
    seq_len: int,
    action_dim: int,
    tokenizer: OrderedPrefixTokenizer,
    seed: int,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    kmax = tokenizer.kmax
    basis = tokenizer.basis

    trajectory_types = rng.choice(3, size=num_samples, p=[0.4, 0.3, 0.3])
    actions = np.zeros((num_samples, seq_len, action_dim), dtype=np.float32)
    obs = np.zeros((num_samples, 8), dtype=np.float32)

    for idx in range(num_samples):
        traj_type = int(trajectory_types[idx])
        complexity = float(rng.beta(2.2, 2.0))
        amplitude = float(rng.uniform(0.6, 1.6))
        residual_noise = float(rng.uniform(0.01, 0.03))
        if traj_type == 0:
            residual_noise *= 0.6
            decay = 1.25 + 0.8 * (1.0 - complexity)
        elif traj_type == 1:
            residual_noise *= 2.8
            decay = 0.35 + 0.35 * (1.0 - complexity)
        else:
            residual_noise *= 1.6
            decay = 0.75 + 0.5 * (1.0 - complexity)

        obs[idx, 0] = complexity
        obs[idx, 1] = amplitude
        obs[idx, 2] = residual_noise
        obs[idx, 3 + traj_type] = 1.0

        coeffs = np.zeros((kmax, action_dim), dtype=np.float32)
        for dim in range(action_dim):
            dim_decay = decay
            if traj_type == 2 and dim >= action_dim // 2:
                dim_decay = 0.45 + 0.30 * (1.0 - complexity)
            coeff_scale = amplitude * np.exp(-dim_decay * np.arange(kmax))
            coeffs[:, dim] = rng.normal(0.0, coeff_scale).astype(np.float32)
            coeffs[0, dim] += amplitude * (0.8 + 0.25 * rng.random())
            coeffs[:, dim] *= 1.0 + 0.05 * dim

        action = np.einsum("kd,kt->td", coeffs, basis, optimize=True).astype(np.float32)
        action += residual_noise * rng.normal(size=action.shape).astype(np.float32)
        if traj_type != 0:
            burst_freq = int(rng.integers(max(2, kmax // 2), kmax + 1))
            burst = np.sin(np.linspace(0.0, burst_freq * np.pi, seq_len, dtype=np.float32))
            action += (0.03 + 0.12 * complexity) * burst[:, None]
        if traj_type == 0:
            action = np.cumsum(action, axis=0).astype(np.float32)
            action /= np.maximum(np.abs(action).max(), 1.0)
        actions[idx] = action

    return SyntheticDataset(obs=obs, actions=actions, trajectory_types=trajectory_types.astype(np.int64))


def train_test_split(dataset: SyntheticDataset, seed: int) -> Tuple[SyntheticDataset, SyntheticDataset]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset.actions.shape[0])
    split = int(round(0.7 * len(indices)))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return (
        SyntheticDataset(
            obs=dataset.obs[train_idx],
            actions=dataset.actions[train_idx],
            trajectory_types=dataset.trajectory_types[train_idx],
        ),
        SyntheticDataset(
            obs=dataset.obs[test_idx],
            actions=dataset.actions[test_idx],
            trajectory_types=dataset.trajectory_types[test_idx],
        ),
    )


def prefix_error_for_selected_k(
    prefix_errors: np.ndarray,
    keep_ks: Sequence[int],
    selected_k: np.ndarray,
) -> np.ndarray:
    index_map = {int(k): idx for idx, k in enumerate(keep_ks)}
    return np.asarray([prefix_errors[i, index_map[int(k)]] for i, k in enumerate(selected_k)], dtype=np.float32)


def evaluate_fixed_method(
    keep_k: int,
    keep_ks: Sequence[int],
    prefix_errors: np.ndarray,
    oracle_keep_k: np.ndarray,
    kmax: int,
) -> Dict[str, object]:
    start = time.perf_counter()
    selected_k = np.full(prefix_errors.shape[0], keep_k, dtype=np.int64)
    recon_mse = float(prefix_error_for_selected_k(prefix_errors, keep_ks, selected_k).mean())
    runtime = time.perf_counter() - start
    return {
        "method": f"fixed_k_{keep_k}",
        "avg_prefix_depth": float(selected_k.mean()),
        "std_prefix_depth": float(selected_k.std()),
        "recon_mse": recon_mse,
        "runtime_sec": runtime,
        "token_ratio": float(selected_k.mean() / kmax),
        "eos_rate": 0.0,
        "halting_accuracy": float(np.mean(selected_k == oracle_keep_k)),
        "selected_k_distribution": distribution_from_values(selected_k, keep_ks),
        "oracle_k_distribution": distribution_from_values(oracle_keep_k, keep_ks),
        "predicted_k_values": selected_k.tolist(),
        "oracle_k_values": oracle_keep_k.tolist(),
        "status": "ok",
        "eval_kind": "synthetic",
    }


def evaluate_random_method(
    keep_ks: Sequence[int],
    prefix_errors: np.ndarray,
    oracle_keep_k: np.ndarray,
    kmax: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    selected_k = rng.choice(np.asarray(keep_ks, dtype=np.int64), size=prefix_errors.shape[0])
    recon_mse = float(prefix_error_for_selected_k(prefix_errors, keep_ks, selected_k).mean())
    runtime = time.perf_counter() - start
    return {
        "method": "random_k",
        "avg_prefix_depth": float(selected_k.mean()),
        "std_prefix_depth": float(selected_k.std()),
        "recon_mse": recon_mse,
        "runtime_sec": runtime,
        "token_ratio": float(selected_k.mean() / kmax),
        "eos_rate": 1.0,
        "halting_accuracy": float(np.mean(selected_k == oracle_keep_k)),
        "selected_k_distribution": distribution_from_values(selected_k, keep_ks),
        "oracle_k_distribution": distribution_from_values(oracle_keep_k, keep_ks),
        "predicted_k_values": selected_k.tolist(),
        "oracle_k_values": oracle_keep_k.tolist(),
        "status": "ok",
        "eval_kind": "synthetic",
    }


def evaluate_oracle_method(
    keep_ks: Sequence[int],
    prefix_errors: np.ndarray,
    oracle_keep_k: np.ndarray,
    kmax: int,
) -> Dict[str, object]:
    start = time.perf_counter()
    selected_k = np.asarray(oracle_keep_k, dtype=np.int64)
    recon_mse = float(prefix_error_for_selected_k(prefix_errors, keep_ks, selected_k).mean())
    runtime = time.perf_counter() - start
    oracle_distribution = distribution_from_values(selected_k, keep_ks)
    return {
        "method": "oracle_k",
        "avg_prefix_depth": float(selected_k.mean()),
        "std_prefix_depth": float(selected_k.std()),
        "recon_mse": recon_mse,
        "runtime_sec": runtime,
        "token_ratio": float(selected_k.mean() / kmax),
        "eos_rate": float(np.mean(selected_k < kmax)),
        "halting_accuracy": 1.0,
        "selected_k_distribution": oracle_distribution,
        "oracle_k_distribution": oracle_distribution,
        "predicted_k_values": selected_k.tolist(),
        "oracle_k_values": selected_k.tolist(),
        "status": "ok",
        "eval_kind": "synthetic",
    }


def evaluate_adaptive_method(
    keep_ks: Sequence[int],
    x_train: np.ndarray,
    oracle_train: np.ndarray,
    x_test: np.ndarray,
    oracle_test: np.ndarray,
    prefix_errors_test: np.ndarray,
    kmax: int,
    seed: int,
) -> Dict[str, object]:
    class_lookup = {int(k): idx for idx, k in enumerate(keep_ks)}
    y_train = np.asarray([class_lookup[int(k)] for k in oracle_train], dtype=np.int64)
    y_test = np.asarray([class_lookup[int(k)] for k in oracle_test], dtype=np.int64)

    classifier = SoftmaxHaltingPolicy(keep_ks=keep_ks, feature_dim=x_train.shape[1], seed=seed)
    train_start = time.perf_counter()
    classifier.fit(x_train, y_train)
    train_runtime = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    predicted_k = classifier.predict_keep_k(x_test)
    infer_runtime = time.perf_counter() - infer_start

    recon_mse = float(prefix_error_for_selected_k(prefix_errors_test, keep_ks, predicted_k).mean())
    policy_loss = classifier.loss(x_test, y_test)
    return {
        "method": "adaptive_halting",
        "avg_prefix_depth": float(predicted_k.mean()),
        "std_prefix_depth": float(predicted_k.std()),
        "recon_mse": recon_mse,
        "runtime_sec": infer_runtime,
        "token_ratio": float(predicted_k.mean() / kmax),
        "eos_rate": 1.0,
        "halting_accuracy": float(np.mean(predicted_k == oracle_test)),
        "policy_loss": float(policy_loss),
        "train_runtime_sec": float(train_runtime),
        "selected_k_distribution": distribution_from_values(predicted_k, keep_ks),
        "oracle_k_distribution": distribution_from_values(oracle_test, keep_ks),
        "predicted_k_values": predicted_k.tolist(),
        "oracle_k_values": oracle_test.tolist(),
        "status": "ok",
        "eval_kind": "synthetic",
    }


def run_synthetic_evaluation(args: argparse.Namespace, logger: logging.Logger) -> Dict[str, List[Dict[str, object]]]:
    keep_ks = sorted(set(args.keep_ks))
    if keep_ks[0] <= 0:
        raise ValueError("keep_ks must be positive")
    kmax = max(keep_ks)
    tokenizer = OrderedPrefixTokenizer(seq_len=32, action_dim=7, kmax=kmax)
    methods_per_run: Dict[str, List[Dict[str, object]]] = {}

    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        set_seed(run_seed)
        dataset = generate_synthetic_dataset(
            num_samples=args.num_samples,
            seq_len=32,
            action_dim=7,
            tokenizer=tokenizer,
            seed=run_seed,
        )
        train_set, test_set = train_test_split(dataset, seed=run_seed)
        prefix_errors_train, _ = tokenizer.prefix_errors(train_set.actions, keep_ks)
        prefix_errors_test, _ = tokenizer.prefix_errors(test_set.actions, keep_ks)
        oracle_train = tokenizer.derive_oracle_keep_k(prefix_errors_train, keep_ks, args.halt_tolerance)
        oracle_test = tokenizer.derive_oracle_keep_k(prefix_errors_test, keep_ks, args.halt_tolerance)

        run_metrics: List[Dict[str, object]] = []
        if args.mode in {"fixed", "all"}:
            for keep_k in keep_ks:
                run_metrics.append(
                    evaluate_fixed_method(
                        keep_k=keep_k,
                        keep_ks=keep_ks,
                        prefix_errors=prefix_errors_test,
                        oracle_keep_k=oracle_test,
                        kmax=kmax,
                    )
                )
            if args.include_random_baseline:
                run_metrics.append(
                    evaluate_random_method(
                        keep_ks=keep_ks,
                        prefix_errors=prefix_errors_test,
                        oracle_keep_k=oracle_test,
                        kmax=kmax,
                        seed=run_seed,
                    )
                )
            run_metrics.append(
                evaluate_oracle_method(
                    keep_ks=keep_ks,
                    prefix_errors=prefix_errors_test,
                    oracle_keep_k=oracle_test,
                    kmax=kmax,
                )
            )
        if args.mode in {"adaptive", "all"}:
            run_metrics.append(
                evaluate_adaptive_method(
                    keep_ks=keep_ks,
                    x_train=train_set.obs,
                    oracle_train=oracle_train,
                    x_test=test_set.obs,
                    oracle_test=oracle_test,
                    prefix_errors_test=prefix_errors_test,
                    kmax=kmax,
                    seed=run_seed,
                )
            )

        methods_per_run[f"run_{run_idx}"] = run_metrics
        logger.info(
            "Synthetic run %d/%d complete | oracle distribution=%s",
            run_idx + 1,
            args.num_runs,
            distribution_from_values(oracle_test, keep_ks),
        )
    return methods_per_run


def probe_import(module_name: str) -> Dict[str, object]:
    try:
        importlib.import_module(module_name)
        return {"available": True, "error": None}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def get_libero_dependency_report() -> Dict[str, Dict[str, object]]:
    return {
        module_name: probe_import(module_name)
        for module_name in LIBERO_DEPENDENCY_MODULES
    }


def get_libero_dependency_flags(report: Dict[str, Dict[str, object]]) -> Dict[str, bool]:
    return {
        module_name: bool(details.get("available", False))
        for module_name, details in report.items()
    }


def summarize_missing_dependencies(report: Dict[str, Dict[str, object]]) -> str:
    missing = []
    for module_name, details in report.items():
        if details.get("available", False):
            continue
        missing.append(f"{module_name}: {details.get('error', 'unknown import error')}")
    return "; ".join(missing)


def save_smoke_status(
    path: Path,
    status: str,
    reason: str,
    dependency_flags: Dict[str, bool],
) -> None:
    payload = {
        "status": status,
        "reason": reason,
        "dependencies": dependency_flags,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_libero_smoke_methods(args: argparse.Namespace) -> List[Tuple[str, Dict[str, object]]]:
    keep_ks = sorted(set(args.keep_ks))
    methods: List[Tuple[str, Dict[str, object]]] = []
    if args.mode in {"fixed", "all"}:
        methods.extend((f"fixed_k_{k}", {"use_k_tokens": k}) for k in keep_ks)
    if args.mode in {"adaptive", "all"}:
        methods.append(("adaptive_halting", {"adaptive_halting": True}))
    return methods


def build_smoke_metric(
    method_name: str,
    keep_ks: Sequence[int],
    status: str,
    notes: str,
    *,
    avg_prefix_depth: Optional[float] = None,
    runtime_sec: Optional[float] = None,
    token_ratio: Optional[float] = None,
    eos_rate: Optional[float] = None,
    success_rate: Optional[float] = None,
    recon_mse: Optional[float] = None,
    selected_dist: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    return {
        "method": method_name,
        "avg_prefix_depth": avg_prefix_depth if avg_prefix_depth is not None else math.nan,
        "std_prefix_depth": 0.0 if avg_prefix_depth is not None else math.nan,
        "recon_mse": recon_mse if recon_mse is not None else math.nan,
        "runtime_sec": runtime_sec if runtime_sec is not None else math.nan,
        "token_ratio": token_ratio if token_ratio is not None else math.nan,
        "eos_rate": eos_rate if eos_rate is not None else math.nan,
        "halting_accuracy": None,
        "selected_k_distribution": selected_dist or {str(k): 0.0 for k in keep_ks},
        "oracle_k_distribution": {},
        "predicted_k_values": [],
        "oracle_k_values": [],
        "success_rate": success_rate if success_rate is not None else math.nan,
        "status": status,
        "failure_reason": notes if status != "passed" else "",
        "notes": notes,
        "eval_kind": "libero_smoke",
    }


def build_skipped_smoke_metrics(
    methods: Sequence[Tuple[str, Dict[str, object]]],
    keep_ks: Sequence[int],
    reason: str,
) -> Dict[str, List[Dict[str, object]]]:
    return {
        "run_0": [
            build_smoke_metric(method_name, keep_ks, status="skipped", notes=reason)
            for method_name, _ in methods
        ]
    }


def format_subprocess_failure(exc: subprocess.CalledProcessError) -> str:
    stderr = (exc.stderr or "").strip()
    stdout = (exc.stdout or "").strip()
    details = stderr or stdout
    if details:
        details = " | ".join(details.splitlines()[-3:])
        return f"command exited with code {exc.returncode}: {details}"
    return f"command exited with code {exc.returncode}"


def try_run_libero_smoke_eval(args: argparse.Namespace, logger: logging.Logger) -> Tuple[Dict[str, List[Dict[str, object]]], Path]:
    output_dir = Path(args.output_dir)
    status_path = output_dir / "libero_smoke_status.json"
    keep_ks = sorted(set(args.keep_ks))
    methods = build_libero_smoke_methods(args)
    dependency_report = get_libero_dependency_report()
    dependency_flags = get_libero_dependency_flags(dependency_report)

    if not all(dependency_flags.values()):
        reason = summarize_missing_dependencies(dependency_report)
        warning = f"Skipping LIBERO smoke evaluation because required dependencies are unavailable: {reason}"
        logger.warning(warning)
        save_smoke_status(status_path, status="skipped", reason=warning, dependency_flags=dependency_flags)
        return build_skipped_smoke_metrics(methods, keep_ks, warning), status_path

    if args.policy_checkpoint is None:
        warning = "Skipping LIBERO smoke evaluation because --policy-checkpoint was not provided."
        logger.warning(warning)
        save_smoke_status(status_path, status="skipped", reason=warning, dependency_flags=dependency_flags)
        return build_skipped_smoke_metrics(methods, keep_ks, warning), status_path

    methods_per_run: Dict[str, List[Dict[str, object]]] = {"run_0": []}

    for method_name, method_kwargs in methods:
        method_output = output_dir / "run_0" / method_name
        command = [
            sys.executable,
            "scripts/eval_policy_sim.py",
            "--checkpoint",
            str(args.policy_checkpoint),
            "--output_dir",
            str(method_output),
            "--num_exp",
            "1",
            "--device",
            args.device,
            "--force",
            "--libero-task-limit",
            str(args.libero_num_tasks),
            "--libero-episodes-per-task",
            str(args.libero_num_episodes),
            "--libero-n-test-vis",
            "0",
            "--libero-n-parallel-envs",
            "1",
        ]
        if "use_k_tokens" in method_kwargs:
            command.extend(["--use_k_tokens", str(method_kwargs["use_k_tokens"])])
        if method_kwargs.get("adaptive_halting", False):
            command.append("--adaptive-halting")

        env = os_environ_headless(args.libero_headless, args.device)
        start = time.perf_counter()
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            runtime = time.perf_counter() - start
            metrics_path = method_output / "eval_log.json"
            if not metrics_path.is_file():
                raise RuntimeError(f"Missing expected eval output: {metrics_path}")
            payload = json.loads(metrics_path.read_text())
            avg_prefix_depth = payload.get("mean_action_tokens_mean", payload.get("mean_action_tokens"))
            token_ratio = payload.get("token_ratio_mean", payload.get("token_ratio"))
            eos_rate = payload.get("eos_prediction_rate_mean", payload.get("eos_prediction_rate", 0.0))
            success_rate = payload.get("mean_success_rate_mean", payload.get("mean_success_rate"))
            selected_dist = {
                str(k): float(payload.get(f"pred_keep_k_{k}_mean", payload.get(f"pred_keep_k_{k}", 0.0)))
                for k in keep_ks
            }
            methods_per_run["run_0"].append(
                build_smoke_metric(
                    method_name,
                    keep_ks,
                    status="passed",
                    notes="completed",
                    avg_prefix_depth=float(avg_prefix_depth) if avg_prefix_depth is not None else None,
                    runtime_sec=float(runtime),
                    token_ratio=float(token_ratio) if token_ratio is not None else None,
                    eos_rate=float(eos_rate) if eos_rate is not None else None,
                    success_rate=float(success_rate) if success_rate is not None else None,
                    recon_mse=payload.get("test_reconst_mse_mean", payload.get("test_reconst_mse")),
                    selected_dist=selected_dist,
                )
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - environment dependent
            runtime = time.perf_counter() - start
            failure_reason = format_subprocess_failure(exc)
            logger.warning("LIBERO smoke eval failed for %s: %s", method_name, failure_reason)
            methods_per_run["run_0"].append(
                build_smoke_metric(
                    method_name,
                    keep_ks,
                    status="failed",
                    notes=failure_reason,
                    runtime_sec=float(runtime),
                )
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            runtime = time.perf_counter() - start
            failure_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("LIBERO smoke eval failed for %s: %s", method_name, failure_reason)
            methods_per_run["run_0"].append(
                build_smoke_metric(
                    method_name,
                    keep_ks,
                    status="failed",
                    notes=failure_reason,
                    runtime_sec=float(runtime),
                )
            )

    all_passed = all(metric.get("status") == "passed" for metric in methods_per_run["run_0"])
    status = "passed" if all_passed else "failed"
    reason = (
        "LIBERO smoke evaluation completed for all requested methods."
        if all_passed
        else "One or more LIBERO smoke methods failed."
    )
    save_smoke_status(status_path, status=status, reason=reason, dependency_flags=dependency_flags)
    return methods_per_run, status_path


def os_environ_headless(headless: bool, device: str) -> Dict[str, str]:
    env = dict(os.environ)
    if headless:
        software_backend = "osmesa" if str(device).lower() == "cpu" else "egl"
        env.setdefault("MUJOCO_GL", software_backend)
        env.setdefault("PYOPENGL_PLATFORM", software_backend)
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    return env


def aggregate_metrics(
    methods_per_run: Dict[str, List[Dict[str, object]]],
    keep_ks: Sequence[int],
) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for run_metrics in methods_per_run.values():
        for metric in run_metrics:
            grouped.setdefault(metric["method"], []).append(metric)

    aggregated: Dict[str, Dict[str, object]] = {}
    for method, metrics_list in grouped.items():
        row: Dict[str, object] = {"method": method}
        for metric_key, alias in NUMERIC_METRIC_SPECS:
            values = []
            for metric in metrics_list:
                value = metric.get(metric_key)
                if value is None:
                    continue
                try:
                    value_f = float(value)
                except Exception:
                    continue
                if math.isnan(value_f):
                    continue
                values.append(value_f)
            mean, std, ci95 = mean_std_ci(values)
            row[f"{alias}_mean"] = mean
            row[f"{alias}_std"] = std
            row[f"{alias}_ci95"] = ci95

        selected_dists = np.asarray(
            [[metric["selected_k_distribution"].get(str(k), 0.0) for k in keep_ks] for metric in metrics_list],
            dtype=np.float64,
        )
        row["selected_k_distribution"] = {str(k): float(selected_dists[:, idx].mean()) for idx, k in enumerate(keep_ks)}

        oracle_dists = [metric.get("oracle_k_distribution", {}) for metric in metrics_list if metric.get("oracle_k_distribution")]
        if oracle_dists:
            oracle_np = np.asarray([[dist.get(str(k), 0.0) for k in keep_ks] for dist in oracle_dists], dtype=np.float64)
            row["oracle_k_distribution"] = {str(k): float(oracle_np[:, idx].mean()) for idx, k in enumerate(keep_ks)}
        else:
            row["oracle_k_distribution"] = {}

        predicted_values = [value for metric in metrics_list for value in metric.get("predicted_k_values", [])]
        oracle_values = [value for metric in metrics_list for value in metric.get("oracle_k_values", [])]
        row["predicted_k_values"] = predicted_values
        row["oracle_k_values"] = oracle_values
        row["num_completed_runs"] = int(
            sum(metric.get("status", "ok") in {"ok", "passed"} for metric in metrics_list)
        )
        aggregated[method] = row
    return aggregated


def write_summary_csv(output_path: Path, aggregated: Dict[str, Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "avg_K_mean", "avg_K_std",
        "recon_mse_mean", "recon_mse_std",
        "token_ratio_mean", "token_ratio_std",
        "runtime_sec_mean", "runtime_sec_std",
        "eos_rate_mean", "eos_rate_std",
        "halting_accuracy_mean", "halting_accuracy_std",
        "selected_k_distribution",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in sorted(aggregated):
            row = aggregated[method]
            writer.writerow({
                "method": method,
                "avg_K_mean": row.get("avg_K_mean"),
                "avg_K_std": row.get("avg_K_std"),
                "recon_mse_mean": row.get("recon_mse_mean"),
                "recon_mse_std": row.get("recon_mse_std"),
                "token_ratio_mean": row.get("token_ratio_mean"),
                "token_ratio_std": row.get("token_ratio_std"),
                "runtime_sec_mean": row.get("runtime_sec_mean"),
                "runtime_sec_std": row.get("runtime_sec_std"),
                "eos_rate_mean": row.get("eos_rate_mean"),
                "eos_rate_std": row.get("eos_rate_std"),
                "halting_accuracy_mean": row.get("halting_accuracy_mean"),
                "halting_accuracy_std": row.get("halting_accuracy_std"),
                "selected_k_distribution": json.dumps(row.get("selected_k_distribution", {}), sort_keys=True),
            })


def write_runs_csv(output_path: Path, methods_per_run: Dict[str, List[Dict[str, object]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "method",
        "avg_prefix_depth",
        "std_prefix_depth",
        "recon_mse",
        "runtime_sec",
        "token_ratio",
        "eos_rate",
        "halting_accuracy",
        "success_rate",
        "status",
        "failure_reason",
        "notes",
        "selected_k_distribution",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_id, run_metrics in methods_per_run.items():
            for metric in run_metrics:
                writer.writerow({
                    "run_id": run_id,
                    "method": metric.get("method"),
                    "avg_prefix_depth": metric.get("avg_prefix_depth"),
                    "std_prefix_depth": metric.get("std_prefix_depth"),
                    "recon_mse": metric.get("recon_mse"),
                    "runtime_sec": metric.get("runtime_sec"),
                    "token_ratio": metric.get("token_ratio"),
                    "eos_rate": metric.get("eos_rate"),
                    "halting_accuracy": metric.get("halting_accuracy"),
                    "success_rate": metric.get("success_rate"),
                    "status": metric.get("status", "ok"),
                    "failure_reason": metric.get("failure_reason", ""),
                    "notes": metric.get("notes", ""),
                    "selected_k_distribution": json.dumps(metric.get("selected_k_distribution", {}), sort_keys=True),
                })


def write_libero_smoke_summary_csv(
    output_path: Path,
    methods_per_run: Dict[str, List[Dict[str, object]]],
    method_order: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_by_method: Dict[str, List[Dict[str, object]]] = {}
    for run_metrics in methods_per_run.values():
        for metric in run_metrics:
            metrics_by_method.setdefault(str(metric.get("method")), []).append(metric)

    fieldnames = [
        "method",
        "success_rate",
        "avg_K",
        "token_ratio",
        "runtime_sec",
        "status",
        "notes",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method_name in method_order:
            metrics = metrics_by_method.get(method_name, [])
            passed_metrics = [metric for metric in metrics if metric.get("status") == "passed"]
            status = "skipped"
            notes = "not evaluated"
            if metrics:
                if all(metric.get("status") == "passed" for metric in metrics):
                    status = "passed"
                    notes = "completed"
                elif any(metric.get("status") == "failed" for metric in metrics):
                    status = "failed"
                    notes = "; ".join(
                        str(metric.get("notes") or metric.get("failure_reason") or "").strip()
                        for metric in metrics
                        if metric.get("status") == "failed"
                    ) or "one or more runs failed"
                else:
                    notes = "; ".join(
                        str(metric.get("notes") or "").strip()
                        for metric in metrics
                        if metric.get("status") == "skipped"
                    ) or "skipped"

            def mean_metric(key: str) -> Optional[float]:
                values = []
                for metric in passed_metrics:
                    value = metric.get(key)
                    if value is None:
                        continue
                    try:
                        value_f = float(value)
                    except Exception:
                        continue
                    if math.isnan(value_f):
                        continue
                    values.append(value_f)
                if not values:
                    return None
                return float(np.mean(values))

            writer.writerow(
                {
                    "method": method_name,
                    "success_rate": mean_metric("success_rate"),
                    "avg_K": mean_metric("avg_prefix_depth"),
                    "token_ratio": mean_metric("token_ratio"),
                    "runtime_sec": mean_metric("runtime_sec"),
                    "status": status,
                    "notes": notes,
                }
            )


def save_metrics_json(
    output_path: Path,
    args: argparse.Namespace,
    methods_per_run: Dict[str, List[Dict[str, object]]],
    aggregated: Dict[str, Dict[str, object]],
) -> None:
    payload = {
        "config": {
            "mode": args.mode,
            "keep_ks": args.keep_ks,
            "num_samples": args.num_samples,
            "num_runs": args.num_runs,
            "device": args.device,
            "seed": args.seed,
            "policy_checkpoint": args.policy_checkpoint,
            "libero_eval": args.libero_eval,
            "libero_smoke_eval": args.libero_smoke_eval,
            "include_random_baseline": args.include_random_baseline,
            "halt_tolerance": args.halt_tolerance,
        },
        "per_run": methods_per_run,
        "aggregate": aggregated,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def print_paper_ready_table(aggregated: Dict[str, Dict[str, object]], logger: logging.Logger) -> None:
    lines = [
        "",
        "Paper-ready summary",
        "method | avg_K mean±std | recon_mse mean±std | token_ratio mean±std | runtime_sec mean±std | eos_rate mean±std | halting_acc mean±std",
        "--- | --- | --- | --- | --- | --- | ---",
    ]
    for method in sorted(aggregated):
        row = aggregated[method]

        def fmt(prefix: str) -> str:
            mean = row.get(f"{prefix}_mean")
            std = row.get(f"{prefix}_std")
            if mean is None:
                return "n/a"
            return f"{float(mean):.4f} ± {float(std or 0.0):.4f}"

        lines.append(
            f"{method} | {fmt('avg_K')} | {fmt('recon_mse')} | {fmt('token_ratio')} | "
            f"{fmt('runtime_sec')} | {fmt('eos_rate')} | {fmt('halting_accuracy')}"
        )
    logger.info("\n%s", "\n".join(lines))


def run_plotter(summary_csv_path: Path, output_dir: Path, logger: logging.Logger) -> None:
    command = [
        sys.executable,
        "experiments/plot_results.py",
        "--input",
        str(summary_csv_path),
        "--output-dir",
        str(output_dir / "plots"),
    ]
    logger.info("Generating plots: %s", " ".join(command))
    subprocess.run(command, check=True)


def log_sanity_checks(aggregated: Dict[str, Dict[str, object]], keep_ks: Sequence[int], logger: logging.Logger) -> None:
    adaptive = aggregated.get("adaptive_halting")
    if adaptive is None:
        return
    predicted = np.asarray(adaptive.get("predicted_k_values", []), dtype=np.int64)
    kmax = max(keep_ks)
    if predicted.size > 0:
        if int(predicted.max()) > kmax:
            raise RuntimeError("Adaptive K exceeded Kmax")
        if np.unique(predicted).size <= 1:
            logger.warning("Adaptive K distribution is degenerate: %s", np.unique(predicted).tolist())
        if float(np.mean(predicted == min(keep_ks))) >= 0.95:
            logger.warning("EOS is almost always triggered at the earliest prefix.")
    fixed_k8 = aggregated.get(f"fixed_k_{kmax}")
    if fixed_k8 and adaptive.get("runtime_sec_mean") and fixed_k8.get("runtime_sec_mean"):
        if float(adaptive["runtime_sec_mean"]) > 5.0 * float(fixed_k8["runtime_sec_mean"]):
            logger.warning(
                "Adaptive runtime increased substantially: %.6fs vs %.6fs",
                adaptive["runtime_sec_mean"],
                fixed_k8["runtime_sec_mean"],
            )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary_csv_path = output_dir / "adaptive_halting_summary.csv"
    runs_csv_path = output_dir / "adaptive_halting_runs.csv"
    metrics_json_path = output_dir / "adaptive_halting_metrics.json"
    logger = setup_logging(Path("research_logs") / "adaptive_halting_eval.log")

    logger.info("Starting adaptive halting evaluation with args=%s", vars(args))
    methods_per_run = run_synthetic_evaluation(args, logger)
    aggregated = aggregate_metrics(methods_per_run, keep_ks=sorted(set(args.keep_ks)))
    write_summary_csv(summary_csv_path, aggregated)
    write_runs_csv(runs_csv_path, methods_per_run)
    save_metrics_json(metrics_json_path, args, methods_per_run, aggregated)
    log_sanity_checks(aggregated, keep_ks=sorted(set(args.keep_ks)), logger=logger)
    print_paper_ready_table(aggregated, logger)

    if not args.skip_plots:
        try:
            run_plotter(summary_csv_path, output_dir, logger)
        except Exception as exc:
            logger.warning("Plot generation failed: %s", exc)

    if args.libero_smoke_eval:
        smoke_metrics, status_path = try_run_libero_smoke_eval(args, logger)
        smoke_summary_path = output_dir / "libero_smoke_summary.csv"
        smoke_runs_path = output_dir / "libero_smoke_runs.csv"
        write_libero_smoke_summary_csv(
            smoke_summary_path,
            smoke_metrics,
            method_order=[method_name for method_name, _ in build_libero_smoke_methods(args)],
        )
        write_runs_csv(smoke_runs_path, smoke_metrics)
        logger.info("LIBERO smoke summary written to %s", smoke_summary_path.resolve())
        logger.info("LIBERO smoke status written to %s", status_path.resolve())

    logger.info("Artifacts written to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
