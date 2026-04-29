#!/usr/bin/env python3

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from oat.dataset.zarr_dataset import ZarrDataset
from oat.tokenizer.oat.tokenizer import OATTok


def find_latest_tinyplus_tokenizer_checkpoint() -> str:
    candidates = sorted(
        Path("output").glob("**/*train_oattok_tinyplus_cpu*/checkpoints/latest.ckpt")
    )
    if not candidates:
        raise FileNotFoundError("Could not find a tinyplus tokenizer checkpoint under output/")
    return str(candidates[-1])


def normalize_distribution(dist: Dict[int, float], support: List[int]) -> Dict[int, float]:
    values = {k: float(dist.get(k, 0.0)) for k in support}
    total = sum(values.values())
    if total <= 0:
        return {k: 0.0 for k in support}
    return {k: v / total for k, v in values.items()}


def distribution_mean(dist: Dict[int, float]) -> float:
    return float(sum(k * v for k, v in dist.items()))


def plot_distribution(
    output_path: Path,
    title: str,
    distributions: Dict[str, Dict[int, float]],
    support: List[int],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    width = 0.35 if len(distributions) > 1 else 0.6
    x = np.arange(len(support))
    offsets = np.linspace(-width * (len(distributions) - 1) / 2, width * (len(distributions) - 1) / 2, len(distributions))

    for offset, (label, dist) in zip(offsets, distributions.items()):
        values = [dist.get(k, 0.0) for k in support]
        plt.bar(x + offset, values, width=width, label=label)

    plt.xticks(x, support)
    plt.xlabel("Prefix depth K")
    plt.ylabel("Probability")
    plt.title(title)
    if len(distributions) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


@click.command()
@click.option("--tokenizer-checkpoint", default=None, type=str, help="Path to tokenizer checkpoint. If omitted, uses the latest tinyplus tokenizer checkpoint.")
@click.option("--dataset", default="data/libero/libero_spatial_tinyplus.zarr", show_default=True, type=str)
@click.option("--output-dir", default="results", show_default=True, type=str)
@click.option("--adaptive-diagnostics", default="output/eval/libero_spatial_tinyplus_adaptive/eos_diagnostics.json", show_default=True, type=str)
@click.option("--batch-size", default=64, show_default=True, type=int)
@click.option("--horizon", default=32, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--tolerance", default=1e-3, show_default=True, type=float)
@click.option("--keep-ks", multiple=True, type=int, default=[1, 2, 4, 8], show_default=True)
def main(
    tokenizer_checkpoint: Optional[str],
    dataset: str,
    output_dir: str,
    adaptive_diagnostics: str,
    batch_size: int,
    horizon: int,
    seed: int,
    tolerance: float,
    keep_ks: List[int],
):
    if tokenizer_checkpoint is None:
        tokenizer_checkpoint = find_latest_tinyplus_tokenizer_checkpoint()

    keep_ks = sorted(set(int(k) for k in keep_ks))
    max_keep_k = max(keep_ks)
    full_keep_ks = list(range(1, max_keep_k + 1))
    keep_k_to_full_idx = {k: idx for idx, k in enumerate(full_keep_ks)}
    sparse_indices = [keep_k_to_full_idx[k] for k in keep_ks]
    oracle_keep_to_sparse_idx = {k: idx for idx, k in enumerate(keep_ks)}

    output_root = Path(output_dir)
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    tokenizer = OATTok.from_checkpoint(tokenizer_checkpoint)
    tokenizer.to(device)
    tokenizer.eval()

    dataset_obj = ZarrDataset(
        zarr_path=dataset,
        obs_keys=[],
        action_key="action",
        n_obs_steps=0,
        n_action_steps=horizon,
        seed=seed,
        val_ratio=0.0,
    )
    dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, num_workers=0)

    total_chunks = 0
    fixed_error_sums = torch.zeros(len(full_keep_ks), dtype=torch.float64)
    oracle_error_sum = 0.0
    oracle_keep_counter: Counter = Counter()

    for batch in dataloader:
        actions = batch["action"].to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            prefix_errors_full = tokenizer.compute_prefix_reconstruction_errors(
                samples=actions,
                keep_ks=full_keep_ks,
            )
            prefix_errors_sparse = prefix_errors_full[:, sparse_indices]
            oracle_keep_k = tokenizer.derive_oracle_keep_k(
                prefix_errors=prefix_errors_sparse,
                keep_ks=keep_ks,
                tolerance=tolerance,
            )

        batch_size_actual = actions.shape[0]
        total_chunks += batch_size_actual
        fixed_error_sums += prefix_errors_full.sum(dim=0).cpu().to(torch.float64)
        oracle_keep_counter.update(int(k) for k in oracle_keep_k.cpu().tolist())

        gather_idx = torch.tensor(
            [oracle_keep_to_sparse_idx[int(k)] for k in oracle_keep_k.cpu().tolist()],
            device=prefix_errors_sparse.device,
            dtype=torch.long,
        )
        oracle_batch_errors = prefix_errors_sparse.gather(1, gather_idx.unsqueeze(1)).squeeze(1)
        oracle_error_sum += float(oracle_batch_errors.sum().item())

    fixed_mean_errors = {k: float(fixed_error_sums[keep_k_to_full_idx[k]].item() / total_chunks) for k in full_keep_ks}
    oracle_distribution = normalize_distribution(
        {k: oracle_keep_counter.get(k, 0) for k in keep_ks},
        keep_ks,
    )
    oracle_mean_k = distribution_mean(oracle_distribution)
    oracle_recon_mse = oracle_error_sum / max(total_chunks, 1)

    oracle_csv_path = output_root / "libero_tinyplus_oracle_diagnostics.csv"
    with oracle_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "value"],
        )
        writer.writeheader()
        writer.writerow({"metric": "num_action_chunks", "value": total_chunks})
        writer.writerow({"metric": "mean_oracle_k", "value": oracle_mean_k})
        writer.writerow({"metric": "oracle_recon_mse", "value": oracle_recon_mse})
        for keep_k in keep_ks:
            writer.writerow({"metric": f"oracle_dist_k_{keep_k}", "value": oracle_distribution.get(keep_k, 0.0)})
        for keep_k in keep_ks:
            writer.writerow({"metric": f"fixed_recon_mse_k_{keep_k}", "value": fixed_mean_errors[keep_k]})

    oracle_json = {
        "tokenizer_checkpoint": tokenizer_checkpoint,
        "dataset": dataset,
        "num_action_chunks": total_chunks,
        "keep_ks": keep_ks,
        "tolerance": tolerance,
        "oracle_k_distribution": {str(k): oracle_distribution.get(k, 0.0) for k in keep_ks},
        "mean_oracle_k": oracle_mean_k,
        "prefix_reconstruction_mse": {str(k): fixed_mean_errors[k] for k in keep_ks},
        "oracle_reconstruction_mse": oracle_recon_mse,
    }
    oracle_json_path = output_root / "libero_tinyplus_oracle_diagnostics.json"
    json.dump(oracle_json, oracle_json_path.open("w"), indent=2, sort_keys=True)

    plot_distribution(
        plots_dir / "libero_tinyplus_oracle_k_distribution.svg",
        "Tinyplus Oracle K Distribution",
        {"oracle": oracle_distribution},
        keep_ks,
    )

    predicted_distribution = None
    comparison_json = None
    adaptive_source = Path(adaptive_diagnostics)
    if adaptive_source.exists():
        adaptive_diag = json.load(adaptive_source.open())
        predicted_distribution = normalize_distribution(
            {int(k): float(v) for k, v in adaptive_diag["selected_k_distribution"].items()},
            list(range(1, max_keep_k + 1)),
        )

        oracle_full_distribution = normalize_distribution(
            {k: oracle_distribution.get(k, 0.0) for k in full_keep_ks},
            full_keep_ks,
        )
        predicted_mean_k = distribution_mean(predicted_distribution)
        oracle_mean_full = distribution_mean(oracle_full_distribution)
        l1_distance = float(sum(abs(predicted_distribution[k] - oracle_full_distribution[k]) for k in full_keep_ks))
        predicted_early_mass = float(sum(predicted_distribution.get(k, 0.0) for k in [1, 2]))
        oracle_deep_mass = float(sum(oracle_full_distribution.get(k, 0.0) for k in [4, 5, 6, 7, 8]))
        collapse_warning = bool(predicted_early_mass >= 0.8 and oracle_deep_mass >= 0.1)

        comparison_rows = [
            {"metric": "oracle_mean_k", "value": oracle_mean_full},
            {"metric": "predicted_mean_k", "value": predicted_mean_k},
            {"metric": "absolute_mean_gap", "value": abs(predicted_mean_k - oracle_mean_full)},
            {"metric": "distribution_l1_distance", "value": l1_distance},
            {"metric": "predicted_mass_k_le_2", "value": predicted_early_mass},
            {"metric": "oracle_mass_k_ge_4", "value": oracle_deep_mass},
            {"metric": "collapse_warning", "value": int(collapse_warning)},
        ]
        comparison_csv_path = output_root / "libero_tinyplus_halting_comparison.csv"
        with comparison_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(comparison_rows)

        comparison_json = {
            "adaptive_diagnostics_path": str(adaptive_source),
            "oracle_distribution": {str(k): oracle_full_distribution.get(k, 0.0) for k in full_keep_ks},
            "predicted_distribution": {str(k): predicted_distribution.get(k, 0.0) for k in full_keep_ks},
            "oracle_mean_k": oracle_mean_full,
            "predicted_mean_k": predicted_mean_k,
            "absolute_mean_gap": abs(predicted_mean_k - oracle_mean_full),
            "distribution_l1_distance": l1_distance,
            "predicted_mass_k_le_2": predicted_early_mass,
            "oracle_mass_k_ge_4": oracle_deep_mass,
            "collapse_warning": collapse_warning,
            "collapse_reason": (
                "Predicted K is heavily concentrated at K<=2 while oracle mass still assigns "
                "non-trivial probability to deeper prefixes."
                if collapse_warning else
                "No strong early-stop collapse signal from the oracle-vs-predicted comparison."
            ),
        }
        comparison_json_path = output_root / "libero_tinyplus_halting_comparison.json"
        json.dump(comparison_json, comparison_json_path.open("w"), indent=2, sort_keys=True)

        plot_distribution(
            plots_dir / "libero_tinyplus_oracle_vs_predicted_k.svg",
            "Tinyplus Oracle vs Predicted K",
            {
                "oracle": oracle_full_distribution,
                "predicted": predicted_distribution,
            },
            full_keep_ks,
        )

    reconstruction_rows = []
    for keep_k in keep_ks:
        reconstruction_rows.append({
            "method": f"fixed_k_{keep_k}",
            "avg_K": float(keep_k),
            "token_ratio": float(keep_k / max_keep_k),
            "recon_mse": fixed_mean_errors[keep_k],
        })

    reconstruction_rows.append({
        "method": "oracle_k",
        "avg_K": oracle_mean_k,
        "token_ratio": float(oracle_mean_k / max_keep_k),
        "recon_mse": oracle_recon_mse,
    })

    adaptive_recon_proxy = None
    if predicted_distribution is not None:
        adaptive_mean_k = distribution_mean(predicted_distribution)
        adaptive_recon_proxy = float(sum(
            predicted_distribution.get(k, 0.0) * fixed_mean_errors[k]
            for k in full_keep_ks
        ))
        reconstruction_rows.append({
            "method": "adaptive_predicted_k",
            "avg_K": adaptive_mean_k,
            "token_ratio": float(adaptive_mean_k / max_keep_k),
            "recon_mse": adaptive_recon_proxy,
        })

    reconstruction_csv_path = output_root / "libero_tinyplus_reconstruction_proxy.csv"
    with reconstruction_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "avg_K", "token_ratio", "recon_mse"])
        writer.writeheader()
        writer.writerows(reconstruction_rows)

    reconstruction_json = {
        "dataset": dataset,
        "tokenizer_checkpoint": tokenizer_checkpoint,
        "methods": reconstruction_rows,
        "notes": (
            "adaptive_predicted_k uses the rollout-predicted K distribution from the adaptive "
            "LIBERO evaluation, not per-sample policy predictions on the dataset."
            if adaptive_recon_proxy is not None else
            "No adaptive rollout diagnostics were available, so the adaptive reconstruction proxy was skipped."
        ),
    }
    reconstruction_json_path = output_root / "libero_tinyplus_reconstruction_proxy.json"
    json.dump(reconstruction_json, reconstruction_json_path.open("w"), indent=2, sort_keys=True)

    oracle_mass_ge4 = float(sum(oracle_distribution.get(k, 0.0) for k in keep_ks if k >= 4))
    predicted_mean_k = None if predicted_distribution is None else distribution_mean(predicted_distribution)
    collapse_warning = False if comparison_json is None else bool(comparison_json["collapse_warning"])

    recommendations = []
    if collapse_warning:
        recommendations.extend([
            "Train the adaptive policy longer before comparing control behavior.",
            "Run evaluation with `--adaptive-halting --adaptive-min-k 2` as a conservative calibration baseline.",
            "Consider a minimum prefix depth during early training or a penalty against EOS before K=2.",
            "Tune `halt_tolerance` and budget regularization so oracle supervision is less aggressively biased toward shallow prefixes if needed.",
        ])
    else:
        recommendations.extend([
            "Keep the current halting setup and gather more rollout data before changing the objective.",
            "Log EOS probability directly if the generate() API is extended.",
        ])

    md_lines = [
        "# Tinyplus Halting Diagnostics",
        "",
        f"- Dataset: `{dataset}`",
        f"- Tokenizer checkpoint: `{tokenizer_checkpoint}`",
        f"- Action chunks analyzed: `{total_chunks}`",
        "",
        "## Answers",
        "",
        f"1. Oracle K distribution mean: `{oracle_mean_k:.4f}`.",
        f"   Oracle distribution over `{keep_ks}`: `{oracle_distribution}`.",
        f"2. Adaptive predicted K mean: `{predicted_mean_k:.4f}`." if predicted_mean_k is not None else "2. Adaptive predicted K distribution was unavailable.",
        f"   Predicted distribution over `1..{max_keep_k}`: `{predicted_distribution}`." if predicted_distribution is not None else "",
        f"3. Adaptive halting collapsed toward early stopping: `{collapse_warning}`.",
        f"   Oracle deep-prefix mass (`K>=4`) is `{oracle_mass_ge4:.4f}`." if predicted_distribution is not None else "",
        "4. Recommended next changes:",
    ]
    for rec in recommendations:
        md_lines.append(f"- {rec}")

    md_lines.extend([
        "",
        "## Reconstruction Proxy",
        "",
    ])
    for row in reconstruction_rows:
        md_lines.append(
            f"- `{row['method']}`: avg_K=`{row['avg_K']:.4f}`, token_ratio=`{row['token_ratio']:.4f}`, recon_mse=`{row['recon_mse']:.6f}`"
        )

    diagnostics_md_path = output_root / "libero_tinyplus_halting_diagnostics.md"
    diagnostics_md_path.write_text("\n".join(line for line in md_lines if line != ""))

    print(f"tokenizer_checkpoint={tokenizer_checkpoint}")
    print(f"dataset={dataset}")
    print(f"oracle_distribution={oracle_distribution}")
    if predicted_distribution is not None:
        print(f"predicted_distribution={predicted_distribution}")
        print(f"collapse_warning={collapse_warning}")


if __name__ == "__main__":
    main()
