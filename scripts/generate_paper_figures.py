#!/usr/bin/env python3
"""Generate paper-ready figures for Entropy-Gated Adaptive Prefix Depth for OAT.

The figures in this script are intentionally conservative:
- synthetic benchmark is the main quantitative result
- LIBERO tiny/tinyplus artifacts are presented only as integration validation
- no claim of LIBERO task-performance improvement is made
"""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper_figures"

SUMMARY_CANDIDATES = [
    ROOT / "results_30runs" / "adaptive_halting_summary.csv",
    ROOT / "results" / "adaptive_halting_summary.csv",
]

TINYPLUS_SUMMARY_CANDIDATES = [
    ROOT / "results" / "libero_tinyplus_summary.csv",
    ROOT / "results" / "libero_tiny_summary.csv",
]

TINYPLUS_DIAGNOSTICS = ROOT / "results" / "libero_tinyplus_diagnostics.json"
TINYPLUS_ORACLE_DIAGNOSTICS_CSV = ROOT / "results" / "libero_tinyplus_oracle_diagnostics.csv"
TINYPLUS_HALTING_COMPARISON_CSV = ROOT / "results" / "libero_tinyplus_halting_comparison.csv"
TINYPLUS_RECON_PROXY_CSV = ROOT / "results" / "libero_tinyplus_reconstruction_proxy.csv"
OBJECT_MEDIUM_COMPARISON_CSV = ROOT / "results" / "libero_object_medium_comparison.csv"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)


METHOD_DISPLAY = {
    "fixed_k_1": "Fixed K=1",
    "fixed_k_2": "Fixed K=2",
    "fixed_k_4": "Fixed K=4",
    "fixed_k_8": "Fixed K=8",
    "random_k": "Random K",
    "oracle_k": "Oracle K*",
    "adaptive_halting": "Adaptive",
}

METHOD_ORDER = [
    "fixed_k_1",
    "fixed_k_2",
    "fixed_k_4",
    "fixed_k_8",
    "random_k",
    "oracle_k",
    "adaptive_halting",
]

COLORS = {
    "fixed": "#9aa0a6",
    "random": "#b07aa1",
    "oracle": "#59a14f",
    "adaptive": "#4e79a7",
    "accent": "#e15759",
    "warning": "#f28e2b",
    "muted": "#bab0ab",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the candidate paths exist: {list(paths)}")


def parse_distribution(value: str | Dict[str, float]) -> Dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if pd.isna(value):
        return {}
    return {str(k): float(v) for k, v in ast.literal_eval(value).items()}


def save_figure(fig: plt.Figure, stem: str) -> None:
    ensure_dir(PAPER_DIR)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(PAPER_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def inventory_artifacts() -> Dict[str, object]:
    summary_path = find_first_existing(SUMMARY_CANDIDATES)
    tinyplus_summary_path = find_first_existing(TINYPLUS_SUMMARY_CANDIDATES)

    inventory = {
        "synthetic_summary": str(summary_path),
        "synthetic_metrics_json": str(summary_path.with_name("adaptive_halting_metrics.json"))
        if summary_path.with_name("adaptive_halting_metrics.json").exists()
        else None,
        "tinyplus_summary": str(tinyplus_summary_path),
        "tinyplus_diagnostics": str(TINYPLUS_DIAGNOSTICS) if TINYPLUS_DIAGNOSTICS.exists() else None,
        "tinyplus_oracle_diagnostics_csv": str(TINYPLUS_ORACLE_DIAGNOSTICS_CSV)
        if TINYPLUS_ORACLE_DIAGNOSTICS_CSV.exists()
        else None,
        "tinyplus_halting_comparison_csv": str(TINYPLUS_HALTING_COMPARISON_CSV)
        if TINYPLUS_HALTING_COMPARISON_CSV.exists()
        else None,
        "tinyplus_reconstruction_proxy_csv": str(TINYPLUS_RECON_PROXY_CSV)
        if TINYPLUS_RECON_PROXY_CSV.exists()
        else None,
        "object_medium_comparison_csv": str(OBJECT_MEDIUM_COMPARISON_CSV)
        if OBJECT_MEDIUM_COMPARISON_CSV.exists()
        else None,
    }
    return inventory


def load_synthetic_summary() -> Tuple[pd.DataFrame, Path]:
    summary_path = find_first_existing(SUMMARY_CANDIDATES)
    df = pd.read_csv(summary_path)
    df["selected_k_distribution_parsed"] = df["selected_k_distribution"].apply(parse_distribution)
    order_map = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    df = df.sort_values(by="method", key=lambda s: s.map(order_map).fillna(999)).reset_index(drop=True)
    return df, summary_path


def load_tinyplus_summary() -> Tuple[pd.DataFrame, Path]:
    summary_path = find_first_existing(TINYPLUS_SUMMARY_CANDIDATES)
    return pd.read_csv(summary_path), summary_path


def load_tinyplus_diagnostics() -> Dict[str, object]:
    if not TINYPLUS_DIAGNOSTICS.exists():
        raise FileNotFoundError(f"Missing tinyplus diagnostics: {TINYPLUS_DIAGNOSTICS}")
    with open(TINYPLUS_DIAGNOSTICS) as f:
        return json.load(f)


def add_box(ax, xy, w, h, text, fc="#ffffff", ec="#333333", fontsize=12, rounded=True):
    boxstyle = "round,pad=0.02,rounding_size=0.02" if rounded else "square,pad=0.02"
    patch = patches.FancyBboxPatch(
        xy, w, h, boxstyle=boxstyle, linewidth=1.5, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def add_arrow(ax, start, end, text=None):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=1.8, color="#333333"),
    )
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.02, text, ha="center", va="bottom")


def generate_method_diagram() -> List[str]:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_positions = [0.90, 0.78, 0.66, 0.54, 0.42, 0.30, 0.18]
    labels = [
        r"Action chunk $a_{1:H}$",
        "OAT encoder + FSQ tokenizer",
        r"Ordered tokens $z_1, z_2, \ldots, z_{K_{max}}$",
        r"Prefix errors $e_1, e_2, e_4, e_8$",
        r"Oracle depth $K^*$",
        r"Train target: [BOS, $z_1,\ldots,z_{K^*}$, EOS]",
        "Inference: generate until EOS,\nthen decode prefix with token_lens",
    ]
    colors = [
        "#f7f7f7",
        "#edf3ff",
        "#edf3ff",
        "#fef3e7",
        "#e9f5e9",
        "#edf3ff",
        "#f7f7f7",
    ]

    for y, label, color in zip(y_positions, labels, colors):
        add_box(ax, (0.18, y - 0.04), 0.64, 0.08, label, fc=color)

    for start_y, end_y in zip(y_positions[:-1], y_positions[1:]):
        add_arrow(ax, (0.50, start_y - 0.04), (0.50, end_y + 0.04))

    add_box(
        ax,
        (0.18, 0.05),
        0.64,
        0.08,
        "Continuous action reconstruction",
        fc="#f7f7f7",
    )
    add_arrow(ax, (0.50, 0.14), (0.50, 0.13))

    ax.text(
        0.5,
        0.985,
        "Entropy-Gated Adaptive Prefix Depth built on prefix-decodable OAT tokens",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.01,
        "Synthetic experiments supervise K* from prefix reconstruction error. Policy learning uses EOS-based adaptive stopping.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#444444",
    )

    save_figure(fig, "fig_method_eg_apd_oat")
    return []


def _draw_token_row(ax, y, active_k, total_k=8, label=""):
    ax.text(0.02, y, label, ha="left", va="center", fontsize=12)
    x0 = 0.22
    gap = 0.075
    for idx in range(total_k):
        fc = COLORS["adaptive"] if idx < active_k else "#ffffff"
        ec = COLORS["adaptive"] if idx < active_k else "#999999"
        rect = patches.FancyBboxPatch(
            (x0 + idx * gap, y - 0.04),
            0.055,
            0.08,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x0 + idx * gap + 0.0275, y, rf"$z_{idx+1}$", ha="center", va="center", fontsize=10, color="#222222" if idx >= active_k else "white")
    return


def generate_fixed_vs_adaptive() -> List[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax in axes:
        ax.set_xlim(0, 0.90)
        ax.set_ylim(0, 1)
        ax.axis("off")

    axes[0].set_title("A. Fixed-depth OAT", loc="left", fontweight="bold")
    _draw_token_row(axes[0], 0.70, 8, label="All samples")
    axes[0].text(0.22, 0.42, "Always decodes the full prefix budget\n(K = 8 for every action chunk).", fontsize=12)
    axes[0].annotate("", xy=(0.70, 0.60), xytext=(0.70, 0.50), arrowprops=dict(arrowstyle="->", lw=1.8))
    axes[0].text(0.70, 0.46, "decode", ha="center", va="top", fontsize=11)

    axes[1].set_title("B. EG-APD OAT", loc="left", fontweight="bold")
    _draw_token_row(axes[1], 0.80, 2, label="Sample 1")
    _draw_token_row(axes[1], 0.60, 4, label="Sample 2")
    _draw_token_row(axes[1], 0.40, 8, label="Sample 3")
    axes[1].text(0.22, 0.12, "Variable-length generation stops with EOS,\nthen decodes only the generated prefix.", fontsize=12)
    axes[1].text(0.78, 0.80, "EOS", fontsize=11, color=COLORS["accent"], fontweight="bold")
    axes[1].text(0.78, 0.60, "EOS", fontsize=11, color=COLORS["accent"], fontweight="bold")
    axes[1].text(0.78, 0.40, "EOS", fontsize=11, color=COLORS["accent"], fontweight="bold")

    save_figure(fig, "fig_fixed_vs_adaptive")
    return []


def _method_color(method: str) -> str:
    if method == "adaptive_halting":
        return COLORS["adaptive"]
    if method == "oracle_k":
        return COLORS["oracle"]
    if method == "random_k":
        return COLORS["random"]
    return COLORS["fixed"]


def generate_bar_plot(
    df: pd.DataFrame,
    value_col: str,
    err_col: str,
    ylabel: str,
    stem: str,
    title: str,
    extra_hline: float | None = None,
    highlight_methods: Iterable[str] = (),
) -> List[str]:
    plot_df = df[df["method"].isin(METHOD_ORDER)].copy()
    plot_df["label"] = plot_df["method"].map(METHOD_DISPLAY)
    plot_df["color"] = plot_df["method"].apply(_method_color)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = range(len(plot_df))
    bars = ax.bar(
        list(x),
        plot_df[value_col],
        yerr=plot_df[err_col],
        color=plot_df["color"],
        edgecolor="#333333",
        linewidth=1.0,
        capsize=4,
    )

    for bar, method in zip(bars, plot_df["method"]):
        if method in highlight_methods:
            bar.set_linewidth(2.2)
            bar.set_edgecolor("#111111")

    if extra_hline is not None:
        ax.axhline(extra_hline, linestyle="--", color=COLORS["accent"], linewidth=1.6, label="Kmax = 8")

    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["label"], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    if extra_hline is not None:
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, stem)
    return []


def generate_efficiency_quality_pareto(df: pd.DataFrame) -> List[str]:
    plot_df = df[df["method"].isin(METHOD_ORDER)].copy()
    plot_df["label"] = plot_df["method"].map(METHOD_DISPLAY)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    fixed_df = plot_df[plot_df["method"].str.startswith("fixed_k_")].copy()
    fixed_df = fixed_df.sort_values("avg_K_mean")
    ax.plot(
        fixed_df["token_ratio_mean"],
        fixed_df["recon_mse_mean"],
        color=COLORS["fixed"],
        linewidth=2.0,
        marker="o",
        label="Fixed-K frontier",
    )

    for _, row in plot_df.iterrows():
        color = _method_color(row["method"])
        size = 140 if row["method"] in {"adaptive_halting", "oracle_k"} else 90
        marker = "D" if row["method"] == "oracle_k" else ("s" if row["method"] == "adaptive_halting" else "o")
        ax.scatter(row["token_ratio_mean"], row["recon_mse_mean"], s=size, color=color, edgecolor="#222222", marker=marker, zorder=3)
        ax.text(
            row["token_ratio_mean"] + 0.012,
            row["recon_mse_mean"] + 0.00018,
            row["label"],
            fontsize=10.5,
        )

    ax.set_xlabel("Average token ratio")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Efficiency-quality trade-off on the synthetic benchmark")
    ax.grid(True, linestyle=":", alpha=0.4)
    legend_handles = [
        Line2D([0], [0], color=COLORS["fixed"], marker="o", lw=2, label="Fixed-K frontier"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["oracle"], markeredgecolor="#222222", label="Oracle K*", markersize=9),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["adaptive"], markeredgecolor="#222222", label="Adaptive", markersize=9),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, "fig_efficiency_quality_pareto")
    return []


def generate_adaptive_distribution(df: pd.DataFrame) -> List[str]:
    row = df[df["method"] == "adaptive_halting"].iloc[0]
    dist = row["selected_k_distribution_parsed"]
    keys = ["1", "2", "4", "8"]
    values = [dist.get(k, 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(keys, values, color=COLORS["adaptive"], edgecolor="#333333", linewidth=1.0)
    ax.set_xlabel("Selected prefix depth K")
    ax.set_ylabel("Probability")
    ax.set_title("Adaptive prefix-depth distribution")
    ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_figure(fig, "fig_adaptive_k_distribution")
    return []


def generate_libero_tinyplus_integration(tinyplus_df: pd.DataFrame, tinyplus_diag: Dict[str, object]) -> List[str]:
    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.9, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Panel A
    methods = ["fixed_k_8", "adaptive_halting"]
    label_map = {"fixed_k_8": "Fixed K=8", "adaptive_halting": "Adaptive"}
    colors = [COLORS["fixed"], COLORS["adaptive"]]
    token_lengths = [
        float(tinyplus_df.loc[tinyplus_df["method"] == method, "avg_token_length"].iloc[0])
        for method in methods
    ]
    ax1.bar([0, 1], token_lengths, color=colors, edgecolor="#333333")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([label_map[m] for m in methods])
    ax1.set_ylabel("Average token length")
    ax1.set_title("A. Token usage")
    ax1.axhline(8, linestyle="--", color=COLORS["accent"], linewidth=1.4)
    ax1.text(1.02, 8.02, "Kmax", color=COLORS["accent"], va="bottom", fontsize=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    # Panel B
    success_values = [
        float(tinyplus_df.loc[tinyplus_df["method"] == method, "success_rate"].iloc[0])
        for method in methods
    ]
    ax2.bar([0, 1], success_values, color=colors, edgecolor="#333333")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([label_map[m] for m in methods])
    ax2.set_ylabel("Success rate")
    ax2.set_title("B. Rollout success")
    ax2.set_ylim(0, max(0.1, max(success_values) + 0.05))
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.text(
        0.5,
        0.92,
        "Integration / smoke test only",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color=COLORS["accent"],
    )

    # Panel C
    selected_dist = tinyplus_diag["adaptive_eos_diagnostics"]["selected_k_distribution"]
    k_labels = ["1", "2", "4", "8"]
    k_values = [float(selected_dist.get(k, 0.0)) for k in k_labels]
    ax3.bar(k_labels, k_values, color=COLORS["adaptive"], edgecolor="#333333")
    ax3.set_xlabel("Selected K")
    ax3.set_ylabel("Probability")
    ax3.set_title("C. Adaptive rollout K distribution")
    ax3.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Tinyplus LIBERO integration validates adaptive halting behavior, not task-level performance")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig_libero_tinyplus_integration")
    return []


def generate_pipeline_status() -> List[str]:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.06, 0.60, 0.26, 0.24, "1. Synthetic benchmark", COLORS["oracle"], "Completed", "30 runs\nfixed / random / oracle / adaptive"),
        (0.37, 0.60, 0.26, 0.24, "2. LIBERO integration", COLORS["adaptive"], "Completed (smoke)", "tiny / tinyplus rollouts\nCPU-only validation"),
        (0.68, 0.60, 0.26, 0.24, "3. Full LIBERO benchmark", COLORS["muted"], "Future work", "requires stronger compute\nand benchmark-scale training"),
    ]

    for x, y, w, h, title, color, status, body in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=color if color != COLORS["muted"] else "#f2f2f2",
            edgecolor="#333333",
            linewidth=1.5,
            alpha=0.15 if color != COLORS["muted"] else 1.0,
        )
        ax.add_patch(rect)
        ax.text(x + 0.02, y + h - 0.05, title, ha="left", va="top", fontsize=13, fontweight="bold")
        ax.text(x + 0.02, y + h - 0.12, status, ha="left", va="top", fontsize=12, color="#111111")
        ax.text(x + 0.02, y + 0.06, body, ha="left", va="bottom", fontsize=11, color="#333333")

    add_arrow(ax, (0.32, 0.72), (0.37, 0.72))
    add_arrow(ax, (0.63, 0.72), (0.68, 0.72))
    ax.text(
        0.50,
        0.28,
        "Claim boundary: synthetic benchmark is the main quantitative validation.\nLIBERO results demonstrate pipeline integration and token-budget behavior only.",
        ha="center",
        va="center",
        fontsize=12,
        color="#333333",
    )

    save_figure(fig, "fig_evaluation_pipeline_status")
    return []


def write_figure_snippets() -> None:
    figures = [
        (
            "fig_method_eg_apd_oat",
            "Method overview of entropy-gated adaptive prefix depth built on ordered, prefix-decodable OAT action tokens. Synthetic experiments supervise the oracle stopping depth $K^*$ from prefix reconstruction error, and policy learning converts adaptive stopping into EOS prediction.",
        ),
        (
            "fig_fixed_vs_adaptive",
            "Fixed-depth OAT allocates the same token budget to every action chunk, whereas the adaptive variant terminates token generation with EOS after a variable number of ordered action tokens.",
        ),
        (
            "fig_efficiency_quality_pareto",
            "Efficiency-quality trade-off on the synthetic benchmark. Fixed-K baselines trace the reconstruction frontier; the adaptive method approaches oracle behavior while reducing token usage relative to full-length decoding.",
        ),
        (
            "fig_reconstruction_mse",
            "Synthetic benchmark reconstruction error across fixed, random, oracle, and adaptive prefix-selection baselines. Error bars show variability across repeated runs.",
        ),
        (
            "fig_avg_token_usage",
            "Average selected prefix depth on the synthetic benchmark. The adaptive method uses fewer than the full $K_{max}=8$ tokens on average.",
        ),
        (
            "fig_token_ratio",
            "Average token ratio relative to the full OAT prefix budget. Adaptive halting reduces average token usage substantially relative to fixed $K=8$ decoding.",
        ),
        (
            "fig_adaptive_k_distribution",
            "Distribution of adaptive prefix depths selected on the synthetic benchmark. The learned policy concentrates on a subset of valid ordered-prefix depths rather than using a single fixed budget.",
        ),
        (
            "fig_libero_tinyplus_integration",
            "Tinyplus LIBERO rollout results. Adaptive halting changes token usage and prefix-length behavior in the real simulator stack, but both methods remain at zero task success; this figure is integration validation rather than benchmark evidence.",
        ),
        (
            "fig_evaluation_pipeline_status",
            "Evaluation status summary. Synthetic experiments provide the main quantitative evidence, while LIBERO results currently validate integration and adaptive stopping behavior under limited CPU-only training.",
        ),
    ]
    lines = []
    for stem, caption in figures:
        lines.extend(
            [
                "\\begin{figure}[t]",
                "  \\centering",
                f"  \\includegraphics[width=\\linewidth]{{paper_figures/{stem}.pdf}}",
                f"  \\caption{{{caption}}}",
                f"  \\label{{fig:{stem}}}",
                "\\end{figure}",
                "",
            ]
        )
    (PAPER_DIR / "figure_snippets.tex").write_text("\n".join(lines))


def write_manifest(data_inventory: Dict[str, object]) -> None:
    lines = [
        "# Figure Manifest",
        "",
        "This manifest records the exact source data and intended claim boundary for each paper figure.",
        "",
    ]
    entries = [
        (
            "fig_method_eg_apd_oat",
            "Conceptual schematic based on the implemented OAT tokenizer, oracle prefix supervision, and EOS-based adaptive policy training/inference.",
            "Method overview of the implemented adaptive-prefix pipeline.",
            "Supports the paper's method description.",
            "Diagrammatic only; not a quantitative result.",
        ),
        (
            "fig_fixed_vs_adaptive",
            "Conceptual comparison using the implemented fixed-K and adaptive-EOS decoding modes.",
            "Illustrates fixed-depth vs variable-depth action decoding.",
            "Supports the conceptual motivation for adaptive prefix depth.",
            "Not derived from a specific numeric dataset.",
        ),
        (
            "fig_reconstruction_mse",
            str(data_inventory["synthetic_summary"]),
            "Synthetic benchmark reconstruction error with repeated-run uncertainty.",
            "Supports the claim that adaptive halting preserves much of the reconstruction quality of deeper fixed prefixes.",
            "Synthetic benchmark only; does not measure robotics success.",
        ),
        (
            "fig_avg_token_usage",
            str(data_inventory["synthetic_summary"]),
            "Synthetic benchmark average selected prefix depth.",
            "Supports the claim that adaptive halting reduces average token budget.",
            "Synthetic benchmark only.",
        ),
        (
            "fig_token_ratio",
            str(data_inventory["synthetic_summary"]),
            "Synthetic benchmark token ratio relative to Kmax.",
            "Supports the efficiency claim for adaptive halting.",
            "Synthetic benchmark only.",
        ),
        (
            "fig_efficiency_quality_pareto",
            str(data_inventory["synthetic_summary"]),
            "Synthetic efficiency-quality trade-off frontier.",
            "Supports the main quantitative paper claim.",
            "Synthetic benchmark is the primary evidence; not a simulator benchmark.",
        ),
        (
            "fig_adaptive_k_distribution",
            str(data_inventory["synthetic_summary"]),
            "Distribution of adaptive prefix depths on the synthetic benchmark.",
            "Supports the claim that the learned policy is not merely a single fixed-K policy.",
            "Distribution is aggregated from summary artifacts.",
        ),
        (
            "fig_libero_tinyplus_integration",
            str(data_inventory["tinyplus_summary"]),
            "Token-length, success-rate, and selected-K behavior for the tinyplus LIBERO integration run.",
            "Supports the claim that adaptive halting executes inside the real LIBERO simulator path.",
            "Success rate remains 0.0; this is not benchmark-level validation.",
        ),
        (
            "fig_evaluation_pipeline_status",
            "Repository artifact inventory and README-documented evaluation pipeline.",
            "Communicates the boundary between completed synthetic validation, completed integration validation, and future full-benchmark work.",
            "Supports scientifically honest framing of results.",
            "Status figure; not a performance result.",
        ),
    ]
    for figure, source, shows, claim, caveat in entries:
        lines.extend(
            [
                f"## {figure}",
                "",
                f"- Filename: `{figure}.svg/.pdf/.png`",
                f"- Source data: `{source}`",
                f"- What it shows: {shows}",
                f"- Claim supported: {claim}",
                f"- Caveat: {caveat}",
                "",
            ]
        )
    (PAPER_DIR / "FIGURE_MANIFEST.md").write_text("\n".join(lines))


def print_inventory(data_inventory: Dict[str, object]) -> None:
    print("Data inventory used for paper figures:")
    for key, value in data_inventory.items():
        print(f"  - {key}: {value}")


def main() -> None:
    ensure_dir(PAPER_DIR)
    inventory = inventory_artifacts()
    print_inventory(inventory)

    synthetic_df, synthetic_summary_path = load_synthetic_summary()
    tinyplus_df, _ = load_tinyplus_summary()
    tinyplus_diag = load_tinyplus_diagnostics()

    generate_method_diagram()
    generate_fixed_vs_adaptive()
    generate_bar_plot(
        synthetic_df,
        value_col="recon_mse_mean",
        err_col="recon_mse_std",
        ylabel="Reconstruction MSE",
        stem="fig_reconstruction_mse",
        title="Synthetic benchmark: reconstruction error",
        highlight_methods=["oracle_k", "adaptive_halting"],
    )
    generate_bar_plot(
        synthetic_df,
        value_col="avg_K_mean",
        err_col="avg_K_std",
        ylabel="Average prefix depth K",
        stem="fig_avg_token_usage",
        title="Synthetic benchmark: average token usage",
        extra_hline=8.0,
        highlight_methods=["oracle_k", "adaptive_halting"],
    )
    generate_bar_plot(
        synthetic_df,
        value_col="token_ratio_mean",
        err_col="token_ratio_std",
        ylabel="Token ratio",
        stem="fig_token_ratio",
        title="Synthetic benchmark: token ratio",
        highlight_methods=["oracle_k", "adaptive_halting"],
    )
    generate_efficiency_quality_pareto(synthetic_df)
    generate_adaptive_distribution(synthetic_df)
    generate_libero_tinyplus_integration(tinyplus_df, tinyplus_diag)
    generate_pipeline_status()

    write_figure_snippets()
    write_manifest(inventory)

    print("\nGenerated files:")
    for path in sorted(PAPER_DIR.iterdir()):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
