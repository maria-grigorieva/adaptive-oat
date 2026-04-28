#!/usr/bin/env python
"""
Create lightweight SVG plots for adaptive halting experiment summaries.

This script reads the summary CSV and renders simple mean±std figures without
requiring matplotlib or seaborn.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot adaptive halting experiment results.")
    parser.add_argument("--input", required=True, help="Path to adaptive_halting_summary.csv")
    parser.add_argument("--output-dir", default="results/plots")
    return parser.parse_args()


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(path: Path, elements: Iterable[str], width: int = 920, height: int = 560) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{svg_header(width, height)}\n" + "\n".join(elements) + "\n</svg>\n")


def draw_axes(
    width: int,
    height: int,
    margin_left: int = 110,
    margin_right: int = 40,
    margin_top: int = 50,
    margin_bottom: int = 90,
) -> Tuple[List[str], Tuple[int, int, int, int]]:
    x0 = margin_left
    y0 = height - margin_bottom
    x1 = width - margin_right
    y1 = margin_top
    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#222" stroke-width="2" />',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222" stroke-width="2" />',
    ]
    return elements, (x0, y0, x1, y1)


def load_summary(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for key, value in list(parsed.items()):
                if value in {"", "n/a", None}:
                    parsed[key] = None
                    continue
                if key == "selected_k_distribution":
                    parsed[key] = json.loads(value)
                    continue
                try:
                    parsed[key] = float(value)
                except Exception:
                    pass
            rows.append(parsed)
    return rows


def draw_y_grid(
    elements: List[str],
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    ymax: float,
) -> None:
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = frac * ymax
        yt = y0 - frac * (y0 - y1)
        elements.append(f'<line x1="{x0}" y1="{yt:.2f}" x2="{x1}" y2="{yt:.2f}" stroke="#e6e6e6" />')
        elements.append(f'<text x="{x0 - 12}" y="{yt + 4:.2f}" font-size="14" text-anchor="end" fill="#222">{value:.4f}</text>')


def plot_metric_bars(
    rows: List[Dict[str, object]],
    mean_key: str,
    std_key: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    rows = [row for row in rows if row.get(mean_key) is not None]
    if not rows:
        return

    width, height = 940, 560
    elements, (x0, y0, x1, y1) = draw_axes(width, height)
    ymax = max(float(row[mean_key]) + float(row.get(std_key) or 0.0) for row in rows)
    ymax = max(ymax, 1e-6)
    bar_width = (x1 - x0) / max(len(rows), 1) * 0.55
    draw_y_grid(elements, x0, x1, y0, y1, ymax)

    for idx, row in enumerate(rows):
        center = x0 + (idx + 0.5) * (x1 - x0) / len(rows)
        mean = float(row[mean_key])
        std = float(row.get(std_key) or 0.0)
        top = y0 - mean / ymax * (y0 - y1)
        err_top = y0 - min(mean + std, ymax) / ymax * (y0 - y1)
        err_bottom = y0 - max(mean - std, 0.0) / ymax * (y0 - y1)
        color = "#d62728" if row["method"] == "adaptive_halting" else "#1f77b4"
        elements.append(
            f'<rect x="{center - bar_width / 2:.2f}" y="{top:.2f}" width="{bar_width:.2f}" '
            f'height="{y0 - top:.2f}" fill="{color}" opacity="0.84" />'
        )
        elements.append(f'<line x1="{center:.2f}" y1="{err_top:.2f}" x2="{center:.2f}" y2="{err_bottom:.2f}" stroke="#111" stroke-width="2" />')
        elements.append(f'<line x1="{center - 8:.2f}" y1="{err_top:.2f}" x2="{center + 8:.2f}" y2="{err_top:.2f}" stroke="#111" stroke-width="2" />')
        elements.append(f'<line x1="{center - 8:.2f}" y1="{err_bottom:.2f}" x2="{center + 8:.2f}" y2="{err_bottom:.2f}" stroke="#111" stroke-width="2" />')
        elements.append(
            f'<text x="{center:.2f}" y="{y0 + 22}" font-size="13" text-anchor="end" '
            f'transform="rotate(-35 {center:.2f} {y0 + 22})" fill="#222">{escape(row["method"])}</text>'
        )
        elements.append(f'<text x="{center:.2f}" y="{top - 8:.2f}" font-size="13" text-anchor="middle" fill="#222">{mean:.4f}</text>')

    elements.extend([
        f'<text x="{width / 2:.2f}" y="30" font-size="22" text-anchor="middle" fill="#111">{escape(title)}</text>',
        f'<text x="28" y="{(y0 + y1) / 2:.2f}" font-size="16" text-anchor="middle" fill="#222" transform="rotate(-90 28 {(y0 + y1) / 2:.2f})">{escape(y_label)}</text>',
    ])
    write_svg(output_path, elements, width=width, height=height)


def plot_adaptive_distribution(rows: List[Dict[str, object]], output_path: Path) -> None:
    adaptive_row = next((row for row in rows if row["method"] == "adaptive_halting"), None)
    if adaptive_row is None:
        return

    selected = adaptive_row.get("selected_k_distribution") or {}
    keep_ks = sorted(int(k) for k in selected.keys())
    if not keep_ks:
        return

    width, height = 900, 540
    elements, (x0, y0, x1, y1) = draw_axes(width, height, margin_bottom=85)
    ymax = max(max(float(v) for v in selected.values()), 1e-6)
    ymax = max(ymax, 0.25)
    draw_y_grid(elements, x0, x1, y0, y1, ymax)
    group_width = (x1 - x0) / len(keep_ks)
    bar_width = group_width * 0.42

    for idx, keep_k in enumerate(keep_ks):
        center = x0 + (idx + 0.5) * group_width
        value = float(selected.get(str(keep_k), 0.0))
        top = y0 - value / ymax * (y0 - y1)
        elements.append(
            f'<rect x="{center - bar_width / 2:.2f}" y="{top:.2f}" width="{bar_width:.2f}" '
            f'height="{y0 - top:.2f}" fill="#d62728" opacity="0.82" />'
        )
        elements.append(f'<text x="{center:.2f}" y="{top - 8:.2f}" font-size="13" text-anchor="middle" fill="#222">{value:.3f}</text>')
        elements.append(f'<text x="{center:.2f}" y="{y0 + 24}" font-size="14" text-anchor="middle" fill="#222">{keep_k}</text>')

    elements.extend([
        f'<text x="{width / 2:.2f}" y="30" font-size="22" text-anchor="middle" fill="#111">Adaptive K Distribution</text>',
        f'<text x="{(x0 + x1) / 2:.2f}" y="{height - 18}" font-size="16" text-anchor="middle" fill="#222">Prefix depth K</text>',
        f'<text x="28" y="{(y0 + y1) / 2:.2f}" font-size="16" text-anchor="middle" fill="#222" transform="rotate(-90 28 {(y0 + y1) / 2:.2f})">Fraction of samples</text>',
    ])
    write_svg(output_path, elements, width=width, height=height)


def main() -> None:
    args = parse_args()
    rows = load_summary(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_bars(
        rows,
        mean_key="recon_mse_mean",
        std_key="recon_mse_std",
        title="Reconstruction MSE (mean ± std)",
        y_label="Reconstruction MSE",
        output_path=output_dir / "reconstruction_mse_errorbars.svg",
    )
    plot_metric_bars(
        rows,
        mean_key="avg_K_mean",
        std_key="avg_K_std",
        title="Average Token Usage (mean ± std)",
        y_label="Average prefix depth K",
        output_path=output_dir / "avg_token_usage_errorbars.svg",
    )
    plot_metric_bars(
        rows,
        mean_key="token_ratio_mean",
        std_key="token_ratio_std",
        title="Token Ratio (mean ± std)",
        y_label="Token ratio avg(K) / Kmax",
        output_path=output_dir / "token_ratio_errorbars.svg",
    )
    plot_adaptive_distribution(rows, output_dir / "adaptive_k_distribution.svg")


if __name__ == "__main__":
    main()
