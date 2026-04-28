#!/usr/bin/env python
"""
Create lightweight SVG plots for adaptive halting experiments.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot adaptive halting experiment results.")
    parser.add_argument("--metrics-json", required=True)
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


def write_svg(path: Path, elements: Iterable[str], width: int = 880, height: int = 520) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(elements)
    path.write_text(f"{svg_header(width, height)}\n{body}\n</svg>\n")


def draw_axes(
    width: int,
    height: int,
    margin_left: int = 90,
    margin_right: int = 40,
    margin_top: int = 40,
    margin_bottom: int = 70,
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


def plot_mse_vs_k(payload: Dict, output_dir: Path) -> None:
    aggregate = payload["aggregate"]
    fixed_rows = []
    adaptive_row = aggregate.get("adaptive_halting")
    for method, row in aggregate.items():
        if method.startswith("fixed_k_"):
            fixed_rows.append((int(method.split("_")[-1]), float(row["recon_mse"])))
    fixed_rows.sort(key=lambda item: item[0])
    if not fixed_rows:
        return

    width, height = 880, 520
    elements, (x0, y0, x1, y1) = draw_axes(width, height)
    x_vals = [item[0] for item in fixed_rows]
    y_vals = [item[1] for item in fixed_rows]
    if adaptive_row and adaptive_row.get("recon_mse") is not None:
        x_vals.append(float(adaptive_row["avg_prefix_depth"]))
        y_vals.append(float(adaptive_row["recon_mse"]))

    xmin, xmax = min(x_vals), max(x_vals)
    ymin, ymax = min(y_vals), max(y_vals)
    if math.isclose(ymin, ymax):
        ymax = ymin + 1e-6

    def sx(value: float) -> float:
        if math.isclose(xmin, xmax):
            return (x0 + x1) / 2
        return x0 + (value - xmin) / (xmax - xmin) * (x1 - x0)

    def sy(value: float) -> float:
        return y0 - (value - ymin) / (ymax - ymin) * (y0 - y1)

    for tick in sorted(set(x_vals)):
        xt = sx(tick)
        elements.append(f'<line x1="{xt:.2f}" y1="{y0}" x2="{xt:.2f}" y2="{y0 + 8}" stroke="#444" />')
        elements.append(f'<text x="{xt:.2f}" y="{y0 + 28}" font-size="14" text-anchor="middle" fill="#222">{tick:.2f}</text>')

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = ymin + frac * (ymax - ymin)
        yt = sy(value)
        elements.append(f'<line x1="{x0}" y1="{yt:.2f}" x2="{x1}" y2="{yt:.2f}" stroke="#e6e6e6" />')
        elements.append(f'<text x="{x0 - 12}" y="{yt + 4:.2f}" font-size="14" text-anchor="end" fill="#222">{value:.5f}</text>')

    fixed_points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in fixed_rows)
    elements.append(f'<polyline points="{fixed_points}" fill="none" stroke="#1f77b4" stroke-width="3" />')
    for x, y in fixed_rows:
        elements.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="5" fill="#1f77b4" />')

    if adaptive_row and adaptive_row.get("recon_mse") is not None:
        ax = sx(float(adaptive_row["avg_prefix_depth"]))
        ay = sy(float(adaptive_row["recon_mse"]))
        elements.append(f'<circle cx="{ax:.2f}" cy="{ay:.2f}" r="7" fill="#d62728" />')
        elements.append(f'<text x="{ax + 12:.2f}" y="{ay - 8:.2f}" font-size="14" fill="#d62728">adaptive</text>')

    elements.extend([
        '<text x="440" y="28" font-size="22" text-anchor="middle" fill="#111">Reconstruction MSE vs Prefix Depth</text>',
        f'<text x="{(x0 + x1) / 2:.2f}" y="{height - 18}" font-size="16" text-anchor="middle" fill="#222">Average prefix depth K</text>',
        f'<text x="22" y="{(y0 + y1) / 2:.2f}" font-size="16" text-anchor="middle" fill="#222" transform="rotate(-90 22 {(y0 + y1) / 2:.2f})">Reconstruction MSE</text>',
    ])
    write_svg(output_dir / "mse_vs_k.svg", elements, width=width, height=height)


def plot_token_usage(payload: Dict, output_dir: Path) -> None:
    aggregate = payload["aggregate"]
    methods = sorted(aggregate.keys())
    values = [aggregate[method].get("token_ratio") for method in methods]
    if not methods:
        return

    width, height = 920, 520
    elements, (x0, y0, x1, y1) = draw_axes(width, height, margin_left=100, margin_bottom=100)
    ymax = max(float(v) for v in values if v is not None)
    ymax = max(ymax, 1.0)
    bar_width = (x1 - x0) / max(len(methods), 1) * 0.6

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = frac * ymax
        yt = y0 - frac * (y0 - y1)
        elements.append(f'<line x1="{x0}" y1="{yt:.2f}" x2="{x1}" y2="{yt:.2f}" stroke="#e6e6e6" />')
        elements.append(f'<text x="{x0 - 12}" y="{yt + 4:.2f}" font-size="14" text-anchor="end" fill="#222">{value:.2f}</text>')

    for idx, method in enumerate(methods):
        center = x0 + (idx + 0.5) * (x1 - x0) / len(methods)
        value = float(aggregate[method]["token_ratio"])
        top = y0 - value / ymax * (y0 - y1)
        color = "#d62728" if method == "adaptive_halting" else "#1f77b4"
        elements.append(
            f'<rect x="{center - bar_width / 2:.2f}" y="{top:.2f}" width="{bar_width:.2f}" '
            f'height="{y0 - top:.2f}" fill="{color}" opacity="0.85" />'
        )
        elements.append(
            f'<text x="{center:.2f}" y="{y0 + 18}" font-size="13" text-anchor="end" '
            f'transform="rotate(-35 {center:.2f} {y0 + 18})" fill="#222">{escape(method)}</text>'
        )
        elements.append(f'<text x="{center:.2f}" y="{top - 8:.2f}" font-size="13" text-anchor="middle" fill="#222">{value:.2f}</text>')

    elements.extend([
        '<text x="460" y="28" font-size="22" text-anchor="middle" fill="#111">Token Usage by Method</text>',
        f'<text x="28" y="{(y0 + y1) / 2:.2f}" font-size="16" text-anchor="middle" fill="#222" transform="rotate(-90 28 {(y0 + y1) / 2:.2f})">Token ratio (avg K / Kmax)</text>',
    ])
    write_svg(output_dir / "token_usage_vs_method.svg", elements, width=width, height=height)


def plot_adaptive_histogram(payload: Dict, output_dir: Path) -> None:
    adaptive = payload["aggregate"].get("adaptive_halting")
    if not adaptive:
        return

    selected = adaptive.get("selected_k_distribution", {})
    oracle = adaptive.get("oracle_k_distribution", {})
    keep_ks = sorted(int(k) for k in selected.keys())
    if not keep_ks:
        return

    width, height = 880, 520
    elements, (x0, y0, x1, y1) = draw_axes(width, height, margin_left=100, margin_bottom=90)
    ymax = max(
        max(float(v) for v in selected.values()),
        max(float(v) for v in oracle.values()) if oracle else 0.0,
        1e-6,
    )
    ymax = max(ymax, 0.25)
    group_width = (x1 - x0) / len(keep_ks)
    bar_width = group_width * 0.28

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = frac * ymax
        yt = y0 - frac * (y0 - y1)
        elements.append(f'<line x1="{x0}" y1="{yt:.2f}" x2="{x1}" y2="{yt:.2f}" stroke="#e6e6e6" />')
        elements.append(f'<text x="{x0 - 12}" y="{yt + 4:.2f}" font-size="14" text-anchor="end" fill="#222">{value:.2f}</text>')

    for idx, keep_k in enumerate(keep_ks):
        center = x0 + (idx + 0.5) * group_width
        pred_value = float(selected.get(str(keep_k), 0.0))
        pred_top = y0 - pred_value / ymax * (y0 - y1)
        elements.append(
            f'<rect x="{center - bar_width - 4:.2f}" y="{pred_top:.2f}" width="{bar_width:.2f}" '
            f'height="{y0 - pred_top:.2f}" fill="#d62728" opacity="0.82" />'
        )
        if oracle:
            oracle_value = float(oracle.get(str(keep_k), 0.0))
            oracle_top = y0 - oracle_value / ymax * (y0 - y1)
            elements.append(
                f'<rect x="{center + 4:.2f}" y="{oracle_top:.2f}" width="{bar_width:.2f}" '
                f'height="{y0 - oracle_top:.2f}" fill="#2ca02c" opacity="0.82" />'
            )
        elements.append(f'<text x="{center:.2f}" y="{y0 + 26}" font-size="14" text-anchor="middle" fill="#222">{keep_k}</text>')

    elements.extend([
        '<text x="440" y="28" font-size="22" text-anchor="middle" fill="#111">Adaptive Prefix Depth Distribution</text>',
        f'<text x="{(x0 + x1) / 2:.2f}" y="{height - 18}" font-size="16" text-anchor="middle" fill="#222">Prefix depth K</text>',
        f'<text x="28" y="{(y0 + y1) / 2:.2f}" font-size="16" text-anchor="middle" fill="#222" transform="rotate(-90 28 {(y0 + y1) / 2:.2f})">Fraction of samples</text>',
        f'<rect x="{x1 - 190}" y="{y1 + 8}" width="16" height="16" fill="#d62728" opacity="0.82" />',
        f'<text x="{x1 - 166}" y="{y1 + 21}" font-size="14" fill="#222">predicted K</text>',
        f'<rect x="{x1 - 92}" y="{y1 + 8}" width="16" height="16" fill="#2ca02c" opacity="0.82" />',
        f'<text x="{x1 - 68}" y="{y1 + 21}" font-size="14" fill="#222">oracle K</text>',
    ])
    write_svg(output_dir / "adaptive_k_distribution.svg", elements, width=width, height=height)


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.metrics_json).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_mse_vs_k(payload, output_dir)
    plot_token_usage(payload, output_dir)
    plot_adaptive_histogram(payload, output_dir)


if __name__ == "__main__":
    main()
