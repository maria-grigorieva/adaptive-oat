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
import pathlib
import click


@click.command()
@click.option("--fixed-log", default="output/eval/libero_spatial_tinyplus_fixedk/eval_log.json", show_default=True)
@click.option("--adaptive-log", default="output/eval/libero_spatial_tinyplus_adaptive/eval_log.json", show_default=True)
@click.option("--adaptive-diagnostics", default="output/eval/libero_spatial_tinyplus_adaptive/eos_diagnostics.json", show_default=True)
@click.option("--summary-csv", default="results/libero_tinyplus_summary.csv", show_default=True)
@click.option("--diagnostics-json", default="results/libero_tinyplus_diagnostics.json", show_default=True)
@click.option("--num-tasks", default=2, show_default=True, type=int)
@click.option("--episodes-per-task", default=2, show_default=True, type=int)
def main(fixed_log, adaptive_log, adaptive_diagnostics, summary_csv, diagnostics_json, num_tasks, episodes_per_task):
    fixed = json.load(open(fixed_log))
    adaptive = json.load(open(adaptive_log))
    adaptive_diag = json.load(open(adaptive_diagnostics)) if os.path.exists(adaptive_diagnostics) else {}

    fixed_tasks = int(fixed.get("libero_task_limit", 0) or num_tasks)
    fixed_eps = int(fixed.get("libero_episodes_per_task", 0) or episodes_per_task)
    adaptive_tasks = int(adaptive.get("libero_task_limit", 0) or num_tasks)
    adaptive_eps = int(adaptive.get("libero_episodes_per_task", 0) or episodes_per_task)

    rows = [
        {
            "method": "fixed_k_8",
            "success_rate": float(fixed.get("mean_success_rate_mean", 0.0)),
            "avg_token_length": float(fixed.get("mean_action_tokens_mean", 0.0)),
            "token_ratio": float(fixed.get("token_ratio_mean", 0.0)),
            "eos_prediction_rate": float(fixed.get("eos_prediction_rate_mean", 0.0)),
            "early_stop_rate": float(fixed.get("early_stop_rate_mean", 0.0)),
            "runtime_sec": float(fixed.get("runtime_sec_mean", 0.0)),
            "num_tasks": fixed_tasks,
            "episodes_per_task": fixed_eps,
            "checkpoint_path": fixed["checkpoint"],
            "notes": "Tinyplus CPU fixed-K rollout.",
        },
        {
            "method": "adaptive_halting",
            "success_rate": float(adaptive.get("mean_success_rate_mean", 0.0)),
            "avg_token_length": float(adaptive.get("mean_action_tokens_mean", 0.0)),
            "token_ratio": float(adaptive.get("token_ratio_mean", 0.0)),
            "eos_prediction_rate": float(adaptive.get("eos_prediction_rate_mean", 0.0)),
            "early_stop_rate": float(adaptive.get("early_stop_rate_mean", 0.0)),
            "runtime_sec": float(adaptive.get("runtime_sec_mean", 0.0)),
            "num_tasks": adaptive_tasks,
            "episodes_per_task": adaptive_eps,
            "checkpoint_path": adaptive["checkpoint"],
            "notes": adaptive_diag.get("notes", "Tinyplus CPU adaptive-halting rollout."),
        },
    ]

    summary_path = pathlib.Path(summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    merged_diagnostics = {
        "fixed_k_8": fixed,
        "adaptive_halting": adaptive,
        "adaptive_eos_diagnostics": adaptive_diag,
        "summary_rows": rows,
    }
    diagnostics_path = pathlib.Path(diagnostics_json)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(merged_diagnostics, diagnostics_path.open("w"), indent=2, sort_keys=True)

    print(summary_path)
    print(diagnostics_path)


if __name__ == "__main__":
    main()
