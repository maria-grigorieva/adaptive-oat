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
import subprocess
import sys
from typing import List

import click


def _format_threshold_tag(threshold: float) -> str:
    return str(threshold).replace(".", "_")


@click.command()
@click.option("-c", "--checkpoint", required=True, help="policy checkpoint to evaluate")
@click.option(
    "-o",
    "--output-dir",
    required=True,
    help="directory where per-threshold eval outputs and the summary CSV will be written",
)
@click.option(
    "--entropy-threshold",
    "entropy_thresholds",
    multiple=True,
    type=float,
    default=[0.5, 1.0, 2.0],
    help="inference entropy threshold to evaluate; can be repeated",
)
@click.option("-d", "--device", default="cpu", help="device passed through to eval_policy_sim.py")
@click.option("-n", "--num-exp", default=1, type=int, help="number of repeated eval runs per threshold")
@click.option("--temperature", default=None, type=float, help="optional temperature override")
@click.option("--topk", default=None, type=int, help="optional top-k override")
@click.option("--use-k-tokens", default=None, type=int, help="optional fixed token budget override")
@click.option("--adaptive-min-k", default=1, type=int, help="minimum adaptive prefix depth")
@click.option("--libero-task-name", default=None, type=str, help="override LiberoRunner task_name")
@click.option("--libero-task-limit", default=None, type=int, help="limit the number of LIBERO subtasks")
@click.option("--libero-episodes-per-task", default=None, type=int, help="episodes to run per selected LIBERO task")
@click.option("--libero-n-test-vis", default=None, type=int, help="override the number of rendered LIBERO evals")
@click.option("--libero-n-parallel-envs", default=None, type=int, help="override the number of parallel LIBERO envs")
@click.option("--libero-max-episode-steps", default=None, type=int, help="override LiberoRunner max_episode_steps")
@click.option("--force", is_flag=True, help="overwrite per-threshold eval directories if they exist")
def main(
    checkpoint: str,
    output_dir: str,
    entropy_thresholds: List[float],
    device: str,
    num_exp: int,
    temperature: float,
    topk: int,
    use_k_tokens: int,
    adaptive_min_k: int,
    libero_task_name: str,
    libero_task_limit: int,
    libero_episodes_per_task: int,
    libero_n_test_vis: int,
    libero_n_parallel_envs: int,
    libero_max_episode_steps: int,
    force: bool,
):
    root_dir = pathlib.Path(__file__).resolve().parent.parent
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = list(entropy_thresholds)
    if len(thresholds) == 0:
        thresholds = [0.5, 1.0, 2.0]

    eval_script = root_dir / "scripts" / "eval_policy_sim.py"
    summary_rows = []

    for threshold in thresholds:
        threshold_tag = _format_threshold_tag(threshold)
        threshold_output_dir = output_dir / f"entropy_{threshold_tag}"
        cmd = [
            sys.executable,
            str(eval_script),
            "--checkpoint",
            checkpoint,
            "--output_dir",
            str(threshold_output_dir),
            "--device",
            device,
            "--num_exp",
            str(num_exp),
            "--adaptive-halting",
            "--adaptive-min-k",
            str(adaptive_min_k),
            "--entropy-threshold",
            str(threshold),
        ]

        if temperature is not None:
            cmd.extend(["--temperature", str(temperature)])
        if topk is not None:
            cmd.extend(["--topk", str(topk)])
        if use_k_tokens is not None:
            cmd.extend(["--use_k_tokens", str(use_k_tokens)])
        if libero_task_name is not None:
            cmd.extend(["--libero-task-name", libero_task_name])
        if libero_task_limit is not None:
            cmd.extend(["--libero-task-limit", str(libero_task_limit)])
        if libero_episodes_per_task is not None:
            cmd.extend(["--libero-episodes-per-task", str(libero_episodes_per_task)])
        if libero_n_test_vis is not None:
            cmd.extend(["--libero-n-test-vis", str(libero_n_test_vis)])
        if libero_n_parallel_envs is not None:
            cmd.extend(["--libero-n-parallel-envs", str(libero_n_parallel_envs)])
        if libero_max_episode_steps is not None:
            cmd.extend(["--libero-max-episode-steps", str(libero_max_episode_steps)])
        if force:
            cmd.append("--force")

        print(f"[sweep] running entropy_threshold={threshold}")
        subprocess.run(cmd, cwd=root_dir, check=True)

        eval_log_path = threshold_output_dir / "eval_log.json"
        eos_diag_path = threshold_output_dir / "eos_diagnostics.json"
        eval_log = json.loads(eval_log_path.read_text())
        eos_diag = json.loads(eos_diag_path.read_text()) if eos_diag_path.exists() else {}

        summary_rows.append({
            "entropy_threshold": threshold,
            "success_rate": float(eval_log.get("mean_success_rate_mean", 0.0)),
            "avg_token_length": float(eval_log.get("mean_action_tokens_mean", 0.0)),
            "eos_rate": float(eval_log.get("eos_prediction_rate_mean", 0.0)),
            "early_stop_rate": float(eval_log.get("early_stop_rate_mean", 0.0)),
            "token_ratio": float(eval_log.get("token_ratio_mean", 0.0)),
            "runtime_sec": float(eval_log.get("runtime_sec_mean", 0.0)),
            "eval_dir": str(threshold_output_dir),
            "checkpoint": checkpoint,
            "selected_k_distribution": json.dumps(
                eos_diag.get("selected_k_distribution", {}),
                sort_keys=True,
            ),
        })

    summary_csv_path = output_dir / "inference_entropy_sweep_summary.csv"
    with summary_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "entropy_threshold",
                "success_rate",
                "avg_token_length",
                "eos_rate",
                "early_stop_rate",
                "token_ratio",
                "runtime_sec",
                "eval_dir",
                "checkpoint",
                "selected_k_distribution",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json_path = output_dir / "inference_entropy_sweep_summary.json"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True))

    print(f"[done] wrote {summary_csv_path}")


if __name__ == "__main__":
    main()
