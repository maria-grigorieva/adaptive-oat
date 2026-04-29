"""
Usage:
python experiments/eval_policy_sim.py --checkpoint path/to/ckpt -o path/to/output_dir
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import shutil
import click
import hydra
import torch
import wandb
import json
import numpy as np
import time
from oat.env_runner.base_runner import BaseRunner
from oat.policy.base_policy import BasePolicy
from typing import List, Optional

@click.command()
@click.option('-c', '--checkpoint', required=True, help="either a .ckpt file or a directory containing .ckpt files")
@click.option('-o', '--output_dir', required=True, help="output directory for eval info dump")
@click.option('-n', '--num_exp', default=1, help="num experiments to run")
@click.option('-d', '--device', default='cuda:0', help="device to run on")
@click.option('--temperature', default=None, type=float, help="temperature for policy inference")
@click.option('--topk', default=None, type=int, help="topk for policy inference")
@click.option('--use_k_tokens', default=None, type=int, help="number of tokens to use for policy inference")
@click.option('--adaptive-halting/--no-adaptive-halting', default=False, help="enable EOS-based adaptive halting")
@click.option('--adaptive-min-k', default=1, type=int, help="ignore adaptive EOS before this prefix depth")
@click.option(
    '--entropy-threshold',
    default=None,
    type=float,
    help=(
        "override adaptive halting entropy threshold at inference time. "
        "Default: None (uses model config). For models trained with "
        "train_entropy_threshold=6.5, sweep between 0.5 and 2.0 for effective early stopping."
    ),
)
@click.option('--force', is_flag=True, help="overwrite output_dir without prompting")
@click.option('--libero-task-name', default=None, type=str, help="override LiberoRunner task_name")
@click.option('--libero-task-limit', default=None, type=int, help="limit the number of LIBERO subtasks")
@click.option('--libero-episodes-per-task', default=None, type=int, help="episodes to run per selected LIBERO task")
@click.option('--libero-n-test-vis', default=None, type=int, help="override the number of rendered LIBERO evals")
@click.option('--libero-n-parallel-envs', default=None, type=int, help="override the number of parallel LIBERO envs")
@click.option('--libero-max-episode-steps', default=None, type=int, help="override LiberoRunner max_episode_steps")
def eval_policy_sim(
    checkpoint: str,
    output_dir: str,
    num_exp: int = 1,
    device: str = 'cuda:0',
    # policy inference args
    temperature: Optional[float] = None,
    topk: Optional[int] = None,
    use_k_tokens: Optional[int] = None,
    adaptive_halting: bool = False,
    adaptive_min_k: int = 1,
    entropy_threshold: Optional[float] = None,
    force: bool = False,
    libero_task_name: Optional[str] = None,
    libero_task_limit: Optional[int] = None,
    libero_episodes_per_task: Optional[int] = None,
    libero_n_test_vis: Optional[int] = None,
    libero_n_parallel_envs: Optional[int] = None,
    libero_max_episode_steps: Optional[int] = None,
):
    if os.path.exists(output_dir):
        if force:
            shutil.rmtree(output_dir)
        else:
            click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
            shutil.rmtree(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # grab all checkpoints
    ckpts: List[str]    # file paths to checkpoints to evaluate
    if os.path.isdir(checkpoint):
        ckpts = [
            os.path.join(checkpoint, f) 
            for f in os.listdir(checkpoint) 
            if f.endswith('.ckpt') and f != 'latest.ckpt'
        ]
    else:
        ckpts = [checkpoint,]

    base_output_dir = output_dir
    for ckpt in ckpts:
        # format output dir
        if len(ckpts) > 1:
            ckpt_name = os.path.basename(ckpt).replace('.ckpt', '')
            output_dir = os.path.join(base_output_dir, ckpt_name)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = base_output_dir
        
        # load checkpoint
        policy, cfg = BasePolicy.from_checkpoint(ckpt, return_configuration=True)
        env_runner_cfg = cfg.task.policy.env_runner
        if libero_task_name is not None:
            env_runner_cfg.task_name = libero_task_name
        if libero_task_limit is not None:
            env_runner_cfg.task_limit = libero_task_limit
        if libero_episodes_per_task is not None:
            env_runner_cfg.episodes_per_task = libero_episodes_per_task
        if libero_n_test_vis is not None:
            env_runner_cfg.n_test_vis = libero_n_test_vis
        if libero_n_parallel_envs is not None:
            env_runner_cfg.n_parallel_envs = libero_n_parallel_envs
        if libero_max_episode_steps is not None:
            env_runner_cfg.max_episode_steps = libero_max_episode_steps
        
        device = torch.device(device)
        policy.to(device)
        policy.eval()
        
        # run eval
        print(f"Running evaluation on {ckpt}")
        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.policy.env_runner,
            output_dir=output_dir,
        )
        
        kwargs = {}
        if temperature is not None:
            kwargs['temperature'] = temperature
        if topk is not None:
            kwargs['topk'] = topk
        if use_k_tokens is not None:
            kwargs['use_k_tokens'] = use_k_tokens
        if adaptive_halting:
            kwargs['adaptive_halting'] = True
            kwargs['adaptive_min_k'] = adaptive_min_k
            if entropy_threshold is not None:
                kwargs['entropy_threshold'] = entropy_threshold
        run_start_time = time.perf_counter()
        runner_log = env_runner.run(
            policy,
            **kwargs
        )
        runner_log['runtime_sec'] = time.perf_counter() - run_start_time
        
        # Store all runs for computing statistics
        all_runs = []
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                runner_log[key] = [value]
        all_runs.append({k: v for k, v in runner_log.items() if not isinstance(v, list)})
        print(f"Exp 1: success rate = {runner_log['mean_success_rate']}")
        
        for i in range(num_exp - 1):
            run_start_time = time.perf_counter()
            this_log = env_runner.run(policy, **kwargs)
            this_log['runtime_sec'] = time.perf_counter() - run_start_time
            print(f"Exp {i + 2}: success rate = {this_log['mean_success_rate']}")
            all_runs.append({k: v for k, v in this_log.items() if not isinstance(v, list)})
            # merge logs
            for key, value in this_log.items():
                assert key in runner_log
                if isinstance(value, wandb.sdk.data_types.video.Video):
                    runner_log[key].append(value)
                else:
                    runner_log[key] += value
        
        # Compute mean and std for all numeric metrics
        numeric_keys = [k for k in all_runs[0].keys()]
        mean_log = {}
        std_log = {}
        
        for key in numeric_keys:
            values = [run[key] for run in all_runs]
            mean_log[key] = np.mean(values)
            if num_exp > 1:
                std_log[key] = np.std(values, ddof=1)  # sample std
        
        env_runner.close()
        
        # dump log to json
        json_log = dict()
        json_log['checkpoint'] = ckpt
        json_log['num_exp'] = num_exp
        json_log['libero_task_limit'] = int(getattr(env_runner_cfg, 'task_limit', 0) or 0)
        json_log['libero_episodes_per_task'] = int(getattr(env_runner_cfg, 'episodes_per_task', 0) or 0)
        json_log['libero_n_test'] = int(getattr(env_runner_cfg, 'n_test', 0) or 0)
        json_log['adaptive_min_k'] = int(adaptive_min_k)
        if entropy_threshold is not None:
            json_log['entropy_threshold_override'] = float(entropy_threshold)
        
        # Add mean values
        for key, value in mean_log.items():
            json_log[f'{key}_mean'] = float(value)
        
        # Add standard deviation & error values if multiple experiments
        if num_exp > 1:
            for key, value in std_log.items():
                json_log[f'{key}_std'] = float(value)
                json_log[f'{key}_stderr'] = float(value / np.sqrt(num_exp))
        
        # Add video paths
        for key, value in runner_log.items():
            if isinstance(value, list):
                for i, video in enumerate(value):
                    assert isinstance(video, wandb.sdk.data_types.video.Video)
                    json_log[f'{key}_{i}'] = video._path
        
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

        if adaptive_halting:
            keep_k_distribution = {
                str(keep_k): float(json_log.get(f'pred_keep_k_{keep_k}_mean', 0.0))
                for keep_k in range(policy.max_seq_len + 1)
            }
            num_sequences = int(json_log.get('num_action_sequences_mean', 0.0))
            eos_diagnostics = {
                'checkpoint': ckpt,
                'adaptive_halting_requested': True,
                'policy_use_adaptive_halting': bool(policy.use_adaptive_halting),
                'eos_token_id': policy.eos_id,
                'max_action_tokens': policy.max_seq_len,
                'adaptive_min_k': int(adaptive_min_k),
                'entropy_threshold_override': (
                    None if entropy_threshold is None else float(entropy_threshold)
                ),
                'mean_action_tokens': float(json_log.get('mean_action_tokens_mean', 0.0)),
                'token_ratio': float(json_log.get('token_ratio_mean', 0.0)),
                'eos_prediction_rate': float(json_log.get('eos_prediction_rate_mean', 0.0)),
                'early_stop_rate': float(json_log.get('early_stop_rate_mean', 0.0)),
                'num_action_sequences': num_sequences,
                'num_sequences_reaching_max_k': int(json_log.get('num_sequences_reaching_max_k_mean', 0.0)),
                'num_sequences_stopping_early': int(json_log.get('num_sequences_stopping_early_mean', 0.0)),
                'selected_k_distribution': keep_k_distribution,
                'mean_eos_probability': None,
                'notes': (
                    'EOS probability diagnostics are unavailable in the current autoregressive '
                    'generate() API; length and EOS-rate diagnostics are logged instead.'
                ),
            }
            diag_path = os.path.join(output_dir, 'eos_diagnostics.json')
            json.dump(eos_diagnostics, open(diag_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    eval_policy_sim()
