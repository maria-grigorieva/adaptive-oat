# Learned Halting for OAT

This document describes the research prototype for learned halting on top of Adaptive Ordered Action Tokenization (OAT).

## What Is Implemented

- tokenizer-side oracle prefix supervision:
  `compute_prefix_reconstruction_errors`
  `derive_oracle_keep_k`
  `compute_oracle_keep_k_from_samples`
- policy-side EOS prediction for adaptive stopping
- fixed-K and adaptive inference support
- a synthetic, CPU-compatible experiment pipeline
- optional integration with the existing LIBERO evaluation script

## Reproducible Evaluation

Synthetic evaluation is the main reproducible result for this prototype. It is CPU-compatible and does not require LIBERO, MuJoCo, robosuite, or a trained checkpoint.

### Synthetic evaluation, statistically robust

```bash
python experiments/run_adaptive_halting_eval.py \
  --mode all \
  --num-runs 10 \
  --output-dir results
```

This writes:

- `results/adaptive_halting_summary.csv`
- `results/adaptive_halting_runs.csv`
- `results/adaptive_halting_metrics.json`
- `results/plots/`
- `research_logs/adaptive_halting_eval.log`

### Plot results

```bash
python experiments/plot_results.py \
  --input results/adaptive_halting_summary.csv \
  --output-dir results/plots
```

## Optional LIBERO Smoke Evaluation

This is not the main quantitative result for the learned adaptive halting prototype. It is only an integration smoke test for the optional robotics path.

Full LIBERO evaluation requires valid checkpoints, a working MuJoCo / robosuite / LIBERO stack, and stronger compute than a typical CPU-only workstation. On CPU-only machines, this smoke path may be skipped gracefully.

### Check the optional environment

```bash
python experiments/check_libero_env.py \
  --output-dir results/libero_smoke
```

This writes:

- `results/libero_smoke/libero_env_check.json`

### Run the optional smoke evaluation

```bash
python experiments/run_adaptive_halting_eval.py \
  --mode all \
  --libero-smoke-eval \
  --libero-num-tasks 1 \
  --libero-num-episodes 1 \
  --output-dir results/libero_smoke
```

This smoke path is optional and may be skipped automatically on CPU-only machines or when LIBERO / MuJoCo / robosuite dependencies are unavailable.

The smoke command writes:

- `results/libero_smoke/libero_smoke_status.json`
- `results/libero_smoke/libero_smoke_summary.csv`

Interpretation:

- `passed`: the requested smoke methods completed and summary/status artifacts were written.
- `skipped`: the optional robotics stack or checkpoint was unavailable, so the script exited cleanly without claiming robotics performance.
- `failed`: the dependencies were present and the script attempted the smoke run, but one or more methods did not complete successfully.

## Tiny CPU-Only LIBERO Smoke Pipeline

This path is meant for smoke and integration validation on a small CPU VM. It is not a benchmark setting and it does not try to reproduce the full OAT LIBERO numbers.

The tiny setup deliberately:

- downloads only `libero_spatial`
- keeps `1` demo per task via `--num_sample_demo 1`
- merges those per-task zarr files into `data/libero/libero_spatial_tiny.zarr`
- uses CPU-friendly batch sizes and only a couple of train, val, and reconstruction steps
- disables rollout during policy training with `lazy_eval: true`

### 1. Prepare a tiny LIBERO dataset

The helper script is defensive: it refuses to download anything unless it can find the LIBERO dataset downloader either in `third_party/LIBERO` or from an installed `libero` package.

```bash
bash scripts/prepare_libero_tiny.sh
```

What it does:

- creates `data/libero/hdf5_datasets`
- downloads only `libero_spatial`
- converts with `python scripts/convert_libero_dataset.py --num_sample_demo 1`
- merges the resulting per-task zarr files into `data/libero/libero_spatial_tiny.zarr`
- prints the final file sizes

If you want to run the steps manually instead:

```bash
python third_party/LIBERO/benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
python scripts/convert_libero_dataset.py --root_dir data/libero --hdf5_dir_name hdf5_datasets --num_sample_demo 1
```

The helper script is preferred because it keeps the merged tiny dataset path deterministic for the smoke configs.

### 2. Validate configs before training

If you only want to confirm that Hydra can compose the smoke configs on a CPU machine, run:

```bash
python scripts/run_workspace.py --config-name=train_oattok_tiny_cpu --cfg job
python scripts/run_workspace.py \
  --config-name=train_oatpolicy_tiny_cpu \
  policy.action_tokenizer.checkpoint=/tmp/oattok-smoke.ckpt \
  --cfg job
```

These commands do not require a real checkpoint and are useful when the VM is too small for a full smoke train.

### 3. Train the tokenizer smoke run

After `data/libero/libero_spatial_tiny.zarr` exists:

```bash
python scripts/run_workspace.py --config-name=train_oattok_tiny_cpu
```

This tiny config keeps training intentionally small:

- CPU-safe dataloaders with `num_workers=0`
- `batch_size=2`
- `num_epochs=2`
- `max_train_steps=2`
- `max_val_steps=1`
- `max_reconst_steps=1`
- `wandb` offline mode

### 4. Train the policy smoke run

Point the policy config at the tokenizer checkpoint you just produced:

```bash
python scripts/run_workspace.py \
  --config-name=train_oatpolicy_tiny_cpu \
  policy.action_tokenizer.checkpoint=/absolute/path/to/oattok.ckpt
```

This config is also CPU-oriented:

- rollout disabled during training through `task.policy.lazy_eval=true`
- `batch_size=2`
- `num_workers=0`
- `num_epochs=2`
- `max_train_steps=2`
- adaptive halting can still be enabled with:

```bash
python scripts/run_workspace.py \
  --config-name=train_oatpolicy_tiny_cpu \
  policy.action_tokenizer.checkpoint=/absolute/path/to/oattok.ckpt \
  policy.use_adaptive_halting=true
```

### 5. Evaluate if a checkpoint exists

If LIBERO, MuJoCo, and robosuite are installed and you already have a policy checkpoint, this is the smallest supported rollout path:

Fixed-K smoke eval:

```bash
python scripts/eval_policy_sim.py \
  --checkpoint /absolute/path/to/oatpolicy.ckpt \
  --output_dir output/eval/libero_spatial_tiny_fixedk \
  --device cpu \
  --num_exp 1 \
  --use_k_tokens 8 \
  --libero-task-name libero_spatial_tiny \
  --libero-task-limit 1 \
  --libero-episodes-per-task 1 \
  --libero-n-test-vis 0 \
  --libero-n-parallel-envs 1 \
  --libero-max-episode-steps 300
```

Adaptive-halting smoke eval:

```bash
python scripts/eval_policy_sim.py \
  --checkpoint /absolute/path/to/oatpolicy.ckpt \
  --output_dir output/eval/libero_spatial_tiny_adaptive \
  --device cpu \
  --num_exp 1 \
  --adaptive-halting \
  --libero-task-name libero_spatial_tiny \
  --libero-task-limit 1 \
  --libero-episodes-per-task 1 \
  --libero-n-test-vis 0 \
  --libero-n-parallel-envs 1 \
  --libero-max-episode-steps 300
```

### Limitations

- This tiny path is for smoke testing only, not for benchmark-quality learning curves.
- `data/libero/libero_spatial_tiny.zarr` contains one demo per downloaded spatial task, not the full LIBERO release.
- Policy training still needs a real tokenizer checkpoint.
- Rollouts require a working LIBERO, MuJoCo, and robosuite stack; if those are missing, stick to config validation plus tokenizer and policy data loading.

### Optional stronger-compute path

If you have valid checkpoints and a proper robotics runtime, you can still use the optional evaluation path that delegates to `eval_policy_sim.py`:

```bash
python experiments/run_adaptive_halting_eval.py \
  --mode all \
  --libero-eval \
  --policy-checkpoint /path/to/oatpolicy.ckpt \
  --output-dir results/libero
```

Full LIBERO validation is not the default reproducible path here. It typically requires stronger compute, more episodes, and real checkpoints.

## Metrics

The experiment runner reports:

- `avg_prefix_depth`: average selected prefix depth
- `std_prefix_depth`: standard deviation of selected prefix depth within a run
- `recon_mse`: tokenizer-style prefix reconstruction error
- `policy_loss`: adaptive classifier loss on held-out data when available
- `runtime_sec`: inference-time runtime for the evaluation method
- `token_ratio`: `avg_prefix_depth / Kmax`
- `eos_rate`: fraction of adaptive samples that emitted STOP/EOS
- `halting_accuracy`: agreement between predicted `K` and oracle `K*`
- `selected_k_distribution`: distribution over used prefix depths
- `oracle_k_distribution`: oracle distribution when available

For the paper-ready aggregate view, the summary CSV reports mean and standard deviation across runs for the main metrics.

## Outputs

The runner writes:

- `results/adaptive_halting_summary.csv`
- `results/adaptive_halting_runs.csv`
- `results/adaptive_halting_metrics.json`
- `results/plots/reconstruction_mse_errorbars.svg`
- `results/plots/avg_token_usage_errorbars.svg`
- `results/plots/token_ratio_errorbars.svg`
- `results/plots/adaptive_k_distribution.svg`
- `research_logs/adaptive_halting_eval.log`

## Interpretation Guidelines

- Adaptive halting is useful when `token_ratio` drops materially below `1.0` while `recon_mse` stays close to the strongest fixed-K baseline.
- `halting_accuracy` measures whether the learned STOP policy matches the oracle prefix decision, not downstream task success directly.
- A non-degenerate `selected_k_distribution` is a good sign; it means the policy is allocating variable compute instead of collapsing to a single prefix depth.
- If adaptive halting collapses to the smallest prefix depth, inspect `oracle_k_distribution` first. The failure may come from the synthetic data or the halting predictor rather than the stopping mechanism itself.
- If runtime increases sharply without reducing `token_ratio`, the adaptive policy is not yet earning its additional control logic.

## Expected Behavior

On the synthetic benchmark:

- smaller fixed-K baselines should use fewer tokens but incur larger reconstruction error
- larger fixed-K baselines should improve reconstruction quality at higher token cost
- adaptive halting should land between these extremes:
  lower average token count than the largest fixed-K baseline
  better reconstruction than overly aggressive small-K baselines
