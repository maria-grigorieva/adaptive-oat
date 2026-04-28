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

### Optional CPU-only LIBERO smoke evaluation

```bash
python experiments/run_adaptive_halting_eval.py \
  --mode all \
  --libero-smoke-eval \
  --libero-num-tasks 1 \
  --libero-num-episodes 1 \
  --output-dir results/libero_smoke
```

This smoke path is optional and may be skipped automatically on weak CPU-only machines or when LIBERO / MuJoCo / robosuite dependencies are unavailable. The script will write a `libero_smoke_status.json` file explaining whether it ran or why it was skipped.

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
