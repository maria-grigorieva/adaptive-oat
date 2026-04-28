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

## Experiments

Run the default synthetic comparison:

```bash
python experiments/run_adaptive_halting_eval.py \
  --mode all \
  --num-runs 3 \
  --output-dir results
```

Useful variants:

```bash
python experiments/run_adaptive_halting_eval.py --mode fixed --keep-ks 1 2 4 8
python experiments/run_adaptive_halting_eval.py --mode adaptive --num-samples 1024
python experiments/run_adaptive_halting_eval.py --mode all --include-random-baseline
```

Optional LIBERO-backed evaluation with an existing policy checkpoint:

```bash
python experiments/run_adaptive_halting_eval.py \
  --mode all \
  --libero-eval \
  --policy-checkpoint /path/to/oatpolicy.ckpt \
  --output-dir results/libero
```

## Metrics

The experiment runner reports:

- `avg_prefix_depth`: average selected prefix depth
- `std_prefix_depth`: standard deviation of selected prefix depth
- `recon_mse`: tokenizer-style prefix reconstruction error
- `policy_loss`: adaptive classifier loss on held-out data when available
- `runtime_sec`: inference-time runtime for the evaluation method
- `token_ratio`: `avg_prefix_depth / Kmax`
- `eos_rate`: fraction of adaptive samples that emitted STOP/EOS
- `halting_accuracy`: agreement between predicted `K` and oracle `K*`
- `selected_k_distribution`: distribution over used prefix depths
- `oracle_k_distribution`: oracle distribution when available

## Outputs

The runner writes:

- `results/adaptive_halting_summary.csv`
- `results/adaptive_halting_metrics.json`
- `results/plots/mse_vs_k.svg`
- `results/plots/token_usage_vs_method.svg`
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
