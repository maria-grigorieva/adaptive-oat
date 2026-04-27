# Adaptive OAT Prototype

This repository now includes a minimal CPU-only research prototype for **Entropy-Gated Adaptive OAT Prefix Length**.

## What Changed

- Added [`oat/tokenizer/oat/adaptive_prefix.py`](/Users/maria/MIPT/AOPT_assignment/adaptive-oat/oat/tokenizer/oat/adaptive_prefix.py) with:
  - `compute_action_complexity(actions)`: mean squared temporal action delta.
  - `select_prefix_depth(complexity, available_depths, thresholds)`: maps per-sample complexity to prefix depth `K`.
  - `budget_regularization(selected_k, target_avg_k)`: placeholder for future routing/budget training.
- Extended [`oat/tokenizer/oat/tokenizer.py`](/Users/maria/MIPT/AOPT_assignment/adaptive-oat/oat/tokenizer/oat/tokenizer.py) so evaluation can:
  - keep the original fixed-`K` behavior;
  - optionally choose `K` adaptively per action chunk;
  - report selected `K`, average token count, reconstruction MSE, complexity, and runtime.
- Kept tensor shapes unchanged by reusing the existing decoder-side `eval_keep_k` masking path.
- Added a CPU experiment script at [`experiments/run_adaptive_prefix_oat.py`](/Users/maria/MIPT/AOPT_assignment/adaptive-oat/experiments/run_adaptive_prefix_oat.py).
- Added sanity tests at [`tests/test_adaptive_prefix_oat.py`](/Users/maria/MIPT/AOPT_assignment/adaptive-oat/tests/test_adaptive_prefix_oat.py).

## How To Run

Run the synthetic CPU experiment:

```bash
python experiments/run_adaptive_prefix_oat.py
```

Run with an existing OAT tokenizer checkpoint on CPU:

```bash
python experiments/run_adaptive_prefix_oat.py \
  --checkpoint /path/to/oattok.ckpt \
  --device cpu
```

Adjust routing thresholds or depth choices:

```bash
python experiments/run_adaptive_prefix_oat.py \
  --available-depths 1 2 4 8 \
  --adaptive-low-threshold 0.01 \
  --adaptive-high-threshold 0.08
```

Run the sanity tests:

```bash
python -m unittest discover -s tests
```

## Output Format

The experiment prints JSON with:

- `metadata`: checkpoint or synthetic setup, latent horizon, device, action shape, thresholds.
- `results`: one entry each for `fixed_k_1`, `fixed_k_2`, `fixed_k_4`, `fixed_k_8` when valid, plus `adaptive_k`.

Each result contains:

- `selected_k`: per-sample prefix depths.
- `avg_k`: average selected prefix depth.
- `avg_token_count`: same as average prefix depth for this prototype.
- `reconstruction_mse`: mean reconstruction error across samples.
- `reconstruction_mse_per_sample`: per-sample reconstruction error.
- `runtime_sec`: wall-clock runtime for the batch.
- `runtime_per_sample_sec`: average runtime per sample.
- `complexity`: per-sample complexity scores in adaptive mode.

## Limitations

- This is an **evaluation/inference-time prototype**, not a trained router.
- The routing heuristic is deliberately simple: temporal action complexity via mean squared delta.
- Thresholds are hand-set or batch-derived rather than learned.
- If no checkpoint is provided, the script uses a tiny synthetic OAT model and a short CPU overfit loop. That is useful for sanity checking the mechanism, not for reporting final research numbers.
- Average token count is currently identical to average selected prefix depth because one ordered token corresponds to one kept prefix position.

## Relation To OAT, BLT, And H-Net

- **OAT** provides the ordered, prefix-decodable action tokens and the existing `eval_keep_k` hook.
- **BLT** motivates allocating more computation or token budget to uncertain inputs.
- **H-Net** motivates lightweight routing and budget-aware selection, but this prototype intentionally avoids a full hierarchical router.

## Notes For Autonomous Research Loops

If this experiment is later plugged into an autonomous research loop such as AutoResearchClaw or a similar `autoresearch` workflow, the natural loop is:

1. Hypothesis: action-complexity-aware prefix selection reduces average token count while preserving reconstruction quality.
2. Code modification: adjust heuristic thresholds, candidate depths, or replace the heuristic with a learned lightweight predictor.
3. Experiment: run fixed-`K` baselines and adaptive routing on the same held-out action chunks.
4. Analysis: compare average `K`, reconstruction MSE, runtime, and the distribution of selected depths.

The current script is structured so that this loop can treat the JSON output as a compact experiment artifact.
