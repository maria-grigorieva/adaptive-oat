# Figure Manifest

This manifest records the exact source data and intended claim boundary for each paper figure.

## fig_method_eg_apd_oat

- Filename: `fig_method_eg_apd_oat.svg/.pdf/.png`
- Source data: `Conceptual schematic based on the implemented OAT tokenizer, oracle prefix supervision, and EOS-based adaptive policy training/inference.`
- What it shows: Method overview of the implemented adaptive-prefix pipeline.
- Claim supported: Supports the paper's method description.
- Caveat: Diagrammatic only; not a quantitative result.

## fig_fixed_vs_adaptive

- Filename: `fig_fixed_vs_adaptive.svg/.pdf/.png`
- Source data: `Conceptual comparison using the implemented fixed-K and adaptive-EOS decoding modes.`
- What it shows: Illustrates fixed-depth vs variable-depth action decoding.
- Claim supported: Supports the conceptual motivation for adaptive prefix depth.
- Caveat: Not derived from a specific numeric dataset.

## fig_reconstruction_mse

- Filename: `fig_reconstruction_mse.svg/.pdf/.png`
- Source data: `/Users/maria/MIPT/AOPT_assignment/adaptive-oat/results_30runs/adaptive_halting_summary.csv`
- What it shows: Synthetic benchmark reconstruction error with repeated-run uncertainty.
- Claim supported: Supports the claim that adaptive halting preserves much of the reconstruction quality of deeper fixed prefixes.
- Caveat: Synthetic benchmark only; does not measure robotics success.

## fig_avg_token_usage

- Filename: `fig_avg_token_usage.svg/.pdf/.png`
- Source data: `/Users/maria/MIPT/AOPT_assignment/adaptive-oat/results_30runs/adaptive_halting_summary.csv`
- What it shows: Synthetic benchmark average selected prefix depth.
- Claim supported: Supports the claim that adaptive halting reduces average token budget.
- Caveat: Synthetic benchmark only.

## fig_token_ratio

- Filename: `fig_token_ratio.svg/.pdf/.png`
- Source data: `/Users/maria/MIPT/AOPT_assignment/adaptive-oat/results_30runs/adaptive_halting_summary.csv`
- What it shows: Synthetic benchmark token ratio relative to Kmax.
- Claim supported: Supports the efficiency claim for adaptive halting.
- Caveat: Synthetic benchmark only.

## fig_efficiency_quality_pareto

- Filename: `fig_efficiency_quality_pareto.svg/.pdf/.png`
- Source data: `/Users/maria/MIPT/AOPT_assignment/adaptive-oat/results_30runs/adaptive_halting_summary.csv`
- What it shows: Synthetic efficiency-quality trade-off frontier.
- Claim supported: Supports the main quantitative paper claim.
- Caveat: Synthetic benchmark is the primary evidence; not a simulator benchmark.

## fig_adaptive_k_distribution

- Filename: `fig_adaptive_k_distribution.svg/.pdf/.png`
- Source data: `/Users/maria/MIPT/AOPT_assignment/adaptive-oat/results_30runs/adaptive_halting_summary.csv`
- What it shows: Distribution of adaptive prefix depths on the synthetic benchmark.
- Claim supported: Supports the claim that the learned policy is not merely a single fixed-K policy.
- Caveat: Distribution is aggregated from summary artifacts.

## fig_libero_tinyplus_integration

- Filename: `fig_libero_tinyplus_integration.svg/.pdf/.png`
- Source data: `/Users/maria/MIPT/AOPT_assignment/adaptive-oat/results/libero_tinyplus_summary.csv`
- What it shows: Token-length, success-rate, and selected-K behavior for the tinyplus LIBERO integration run.
- Claim supported: Supports the claim that adaptive halting executes inside the real LIBERO simulator path.
- Caveat: Success rate remains 0.0; this is not benchmark-level validation.

## fig_evaluation_pipeline_status

- Filename: `fig_evaluation_pipeline_status.svg/.pdf/.png`
- Source data: `Repository artifact inventory and README-documented evaluation pipeline.`
- What it shows: Communicates the boundary between completed synthetic validation, completed integration validation, and future full-benchmark work.
- Claim supported: Supports scientifically honest framing of results.
- Caveat: Status figure; not a performance result.
