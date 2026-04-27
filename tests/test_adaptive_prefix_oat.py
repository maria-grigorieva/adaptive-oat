import unittest

import numpy as np

from experiments.run_adaptive_prefix_oat import (
    allocation_distribution,
    calibrate_quantile_thresholds,
    compute_complexity_scores,
    exact_sign_flip_pvalue,
    map_scores_to_keep_k,
    validate_keep_ks,
    validate_thresholds,
)


class AdaptivePrefixHelpersTest(unittest.TestCase):
    def test_validate_keep_ks_rejects_non_pow2(self):
        with self.assertRaises(ValueError):
            validate_keep_ks([1, 3, 8])

    def test_validate_thresholds_checks_length(self):
        with self.assertRaises(ValueError):
            validate_thresholds([0.5], [1, 2, 4, 8])

    def test_delta_complexity_exceeds_constant_sequence(self):
        constant = np.zeros((1, 4, 2), dtype=np.float32)
        jagged = np.array(
            [[[0.0, 0.0], [1.0, -1.0], [0.5, 0.5], [2.0, -2.0]]],
            dtype=np.float32,
        )
        scores = compute_complexity_scores(np.concatenate([constant, jagged], axis=0), "delta_l2_mean")
        self.assertLess(scores[0], scores[1])

    def test_quantile_mapping_covers_all_buckets(self):
        calibration = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        thresholds = calibrate_quantile_thresholds(calibration, [0.25, 0.5, 0.75])
        mapped = map_scores_to_keep_k(
            scores=np.array([0.05, 0.2, 0.3, 0.45], dtype=np.float32),
            keep_ks=[1, 2, 4, 8],
            strategy="quantile",
            calibration_scores=calibration,
            quantiles=[0.25, 0.5, 0.75],
        )
        self.assertEqual(mapped.tolist(), [1, 2, 4, 8])
        self.assertEqual(len(thresholds), 3)

    def test_linear_mapping_hits_extremes(self):
        mapped = map_scores_to_keep_k(
            scores=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            keep_ks=[1, 2, 4, 8],
            strategy="linear",
            calibration_scores=np.array([0.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(mapped.tolist(), [1, 4, 8])

    def test_allocation_distribution_sums_to_one(self):
        dist = allocation_distribution(np.array([1, 2, 2, 8]), [1, 2, 4, 8])
        self.assertAlmostEqual(sum(dist.values()), 1.0)
        self.assertEqual(dist["4"], 0.0)

    def test_exact_sign_flip_pvalue_prefers_negative_shift(self):
        strong_negative = [-0.4, -0.2, -0.3, -0.1]
        pvalue = exact_sign_flip_pvalue(strong_negative, alternative="less")
        self.assertLess(pvalue, 0.1)


if __name__ == "__main__":
    unittest.main()
