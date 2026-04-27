from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch


def compute_action_complexity(actions: torch.Tensor) -> torch.Tensor:
    """Compute a per-sample temporal complexity score.

    The current heuristic is the mean squared delta across adjacent timesteps.
    Low-motion / smooth trajectories produce lower scores, while noisy or rapidly
    changing trajectories produce higher scores.
    """
    if actions.ndim != 3:
        raise ValueError(
            f"Expected actions with shape [B, T, D], but received {tuple(actions.shape)}."
        )

    if actions.shape[1] < 2:
        return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

    deltas = actions[:, 1:] - actions[:, :-1]
    return deltas.square().mean(dim=(1, 2))


def build_adaptive_thresholds(
    available_depths: Sequence[int],
    low_threshold: float,
    high_threshold: float,
) -> List[float]:
    """Interpolate monotonic thresholds spanning [low_threshold, high_threshold]."""
    if len(available_depths) < 2:
        return []

    if high_threshold < low_threshold:
        raise ValueError(
            f"Expected high_threshold >= low_threshold, got {high_threshold} < {low_threshold}."
        )

    threshold_count = len(available_depths) - 1
    if threshold_count == 1:
        return [float(low_threshold)]

    return torch.linspace(
        low_threshold,
        high_threshold,
        steps=threshold_count,
        dtype=torch.float32,
    ).tolist()


def select_prefix_depth(
    complexity: torch.Tensor,
    available_depths: Sequence[int] = (1, 2, 4, 8),
    thresholds: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Map complexity scores to prefix depths.

    If thresholds are omitted, the boundaries are computed from evenly spaced
    quantiles of the current batch, which keeps the heuristic scale-agnostic.
    """
    if complexity.ndim != 1:
        raise ValueError(
            f"Expected complexity with shape [B], but received {tuple(complexity.shape)}."
        )

    if len(available_depths) == 0:
        raise ValueError("available_depths must contain at least one depth.")

    depths = list(sorted(int(depth) for depth in available_depths))
    if any(depth <= 0 for depth in depths):
        raise ValueError(f"All prefix depths must be positive, got {depths}.")

    if thresholds is None:
        if len(depths) == 1:
            thresholds_tensor = complexity.new_empty((0,))
        else:
            quantiles = torch.linspace(
                0.0,
                1.0,
                steps=len(depths) + 1,
                device=complexity.device,
                dtype=complexity.dtype,
            )[1:-1]
            thresholds_tensor = torch.quantile(complexity.detach(), quantiles)
    else:
        if len(thresholds) != len(depths) - 1:
            raise ValueError(
                "Expected len(thresholds) == len(available_depths) - 1, "
                f"got {len(thresholds)} thresholds for {len(depths)} depths."
            )
        thresholds_tensor = torch.as_tensor(
            list(thresholds),
            device=complexity.device,
            dtype=complexity.dtype,
        )

    bucket_ids = torch.bucketize(complexity, thresholds_tensor)
    depth_tensor = torch.as_tensor(depths, device=complexity.device, dtype=torch.long)
    return depth_tensor[bucket_ids]


def budget_regularization(
    selected_k: torch.Tensor | Iterable[int],
    target_avg_k: float,
) -> torch.Tensor:
    """Placeholder regularizer for future training-time routing experiments."""
    selected_k_tensor = torch.as_tensor(list(selected_k), dtype=torch.float32)
    return (selected_k_tensor.mean() - float(target_avg_k)).square()
