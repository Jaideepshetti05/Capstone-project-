"""
Unit Test Suite for Byzantine-Robust Federated Aggregation Algorithms.
Tests:
  1. Coordinate-wise Median exact mathematical calculation.
  2. Coordinate-wise Median resilience against extreme Byzantine outliers.
  3. Trimmed Mean exact sorting and coordinate trimming.
  4. Trimmed Mean resilience against bilateral Byzantine outliers.
  5. Handling of non-float state dict buffers.
  6. Input validation (empty updates, invalid trimming fraction beta).
  7. Unified robust aggregation router dispatching.
"""

import math
import torch
import pytest
import numpy as np

from src.federated.robust_aggregation import (
    aggregate_coordinate_median,
    aggregate_trimmed_mean,
    aggregate_robust,
)
from src.federated.fedavg import aggregate_fedavg


def test_coordinate_median_exactness():
    """Verify coordinate-wise median produces exact mathematical median per parameter."""
    # 5 clients with 1D parameter vectors
    # coordinate 0 values: [1.0, 2.0, 3.0, 4.0, 5.0] -> median 3.0
    # coordinate 1 values: [10.0, 20.0, 50.0, 40.0, 30.0] -> sorted [10, 20, 30, 40, 50] -> median 30.0
    client_updates = [
        ({"w": torch.tensor([1.0, 10.0])}, 100),
        ({"w": torch.tensor([2.0, 20.0])}, 100),
        ({"w": torch.tensor([3.0, 50.0])}, 100),
        ({"w": torch.tensor([4.0, 40.0])}, 100),
        ({"w": torch.tensor([5.0, 30.0])}, 100),
    ]

    agg = aggregate_coordinate_median(client_updates)
    assert torch.allclose(agg["w"], torch.tensor([3.0, 30.0])), f"Unexpected median: {agg['w']}"


def test_coordinate_median_byzantine_resilience():
    """Verify coordinate-wise median is immune to 20% extreme poisoning while FedAvg fails."""
    # 10 clients: 8 honest around 1.0, 2 malicious with extreme magnitude 1,000,000.0
    honest_updates = [({"w": torch.tensor([1.0 + 0.05 * i])}, 100) for i in range(8)]
    malicious_updates = [
        ({"w": torch.tensor([1_000_000.0])}, 100),
        ({"w": torch.tensor([1_000_000.0])}, 100),
    ]
    all_updates = honest_updates + malicious_updates

    # FedAvg should be catastrophically corrupted
    fedavg_res = aggregate_fedavg(all_updates)
    assert fedavg_res["w"].item() > 100_000.0, "FedAvg should have been corrupted by large outliers"

    # Coordinate-wise Median should remain strictly within honest distribution [1.0, 1.35]
    median_res = aggregate_coordinate_median(all_updates)
    assert 1.0 <= median_res["w"].item() <= 1.4, (
        f"Coordinate median failed Byzantine resilience test: got {median_res['w'].item()}"
    )


def test_trimmed_mean_exactness():
    """Verify trimmed mean removes exactly floor(beta * K) elements from both ends."""
    # 10 clients with values 1.0 to 10.0
    # With beta = 0.2, num_to_trim = floor(0.2 * 10) = 2
    # Trims lowest 2 (1.0, 2.0) and highest 2 (9.0, 10.0)
    # Remaining values: 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 -> mean = 33.0 / 6 = 5.5
    client_updates = [({"w": torch.tensor([float(i)])}, 100) for i in range(1, 11)]

    agg = aggregate_trimmed_mean(client_updates, beta=0.2)
    expected_val = 5.5
    assert abs(agg["w"].item() - expected_val) < 1e-6, (
        f"Trimmed mean expected {expected_val}, got {agg['w'].item()}"
    )


def test_trimmed_mean_byzantine_resilience():
    """Verify trimmed mean trims extreme positive and negative outliers cleanly."""
    # 10 clients: 8 honest around 5.0, 1 extreme negative (-10,000.0), 1 extreme positive (+10,000.0)
    updates = [({"w": torch.tensor([5.0 + 0.1 * i])}, 100) for i in range(8)]
    updates.append(({"w": torch.tensor([-10_000.0])}, 100))
    updates.append(({"w": torch.tensor([10_000.0])}, 100))

    # beta = 0.1 trims 1 from each end (exactly the 2 outliers)
    agg = aggregate_trimmed_mean(updates, beta=0.1)
    honest_mean = np.mean([5.0 + 0.1 * i for i in range(8)])
    assert abs(agg["w"].item() - honest_mean) < 1e-5, (
        f"Trimmed mean failed outlier elimination: expected {honest_mean}, got {agg['w'].item()}"
    )


def test_robust_aggregation_router_and_validation():
    """Verify input validation and unified routing logic."""
    # Empty input
    with pytest.raises(ValueError, match="No client updates provided"):
        aggregate_coordinate_median([])

    with pytest.raises(ValueError, match="No client updates provided"):
        aggregate_trimmed_mean([])

    # Invalid beta
    with pytest.raises(ValueError, match="Trimming fraction beta must be in"):
        aggregate_trimmed_mean([({"w": torch.tensor([1.0])}, 10)], beta=0.6)

    with pytest.raises(ValueError, match="Trimming fraction beta must be in"):
        aggregate_trimmed_mean([({"w": torch.tensor([1.0])}, 10)], beta=-0.1)

    # Router tests
    dummy_updates = [
        ({"w": torch.tensor([1.0]), "step": torch.tensor(10, dtype=torch.long)}, 50),
        ({"w": torch.tensor([3.0]), "step": torch.tensor(10, dtype=torch.long)}, 50),
        ({"w": torch.tensor([2.0]), "step": torch.tensor(10, dtype=torch.long)}, 50),
    ]

    med_res = aggregate_robust(dummy_updates, method="median")
    assert med_res["w"].item() == 2.0
    assert med_res["step"].item() == 10

    trim_res = aggregate_robust(dummy_updates, method="trimmed_mean", beta=0.2)
    assert 1.0 <= trim_res["w"].item() <= 3.0

    avg_res = aggregate_robust(dummy_updates, method="fedavg")
    assert abs(avg_res["w"].item() - 2.0) < 1e-6

    with pytest.raises(ValueError, match="Unknown robust aggregation method"):
        aggregate_robust(dummy_updates, method="invalid_method")
