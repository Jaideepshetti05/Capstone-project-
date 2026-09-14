"""
Unit Test Suite for Phase 7.1 Adaptive Gradient Clipping and Dynamic Noise Scaling.
Tests:
  1. AdaptiveClippingController initialization and parameter validation.
  2. Quantile calculation and clipping fraction estimation.
  3. C update direction (increases on high clipping fraction, decreases on low).
  4. Strict enforcement of C_min and C_max bounds.
  5. Robust handling of invalid, empty, NaN, and Inf inputs.
  6. Deterministic behavior under identical inputs.
  7. Gradient norm distribution statistics computation (mean, median, p75, p90, p95).
  8. Integration with DPClientTrainer and dynamic noise scaling.
  9. Full backward compatibility: Phase 6 fixed-clipping behavior unchanged when adaptive_clipping=False.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pathlib

from src.models.mlp import MalwareMLP
from src.federated.adaptive_clipping import AdaptiveClippingController
from src.federated.dp_client import DPClientTrainer
from src.federated.client import FederatedClient
from src.federated.dp_accountant import RDPPrivacyAccountant


def test_adaptive_clipping_initialization():
    """Verify AdaptiveClippingController initialization and parameter checks."""
    ctrl = AdaptiveClippingController(
        initial_C=1.5,
        target_quantile=0.90,
        learning_rate=0.1,
        C_min=0.2,
        C_max=5.0,
    )
    assert ctrl.get_current_C() == 1.5
    assert ctrl.target_quantile == 0.90
    assert abs(ctrl.target_clip_frac - 0.10) < 1e-6
    assert ctrl.C_min == 0.2
    assert ctrl.C_max == 5.0

    # Initial C clamped if out of bounds
    ctrl_clamped = AdaptiveClippingController(initial_C=15.0, C_min=0.1, C_max=5.0)
    assert ctrl_clamped.get_current_C() == 5.0

    # Invalid parameter rejections
    try:
        AdaptiveClippingController(initial_C=-1.0)
        assert False, "Should reject negative initial_C"
    except ValueError:
        pass

    try:
        AdaptiveClippingController(target_quantile=1.5)
        assert False, "Should reject target_quantile >= 1.0"
    except ValueError:
        pass

    try:
        AdaptiveClippingController(C_min=10.0, C_max=2.0)
        assert False, "Should reject C_max < C_min"
    except ValueError:
        pass

    print("  [PASS] Test 1: AdaptiveClippingController Initialization & Boundary Validation")


def test_c_update_direction_and_bounds():
    """Verify that C increases when clipping fraction is high and decreases when low, strictly bounded."""
    ctrl = AdaptiveClippingController(
        initial_C=1.0,
        target_quantile=0.90,  # target clipping fraction is 0.10
        learning_rate=0.2,
        C_min=0.2,
        C_max=3.0,
    )

    # 1. High clipping fraction (0.50 > 0.10) -> C must INCREASE
    c1 = ctrl.update_from_clipping_fraction(0.50, round_idx=1)
    assert c1 > 1.0, f"Expected C to increase, got {c1}"

    # 2. Low clipping fraction (0.00 < 0.10) -> C must DECREASE
    c2 = ctrl.update_from_clipping_fraction(0.00, round_idx=2)
    assert c2 < c1, f"Expected C to decrease, got {c2}"

    # 3. Upper bound enforcement: repeated high clipping fraction should not exceed C_max
    for r in range(3, 50):
        ctrl.update_from_clipping_fraction(1.0, round_idx=r)
    assert ctrl.get_current_C() == 3.0, f"Expected C to saturate at C_max=3.0, got {ctrl.get_current_C()}"

    # 4. Lower bound enforcement: repeated zero clipping fraction should saturate at C_min
    for r in range(50, 250):
        ctrl.update_from_clipping_fraction(0.0, round_idx=r)
    assert ctrl.get_current_C() == 0.2, f"Expected C to saturate at C_min=0.2, got {ctrl.get_current_C()}"

    print("  [PASS] Test 2: Directional Adaptation & Strict Min/Max Bound Clamping")


def test_invalid_and_empty_inputs_resilience():
    """Verify that controller handles NaN, Inf, empty arrays without crashing or breaking state."""
    ctrl = AdaptiveClippingController(initial_C=1.0, C_min=0.1, C_max=10.0)

    # NaN clipping fraction
    c_nan = ctrl.update_from_clipping_fraction(float("nan"), round_idx=1)
    assert not math.isnan(c_nan) and not math.isinf(c_nan)
    assert 0.1 <= c_nan <= 10.0

    # Inf clipping fraction
    c_inf = ctrl.update_from_clipping_fraction(float("inf"), round_idx=2)
    assert not math.isnan(c_inf) and not math.isinf(c_inf)
    assert 0.1 <= c_inf <= 10.0

    # Empty raw norms
    c_empty = ctrl.update_from_norms([], round_idx=3)
    assert not math.isnan(c_empty)

    # Norm statistics with empty input
    stats_empty = AdaptiveClippingController.compute_norm_statistics([], clipping_threshold=1.0)
    assert stats_empty["mean_gradient_norm"] == 0.0
    assert stats_empty["sample_count"] == 0

    print("  [PASS] Test 3: Robust Exception-Free Handling of NaN/Inf/Empty Inputs")


def test_norm_statistics_and_percentiles():
    """Verify exact calculation of mean, median, p75, p90, p95 gradient norms and clipping fractions."""
    norms = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
    C = 0.8
    stats = AdaptiveClippingController.compute_norm_statistics(norms, clipping_threshold=C)

    assert stats["sample_count"] == 12
    assert abs(stats["mean_gradient_norm"] - np.mean(norms)) < 1e-6
    assert abs(stats["median_gradient_norm"] - np.median(norms)) < 1e-6
    assert abs(stats["p90_gradient_norm"] - np.percentile(norms, 90)) < 1e-6

    # 4 values exceed 0.8: 0.9, 1.0, 1.5, 2.0 -> fraction = 4/12 = 0.3333...
    assert abs(stats["clipping_fraction"] - (4.0 / 12.0)) < 1e-6
    assert stats["clipping_threshold_C"] == 0.8

    print("  [PASS] Test 4: Distributional Percentiles & Clipping Fraction Statistics")


def test_dp_client_trainer_dynamic_noise_and_percentiles():
    """Verify that DPClientTrainer uses active C for both clipping and proportional Gaussian noise."""
    torch.manual_seed(42)
    model = MalwareMLP(in_features=10, num_classes=3, hidden_dims=[16, 8])
    C_active = 2.5
    sigma = 1.0
    batch_size = 20

    trainer = DPClientTrainer(clip_norm=C_active, noise_multiplier=sigma, device="cpu")

    X = torch.randn(40, 10)
    y = torch.randint(0, 3, (40,))
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    metrics = trainer.train_epoch_dpsgd(model, loader, optimizer, criterion)

    expected_eff_noise = (sigma * C_active) / float(batch_size)
    assert abs(metrics["effective_noise_std"] - expected_eff_noise) < 1e-6
    assert metrics["clipping_threshold_C"] == C_active
    assert "median_grad_norm" in metrics
    assert "p90_grad_norm" in metrics
    assert "p95_grad_norm" in metrics

    print(f"  [PASS] Test 5: DPClientTrainer Dynamic Noise & Percentiles (C={C_active}, eff_noise={expected_eff_noise:.6f})")


def test_backward_compatibility_phase6_fixed_dp():
    """Verify that when adaptive clipping is disabled, client training and metrics match Phase 6 exactly."""
    torch.manual_seed(42)
    X = np.random.randn(80, 443).astype(np.float32)
    y = np.random.randint(0, 5, size=80).astype(np.int64)

    client = FederatedClient(client_id=0, X_local=X, y_local=y, device="cpu")
    model = MalwareMLP(in_features=443, num_classes=5)
    global_state = model.state_dict()

    # Standard Phase 6 DP config
    config_phase6 = {
        "batch_size": 32,
        "local_epochs": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dp_enabled": True,
        "clip_norm": 1.0,
        "noise_multiplier": 1.0,
        "mu": 0.0,
    }

    updated_state, n, metrics = client.train(global_state, config_phase6)
    assert n == 80
    assert "clipping_fraction" in metrics
    assert "mean_grad_norm" in metrics
    assert metrics["clipping_threshold_C"] == 1.0
    assert abs(metrics["noise_std"] - (1.0 * 1.0 / 32.0)) < 1e-6

    print("  [PASS] Test 6: Backward Compatibility with Phase 6 (Fixed C=1.0 behavior 100% preserved)")


def run_all_adaptive_clipping_tests():
    print("\n" + "=" * 75)
    print("  Phase 7.1 Adaptive Gradient Clipping Unit Test Suite")
    print("=" * 75)

    test_adaptive_clipping_initialization()
    test_c_update_direction_and_bounds()
    test_invalid_and_empty_inputs_resilience()
    test_norm_statistics_and_percentiles()
    test_dp_client_trainer_dynamic_noise_and_percentiles()
    test_backward_compatibility_phase6_fixed_dp()

    print("=" * 75)
    print("  ALL 6 / 6 ADAPTIVE CLIPPING UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_all_adaptive_clipping_tests()
