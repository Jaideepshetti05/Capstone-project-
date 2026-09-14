"""
Unit Test Suite for Poisoning Attacks in Federated Learning.
Tests:
  1. Label-flipping targeted evasion attack (malware -> benign).
  2. Label-flipping cyclic permutation attack.
  3. Weight poisoning Gaussian noise injection and parameter perturbation.
  4. Weight poisoning sign inversion.
  5. Weight poisoning scaling.
"""

import torch
import numpy as np
import pytest

from src.federated.attacks import LabelFlipAttack, WeightPoisonAttack


def test_label_flip_malware_to_benign():
    """Verify all malware labels [0, 1, 2, 3] are flipped to Benign [4]."""
    y_honest = np.array([0, 1, 2, 3, 4, 0, 2, 4], dtype=np.int64)
    attack = LabelFlipAttack(attack_type="malware_to_benign", target_class=4)
    y_poisoned = attack.apply(y_honest)

    # All elements must now be 4
    assert np.all(y_poisoned == 4), f"Expected all 4s, got {y_poisoned}"
    assert len(y_poisoned) == len(y_honest)


def test_label_flip_cyclic():
    """Verify cyclic label permutation y -> (y + 1) % 5."""
    y_honest = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    attack = LabelFlipAttack(attack_type="cyclic", num_classes=5)
    y_poisoned = attack.apply(y_honest)

    expected = np.array([1, 2, 3, 4, 0], dtype=np.int64)
    assert np.array_equal(y_poisoned, expected), f"Expected {expected}, got {y_poisoned}"


def test_weight_poison_gaussian():
    """Verify Gaussian noise perturbation changes parameter values with expected magnitude."""
    state = {
        "fc.weight": torch.zeros((10, 10), dtype=torch.float32),
        "fc.bias": torch.zeros(10, dtype=torch.float32),
        "num_batches": torch.tensor(100, dtype=torch.long),
    }

    attack = WeightPoisonAttack(attack_type="gaussian_noise", noise_std=2.0, seed=42)
    poisoned = attack.apply(state)

    # Float parameters should have non-zero variance around 2.0
    w_noise = poisoned["fc.weight"]
    assert not torch.allclose(w_noise, state["fc.weight"])
    assert 1.5 < torch.std(w_noise).item() < 2.5, f"Expected std ~2.0, got {torch.std(w_noise).item()}"

    # Non-float tensors should be unchanged
    assert poisoned["num_batches"].item() == 100


def test_weight_poison_sign_inversion():
    """Verify sign inversion reverses model gradient direction relative to global model."""
    global_state = {"fc.weight": torch.tensor([[1.0, 1.0], [1.0, 1.0]])}
    # Local client moved from 1.0 to 1.5 (delta = +0.5)
    local_state = {"fc.weight": torch.tensor([[1.5, 1.5], [1.5, 1.5]])}

    attack = WeightPoisonAttack(attack_type="sign_inversion", scale_factor=2.0)
    poisoned = attack.apply(local_state, global_state_dict=global_state)

    # w_mal = g - 2.0 * (1.5 - 1.0) = 1.0 - 1.0 = 0.0
    expected = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(poisoned["fc.weight"], expected), (
        f"Expected {expected}, got {poisoned['fc.weight']}"
    )


def test_weight_poison_scaling():
    """Verify scaling multiplies parameter tensors by scale factor."""
    local_state = {"fc.weight": torch.tensor([1.0, 2.0, 3.0])}
    attack = WeightPoisonAttack(attack_type="scaling", scale_factor=5.0)
    poisoned = attack.apply(local_state)

    expected = torch.tensor([5.0, 10.0, 15.0])
    assert torch.allclose(poisoned["fc.weight"], expected)
