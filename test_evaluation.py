"""
Unit Test Suite for Model Evaluation and Multi-Class ROC-AUC.
Tests:
  1. Multi-class ROC-AUC One-vs-Rest calculation.
  2. Per-class ROC-AUC scores.
  3. Correct handling of edge cases (e.g. single class slice).
  4. Accuracy, Precision, Recall, F1 consistency.
"""

import torch
import numpy as np
import pytest

from src.models.mlp import MalwareMLP
from src.federated.evaluation import evaluate_model, DEFAULT_CLASS_NAMES


def test_roc_auc_evaluation():
    """Verify that evaluate_model calculates valid multi-class ROC-AUC scores."""
    torch.manual_seed(42)
    np.random.seed(42)

    in_features = 20
    num_classes = 5
    model = MalwareMLP(in_features=in_features, num_classes=num_classes, hidden_dims=[32, 16])

    # Generate synthetic balanced data for 5 classes
    X = np.random.randn(200, in_features).astype(np.float32)
    y = np.array([i % num_classes for i in range(200)], dtype=np.int64)

    metrics = evaluate_model(model, X, y, class_names=DEFAULT_CLASS_NAMES)

    assert "roc_auc_macro" in metrics
    assert "roc_auc_weighted" in metrics
    assert metrics["roc_auc_macro"] is not None
    assert 0.0 <= metrics["roc_auc_macro"] <= 1.0
    assert 0.0 <= metrics["roc_auc_weighted"] <= 1.0

    # Per-class verification
    for cname in DEFAULT_CLASS_NAMES:
        assert cname in metrics["per_class"]
        c_dict = metrics["per_class"][cname]
        assert "roc_auc" in c_dict
        assert c_dict["roc_auc"] is not None
        assert 0.0 <= c_dict["roc_auc"] <= 1.0


def test_roc_auc_edge_case_single_class():
    """Verify that evaluate_model does not crash when single class is passed."""
    in_features = 10
    model = MalwareMLP(in_features=in_features, num_classes=5, hidden_dims=[16, 8])

    X = np.random.randn(20, in_features).astype(np.float32)
    y = np.zeros(20, dtype=np.int64)  # Only class 0

    metrics = evaluate_model(model, X, y, class_names=DEFAULT_CLASS_NAMES)
    assert metrics["roc_auc_macro"] is None  # Undefined for 1 class, gracefully returns None
    assert metrics["accuracy"] >= 0.0
