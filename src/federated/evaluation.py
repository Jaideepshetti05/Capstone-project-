"""
Evaluation module for centralized validation and test datasets during Federated Learning.
Computes comprehensive classification metrics, confusion matrices, and inference latency.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DEFAULT_CLASS_NAMES = [
    "Adware",
    "Banking malware",
    "SMS malware",
    "Riskware",
    "Benign",
]


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a PyTorch model on the provided dataset and return complete metrics.

    Args:
        model: PyTorch model to evaluate.
        X: Feature matrix of shape (N, D).
        y: Ground truth integer class labels of shape (N,).
        batch_size: Batch size for forward pass evaluation.
        device: 'cpu' or 'cuda'.
        class_names: Names for each class index (0..4).

    Returns:
        Dictionary with accuracy, macro/weighted metrics, per-class metrics, and confusion matrix.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    model_device = torch.device(device)
    model.eval()
    model.to(model_device)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(model_device), by.to(model_device)
            logits = model(bx)
            loss = criterion(logits, by)
            total_loss += loss.item() * len(bx)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(by.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    inf_time = time.perf_counter() - t0

    avg_loss = total_loss / max(len(dataset), 1)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Multi-class ROC-AUC calculation (One-vs-Rest)
    try:
        if len(np.unique(y_true)) > 1 and y_prob.shape[1] == len(class_names):
            roc_auc_macro = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            roc_auc_weighted = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        else:
            roc_auc_macro = None
            roc_auc_weighted = None
    except Exception:
        roc_auc_macro = None
        roc_auc_weighted = None

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    per_class = {}
    for idx, cname in enumerate(class_names):
        # Per-class binary ROC-AUC
        try:
            binary_target = (y_true == idx).astype(int)
            if len(np.unique(binary_target)) == 2:
                class_auc = float(roc_auc_score(binary_target, y_prob[:, idx]))
            else:
                class_auc = None
        except Exception:
            class_auc = None

        per_class[cname] = {
            "precision": float(report_dict[cname]["precision"]),
            "recall": float(report_dict[cname]["recall"]),
            "f1_score": float(report_dict[cname]["f1-score"]),
            "roc_auc": class_auc,
            "support": int(report_dict[cname]["support"]),
        }

    latency_ms_per_sample = (inf_time / max(len(X), 1)) * 1000.0

    return {
        "loss": round(float(avg_loss), 4),
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "per_class": per_class,
        "confusion_matrix": cm,
        "inference_time_seconds": round(inf_time, 6),
        "latency_ms_per_sample": round(latency_ms_per_sample, 4),
    }

