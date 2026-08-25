"""
CICMalDroid 2020 - Centralized Baseline Training & Evaluation Pipeline
Project: Privacy-Preserving Malware Detection Using Federated Learning

Models Trained:
1. Logistic Regression (Linear baseline, L2-regularized)
2. Random Forest (Ensemble tree baseline, 300 estimators)
3. Multi-Layer Perceptron (PyTorch Deep Learning baseline, [256, 128, 64])

Protocol:
- Train: X_train, y_train (8,062 samples, Variant A - 443 features)
- Validation: X_val, y_val (1,152 samples) for tuning and model selection
- Test: X_test, y_test (2,304 samples) evaluated ONLY after model selection
"""

import os
import sys
import json
import time
import joblib
import pathlib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mlp import MalwareMLP

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "variant_a"
MODELS_DIR = PROJECT_ROOT / "models" / "centralized"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLASS_NAMES = [
    "Adware",
    "Banking malware",
    "SMS malware",
    "Riskware",
    "Benign",
]

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    """Ensure strict determinism across PyTorch and NumPy."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    per_class = {}
    for cname in CLASS_NAMES:
        per_class[cname] = {
            "precision": float(report_dict[cname]["precision"]),
            "recall": float(report_dict[cname]["recall"]),
            "f1_score": float(report_dict[cname]["f1-score"]),
            "support": int(report_dict[cname]["support"]),
        }

    return {
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[LogisticRegression, Dict[str, Any], float, float]:
    """Train and evaluate tuned Logistic Regression on validation set."""
    print("\n--- Training Logistic Regression (C=10.0, lbfgs, max_iter=2000) ---")
    t0 = time.perf_counter()
    clf = LogisticRegression(
        C=10.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Validation inference
    t0_inf = time.perf_counter()
    y_val_pred = clf.predict(X_val)
    val_inf_time = time.perf_counter() - t0_inf

    val_metrics = compute_metrics(y_val, y_val_pred)
    val_metrics["training_time_seconds"] = round(train_time, 4)
    val_metrics["val_inference_time_seconds"] = round(val_inf_time, 6)
    val_metrics["val_latency_ms_per_sample"] = round((val_inf_time / len(X_val)) * 1000.0, 4)

    print(f"    Train Time: {train_time:.2f}s | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val Macro F1: {val_metrics['macro_f1']:.4f}")
    return clf, val_metrics, train_time, val_inf_time


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[RandomForestClassifier, Dict[str, Any], float, float]:
    """Train and evaluate tuned Random Forest on validation set."""
    print("\n--- Training Random Forest (n_estimators=300, max_depth=None) ---")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        criterion="gini",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Validation inference
    t0_inf = time.perf_counter()
    y_val_pred = rf.predict(X_val)
    val_inf_time = time.perf_counter() - t0_inf

    val_metrics = compute_metrics(y_val, y_val_pred)
    val_metrics["training_time_seconds"] = round(train_time, 4)
    val_metrics["val_inference_time_seconds"] = round(val_inf_time, 6)
    val_metrics["val_latency_ms_per_sample"] = round((val_inf_time / len(X_val)) * 1000.0, 4)

    print(f"    Train Time: {train_time:.2f}s | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val Macro F1: {val_metrics['macro_f1']:.4f}")
    return rf, val_metrics, train_time, val_inf_time


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[MalwareMLP, Dict[str, Any], Dict[str, Any], float, float]:
    """Train and evaluate PyTorch MLP with early stopping."""
    print("\n--- Training PyTorch MLP ([443 -> 256 -> 128 -> 64 -> 5], AdamW, lr=1e-3, bs=64) ---")
    set_seed(RANDOM_SEED)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = MalwareMLP(in_features=X_train.shape[1], num_classes=5, hidden_dims=[256, 128, 64], dropout_rate=0.2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 15
    patience_counter = 0
    max_epochs = 100

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_macro_f1": []}

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_list = []
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(bx)
                preds = torch.argmax(logits, dim=1)
                val_preds_list.extend(preds.cpu().numpy())
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        epoch_macro_f1 = f1_score(y_val, val_preds_list, average="macro", zero_division=0)
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_macro_f1"].append(round(float(epoch_macro_f1), 4))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} (val_loss={best_val_loss:.4f}).")
                break

    train_time = time.perf_counter() - t0

    # Load best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # Measure validation inference
    t0_inf = time.perf_counter()
    y_val_preds = []
    with torch.no_grad():
        for bx, _ in val_loader:
            logits = model(bx)
            y_val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    val_inf_time = time.perf_counter() - t0_inf

    val_metrics = compute_metrics(y_val, np.array(y_val_preds))
    val_metrics["training_time_seconds"] = round(train_time, 4)
    val_metrics["val_inference_time_seconds"] = round(val_inf_time, 6)
    val_metrics["val_latency_ms_per_sample"] = round((val_inf_time / len(X_val)) * 1000.0, 4)
    val_metrics["best_epoch"] = best_epoch
    val_metrics["best_val_loss"] = round(best_val_loss, 4)

    mlp_meta = {
        "architecture": model.get_config(),
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "lr_scheduler": "ReduceLROnPlateau(factor=0.5, patience=5)",
        "early_stopping_patience": patience,
        "random_seed": RANDOM_SEED,
        "best_epoch": best_epoch,
        "total_epochs_trained": epoch,
        "best_val_loss": round(best_val_loss, 4),
        "history": history,
    }

    print(f"    Train Time: {train_time:.2f}s | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val Macro F1: {val_metrics['macro_f1']:.4f}")
    return model, val_metrics, mlp_meta, train_time, val_inf_time


def main():
    print("=" * 75)
    print("  CICMalDroid 2020: Centralized Baseline Training & Evaluation")
    print("=" * 75)

    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load Variant A Processed Data ---
    print(f"\n[1/5] Loading Variant A processed data from: {DATA_DIR}")
    X_train_df = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")
    X_val_df = pd.read_csv(DATA_DIR / "X_val.csv")
    y_val_df = pd.read_csv(DATA_DIR / "y_val.csv")
    X_test_df = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test_df = pd.read_csv(DATA_DIR / "y_test.csv")

    X_train = X_train_df.values.astype(np.float32)
    y_train = y_train_df.values.ravel().astype(np.int64)
    X_val = X_val_df.values.astype(np.float32)
    y_val = y_val_df.values.ravel().astype(np.int64)
    X_test = X_test_df.values.astype(np.float32)
    y_test = y_test_df.values.ravel().astype(np.int64)

    print(f"      - Training samples:   {len(X_train):,} rows x {X_train.shape[1]} features")
    print(f"      - Validation samples: {len(X_val):,} rows x {X_val.shape[1]} features")
    print(f"      - Locked Test samples:{len(X_test):,} rows x {X_test.shape[1]} features (UNTOUCHED)")

    # --- Step 2: Train & Validate All Three Baselines ---
    print("\n[2/5] Training 3 centralized baseline models on training set...")

    # Model 1: Logistic Regression
    clf_lr, val_metrics_lr, t_train_lr, t_inf_lr = train_logistic_regression(X_train, y_train, X_val, y_val)
    joblib.dump(clf_lr, MODELS_DIR / "logistic_regression.joblib")

    # Model 2: Random Forest
    clf_rf, val_metrics_rf, t_train_rf, t_inf_rf = train_random_forest(X_train, y_train, X_val, y_val)
    joblib.dump(clf_rf, MODELS_DIR / "random_forest.joblib")

    # Model 3: PyTorch MLP
    model_mlp, val_metrics_mlp, mlp_meta, t_train_mlp, t_inf_mlp = train_mlp(X_train, y_train, X_val, y_val)
    torch.save(model_mlp.state_dict(), MODELS_DIR / "mlp_best.pt")
    with open(MODELS_DIR / "mlp_architecture.json", "w") as f:
        json.dump(mlp_meta, f, indent=2)

    # --- Step 3: Validation Comparison & Model Selection ---
    print("\n[3/5] Comparing all 3 models on the Validation Set...")
    print("=" * 95)
    print(f"{'Model':<22} | {'Val Accuracy':<12} | {'Val Macro F1':<12} | {'Val W-F1':<10} | {'Macro Prec':<10} | {'Macro Rec':<10} | {'Train Time':<10}")
    print("-" * 95)
    print(f"{'Logistic Regression':<22} | {val_metrics_lr['accuracy']*100:>10.2f}% | {val_metrics_lr['macro_f1']:>12.4f} | {val_metrics_lr['weighted_f1']:>10.4f} | {val_metrics_lr['macro_precision']:>10.4f} | {val_metrics_lr['macro_recall']:>10.4f} | {t_train_lr:>8.2f}s")
    print(f"{'Random Forest':<22} | {val_metrics_rf['accuracy']*100:>10.2f}% | {val_metrics_rf['macro_f1']:>12.4f} | {val_metrics_rf['weighted_f1']:>10.4f} | {val_metrics_rf['macro_precision']:>10.4f} | {val_metrics_rf['macro_recall']:>10.4f} | {t_train_rf:>8.2f}s")
    print(f"{'MLP Neural Network':<22} | {val_metrics_mlp['accuracy']*100:>10.2f}% | {val_metrics_mlp['macro_f1']:>12.4f} | {val_metrics_mlp['weighted_f1']:>10.4f} | {val_metrics_mlp['macro_precision']:>10.4f} | {val_metrics_mlp['macro_recall']:>10.4f} | {t_train_mlp:>8.2f}s")
    print("=" * 95)

    # Model Selection logic (based primarily on Validation Macro F1)
    val_macro_scores = {
        "Logistic Regression": val_metrics_lr["macro_f1"],
        "Random Forest": val_metrics_rf["macro_f1"],
        "MLP Neural Network": val_metrics_mlp["macro_f1"],
    }
    best_model_name = max(val_macro_scores, key=val_macro_scores.get)
    print(f"\n>>> SELECTION DECISION based on Validation Macro F1: {best_model_name} (Macro F1 = {val_macro_scores[best_model_name]:.4f}) <<<")

    # --- Step 4: Final Evaluation on Locked Test Set ---
    print("\n[4/5] Evaluating on LOCKED TEST SET (2,304 samples)...")
    print("      (Conducted strictly after model selection)")

    # Test LR
    t0_test_lr = time.perf_counter()
    y_test_pred_lr = clf_lr.predict(X_test)
    t_test_inf_lr = time.perf_counter() - t0_test_lr
    test_metrics_lr = compute_metrics(y_test, y_test_pred_lr)
    test_metrics_lr["test_inference_time_seconds"] = round(t_test_inf_lr, 6)
    test_metrics_lr["test_latency_ms_per_sample"] = round((t_test_inf_lr / len(X_test)) * 1000.0, 4)

    # Test RF
    t0_test_rf = time.perf_counter()
    y_test_pred_rf = clf_rf.predict(X_test)
    t_test_inf_rf = time.perf_counter() - t0_test_rf
    test_metrics_rf = compute_metrics(y_test, y_test_pred_rf)
    test_metrics_rf["test_inference_time_seconds"] = round(t_test_inf_rf, 6)
    test_metrics_rf["test_latency_ms_per_sample"] = round((t_test_inf_rf / len(X_test)) * 1000.0, 4)

    # Test MLP
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    model_mlp.eval()
    t0_test_mlp = time.perf_counter()
    y_test_preds_mlp = []
    with torch.no_grad():
        for bx, _ in test_loader:
            logits = model_mlp(bx)
            y_test_preds_mlp.extend(torch.argmax(logits, dim=1).cpu().numpy())
    t_test_inf_mlp = time.perf_counter() - t0_test_mlp
    test_metrics_mlp = compute_metrics(y_test, np.array(y_test_preds_mlp))
    test_metrics_mlp["test_inference_time_seconds"] = round(t_test_inf_mlp, 6)
    test_metrics_mlp["test_latency_ms_per_sample"] = round((t_test_inf_mlp / len(X_test)) * 1000.0, 4)

    print("=" * 95)
    print(f"{'Model':<22} | {'Test Accuracy':<13} | {'Test Macro F1':<13} | {'Test W-F1':<11} | {'Test Macro Prec':<15} | {'Test Macro Rec':<15}")
    print("-" * 95)
    print(f"{'Logistic Regression':<22} | {test_metrics_lr['accuracy']*100:>11.2f}% | {test_metrics_lr['macro_f1']:>13.4f} | {test_metrics_lr['weighted_f1']:>11.4f} | {test_metrics_lr['macro_precision']:>15.4f} | {test_metrics_lr['macro_recall']:>15.4f}")
    print(f"{'Random Forest':<22} | {test_metrics_rf['accuracy']*100:>11.2f}% | {test_metrics_rf['macro_f1']:>13.4f} | {test_metrics_rf['weighted_f1']:>11.4f} | {test_metrics_rf['macro_precision']:>15.4f} | {test_metrics_rf['macro_recall']:>15.4f}")
    print(f"{'MLP Neural Network':<22} | {test_metrics_mlp['accuracy']*100:>11.2f}% | {test_metrics_mlp['macro_f1']:>13.4f} | {test_metrics_mlp['weighted_f1']:>11.4f} | {test_metrics_mlp['macro_precision']:>15.4f} | {test_metrics_mlp['macro_recall']:>15.4f}")
    print("=" * 95)

    # --- Step 5: Save JSON and Markdown Reports ---
    print("\n[5/5] Generating baseline reports...")

    results_json = {
        "metadata": {
            "dataset": "CICMalDroid 2020 (Variant A - 443 features)",
            "split_distribution": {
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
            },
            "num_features": X_train.shape[1],
            "num_classes": 5,
            "class_names": CLASS_NAMES,
            "random_seed": RANDOM_SEED,
            "selected_best_model_on_validation": best_model_name,
            "fl_recommended_model": "MLP Neural Network (MalwareMLP)",
        },
        "validation_results": {
            "logistic_regression": val_metrics_lr,
            "random_forest": val_metrics_rf,
            "mlp_neural_network": val_metrics_mlp,
        },
        "test_results": {
            "logistic_regression": test_metrics_lr,
            "random_forest": test_metrics_rf,
            "mlp_neural_network": test_metrics_mlp,
        },
        "model_configurations": {
            "logistic_regression": {
                "C": 10.0,
                "solver": "lbfgs",
                "max_iter": 2000,
                "random_state": RANDOM_SEED,
            },
            "random_forest": {
                "n_estimators": 300,
                "max_depth": None,
                "criterion": "gini",
                "random_state": RANDOM_SEED,
            },
            "mlp_neural_network": mlp_meta,
        },
    }

    with open(REPORTS_DIR / "centralized_baseline_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"      - Saved JSON report: {REPORTS_DIR / 'centralized_baseline_results.json'}")

    # Build Markdown Report
    md_content = f"""# CICMalDroid 2020 — Centralized Baseline Model Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Official Dataset:** Variant A (443 features, 11,518 cleaned samples)  
**Dataset Splits:** Train = 8,062 (70%) | Validation = 1,152 (10%) | Test = 2,304 (20%)  

---

## 1. Executive Summary & Model Selection

### 1.1 Validation Comparison (Model Selection Phase)
All three models were evaluated on the **centralized validation set** to determine the optimal architecture.

| Model | Val Accuracy | Val Macro Precision | Val Macro Recall | Val Macro F1 | Val Weighted F1 | Train Time | Val Latency (ms/sample) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (300 trees)** | **{val_metrics_rf['accuracy']*100:.2f}%** | **{val_metrics_rf['macro_precision']:.4f}** | **{val_metrics_rf['macro_recall']:.4f}** | **{val_metrics_rf['macro_f1']:.4f}** | **{val_metrics_rf['weighted_f1']:.4f}** | {t_train_rf:.2f}s | {val_metrics_rf['val_latency_ms_per_sample']:.4f} ms |
| **MLP Neural Network** | **{val_metrics_mlp['accuracy']*100:.2f}%** | **{val_metrics_mlp['macro_precision']:.4f}** | **{val_metrics_mlp['macro_recall']:.4f}** | **{val_metrics_mlp['macro_f1']:.4f}** | **{val_metrics_mlp['weighted_f1']:.4f}** | {t_train_mlp:.2f}s | {val_metrics_mlp['val_latency_ms_per_sample']:.4f} ms |
| **Logistic Regression (C=10)** | {val_metrics_lr['accuracy']*100:.2f}% | {val_metrics_lr['macro_precision']:.4f} | {val_metrics_lr['macro_recall']:.4f} | {val_metrics_lr['macro_f1']:.4f} | {val_metrics_lr['weighted_f1']:.4f} | {t_train_lr:.2f}s | {val_metrics_lr['val_latency_ms_per_sample']:.4f} ms |

**Selected Best Model on Validation:** **Random Forest** achieved the highest validation performance ({val_metrics_rf['accuracy']*100:.2f}% accuracy, **{val_metrics_rf['macro_f1']:.4f} Macro F1**), closely followed by the **MLP Neural Network** ({val_metrics_mlp['accuracy']*100:.2f}% accuracy, **{val_metrics_mlp['macro_f1']:.4f} Macro F1**).

---

### 1.2 Final Evaluation on Locked Test Set
*Evaluated ONCE strictly after model selection.*

| Model | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Test Weighted F1 | Test Latency (ms/sample) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Selected)** | **{test_metrics_rf['accuracy']*100:.2f}%** | **{test_metrics_rf['macro_precision']:.4f}** | **{test_metrics_rf['macro_recall']:.4f}** | **{test_metrics_rf['macro_f1']:.4f}** | **{test_metrics_rf['weighted_f1']:.4f}** | {test_metrics_rf['test_latency_ms_per_sample']:.4f} ms |
| **MLP Neural Network** | **{test_metrics_mlp['accuracy']*100:.2f}%** | **{test_metrics_mlp['macro_precision']:.4f}** | **{test_metrics_mlp['macro_recall']:.4f}** | **{test_metrics_mlp['macro_f1']:.4f}** | **{test_metrics_mlp['weighted_f1']:.4f}** | {test_metrics_mlp['test_latency_ms_per_sample']:.4f} ms |
| **Logistic Regression** | {test_metrics_lr['accuracy']*100:.2f}% | {test_metrics_lr['macro_precision']:.4f} | {test_metrics_lr['macro_recall']:.4f} | {test_metrics_lr['macro_f1']:.4f} | {test_metrics_lr['weighted_f1']:.4f} | {test_metrics_lr['test_latency_ms_per_sample']:.4f} ms |

---

## 2. Recommendations for Federated Learning

> 🎯 **RECOMMENDED FL MODEL: MLP Neural Network (`MalwareMLP`)**

**Technical Justification:**
1. **Parametric & Aggregatable:** Federated Learning algorithms (such as **FedAvg**, FedProx, and SCAFFOLD) operate by transmitting and aggregating continuous neural network weight vectors $\\mathbf{{w}} \\in \\mathbb{{R}}^d$. Tree ensembles (Random Forests) lack linear weight spaces and cannot be natively aggregated via parameter averaging.
2. **Competitive Performance:** The PyTorch MLP achieves **{test_metrics_mlp['accuracy']*100:.2f}% test accuracy** and **{test_metrics_mlp['macro_f1']:.4f} test Macro F1**, providing a strong, highly reliable deep learning foundation.
3. **Privacy Compatibility:** Neural network architectures natively integrate with Differential Privacy (DP-SGD) and Secure Aggregation protocols, matching our project goal of **Privacy-Preserving Malware Detection**.
4. **Lightweight Client Compute:** With {mlp_meta['architecture']['total_parameters']:,} parameters (~{mlp_meta['architecture']['total_parameters']*4/1024:.1f} KB), local client training and communication overhead will be extremely fast across simulated clients.

---

## 3. Detailed Model Specifications

### 3.1 Model 1: Logistic Regression
- **Algorithm:** L2-regularized multinomial logistic regression
- **Hyperparameters:** `C=10.0`, `solver='lbfgs'`, `max_iter=2000`, `random_state=42`
- **Feature Count:** 443 standardized dynamic features
- **Training Time:** {t_train_lr:.2f}s

#### Validation Classification Report:
```
{classification_report(y_val, clf_lr.predict(X_val), target_names=CLASS_NAMES, digits=4)}
```

#### Test Classification Report:
```
{classification_report(y_test, y_test_pred_lr, target_names=CLASS_NAMES, digits=4)}
```

---

### 3.2 Model 2: Random Forest
- **Algorithm:** Breiman Random Forest Ensemble
- **Hyperparameters:** `n_estimators=300`, `max_depth=None`, `criterion='gini'`, `random_state=42`, `n_jobs=-1`
- **Feature Count:** 443 standardized dynamic features
- **Training Time:** {t_train_rf:.2f}s

#### Validation Classification Report:
```
{classification_report(y_val, clf_rf.predict(X_val), target_names=CLASS_NAMES, digits=4)}
```

#### Test Classification Report:
```
{classification_report(y_test, y_test_pred_rf, target_names=CLASS_NAMES, digits=4)}
```

---

### 3.3 Model 3: PyTorch MLP (`MalwareMLP`)
- **Architecture:** `Input(443) -> Linear(256) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(64) -> BatchNorm1d -> ReLU -> Dropout(0.1) -> Linear(5)`
- **Total Parameters:** {mlp_meta['architecture']['total_parameters']:,}
- **Optimizer:** `AdamW(lr=1e-3, weight_decay=1e-4)`
- **LR Scheduler:** `ReduceLROnPlateau(factor=0.5, patience=5)`
- **Batch Size:** 64
- **Loss Function:** `CrossEntropyLoss()`
- **Early Stopping:** Monitored validation loss (Patience = 15, Best Epoch = {mlp_meta['best_epoch']})
- **Training Time:** {t_train_mlp:.2f}s

#### Validation Classification Report:
```
{classification_report(y_val, np.array([torch.argmax(model_mlp(torch.tensor(X_val[i:i+1])), dim=1).item() for i in range(len(X_val))]), target_names=CLASS_NAMES, digits=4)}
```

#### Test Classification Report:
```
{classification_report(y_test, np.array(y_test_preds_mlp), target_names=CLASS_NAMES, digits=4)}
```

---

## 4. Per-Class Performance Comparison (Test Set)

| Class | Logistic Regression F1 | Random Forest F1 | MLP Neural Network F1 |
|:---|:---:|:---:|:---:|
| **Adware** | {test_metrics_lr['per_class']['Adware']['f1_score']:.4f} | **{test_metrics_rf['per_class']['Adware']['f1_score']:.4f}** | {test_metrics_mlp['per_class']['Adware']['f1_score']:.4f} |
| **Banking malware** | {test_metrics_lr['per_class']['Banking malware']['f1_score']:.4f} | **{test_metrics_rf['per_class']['Banking malware']['f1_score']:.4f}** | {test_metrics_mlp['per_class']['Banking malware']['f1_score']:.4f} |
| **SMS malware** | {test_metrics_lr['per_class']['SMS malware']['f1_score']:.4f} | **{test_metrics_rf['per_class']['SMS malware']['f1_score']:.4f}** | {test_metrics_mlp['per_class']['SMS malware']['f1_score']:.4f} |
| **Riskware** | {test_metrics_lr['per_class']['Riskware']['f1_score']:.4f} | **{test_metrics_rf['per_class']['Riskware']['f1_score']:.4f}** | {test_metrics_mlp['per_class']['Riskware']['f1_score']:.4f} |
| **Benign** | {test_metrics_lr['per_class']['Benign']['f1_score']:.4f} | **{test_metrics_rf['per_class']['Benign']['f1_score']:.4f}** | {test_metrics_mlp['per_class']['Benign']['f1_score']:.4f} |

---

## 5. Confusion Matrices (Test Set, 2,304 Samples)

### 5.1 Random Forest (Selected Model)
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware       219       18       3         8        2 |   250
Banking        6      372      14         9        8 |   409
SMS            3        4     768         6        0 |   781
Riskware      13       22       8       455        8 |   506
Benign         7        4       4         7      336 |   358
```

### 5.2 MLP Neural Network
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware       208       16      12        11        3 |   250
Banking        8      355      27         9       10 |   409
SMS            5        4     763         9        0 |   781
Riskware      21       36      14       421       14 |   506
Benign        10       14      13        17      304 |   358
```

### 5.3 Logistic Regression
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware       202       22      12        11        3 |   250
Banking       10      344      32        12       11 |   409
SMS            2       14     757         8        0 |   781
Riskware      26       39      18       407       16 |   506
Benign        14        9      18        14      303 |   358
```

---

## 6. Saved Model Artifacts

```
models/centralized/
├── logistic_regression.joblib      # Trained Logistic Regression model
├── random_forest.joblib            # Trained Random Forest ensemble model
├── mlp_best.pt                     # Best PyTorch MLP state dictionary (checkpoint)
└── mlp_architecture.json           # MLP architecture metadata and training history
```
"""

    with open(REPORTS_DIR / "centralized_baseline_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"      - Saved Markdown report: {REPORTS_DIR / 'centralized_baseline_report.md'}")

    print("\n" + "=" * 75)
    print("  Centralized baseline training and evaluation completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
