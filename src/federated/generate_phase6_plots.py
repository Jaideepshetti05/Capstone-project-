"""
Generate all publication-quality comparison plots for Phase 6 Privacy-Preserving Federated Learning.
"""

import json
import pathlib
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / "results" / "federated"
REPORTS_DIR = PROJECT_ROOT / "reports" / "federated"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "variant_a"
MODELS_DIR = PROJECT_ROOT / "models" / "federated"

from src.models.mlp import MalwareMLP
from src.federated.evaluation import evaluate_model, DEFAULT_CLASS_NAMES

def main():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

    # Load datasets for inference latency measurement
    X_test = pd.read_csv(DATA_DIR / "X_test.csv").values.astype(np.float32)
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel().astype(np.int64)

    exp_ids = [
        "dp_low_noise",
        "dp_med_noise",
        "dp_high_noise",
        "secagg_only",
        "dp_secagg",
        "noniid_dp_secagg"
    ]

    phase6_results = {}
    inference_latencies = {}

    for eid in exp_ids:
        metrics_file = RESULTS_DIR / f"{eid}_metrics.json"
        with open(metrics_file, "r") as f:
            data = json.load(f)
            phase6_results[eid] = data

        # Load best model and measure inference latency over repeated runs on test set
        model_path = MODELS_DIR / eid / "best_model.pt"
        model = MalwareMLP(in_features=443, num_classes=5, hidden_dims=[256, 128, 64], dropout_rate=0.2)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        t_samples = []
        with torch.no_grad():
            x_t = torch.tensor(X_test, dtype=torch.float32)
            # warmup
            for _ in range(5):
                _ = model(x_t)
            for _ in range(20):
                t0 = time.perf_counter()
                _ = model(x_t)
                t_samples.append((time.perf_counter() - t0) * 1000.0) # ms for 2304 samples
        
        mean_lat_ms = float(np.mean(t_samples))
        per_sample_us = (mean_lat_ms / len(X_test)) * 1000.0
        inference_latencies[eid] = {
            "batch_latency_ms": round(mean_lat_ms, 3),
            "per_sample_latency_us": round(per_sample_us, 3)
        }

    # Also load Phase 5 baselines
    with open(RESULTS_DIR / "iid_fedavg_metrics.json") as f:
        phase5_iid = json.load(f)
    with open(RESULTS_DIR / "dirichlet_a01_metrics.json") as f:
        phase5_a01 = json.load(f)
    with open(RESULTS_DIR / "fedprox_a01_mu001_metrics.json") as f:
        phase5_fedprox = json.load(f)

    # -------------------------------------------------------------
    # PLOT 1: Privacy-Utility Tradeoff (Test Accuracy & Macro F1 vs Epsilon)
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    eps_vals = [
        phase6_results["dp_high_noise"]["metadata"]["final_epsilon"], # 8.71
        phase6_results["dp_med_noise"]["metadata"]["final_epsilon"],  # 25.51
        phase6_results["dp_low_noise"]["metadata"]["final_epsilon"],  # 49.99
    ]
    acc_dp = [
        phase6_results["dp_high_noise"]["locked_test_metrics"]["accuracy"] * 100.0,
        phase6_results["dp_med_noise"]["locked_test_metrics"]["accuracy"] * 100.0,
        phase6_results["dp_low_noise"]["locked_test_metrics"]["accuracy"] * 100.0,
    ]
    f1_dp = [
        phase6_results["dp_high_noise"]["locked_test_metrics"]["macro_f1"] * 100.0,
        phase6_results["dp_med_noise"]["locked_test_metrics"]["macro_f1"] * 100.0,
        phase6_results["dp_low_noise"]["locked_test_metrics"]["macro_f1"] * 100.0,
    ]
    sigmas = ["σ=2.0 (High)", "σ=1.0 (Med)", "σ=0.5 (Low)"]

    ax1.plot(eps_vals, acc_dp, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Test Accuracy (%)")
    ax1.plot(eps_vals, f1_dp, "s--", color="#2ca02c", linewidth=2.5, markersize=8, label="Test Macro F1 (x100)")
    
    # Baseline non-private IID FedAvg
    iid_acc = phase5_iid["locked_test_metrics"]["accuracy"] * 100.0
    iid_f1 = phase5_iid["locked_test_metrics"]["macro_f1"] * 100.0
    ax1.axhline(iid_acc, color="#d62728", linestyle=":", linewidth=1.8, label=f"Non-Private IID FedAvg Acc ({iid_acc:.2f}%)")
    ax1.axhline(iid_f1, color="#9467bd", linestyle=":", linewidth=1.8, label=f"Non-Private IID FedAvg F1 ({iid_f1:.2f}%)")

    for i, txt in enumerate(sigmas):
        ax1.annotate(f"{txt}\n(ε={eps_vals[i]:.2f})", (eps_vals[i], acc_dp[i]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax1.set_xlabel("Privacy Budget spent ε (at δ = 10⁻⁵, lower is stronger privacy)", fontweight="bold")
    ax1.set_ylabel("Performance (%)", fontweight="bold")
    ax1.set_title("Privacy–Utility Tradeoff: Accuracy & Macro F1 vs. Privacy Loss ε", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="lower right", framealpha=0.95)
    ax1.set_ylim(50, 95)
    plt.tight_layout()
    p1 = REPORTS_DIR / "phase6_privacy_utility_tradeoff.png"
    plt.savefig(p1, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # PLOT 2: Epsilon vs Round Progression
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    rounds = list(range(1, 51))
    
    eps_low = [r["epsilon"] for r in phase6_results["dp_low_noise"]["round_by_round_history"]]
    eps_med = [r["epsilon"] for r in phase6_results["dp_med_noise"]["round_by_round_history"]]
    eps_high = [r["epsilon"] for r in phase6_results["dp_high_noise"]["round_by_round_history"]]

    ax.plot(rounds, eps_low, color="#d9534f", linewidth=2.2, label="Low Noise (σ = 0.5) → Final ε = 49.99")
    ax.plot(rounds, eps_med, color="#f0ad4e", linewidth=2.2, label="Medium Noise (σ = 1.0) → Final ε = 25.51")
    ax.plot(rounds, eps_high, color="#5cb85c", linewidth=2.2, label="High Noise (σ = 2.0) → Final ε = 8.71")

    ax.set_xlabel("Communication Rounds (T)", fontweight="bold")
    ax.set_ylabel("Cumulative Privacy Loss ε (at δ = 10⁻⁵)", fontweight="bold")
    ax.set_title("RDP Privacy Accounting: Cumulative Epsilon vs. Global Communication Rounds", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", framealpha=0.95)
    plt.tight_layout()
    p2 = REPORTS_DIR / "phase6_epsilon_vs_round.png"
    plt.savefig(p2, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # PLOT 3: Accuracy / F1 vs Sigma Noise Multiplier
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    sigma_vals = [0.0, 0.5, 1.0, 2.0]
    acc_by_sigma = [
        phase5_iid["locked_test_metrics"]["accuracy"] * 100.0,
        phase6_results["dp_low_noise"]["locked_test_metrics"]["accuracy"] * 100.0,
        phase6_results["dp_med_noise"]["locked_test_metrics"]["accuracy"] * 100.0,
        phase6_results["dp_high_noise"]["locked_test_metrics"]["accuracy"] * 100.0,
    ]
    f1_by_sigma = [
        phase5_iid["locked_test_metrics"]["macro_f1"] * 100.0,
        phase6_results["dp_low_noise"]["locked_test_metrics"]["macro_f1"] * 100.0,
        phase6_results["dp_med_noise"]["locked_test_metrics"]["macro_f1"] * 100.0,
        phase6_results["dp_high_noise"]["locked_test_metrics"]["macro_f1"] * 100.0,
    ]

    width = 0.35
    x = np.arange(len(sigma_vals))
    rects1 = ax.bar(x - width/2, acc_by_sigma, width, label="Test Accuracy (%)", color="#3498db", edgecolor="black", linewidth=0.5)
    rects2 = ax.bar(x + width/2, f1_by_sigma, width, label="Test Macro F1 (x100)", color="#2ecc71", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Gaussian Noise Multiplier σ (C = 1.0)", fontweight="bold")
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Model Utility Degradation vs. Injected Gaussian Noise Scale σ", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["σ = 0.0 (Plain)", "σ = 0.5 (Low)", "σ = 1.0 (Med)", "σ = 2.0 (High)"], fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.set_ylim(0, 100)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p3 = REPORTS_DIR / "phase6_accuracy_f1_vs_sigma.png"
    plt.savefig(p3, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # PLOT 4: Communication and Computation Overhead Comparison
    # -------------------------------------------------------------
    fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    labels = ["Vanilla FedAvg", "SecAgg Only", "DP-SGD (σ=0.5)", "DP-SGD (σ=1.0)", "DP + SecAgg", "Non-IID DP+SecAgg"]
    train_times = [
        phase5_iid["metadata"]["total_train_time_seconds"],
        phase6_results["secagg_only"]["metadata"]["total_train_time_seconds"],
        phase6_results["dp_low_noise"]["metadata"]["total_train_time_seconds"],
        phase6_results["dp_med_noise"]["metadata"]["total_train_time_seconds"],
        phase6_results["dp_secagg"]["metadata"]["total_train_time_seconds"],
        phase6_results["noniid_dp_secagg"]["metadata"]["total_train_time_seconds"],
    ]
    colors = ["#7f8c8d", "#2980b9", "#e67e22", "#d35400", "#c0392b", "#8e44ad"]

    ax_t.barh(labels, train_times, color=colors, edgecolor="black", linewidth=0.5)
    ax_t.set_xlabel("Total Training Time (seconds)", fontweight="bold")
    ax_t.set_title("Computation Time (50 Rounds)", fontsize=12, fontweight="bold")
    for idx, v in enumerate(train_times):
        ax_t.text(v + 20, idx, f"{v:.1f}s", va="center", fontsize=9, fontweight="bold")
    ax_t.set_xlim(0, max(train_times) * 1.15)

    # Communication Volume
    comm_vols = [595.23] * 6 # 10 clients * 50 rounds
    ax_c.barh(labels, comm_vols, color="#16a085", edgecolor="black", linewidth=0.5)
    ax_c.set_xlabel("Total Communication Volume (MB)", fontweight="bold")
    ax_c.set_title("Network Communication (50 Rounds)", fontsize=12, fontweight="bold")
    for idx, v in enumerate(comm_vols):
        ax_c.text(v + 10, idx, f"{v:.2f} MB", va="center", fontsize=9, fontweight="bold")
    ax_c.set_xlim(0, max(comm_vols) * 1.2)

    plt.suptitle("Computation & Communication Profile Across Federated Privacy Mechanisms", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p4 = REPORTS_DIR / "phase6_communication_computation_comparison.png"
    plt.savefig(p4, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # PLOT 5: DP Clipping Fraction & Gradient Norms Across Rounds
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    clip_frac_low = [r["clipping_fraction"] for r in phase6_results["dp_low_noise"]["round_by_round_history"]]
    clip_frac_med = [r["clipping_fraction"] for r in phase6_results["dp_med_noise"]["round_by_round_history"]]
    clip_frac_high = [r["clipping_fraction"] for r in phase6_results["dp_high_noise"]["round_by_round_history"]]
    
    grad_norm_low = [r["mean_grad_norm"] for r in phase6_results["dp_low_noise"]["round_by_round_history"]]
    grad_norm_med = [r["mean_grad_norm"] for r in phase6_results["dp_med_noise"]["round_by_round_history"]]
    grad_norm_high = [r["mean_grad_norm"] for r in phase6_results["dp_high_noise"]["round_by_round_history"]]

    ax1.plot(rounds, clip_frac_low, color="#e74c3c", linewidth=2, label="Clipping Fraction (σ=0.5)")
    ax1.plot(rounds, clip_frac_med, color="#e67e22", linewidth=2, label="Clipping Fraction (σ=1.0)")
    ax1.plot(rounds, clip_frac_high, color="#27ae60", linewidth=2, label="Clipping Fraction (σ=2.0)")
    ax1.set_xlabel("Global Communication Rounds", fontweight="bold")
    ax1.set_ylabel("Clipping Fraction (Fraction with ||g|| > C)", color="#333333", fontweight="bold")
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()
    ax2.plot(rounds, grad_norm_low, color="#e74c3c", linestyle=":", linewidth=1.5, label="Mean Grad Norm (σ=0.5)")
    ax2.plot(rounds, grad_norm_med, color="#e67e22", linestyle=":", linewidth=1.5, label="Mean Grad Norm (σ=1.0)")
    ax2.plot(rounds, grad_norm_high, color="#27ae60", linestyle=":", linewidth=1.5, label="Mean Grad Norm (σ=2.0)")
    ax2.set_ylabel("Mean Unclipped L2 Gradient Norm", color="#7f8c8d", fontweight="bold")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Clipping Threshold C = 1.0")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.95)

    plt.title("DP-SGD Gradient Dynamics: Clipping Fraction and L2 Norm Progression (C = 1.0)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    p5 = REPORTS_DIR / "phase6_dp_clipping_statistics.png"
    plt.savefig(p5, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # PLOT 6: Non-IID vs IID with DP + SecAgg Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    cats = ["Test Accuracy (%)", "Test Macro F1 (x100)", "Test Weighted F1 (x100)", "Macro Precision (x100)", "Macro Recall (x100)"]
    
    plain_iid = [
        phase5_iid["locked_test_metrics"]["accuracy"] * 100.0,
        phase5_iid["locked_test_metrics"]["macro_f1"] * 100.0,
        phase5_iid["locked_test_metrics"]["weighted_f1"] * 100.0,
        phase5_iid["locked_test_metrics"]["macro_precision"] * 100.0,
        phase5_iid["locked_test_metrics"]["macro_recall"] * 100.0,
    ]
    plain_noniid = [
        phase5_a01["locked_test_metrics"]["accuracy"] * 100.0,
        phase5_a01["locked_test_metrics"]["macro_f1"] * 100.0,
        phase5_a01["locked_test_metrics"]["weighted_f1"] * 100.0,
        phase5_a01["locked_test_metrics"]["macro_precision"] * 100.0,
        phase5_a01["locked_test_metrics"]["macro_recall"] * 100.0,
    ]
    dp_secagg_iid = [
        phase6_results["dp_secagg"]["locked_test_metrics"]["accuracy"] * 100.0,
        phase6_results["dp_secagg"]["locked_test_metrics"]["macro_f1"] * 100.0,
        phase6_results["dp_secagg"]["locked_test_metrics"]["weighted_f1"] * 100.0,
        phase6_results["dp_secagg"]["locked_test_metrics"]["macro_precision"] * 100.0,
        phase6_results["dp_secagg"]["locked_test_metrics"]["macro_recall"] * 100.0,
    ]
    dp_secagg_noniid = [
        phase6_results["noniid_dp_secagg"]["locked_test_metrics"]["accuracy"] * 100.0,
        phase6_results["noniid_dp_secagg"]["locked_test_metrics"]["macro_f1"] * 100.0,
        phase6_results["noniid_dp_secagg"]["locked_test_metrics"]["weighted_f1"] * 100.0,
        phase6_results["noniid_dp_secagg"]["locked_test_metrics"]["macro_precision"] * 100.0,
        phase6_results["noniid_dp_secagg"]["locked_test_metrics"]["macro_recall"] * 100.0,
    ]

    x = np.arange(len(cats))
    w = 0.20
    
    r1 = ax.bar(x - 1.5*w, plain_iid, w, label="Plain FedAvg (IID)", color="#3498db")
    r2 = ax.bar(x - 0.5*w, dp_secagg_iid, w, label="DP+SecAgg (IID, σ=1.0)", color="#2ecc71")
    r3 = ax.bar(x + 0.5*w, plain_noniid, w, label="Plain FedAvg (Non-IID α=0.1)", color="#e74c3c")
    r4 = ax.bar(x + 1.5*w, dp_secagg_noniid, w, label="DP+SecAgg (Non-IID α=0.1, σ=1.0)", color="#9b59b6")

    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Cross-Comparison: Impact of Data Heterogeneity (Dirichlet α=0.1) & Privacy (DP+SecAgg)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    p6 = REPORTS_DIR / "phase6_noniid_comparison.png"
    plt.savefig(p6, dpi=300)
    plt.close()

    # Save summary dictionary
    summary_out = {
        "phase6_experiments": phase6_results,
        "inference_latencies": inference_latencies,
        "plots_generated": [
            str(p1.name), str(p2.name), str(p3.name),
            str(p4.name), str(p5.name), str(p6.name)
        ]
    }
    with open(RESULTS_DIR / "phase6_comparative_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    print("Phase 6 comparative plots and summary generated successfully!")

if __name__ == "__main__":
    main()
