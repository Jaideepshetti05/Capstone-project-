"""
Publication-Quality Visualization Pipeline for Federated Malware Detection.
Generates comprehensive figures and charts from actual saved experimental results.
"""

import os
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "federated"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 13


def load_metric(exp_name: str):
    fpath = RESULTS_DIR / f"{exp_name}_metrics.json"
    if not fpath.exists():
        return None
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_centralized_vs_federated():
    """Figure 1: Centralized vs Federated Baselines Comparison."""
    cent_path = REPORTS_DIR / "centralized_baseline_results.json"
    if not cent_path.exists():
        return
    with open(cent_path, "r") as f:
        cent = json.load(f)

    fed_iid = load_metric("iid_fedavg") or load_metric("iid")
    if not fed_iid:
        return

    models = ["Logistic Reg.", "Random Forest", "Centralized MLP", "FedAvg (IID)"]
    accs = [
        cent["logistic_regression"]["test"]["accuracy"] * 100,
        cent["random_forest"]["test"]["accuracy"] * 100,
        cent["mlp"]["test"]["accuracy"] * 100,
        fed_iid["locked_test_metrics"]["accuracy"] * 100,
    ]
    f1s = [
        cent["logistic_regression"]["test"]["macro_f1"],
        cent["random_forest"]["test"]["macro_f1"],
        cent["mlp"]["test"]["macro_f1"],
        fed_iid["locked_test_metrics"]["macro_f1"],
    ]

    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    x = np.arange(len(models))
    width = 0.35

    rects1 = ax1.bar(x - width/2, accs, width, label="Test Accuracy (%)", color="#1f77b4", alpha=0.9)
    ax1.set_ylabel("Accuracy (%)", color="#1f77b4", fontweight="bold")
    ax1.set_ylim(80, 100)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, f1s, width, label="Macro F1 Score", color="#ff7f0e", alpha=0.9)
    ax2.set_ylabel("Macro F1", color="#ff7f0e", fontweight="bold")
    ax2.set_ylim(0.80, 1.0)
    ax2.grid(False)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_title("Centralized vs Federated Detection Performance (CICMalDroid 2020)", pad=15)

    # Annotations
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig1_centralized_vs_federated.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_noniid_convergence():
    """Figure 2: Non-IID Dirichlet Skew Convergence & FedProx Mitigation."""
    exps = {
        "IID FedAvg": "iid_fedavg",
        "Dirichlet (alpha=1.0)": "dirichlet_a10",
        "Dirichlet (alpha=0.5)": "dirichlet_a05",
        "Dirichlet (alpha=0.1)": "dirichlet_a01",
        "FedProx (alpha=0.1, mu=0.01)": "fedprox_a01_mu001",
    }
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    for (label, ename), col in zip(exps.items(), colors):
        d = load_metric(ename)
        if not d:
            continue
        rounds = [r["round"] for r in d["round_by_round_history"]]
        f1s = [r["val_macro_f1"] for r in d["round_by_round_history"]]
        losses = [r["val_loss"] for r in d["round_by_round_history"]]

        ax1.plot(rounds, f1s, label=label, color=col, linewidth=1.8)
        ax2.plot(rounds, losses, label=label, color=col, linewidth=1.8)

    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Validation Macro F1")
    ax1.set_title("Validation Macro F1 Convergence")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0.4, 0.95)

    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Validation Cross-Entropy Loss")
    ax2.set_title("Validation Loss Trajectory")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0.2, 1.8)

    fig.suptitle("Impact of Non-IID Data Heterogeneity & FedProx Regularization", y=1.02)
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig2_noniid_convergence.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_privacy_utility_tradeoff():
    """Figure 3: Privacy Budget (epsilon) vs Accuracy & Macro F1 Pareto Curve."""
    dp_exps = [
        ("High Privacy (sigma=2.0)", "dp_high_noise"),
        ("Moderate (sigma=1.0)", "dp_med_noise"),
        ("Low Privacy (sigma=0.5)", "dp_low_noise"),
    ]

    eps_vals = []
    acc_vals = []
    f1_vals = []
    labels = []

    for name, ename in dp_exps:
        d = load_metric(ename)
        if d:
            eps_vals.append(d["privacy_accounting"]["epsilon"])
            acc_vals.append(d["locked_test_metrics"]["accuracy"] * 100)
            f1_vals.append(d["locked_test_metrics"]["macro_f1"])
            labels.append(name)

    # Also plain FedAvg (eps = Inf)
    fed_iid = load_metric("iid_fedavg") or load_metric("iid")
    if fed_iid:
        labels.append("Plain FedAvg (No DP)")
        acc_vals.append(fed_iid["locked_test_metrics"]["accuracy"] * 100)
        f1_vals.append(fed_iid["locked_test_metrics"]["macro_f1"])
        eps_vals.append(70.0)  # surrogate marker for plot

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(eps_vals[:-1], acc_vals[:-1], "o-", color="#1f77b4", linewidth=2.0, markersize=8, label="Test Accuracy (%)")
    ax.plot(eps_vals[:-1], [f * 100 for f in f1_vals[:-1]], "s--", color="#ff7f0e", linewidth=2.0, markersize=8, label="Test Macro F1 (x100)")

    for i in range(len(eps_vals) - 1):
        ax.annotate(f"{labels[i]}\n(eps={eps_vals[i]:.2f}, Acc={acc_vals[i]:.1f}%)",
                    (eps_vals[i], acc_vals[i]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax.set_xlabel(r"Differential Privacy Budget Spent $\varepsilon$ ($\delta=10^{-5}$)")
    ax.set_ylabel("Utility Metric (%)")
    ax.set_title(r"Privacy–Utility Pareto Tradeoff under DP-SGD ($\sigma \in \{0.5, 1.0, 2.0\}$)", pad=15)
    ax.legend(loc="lower right")
    ax.set_ylim(50, 95)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig3_privacy_utility_tradeoff.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_robustness_comparison():
    """Figure 4: Byzantine Robustness under Poisoning Attacks (FedAvg vs Trimmed Mean vs Median)."""
    attacks = ["Label-Flipping (Malware -> Benign)", "Weight Poisoning (Gaussian Noise)"]
    algos = ["FedAvg (No Defense)", "Trimmed Mean (beta=0.2)", "Coordinate Median"]

    # Load actual metrics if available
    exp_matrix = [
        ["attack_labelflip_fedavg", "attack_labelflip_trimmed_mean", "attack_labelflip_median"],
        ["attack_weightpoison_fedavg", "attack_weightpoison_trimmed_mean", "attack_weightpoison_median"],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    for ax, atk_name, exps in zip([ax1, ax2], attacks, exp_matrix):
        accs = []
        f1s = []
        for ename in exps:
            d = load_metric(ename)
            if d:
                accs.append(d["locked_test_metrics"]["accuracy"] * 100)
                f1s.append(d["locked_test_metrics"]["macro_f1"])
            else:
                accs.append(0.0)
                f1s.append(0.0)

        x = np.arange(len(algos))
        w = 0.35
        rects1 = ax.bar(x - w/2, accs, w, label="Test Accuracy (%)", color="#1f77b4", alpha=0.9)
        rects2 = ax.bar(x + w/2, [f * 100 for f in f1s], w, label="Macro F1 (x100)", color="#2ca02c", alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Metric (%)")
        ax.set_title(f"{atk_name}\n(20% Malicious Clients)", pad=10)
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left")

        for rect in rects1:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        for rect in rects2:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h/100:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Byzantine Attack Impact & Robust Aggregation Defenses", y=1.03)
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig4_byzantine_robustness.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_all_plots():
    print("Generating comprehensive figures...")
    plot_centralized_vs_federated()
    plot_noniid_convergence()
    plot_privacy_utility_tradeoff()
    plot_robustness_comparison()
    print("All figures successfully created in reports/figures/")


if __name__ == "__main__":
    generate_all_plots()
