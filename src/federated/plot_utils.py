"""
Visualization module for Federated Learning experiments.
Generates publication-quality figures for:
1. Convergence curves (Loss, Accuracy, Macro F1 across communication rounds).
2. Non-IID client class distributions (stacked bar charts).
3. Test set confusion matrix heatmaps.
4. Multi-experiment comparative bar charts (FedAvg vs FedProx, Per-class F1).
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional


def generate_experiment_plots(
    results: Dict[str, Any],
    output_dir: pathlib.Path,
    class_names: List[str] = None,
) -> Dict[str, pathlib.Path]:
    """
    Generate and save convergence, class distribution, and confusion matrix plots.
    """
    if class_names is None:
        class_names = ["Adware", "Banking", "SMS", "Riskware", "Benign"]

    output_dir.mkdir(parents=True, exist_ok=True)
    exp_name = results["metadata"].get("experiment_name", "dirichlet_a10")
    history = results["round_by_round_history"]
    best_round = results["metadata"].get("best_round", 1)

    generated_plots = {}

    # Set style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

    # --- 1. Convergence Curve ---
    rounds = [r["round"] for r in history]
    losses = [r["val_loss"] for r in history]
    accuracies = [r["val_accuracy"] * 100.0 for r in history]
    macro_f1s = [r["val_macro_f1"] for r in history]

    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)

    color_loss = "#d9534f"
    color_acc = "#0275d8"
    color_f1 = "#5cb85c"

    ax1.set_xlabel("Global Communication Rounds", fontweight="bold")
    ax1.set_ylabel("Validation Loss", color=color_loss, fontweight="bold")
    line1 = ax1.plot(rounds, losses, color=color_loss, linewidth=2, label="Validation Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Accuracy (%) & Macro F1 (x100)", color="#333333", fontweight="bold")
    line2 = ax2.plot(rounds, accuracies, color=color_acc, linewidth=2, label="Validation Accuracy (%)")
    line3 = ax2.plot(rounds, [f * 100.0 for f in macro_f1s], color=color_f1, linewidth=2, linestyle="--", label="Validation Macro F1 (x100)")
    ax2.tick_params(axis="y", labelcolor="#333333")

    # Mark best round
    best_val_f1 = history[best_round - 1]["val_macro_f1"] * 100.0
    ax2.scatter([best_round], [best_val_f1], color="#f0ad4e", s=130, zorder=5, edgecolors="black", linewidth=1.5,
                label=f"Best Checkpoint (Round {best_round})")

    # Combine legends
    lines = line1 + line2 + line3 + [ax2.collections[0]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.95)

    title_str = f"Federated Learning Convergence ({results['metadata'].get('algorithm', 'FedAvg')} - {exp_name.replace('_', ' ').title()})"
    plt.title(title_str, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    conv_path = output_dir / f"{exp_name}_convergence.png"
    plt.savefig(conv_path, dpi=300)
    plt.close()
    generated_plots["convergence_plot"] = conv_path

    # --- 2. Client Class Distribution (Stacked Bar Chart) ---
    part = results["partition_validation"]["client_stats"]
    client_ids = [f"Client {int(k.replace('client_', '')):02d}" for k in part.keys()]
    num_clients = len(client_ids)
    palette = ["#4A90E2", "#E94E77", "#F5A623", "#7ED321", "#9013FE"]

    counts_matrix = np.zeros((len(class_names), num_clients))
    for col_idx, (c_key, c_val) in enumerate(part.items()):
        for row_idx in range(len(class_names)):
            counts_matrix[row_idx, col_idx] = c_val["class_counts"][row_idx]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    bottom = np.zeros(num_clients)

    for row_idx, cname in enumerate(class_names):
        values = counts_matrix[row_idx]
        ax.bar(client_ids, values, bottom=bottom, label=cname, color=palette[row_idx % len(palette)], width=0.65, edgecolor="black", linewidth=0.5)
        bottom += values

    # Annotate total samples on top of each bar
    for col_idx, total in enumerate(bottom):
        ax.text(col_idx, total + 15, f"{int(total)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Local Sample Count", fontweight="bold")
    ax.set_xlabel("Federated Clients", fontweight="bold")
    alpha_str = f"Dirichlet α = {results['metadata'].get('dirichlet_alpha', '1.0')}" if "dirichlet" in exp_name or "fedprox" in exp_name else "Stratified IID"
    ax.set_title(f"Client Data Partition & Label Skew ({alpha_str}, K=10 Clients, N=8,062 Samples)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(title="Malware Category", loc="upper right", framealpha=0.95)
    ax.set_ylim(0, max(bottom) * 1.15)
    plt.xticks(rotation=15)
    plt.tight_layout()

    dist_path = output_dir / f"{exp_name}_class_distribution.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    generated_plots["class_distribution_plot"] = dist_path

    # --- 3. Confusion Matrix Heatmap ---
    cm = np.array(results["locked_test_metrics"]["confusion_matrix"])
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-9)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=1,
        linecolor="white",
    )
    ax.set_xlabel("Predicted Class", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Class", fontweight="bold", labelpad=8)
    acc = results["locked_test_metrics"]["accuracy"] * 100.0
    macro_f1 = results["locked_test_metrics"]["macro_f1"]
    ax.set_title(f"Locked Test Set Confusion Matrix\n({exp_name.replace('_', ' ').title()} — Accuracy: {acc:.2f}%, Macro F1: {macro_f1:.4f})", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    cm_path = output_dir / f"{exp_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    generated_plots["confusion_matrix_plot"] = cm_path

    return generated_plots


def generate_multi_experiment_comparisons(output_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    """
    Generate publication-quality comparative plots across all 6 experimental conditions:
    1. FedAvg vs FedProx under Dirichlet a=0.1
    2. Per-class F1 scores across all 6 conditions
    3. Multi-experiment test accuracy and Macro F1 summary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

    plots = {}

    # --- 1. FedAvg vs FedProx (Dirichlet a=0.1) Comparison ---
    categories = ["Accuracy (%)", "Macro F1 (x100)", "Weighted F1 (x100)", "Macro Prec (x100)", "Macro Rec (x100)"]
    fedavg_a01 = [69.92, 58.36, 64.97, 57.56, 64.24]
    fedprox_a01 = [72.92, 61.38, 68.36, 60.03, 67.86]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    rects1 = ax.bar(x - width/2, fedavg_a01, width, label="FedAvg (α = 0.1)", color="#e74c3c", edgecolor="black", linewidth=0.5)
    rects2 = ax.bar(x + width/2, fedprox_a01, width, label="FedProx (α = 0.1, μ = 0.01)", color="#2ecc71", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Extreme Non-IID Optimization: FedAvg vs. FedProx (Dirichlet α = 0.1)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_ylim(0, 90)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    comp1_path = output_dir / "fedavg_vs_fedprox_a01_comparison.png"
    plt.savefig(comp1_path, dpi=300)
    plt.close()
    plots["fedavg_vs_fedprox_comparison"] = comp1_path

    # --- 2. Per-Class F1 Comparison Across All 6 Conditions ---
    classes = ["Adware", "Banking", "SMS", "Riskware", "Benign"]
    f1_centralized = [0.8596, 0.8878, 0.9634, 0.8812, 0.8863]
    f1_iid = [0.8407, 0.8659, 0.9575, 0.8809, 0.8815]
    f1_a10 = [0.7821, 0.8575, 0.9540, 0.8933, 0.9038]
    f1_a05 = [0.8373, 0.8645, 0.9544, 0.8746, 0.8919]
    f1_a01 = [0.5409, 0.7710, 0.8494, 0.7569, 0.0000]
    f1_prox = [0.5330, 0.8593, 0.8791, 0.7978, 0.0000]

    x = np.arange(len(classes))
    w = 0.13

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.bar(x - 2.5*w, f1_centralized, w, label="Centralized MLP", color="#34495e")
    ax.bar(x - 1.5*w, f1_iid, w, label="FedAvg IID", color="#3498db")
    ax.bar(x - 0.5*w, f1_a10, w, label="FedAvg α=1.0", color="#9b59b6")
    ax.bar(x + 0.5*w, f1_a05, w, label="FedAvg α=0.5", color="#f39c12")
    ax.bar(x + 1.5*w, f1_a01, w, label="FedAvg α=0.1", color="#e74c3c")
    ax.bar(x + 2.5*w, f1_prox, w, label="FedProx α=0.1", color="#2ecc71")

    ax.set_ylabel("Test F1-Score", fontweight="bold")
    ax.set_title("Per-Class Test F1-Score Across All 6 Experimental Baselines", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.set_ylim(0, 1.15)
    plt.tight_layout()

    comp2_path = output_dir / "per_class_f1_comparison.png"
    plt.savefig(comp2_path, dpi=300)
    plt.close()
    plots["per_class_f1_comparison"] = comp2_path

    # --- 3. Multi-Experiment Overall Benchmark Comparison ---
    exp_labels = [
        "1. Centralized MLP",
        "2. FedAvg (IID)",
        "3. FedAvg (α=1.0)",
        "4. FedAvg (α=0.5)",
        "5. FedAvg (α=0.1)",
        "6. FedProx (α=0.1)",
    ]
    test_accs = [91.28, 90.19, 89.80, 89.93, 69.92, 72.92]
    test_macro_f1s = [89.57, 88.53, 87.81, 88.45, 58.36, 61.38]

    x = np.arange(len(exp_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    r1 = ax.bar(x - width/2, test_accs, width, label="Test Accuracy (%)", color="#2980b9", edgecolor="black", linewidth=0.5)
    r2 = ax.bar(x + width/2, test_macro_f1s, width, label="Test Macro F1 (x100)", color="#27ae60", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Overall Test Performance Across All 6 Capstone Project Benchmarks", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, rotation=20, ha="right", fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.set_ylim(0, 105)

    for rect in r1 + r2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    comp3_path = output_dir / "multi_experiment_benchmark_comparison.png"
    plt.savefig(comp3_path, dpi=300)
    plt.close()
    plots["multi_experiment_benchmark"] = comp3_path

    return plots
