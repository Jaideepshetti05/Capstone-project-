"""
Visualization module for Federated Learning experiments.
Generates publication-quality figures for:
1. Convergence curves (Loss, Accuracy, Macro F1 across communication rounds).
2. Non-IID client class distributions (stacked bar charts).
3. Test set confusion matrix heatmaps.
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List


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
    alpha_str = f"Dirichlet α = {results['metadata'].get('dirichlet_alpha', '1.0')}" if "dirichlet" in exp_name else "Stratified IID"
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
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

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
