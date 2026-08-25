"""
Federated Learning Experiment Orchestrator.
Manages partitioning, client lifecycle, communication rounds, FedAvg / FedProx / DP / SecAgg aggregation,
round-by-round validation tracking, privacy accounting, and final locked test evaluation.
"""

import os
import copy
import json
import time
import pathlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.mlp import MalwareMLP
from .partition import create_iid_partition, create_dirichlet_partition, validate_partition
from .client import FederatedClient
from .fedavg import aggregate_fedavg
from .evaluation import evaluate_model, DEFAULT_CLASS_NAMES
from .utils import set_seed, get_system_metadata, compute_communication_cost
from .plot_utils import generate_experiment_plots
from .dp_accountant import RDPPrivacyAccountant
from .secure_aggregation import SecureAggregationEngine, verify_secure_aggregation_cancellation


class FederatedExperiment:
    """
    Federated Learning Experiment Manager.
    Supports FedAvg, FedProx, Differential Privacy (DP-SGD), and Secure Aggregation (SecAgg).
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        config: Dict[str, Any],
        output_dir: Optional[pathlib.Path] = None,
        report_dir: Optional[pathlib.Path] = None,
        result_dir: Optional[pathlib.Path] = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.config = config

        self.algorithm = config.get("algorithm", "FedAvg")
        self.mu = float(config.get("mu", 0.0))
        self.dp_enabled = bool(config.get("dp_enabled", False))
        self.clip_norm = float(config.get("clip_norm", 1.0))
        self.noise_multiplier = float(config.get("noise_multiplier", 1.0))
        self.target_delta = float(config.get("target_delta", 1e-5))
        self.secagg_enabled = bool(config.get("secagg_enabled", False))

        default_name = config.get("experiment_name", "dirichlet_a01")
        self.output_dir = output_dir or pathlib.Path(f"models/federated/{default_name}")
        self.report_dir = report_dir or pathlib.Path("reports/federated")
        self.result_dir = result_dir or pathlib.Path("results/federated")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.seed = config.get("seed", 42)
        set_seed(self.seed)

        self.num_clients = config.get("num_clients", 10)
        self.client_fraction = config.get("client_fraction", 1.0)
        self.num_rounds = config.get("num_rounds", 50)
        self.local_epochs = config.get("local_epochs", 2)
        self.batch_size = config.get("batch_size", 64)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.weight_decay = config.get("weight_decay", 1e-4)
        self.partition_type = config.get("partition_type", "dirichlet")
        self.dirichlet_alpha = config.get("dirichlet_alpha", 0.1)
        self.device = config.get("device", "cpu")

        # Privacy Accountant initialization
        if self.dp_enabled:
            self.accountant = RDPPrivacyAccountant(target_delta=self.target_delta)
        else:
            self.accountant = None

        # Secure Aggregation Engine initialization
        if self.secagg_enabled:
            self.secagg_engine = SecureAggregationEngine(
                num_clients=self.num_clients,
                mask_variance=10.0,
                seed=self.seed,
            )
        else:
            self.secagg_engine = None

    def run_pre_training_assertions(self) -> None:
        """
        Automated data validation checks before training begins.
        Stops execution with AssertionError if any integrity violation is detected.
        """
        print("    [Validation 1/5] Checking dataset shapes and dimensions...")
        assert self.X_train.shape == (8062, 443), f"X_train shape error: {self.X_train.shape}"
        assert len(self.y_train) == 8062, f"y_train length error: {len(self.y_train)}"
        assert self.X_val.shape == (1152, 443), f"X_val shape error: {self.X_val.shape}"
        assert len(self.y_val) == 1152, f"y_val length error: {len(self.y_val)}"
        assert self.X_test.shape == (2304, 443), f"X_test shape error: {self.X_test.shape}"
        assert len(self.y_test) == 2304, f"y_test length error: {len(self.y_test)}"

        print("    [Validation 2/5] Checking for NaN and Infinite values...")
        assert not np.isnan(self.X_train).any(), "NaN found in X_train"
        assert not np.isnan(self.X_val).any(), "NaN found in X_val"
        assert not np.isnan(self.X_test).any(), "NaN found in X_test"
        assert not np.isinf(self.X_train).any(), "Inf found in X_train"
        assert not np.isinf(self.X_val).any(), "Inf found in X_val"
        assert not np.isinf(self.X_test).any(), "Inf found in X_test"

        print("    [Validation 3/5] Checking target label bounds [0..4]...")
        assert set(np.unique(self.y_train)).issubset({0, 1, 2, 3, 4}), f"Invalid y_train labels: {np.unique(self.y_train)}"
        assert set(np.unique(self.y_val)).issubset({0, 1, 2, 3, 4}), f"Invalid y_val labels: {np.unique(self.y_val)}"
        assert set(np.unique(self.y_test)).issubset({0, 1, 2, 3, 4}), f"Invalid y_test labels: {np.unique(self.y_test)}"

        print("    [Validation 4/5] Checking class representation...")
        assert len(np.unique(self.y_train)) == 5, "Missing classes in y_train"
        assert len(np.unique(self.y_val)) == 5, "Missing classes in y_val"
        assert len(np.unique(self.y_test)) == 5, "Missing classes in y_test"

        print("    [Validation 5/5] All pre-training data integrity assertions passed successfully.")

    def run(self) -> Dict[str, Any]:
        """
        Execute full federated learning training workflow.
        """
        exp_name = self.config.get("experiment_name", "dirichlet_a01")
        privacy_tags = []
        if self.dp_enabled:
            privacy_tags.append(f"DP(sigma={self.noise_multiplier}, C={self.clip_norm})")
        if self.secagg_enabled:
            privacy_tags.append("SecAgg")
        if self.mu > 0.0:
            privacy_tags.append(f"FedProx(mu={self.mu})")
        tag_str = ", ".join(privacy_tags) if privacy_tags else "Plaintext FedAvg"

        print("\n" + "=" * 80)
        print(f"  Federated Learning Experiment: {exp_name} [{tag_str}]")
        print("=" * 80)

        # 1. Run assertions
        print("\n[Step 1/6] Running automated data validation checks...")
        self.run_pre_training_assertions()

        # 2. Partition data
        print(f"\n[Step 2/6] Generating {self.partition_type.upper()} (alpha={self.dirichlet_alpha}) partition across {self.num_clients} clients...")
        if self.partition_type == "iid":
            partition = create_iid_partition(self.y_train, num_clients=self.num_clients, seed=self.seed)
        elif self.partition_type == "dirichlet":
            alpha = self.dirichlet_alpha if self.dirichlet_alpha is not None else 0.1
            partition = create_dirichlet_partition(self.y_train, num_clients=self.num_clients, alpha=alpha, seed=self.seed)
        else:
            raise ValueError(f"Unknown partition type: {self.partition_type}")

        # Validate partition
        partition_validation = validate_partition(
            partition,
            self.y_train,
            total_expected=len(self.X_train),
            class_names=DEFAULT_CLASS_NAMES,
        )
        print(f"    - Partition validated: {partition_validation['total_samples']} samples assigned with 0 duplication.")
        het = partition_validation["heterogeneity_summary"]
        print(f"    - Client size range: [{het['min_client_size']}..{het['max_client_size']}] samples (Mean: {het['mean_client_size']}, Std: {het['std_client_size']})")
        print(f"    - Mean Client Class Entropy: {het['mean_client_entropy_bits']:.4f} bits (Max uniform: {het['max_possible_entropy_bits']:.4f} bits)")
        print(f"    - Mean Total Variation Distance (TVD) from global prior: {het['mean_tvd_from_global_prior']:.4f}")

        # Save partition JSON in model directory
        with open(self.output_dir / "partition.json", "w") as f:
            json.dump(partition, f, indent=2)

        # 3. Instantiate Clients
        print(f"\n[Step 3/6] Instantiating {self.num_clients} isolated Federated Client workers...")
        clients: List[FederatedClient] = []
        for client_id in range(self.num_clients):
            c_indices = partition[client_id]
            client = FederatedClient(
                client_id=client_id,
                X_local=self.X_train[c_indices],
                y_local=self.y_train[c_indices],
                device=self.device,
            )
            clients.append(client)

        # 4. Initialize Global Server Model
        print("\n[Step 4/6] Initializing Global PyTorch MLP on Server...")
        global_model = MalwareMLP(
            in_features=self.X_train.shape[1],
            num_classes=5,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.2,
        ).to(self.device)

        model_config = global_model.get_config()
        num_params = model_config["total_parameters"]
        print(f"    - Global model parameters: {num_params:,}")

        # Compute Communication Cost profile
        num_active_clients = max(1, int(self.num_clients * self.client_fraction))
        comm_cost = compute_communication_cost(
            num_parameters=num_params,
            num_clients_per_round=num_active_clients,
            num_rounds=self.num_rounds,
        )
        print(f"    - Model payload size: {comm_cost['model_size_kb']:.2f} KB ({comm_cost['model_size_bytes']:,} bytes)")
        print(f"    - Per-round communication (all {num_active_clients} clients): {comm_cost['total_mb_per_round']:.4f} MB")
        print(f"    - Total {self.num_rounds}-round communication: {comm_cost['total_mb_all_rounds']:.2f} MB")

        # 5. Federated Training Loop
        print(f"\n[Step 5/6] Starting Federated Training ({tag_str}, {self.num_rounds} global rounds)...")
        print("=" * 105)
        print(f"{'Round':<7} | {'Val Loss':<10} | {'Val Accuracy':<14} | {'Val Macro F1':<14} | {'Val W-F1':<10} | {'Epsilon (ε)':<12} | {'Round Time':<11} | {'Cumul. MB':<10}")
        print("-" * 105)

        round_history = []
        best_val_macro_f1 = 0.0
        best_val_loss = float("inf")
        best_round = 0
        best_model_state = None

        experiment_t0 = time.perf_counter()

        client_train_config = {
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "mu": self.mu,
            "dp_enabled": self.dp_enabled,
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_multiplier,
        }

        # Calculate average steps per round across clients for privacy accountant
        avg_client_samples = len(self.X_train) / self.num_clients
        steps_per_client_per_round = int(self.local_epochs * math.ceil(avg_client_samples / self.batch_size))
        sampling_rate_q = min(1.0, float(self.batch_size) / float(avg_client_samples))

        secagg_verification_results = []

        for round_idx in range(1, self.num_rounds + 1):
            round_t0 = time.perf_counter()

            # 5a. Broadcast global state dict to participating clients
            current_global_state = copy.deepcopy(global_model.state_dict())
            client_updates = []
            client_metrics_list = []

            # 5b. Local training on each client (DP-SGD or Standard SGD/AdamW)
            for client in clients:
                updated_state, n_k, local_metrics = client.train(
                    current_global_state,
                    client_train_config,
                )
                client_updates.append((updated_state, n_k))
                client_metrics_list.append(local_metrics)

            # 5c. Server Aggregation: Secure Aggregation vs Standard Plaintext
            total_train_samples = sum(n_k for _, n_k in client_updates)
            client_ids = list(range(len(client_updates)))

            if self.secagg_enabled:
                # Mask client updates
                masked_updates = []
                for c_id, (c_state, n_k) in enumerate(client_updates):
                    w = float(n_k) / float(total_train_samples)
                    masked_u = self.secagg_engine.mask_client_update(
                        client_id=c_id,
                        client_state_dict=c_state,
                        weight=w,
                        participating_client_ids=client_ids,
                        round_idx=round_idx,
                    )
                    masked_updates.append(masked_u)

                # Server aggregates masked updates (plaintext individual updates are never accessed)
                new_global_state, _ = self.secagg_engine.aggregate_masked_updates(
                    masked_updates=masked_updates,
                    template_state_dict=current_global_state,
                )

                # Verification check in round 1 and every 10 rounds
                if round_idx == 1 or round_idx % 10 == 0:
                    v_res = verify_secure_aggregation_cancellation(client_updates, self.secagg_engine, round_idx=round_idx)
                    secagg_verification_results.append({
                        "round": round_idx,
                        "max_cancellation_error": v_res["max_cancellation_error"],
                        "passed": v_res["passed"],
                    })
            else:
                new_global_state = aggregate_fedavg(client_updates)

            global_model.load_state_dict(new_global_state)

            # 5d. Privacy Accounting Step
            if self.dp_enabled:
                p_record = self.accountant.step(
                    q=sampling_rate_q,
                    sigma=self.noise_multiplier,
                    steps_in_round=steps_per_client_per_round,
                    round_idx=round_idx,
                )
                current_eps_str = f"{p_record['cumulative_epsilon']:.2f}"
                current_eps_val = p_record["cumulative_epsilon"]
            else:
                current_eps_str = "None (Plain)"
                current_eps_val = None

            round_elapsed = time.perf_counter() - round_t0
            cumul_comm_mb = round((comm_cost["total_bytes_per_round"] * round_idx) / (1024.0 * 1024.0), 2)

            # 5e. Centralized Validation Evaluation
            val_metrics = evaluate_model(
                global_model,
                self.X_val,
                self.y_val,
                device=self.device,
                class_names=DEFAULT_CLASS_NAMES,
            )

            val_acc = val_metrics["accuracy"]
            val_macro_f1 = val_metrics["macro_f1"]
            val_loss = val_metrics["loss"]
            val_w_f1 = val_metrics["weighted_f1"]

            # Save checkpoint if best validation Macro F1
            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_val_loss = val_loss
                best_round = round_idx
                best_model_state = copy.deepcopy(new_global_state)
                is_best = "*"
            else:
                is_best = " "

            round_record = {
                "round": round_idx,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_weighted_f1": val_w_f1,
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "epsilon": current_eps_val,
                "delta": self.target_delta if self.dp_enabled else None,
                "per_class_f1": {c: val_metrics["per_class"][c]["f1_score"] for c in DEFAULT_CLASS_NAMES},
                "round_time_seconds": round(round_elapsed, 2),
                "cumulative_comm_mb": cumul_comm_mb,
            }
            round_history.append(round_record)

            if round_idx % 5 == 0 or round_idx == 1 or round_idx == self.num_rounds or is_best == "*":
                print(f"Round {round_idx:2d}{is_best}| {val_loss:>8.4f}   | {val_acc*100:>11.2f}%   | {val_macro_f1:>12.4f}   | {val_w_f1:>8.4f}   | {current_eps_str:>12} | {round_elapsed:>8.2f}s    | {cumul_comm_mb:>8.2f} MB")

        total_train_time = time.perf_counter() - experiment_t0
        print("=" * 105)
        print(f"    Federated training finished in {total_train_time:.2f}s ({self.num_rounds} rounds).")
        print(f"    Best Validation Round: Round {best_round} (Val Macro F1 = {best_val_macro_f1:.4f}, Val Acc = {round_history[best_round-1]['val_accuracy']*100:.2f}%)")

        # Save model checkpoints
        final_model_state = copy.deepcopy(global_model.state_dict())
        torch.save(best_model_state, self.output_dir / "best_model.pt")
        torch.save(final_model_state, self.output_dir / "final_model.pt")

        # Extract privacy spent
        privacy_summary = self.accountant.get_privacy_spent() if self.dp_enabled else None
        if self.dp_enabled:
            with open(self.output_dir / "privacy_accounting.json", "w") as f:
                json.dump(privacy_summary, f, indent=2)

        # Save convergence JSON and training_history.json in model directory
        history_data = {
            "best_round": best_round,
            "best_val_macro_f1": best_val_macro_f1,
            "best_val_accuracy": round_history[best_round - 1]["val_accuracy"],
            "best_val_loss": round_history[best_round - 1]["val_loss"],
            "best_val_macro_precision": round_history[best_round - 1]["val_macro_precision"],
            "best_val_macro_recall": round_history[best_round - 1]["val_macro_recall"],
            "best_val_weighted_f1": round_history[best_round - 1]["val_weighted_f1"],
            "final_epsilon": privacy_summary["epsilon"] if self.dp_enabled else None,
            "final_delta": self.target_delta if self.dp_enabled else None,
            "total_train_time_seconds": round(total_train_time, 2),
            "history": round_history,
        }
        with open(self.output_dir / "convergence.json", "w") as f:
            json.dump(history_data, f, indent=2)
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history_data, f, indent=2)

        # 6. Final Evaluation on Locked Test Set (Conducted strictly after training completes)
        print(f"\n[Step 6/6] Evaluating best global model (Round {best_round}) on the LOCKED TEST SET (2,304 samples)...")
        global_model.load_state_dict(best_model_state)
        test_metrics = evaluate_model(
            global_model,
            self.X_test,
            self.y_test,
            device=self.device,
            class_names=DEFAULT_CLASS_NAMES,
        )

        print(f"    - Test Accuracy:        {test_metrics['accuracy']*100:.2f}%")
        print(f"    - Test Macro F1:        {test_metrics['macro_f1']:.4f}")
        print(f"    - Test Weighted F1:     {test_metrics['weighted_f1']:.4f}")
        print(f"    - Test Macro Precision: {test_metrics['macro_precision']:.4f}")
        print(f"    - Test Macro Recall:    {test_metrics['macro_recall']:.4f}")

        # Package full experiment results
        results = {
            "metadata": {
                "experiment_name": exp_name,
                "algorithm": self.algorithm,
                "mu": self.mu,
                "dp_enabled": self.dp_enabled,
                "clip_norm": self.clip_norm if self.dp_enabled else None,
                "noise_multiplier": self.noise_multiplier if self.dp_enabled else None,
                "target_delta": self.target_delta if self.dp_enabled else None,
                "final_epsilon": privacy_summary["epsilon"] if self.dp_enabled else None,
                "secagg_enabled": self.secagg_enabled,
                "partition_type": self.partition_type,
                "dirichlet_alpha": self.dirichlet_alpha,
                "num_clients": self.num_clients,
                "client_fraction": self.client_fraction,
                "num_rounds": self.num_rounds,
                "local_epochs": self.local_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "seed": self.seed,
                "total_train_time_seconds": round(total_train_time, 2),
                "best_round": best_round,
                "system_metadata": get_system_metadata(),
                "model_architecture": model_config,
            },
            "partition_validation": partition_validation,
            "communication_cost": comm_cost,
            "privacy_accounting": privacy_summary,
            "secagg_verification": secagg_verification_results,
            "best_validation_metrics": {
                "round": best_round,
                "accuracy": round_history[best_round - 1]["val_accuracy"],
                "macro_f1": round_history[best_round - 1]["val_macro_f1"],
                "weighted_f1": round_history[best_round - 1]["val_weighted_f1"],
                "macro_precision": round_history[best_round - 1]["val_macro_precision"],
                "macro_recall": round_history[best_round - 1]["val_macro_recall"],
                "loss": round_history[best_round - 1]["val_loss"],
                "per_class_f1": round_history[best_round - 1]["per_class_f1"],
            },
            "final_round_validation_metrics": {
                "round": self.num_rounds,
                "accuracy": round_history[-1]["val_accuracy"],
                "macro_f1": round_history[-1]["val_macro_f1"],
                "weighted_f1": round_history[-1]["val_weighted_f1"],
                "loss": round_history[-1]["val_loss"],
            },
            "locked_test_metrics": test_metrics,
            "round_by_round_history": round_history,
        }

        # Generate plots
        print("\n    - Generating publication-quality figures...")
        plots = generate_experiment_plots(results, self.report_dir, class_names=DEFAULT_CLASS_NAMES)
        for p_name, p_path in plots.items():
            print(f"      * {p_path.name}")

        # Save Metrics JSON
        results_file = self.result_dir / f"{exp_name}_metrics.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n    - Saved JSON metrics to: {results_file}")

        # Save Configuration JSON in model dir
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        # Generate Markdown Report
        report_file = self.report_dir / f"{exp_name}_report.md"
        self._write_markdown_report(report_file, results)
        print(f"    - Saved Markdown report to: {report_file}")

        return results

    def _write_markdown_report(self, report_file: pathlib.Path, results: Dict[str, Any]) -> None:
        """
        Generate comprehensive scientific markdown report.
        """
        meta = results["metadata"]
        comm = results["communication_cost"]
        val_best = results["best_validation_metrics"]
        test = results["locked_test_metrics"]
        part = results["partition_validation"]
        het = part["heterogeneity_summary"]
        exp_name = meta.get("experiment_name", "privacy_exp")
        dp_on = meta.get("dp_enabled", False)
        secagg_on = meta.get("secagg_enabled", False)

        # Build client distribution table
        client_table_rows = []
        for c_key, c_info in part["client_stats"].items():
            cid = c_info["client_id"]
            cnt = c_info["sample_count"]
            pct = c_info["sample_percentage"]
            counts = [c_info["class_counts"][i] for i in range(5)]
            dom_str = f"{c_info['dominant_class']} ({c_info['dominant_class_percentage']}%)"
            ent_str = f"{c_info['entropy_bits']:.2f} b"
            tvd_str = f"{c_info['tvd_from_global_prior']:.3f}"
            row_str = f"| Client {cid:02d} | {cnt} ({pct:.1f}%) | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} | {counts[4]} | {dom_str} | {ent_str} | {tvd_str} |"
            client_table_rows.append(row_str)
        client_table_md = "\n".join(client_table_rows)

        # Build round progression sample table (every 5 rounds + key rounds)
        progression_rows = []
        for r in results["round_by_round_history"]:
            if r["round"] % 5 == 0 or r["round"] == 1 or r["round"] == meta["best_round"] or r["round"] == meta["num_rounds"]:
                star = " 🌟 (Best)" if r["round"] == meta["best_round"] else ""
                eps_val = f"{r['epsilon']:.2f}" if r["epsilon"] is not None else "N/A"
                progression_rows.append(
                    f"| Round {r['round']:02d}{star} | {r['val_loss']:.4f} | {r['val_accuracy']*100:.2f}% | {r['val_macro_f1']:.4f} | {r['val_weighted_f1']:.4f} | {eps_val} | {r['cumulative_comm_mb']:.2f} MB |"
                )
        progression_md = "\n".join(progression_rows)

        # Per-class test comparison
        test_per_class_rows = []
        for cname in DEFAULT_CLASS_NAMES:
            cdata = test["per_class"][cname]
            test_per_class_rows.append(
                f"| **{cname}** | {cdata['precision']:.4f} | {cdata['recall']:.4f} | **{cdata['f1_score']:.4f}** | {cdata['support']} |"
            )
        test_per_class_md = "\n".join(test_per_class_rows)

        title_header = f"# Experiment Report: {exp_name}"

        lines = [
            title_header,
            f"",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Project:** Privacy-Preserving Malware Detection Using Federated Learning  ",
            f"**Algorithm:** {meta['algorithm']} (DP: {dp_on}, SecAgg: {secagg_on})  ",
            f"**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  ",
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary",
            f"",
            f"| Metric | Value |",
            f"|:---|:---|",
            f"| **Test Accuracy** | **{test['accuracy']*100:.2f}%** |",
            f"| **Test Macro F1** | **{test['macro_f1']:.4f}** |",
            f"| **Test Weighted F1** | **{test['weighted_f1']:.4f}** |",
            f"| **Test Macro Precision** | **{test['macro_precision']:.4f}** |",
            f"| **Test Macro Recall** | **{test['macro_recall']:.4f}** |",
            f"| **Differential Privacy Guarantee** | **ε = {meta.get('final_epsilon', 'N/A')}, δ = {meta.get('target_delta', 'N/A')}** |",
            f"| **Secure Aggregation** | **{'Enabled (Pairwise Additive Masking)' if secagg_on else 'Disabled'}** |",
            f"| **Best Validation Round** | **Round {val_best['round']}** (Val F1: {val_best['macro_f1']:.4f}, Val Acc: {val_best['accuracy']*100:.2f}%) |",
            f"| **Training Time** | **{meta['total_train_time_seconds']:.2f}s** |",
            f"| **Total Communication** | **{comm['total_mb_all_rounds']:.2f} MB** |",
            f"",
            f"> [!NOTE]",
            f"> **Privacy Definition:** DP bounds the probability ratio of outputs on adjacent datasets by exp(ε). Secure Aggregation provides cryptographic confidentiality by masking individual updates so the server only observes the aggregated sum.",
            f"",
            f"---",
            f"",
            f"## 2. Configuration",
            f"",
            f"| Parameter | Value |",
            f"|:---|:---|",
            f"| **Algorithm** | {meta['algorithm']} |",
            f"| **DP Enabled** | {dp_on} |",
            f"| **Noise Multiplier (sigma)** | {meta.get('noise_multiplier', 'N/A')} |",
            f"| **Clipping Norm (C)** | {meta.get('clip_norm', 'N/A')} |",
            f"| **Target Delta** | {meta.get('target_delta', 'N/A')} |",
            f"| **SecAgg Enabled** | {secagg_on} |",
            f"| **Partition** | {meta['partition_type']} (alpha = {meta.get('dirichlet_alpha', 'N/A')}) |",
            f"| **Clients (K)** | {meta['num_clients']} |",
            f"| **Rounds (T)** | {meta['num_rounds']} |",
            f"| **Local Epochs (E)** | {meta['local_epochs']} |",
            f"| **Batch Size (B)** | {meta['batch_size']} |",
            f"| **Learning Rate** | {meta['learning_rate']} |",
            f"| **Optimizer** | AdamW (wd = {meta['weight_decay']}) |",
            f"",
            f"---",
            f"",
            f"## 3. Client Data Distribution",
            f"",
            f"| Client ID | Total Samples | Adware (0) | Banking (1) | SMS (2) | Riskware (3) | Benign (4) | Dominant Class (% of Client) | Entropy | TVD to Prior |",
            f"|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---:|",
            client_table_md,
            f"| **Total Global Train** | **8,062** | **876** | **1,429** | **2,731** | **1,772** | **1,254** | **SMS (33.87%)** | **2.20 b** | **0.000** |",
            f"",
            f"---",
            f"",
            f"## 4. Round-by-Round Validation History",
            f"",
            f"| Global Round | Validation Loss | Validation Accuracy | Validation Macro F1 | Validation Weighted F1 | Epsilon (ε) | Cumulative Comm. |",
            f"|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
            progression_md,
            f"",
            f"---",
            f"",
            f"## 5. Locked Test Set Performance (2,304 Samples)",
            f"",
            f"| Class Name | Precision | Recall | F1-Score | Support |",
            f"|:---|:---:|:---:|:---:|:---:|",
            test_per_class_md,
            f"| **Macro Average** | **{test['macro_precision']:.4f}** | **{test['macro_recall']:.4f}** | **{test['macro_f1']:.4f}** | 2,304 |",
            f"| **Weighted Average** | **{test['macro_precision']:.4f}** | **{test['macro_recall']:.4f}** | **{test['weighted_f1']:.4f}** | 2,304 |",
            f"",
            f"### Confusion Matrix",
            f"```",
            f"Pred ->   Adware  Banking     SMS  Riskware   Benign | Total",
            f"Adware       {test['confusion_matrix'][0][0]:4d}     {test['confusion_matrix'][0][1]:4d}    {test['confusion_matrix'][0][2]:4d}       {test['confusion_matrix'][0][3]:4d}     {test['confusion_matrix'][0][4]:4d} |   250",
            f"Banking      {test['confusion_matrix'][1][0]:4d}     {test['confusion_matrix'][1][1]:4d}    {test['confusion_matrix'][1][2]:4d}       {test['confusion_matrix'][1][3]:4d}     {test['confusion_matrix'][1][4]:4d} |   409",
            f"SMS          {test['confusion_matrix'][2][0]:4d}     {test['confusion_matrix'][2][1]:4d}    {test['confusion_matrix'][2][2]:4d}       {test['confusion_matrix'][2][3]:4d}     {test['confusion_matrix'][2][4]:4d} |   781",
            f"Riskware     {test['confusion_matrix'][3][0]:4d}     {test['confusion_matrix'][3][1]:4d}    {test['confusion_matrix'][3][2]:4d}       {test['confusion_matrix'][3][3]:4d}     {test['confusion_matrix'][3][4]:4d} |   506",
            f"Benign       {test['confusion_matrix'][4][0]:4d}     {test['confusion_matrix'][4][1]:4d}    {test['confusion_matrix'][4][2]:4d}       {test['confusion_matrix'][4][3]:4d}     {test['confusion_matrix'][4][4]:4d} |   358",
            f"```",
        ]

        md_text = "\n".join(lines) + "\n"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(md_text)
