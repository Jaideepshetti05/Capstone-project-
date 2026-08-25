"""
Runner script for Federated Learning experiments on CICMalDroid 2020.
Supports running IID, Non-IID Dirichlet FedAvg, FedProx, Differential Privacy (DP-SGD), and Secure Aggregation (SecAgg).
"""

import sys
import argparse
import pathlib
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.federated.experiment import FederatedExperiment

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "variant_a"


def load_datasets():
    """Load preprocessed Variant A datasets."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv").values.astype(np.float32)
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel().astype(np.int64)
    X_val = pd.read_csv(DATA_DIR / "X_val.csv").values.astype(np.float32)
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").values.ravel().astype(np.int64)
    X_test = pd.read_csv(DATA_DIR / "X_test.csv").values.astype(np.float32)
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel().astype(np.int64)
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiment on CICMalDroid 2020")
    parser.add_argument(
        "--experiment",
        type=str,
        default="dirichlet_a10",
        choices=[
            "iid", "dirichlet_a10", "dirichlet_a1", "dirichlet_a05", "dirichlet_a01",
            "fedprox_a01_mu001", "fedprox_a01", "fedprox",
            "dp_low_noise", "dp_med_noise", "dp_high_noise", "secagg_only", "dp_secagg", "noniid_dp_secagg",
            "smoke_test_dp", "custom"
        ],
        help="Experiment configuration to execute",
    )
    parser.add_argument("--num_clients", type=int, default=10, help="Total number of clients")
    parser.add_argument("--num_rounds", type=int, default=50, help="Total global communication rounds")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local training epochs per client per round")
    parser.add_argument("--batch_size", type=int, default=64, help="Local training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local AdamW learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Local AdamW weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--alpha", type=float, default=None, help="Dirichlet concentration parameter alpha")
    parser.add_argument("--mu", type=float, default=None, help="FedProx proximal coefficient mu")
    parser.add_argument("--dp", action="store_true", help="Enable Differential Privacy (DP-SGD)")
    parser.add_argument("--sigma", type=float, default=1.0, help="DP noise multiplier sigma")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="DP per-sample clipping norm C")
    parser.add_argument("--delta", type=float, default=1e-5, help="DP target delta")
    parser.add_argument("--secagg", action="store_true", help="Enable Secure Aggregation (SecAgg)")

    args = parser.parse_args()

    # Presets for Phase 6 Experiments
    if args.experiment == "dp_low_noise":
        args.dp = True
        args.sigma = 0.5
        args.clip_norm = 1.0
        exp_name = "dp_low_noise"
        part_type = "iid"
        alpha_val = None
    elif args.experiment == "dp_med_noise":
        args.dp = True
        args.sigma = 1.0
        args.clip_norm = 1.0
        exp_name = "dp_med_noise"
        part_type = "iid"
        alpha_val = None
    elif args.experiment == "dp_high_noise":
        args.dp = True
        args.sigma = 2.0
        args.clip_norm = 1.0
        exp_name = "dp_high_noise"
        part_type = "iid"
        alpha_val = None
    elif args.experiment == "secagg_only":
        args.secagg = True
        args.dp = False
        exp_name = "secagg_only"
        part_type = "iid"
        alpha_val = None
    elif args.experiment == "dp_secagg":
        args.dp = True
        args.sigma = 1.0
        args.clip_norm = 1.0
        args.secagg = True
        exp_name = "dp_secagg"
        part_type = "iid"
        alpha_val = None
    elif args.experiment == "noniid_dp_secagg":
        args.dp = True
        args.sigma = 1.0
        args.clip_norm = 1.0
        args.secagg = True
        exp_name = "noniid_dp_secagg"
        part_type = "dirichlet"
        alpha_val = 0.1
    elif args.experiment == "smoke_test_dp":
        args.dp = True
        args.sigma = 1.0
        args.clip_norm = 1.0
        args.secagg = True
        args.num_rounds = 2
        exp_name = "smoke_test_dp"
        part_type = "iid"
        alpha_val = None
    elif args.experiment == "iid":
        exp_name = "iid_fedavg"
        part_type = "iid"
        alpha_val = None
    elif args.experiment in ["dirichlet_a10", "dirichlet_a1"]:
        exp_name = "dirichlet_a10"
        part_type = "dirichlet"
        alpha_val = args.alpha if args.alpha is not None else 1.0
    elif args.experiment == "dirichlet_a05":
        exp_name = "dirichlet_a05"
        part_type = "dirichlet"
        alpha_val = args.alpha if args.alpha is not None else 0.5
    elif args.experiment == "dirichlet_a01":
        exp_name = "dirichlet_a01"
        part_type = "dirichlet"
        alpha_val = args.alpha if args.alpha is not None else 0.1
    elif args.experiment in ["fedprox_a01_mu001", "fedprox_a01", "fedprox"]:
        exp_name = "fedprox_a01_mu001"
        part_type = "dirichlet"
        alpha_val = args.alpha if args.alpha is not None else 0.1
        args.mu = args.mu if args.mu is not None else 0.01
    else:
        exp_name = "custom_exp"
        part_type = "dirichlet" if args.alpha is not None else "iid"
        alpha_val = args.alpha

    algo_name = "FedProx" if (args.mu is not None and args.mu > 0.0) else "FedAvg"

    config = {
        "experiment_name": exp_name,
        "algorithm": algo_name,
        "partition_type": part_type,
        "dirichlet_alpha": alpha_val,
        "mu": args.mu if args.mu is not None else 0.0,
        "dp_enabled": args.dp,
        "noise_multiplier": args.sigma,
        "clip_norm": args.clip_norm,
        "target_delta": args.delta,
        "secagg_enabled": args.secagg,
        "num_clients": args.num_clients,
        "client_fraction": 1.0,
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": "cpu",
    }
    out_dir = PROJECT_ROOT / "models" / "federated" / exp_name

    print("Loading preprocessed datasets...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets()

    experiment = FederatedExperiment(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        config=config,
        output_dir=out_dir,
        report_dir=PROJECT_ROOT / "reports" / "federated",
        result_dir=PROJECT_ROOT / "results" / "federated",
    )

    results = experiment.run()
    return results


if __name__ == "__main__":
    main()
