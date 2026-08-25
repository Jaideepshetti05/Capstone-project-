"""
Runner script for Federated Learning experiments on CICMalDroid 2020.
Supports running IID, Non-IID Dirichlet FedAvg, and FedProx experiments.
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
        choices=["iid", "dirichlet_a10", "dirichlet_a1", "dirichlet_a05", "dirichlet_a01", "fedprox_a01_mu001", "fedprox_a01", "fedprox"],
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

    args = parser.parse_args()

    # Define experiment configuration
    if args.experiment == "iid":
        config = {
            "experiment_name": "iid_fedavg",
            "algorithm": "FedAvg",
            "partition_type": "iid",
            "dirichlet_alpha": None,
            "mu": 0.0,
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
        out_dir = PROJECT_ROOT / "models" / "federated" / "iid"
    elif args.experiment in ["dirichlet_a10", "dirichlet_a1"]:
        config = {
            "experiment_name": "dirichlet_a10",
            "algorithm": "FedAvg",
            "partition_type": "dirichlet",
            "dirichlet_alpha": args.alpha if args.alpha is not None else 1.0,
            "mu": 0.0,
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
        out_dir = PROJECT_ROOT / "models" / "federated" / "dirichlet_a10"
    elif args.experiment == "dirichlet_a05":
        config = {
            "experiment_name": "dirichlet_a05",
            "algorithm": "FedAvg",
            "partition_type": "dirichlet",
            "dirichlet_alpha": args.alpha if args.alpha is not None else 0.5,
            "mu": 0.0,
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
        out_dir = PROJECT_ROOT / "models" / "federated" / "dirichlet_a05"
    elif args.experiment == "dirichlet_a01":
        config = {
            "experiment_name": "dirichlet_a01",
            "algorithm": "FedAvg",
            "partition_type": "dirichlet",
            "dirichlet_alpha": args.alpha if args.alpha is not None else 0.1,
            "mu": 0.0,
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
        out_dir = PROJECT_ROOT / "models" / "federated" / "dirichlet_a01"
    elif args.experiment in ["fedprox_a01_mu001", "fedprox_a01", "fedprox"]:
        mu_val = args.mu if args.mu is not None else 0.01
        alpha_val = args.alpha if args.alpha is not None else 0.1
        config = {
            "experiment_name": "fedprox_a01_mu001",
            "algorithm": "FedProx",
            "partition_type": "dirichlet",
            "dirichlet_alpha": alpha_val,
            "mu": mu_val,
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
        out_dir = PROJECT_ROOT / "models" / "federated" / "fedprox_a01_mu001"

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
