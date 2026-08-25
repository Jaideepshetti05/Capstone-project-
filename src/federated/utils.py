"""
Utility functions for Federated Learning: random seeds, system metadata,
communication cost calculation, and logging.
"""

import sys
import platform
import random
import numpy as np
import torch
from typing import Dict, Any


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_system_metadata() -> Dict[str, Any]:
    """
    Collect execution environment metadata for reproducible scientific reporting.
    """
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def compute_communication_cost(
    num_parameters: int,
    num_clients_per_round: int,
    num_rounds: int,
    bytes_per_param: int = 4,  # standard uncompressed IEEE 754 float32
) -> Dict[str, Any]:
    """
    Calculate theoretical communication cost for uncompressed parameter transmission.

    For each participating client in each round:
      - Downlink (Server -> Client): 1 full model parameter payload
      - Uplink (Client -> Server): 1 full model parameter payload
    Total per round = 2 * num_clients_per_round * (num_parameters * bytes_per_param)
    """
    model_size_bytes = num_parameters * bytes_per_param
    model_size_kb = model_size_bytes / 1024.0
    model_size_mb = model_size_kb / 1024.0

    bytes_downlink_per_round = num_clients_per_round * model_size_bytes
    bytes_uplink_per_round = num_clients_per_round * model_size_bytes
    bytes_per_round = bytes_downlink_per_round + bytes_uplink_per_round

    total_bytes = bytes_per_round * num_rounds
    total_mb = total_bytes / (1024.0 * 1024.0)

    return {
        "num_parameters": num_parameters,
        "bytes_per_parameter": bytes_per_param,
        "format": "uncompressed_float32",
        "model_size_bytes": model_size_bytes,
        "model_size_kb": round(model_size_kb, 2),
        "model_size_mb": round(model_size_mb, 4),
        "downlink_bytes_per_round": bytes_downlink_per_round,
        "uplink_bytes_per_round": bytes_uplink_per_round,
        "total_bytes_per_round": bytes_per_round,
        "total_mb_per_round": round(bytes_per_round / (1024.0 * 1024.0), 4),
        "total_bytes_all_rounds": total_bytes,
        "total_mb_all_rounds": round(total_mb, 2),
    }
