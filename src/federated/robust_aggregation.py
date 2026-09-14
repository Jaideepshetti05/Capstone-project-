"""
Robust Aggregation Algorithms for Byzantine-Resilient Federated Learning.

Provides defenses against model poisoning and label-flipping attacks:
1. Coordinate-wise Median (FedMedian) - Yin et al. (2018), "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
2. Trimmed Mean (FedTrimmedMean) - Yin et al. (2018)
"""

import math
import torch
from typing import List, Tuple, Dict, Any, Optional


def aggregate_coordinate_median(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    """
    Perform Coordinate-wise Median aggregation across client model updates.

    For each model parameter tensor and each coordinate j:
      w_global[j] = median({w_1[j], w_2[j], ..., w_K[j]})

    Provides Byzantine robustness against up to 50% arbitrary malicious/poisoned clients.

    Args:
        client_updates: List of tuples (client_state_dict, client_num_samples).

    Returns:
        Aggregated global state dictionary.
    """
    if not client_updates:
        raise ValueError("Robust Aggregation Error: No client updates provided.")

    num_clients = len(client_updates)
    first_state, _ = client_updates[0]
    global_state: Dict[str, torch.Tensor] = {}

    for key, template_tensor in first_state.items():
        if template_tensor.dtype in [torch.float32, torch.float64, torch.float16]:
            # Stack parameter tensors across clients: shape (K, *param_shape)
            stacked = torch.stack(
                [c_state[key] for c_state, _ in client_updates],
                dim=0,
            )
            # Compute median along dimension 0 (across clients)
            # torch.median returns values and indices; we need values
            median_vals, _ = torch.median(stacked, dim=0)
            global_state[key] = median_vals.to(template_tensor.dtype)
        else:
            # Non-float tracking buffers (e.g. integer steps/counts)
            global_state[key] = template_tensor.clone()

    return global_state


def aggregate_trimmed_mean(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    beta: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Perform Coordinate-wise Trimmed Mean aggregation across client model updates.

    For each parameter coordinate j:
      Sort client values w_1[j] <= w_2[j] <= ... <= w_K[j]
      Remove m = floor(beta * K) lowest and highest values
      Average the remaining K - 2m values.

    Args:
        client_updates: List of tuples (client_state_dict, client_num_samples).
        beta: Trimming fraction in [0.0, 0.5). Typically 0.1 to 0.2.

    Returns:
        Aggregated global state dictionary.
    """
    if not client_updates:
        raise ValueError("Robust Aggregation Error: No client updates provided.")
    if not (0.0 <= beta < 0.5):
        raise ValueError(f"Trimming fraction beta must be in [0.0, 0.5), got {beta}")

    num_clients = len(client_updates)
    first_state, _ = client_updates[0]
    global_state: Dict[str, torch.Tensor] = {}

    # Number of elements to trim from each end
    num_to_trim = int(math.floor(beta * num_clients))

    # Guard: if trimming would eliminate all or leave 0 clients, fall back to median or average
    if 2 * num_to_trim >= num_clients:
        num_to_trim = max(0, (num_clients - 1) // 2)

    for key, template_tensor in first_state.items():
        if template_tensor.dtype in [torch.float32, torch.float64, torch.float16]:
            stacked = torch.stack(
                [c_state[key] for c_state, _ in client_updates],
                dim=0,
            )  # (K, *shape)

            if num_to_trim == 0:
                # Standard mean across clients
                global_state[key] = torch.mean(stacked, dim=0).to(template_tensor.dtype)
            else:
                # Sort along client dimension
                sorted_vals, _ = torch.sort(stacked, dim=0)
                # Slice out the m smallest and m largest
                trimmed = sorted_vals[num_to_trim : num_clients - num_to_trim]
                # Mean of remaining values
                global_state[key] = torch.mean(trimmed, dim=0).to(template_tensor.dtype)
        else:
            global_state[key] = template_tensor.clone()

    return global_state


def aggregate_robust(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    method: str = "trimmed_mean",
    beta: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Unified router for robust Byzantine-resilient aggregation methods.

    Supported methods:
      - 'median' / 'coordinate_median' / 'fedmedian': Coordinate-wise median
      - 'trimmed_mean' / 'fedtrimmedmean': Coordinate-wise trimmed mean
      - 'fedavg': Standard sample-weighted FedAvg
    """
    method_lower = method.lower().strip()
    if method_lower in ["median", "coordinate_median", "fedmedian"]:
        return aggregate_coordinate_median(client_updates)
    elif method_lower in ["trimmed_mean", "fedtrimmedmean", "trimmed"]:
        return aggregate_trimmed_mean(client_updates, beta=beta)
    elif method_lower in ["fedavg", "average", "mean"]:
        from .fedavg import aggregate_fedavg
        return aggregate_fedavg(client_updates)
    else:
        raise ValueError(
            f"Unknown robust aggregation method: '{method}'. "
            "Supported options: 'median', 'trimmed_mean', 'fedavg'."
        )
