"""
Federated Averaging (FedAvg) aggregation algorithm.
Computes sample-weighted parameter aggregation across participating client updates:
  w_global = sum_{k=1}^K (n_k / N) * w_k
"""

import copy
import torch
from typing import List, Tuple, Dict


def aggregate_fedavg(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    """
    Perform weighted Federated Averaging (FedAvg) on client model updates.

    Args:
        client_updates: List of tuples (client_state_dict, client_num_samples)
                        from all participating clients in the round.

    Returns:
        Aggregated global state dictionary.

    Raises:
        ValueError: If client_updates is empty or total sample count is zero.
    """
    if not client_updates:
        raise ValueError("FedAvg Error: No client updates provided for aggregation.")

    total_samples = sum(n_k for _, n_k in client_updates)
    if total_samples <= 0:
        raise ValueError("FedAvg Error: Total client sample count must be positive.")

    # Initialize aggregated state dict with zeros matching the first client's structure
    first_state, _ = client_updates[0]
    global_state: Dict[str, torch.Tensor] = {}

    for key, tensor in first_state.items():
        if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
            global_state[key] = torch.zeros_like(tensor, dtype=tensor.dtype)
        else:
            # For integer tracking buffers (e.g. BatchNorm num_batches_tracked)
            global_state[key] = tensor.clone()

    # Accumulate weighted parameters
    for client_state, n_k in client_updates:
        weight = float(n_k) / float(total_samples)

        for key in global_state.keys():
            if global_state[key].dtype in [torch.float32, torch.float64, torch.float16]:
                global_state[key] += client_state[key].to(global_state[key].dtype) * weight
            else:
                # Keep latest buffer value for non-float tensors
                global_state[key] = client_state[key].clone()

    return global_state
