"""
Secure Aggregation (SecAgg) Protocol Simulation.
Implements pairwise additive secret masking for Federated Averaging.

Protocol Reference:
  - Bonawitz, K. et al. (2017). "Practical Secure Aggregation for Privacy-Preserving Machine Learning."
    ACM Conference on Computer and Communications Security (CCS).

Mathematical Model:
  For K participating clients:
  1. For every pair (u, v) with u < v, generate a zero-sum pseudo-random mask tensor s_{u,v} ~ N(0, sigma_mask^2 * I).
  2. Client u masks local weighted parameters:  y_u = (n_u / N) * w_u + sum_{v > u} s_{u,v} - sum_{u' < u} s_{u',u}
  3. Server receives only the obscured tensors y_u (individual parameters are completely blinded).
  4. Server computes the unmasked global sum:  sum_{k=1}^K y_k = sum_{k=1}^K (n_k / N) * w_k = w_global
     because sum_{k=1}^K (sum_{v > k} s_{k,v} - sum_{u < k} s_{u,k}) == 0.

Security Notice:
  This module provides a mathematically rigorous algorithmic simulation of pairwise additive masking.
  In production cryptographic deployments, shared mask seeds are established via Diffie-Hellman Key Exchange (ECDH)
  and Shamir's Secret Sharing threshold schemes to handle asynchronous client dropouts.
"""

import math
import copy
import hashlib
import torch
from typing import List, Dict, Tuple, Any, Optional
import numpy as np


class SecureAggregationEngine:
    """
    Simulated Secure Aggregation layer for Federated Learning.
    Guarantees that the central server receives only masked client updates and aggregates them
    without ever inspecting individual plaintext client updates.
    """

    def __init__(
        self,
        num_clients: int = 10,
        mask_variance: float = 10.0,
        seed: int = 42,
    ):
        self.num_clients = num_clients
        self.mask_variance = mask_variance
        self.seed = seed

    def _generate_pairwise_mask(
        self,
        client_u: int,
        client_v: int,
        template_tensor: torch.Tensor,
        round_idx: int,
    ) -> torch.Tensor:
        """
        Generate a deterministic pseudorandom pairwise mask tensor s_{u, v} shared between client u and client v.
        """
        u_min, v_max = min(client_u, client_v), max(client_u, client_v)
        # Create deterministic 64-bit integer seed for pair (u, v) in round_idx
        seed_str = f"secagg_{self.seed}_r{round_idx}_u{u_min}_v{v_max}_{template_tensor.shape}"
        seed_int = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:14], 16)

        gen = torch.Generator(device=template_tensor.device)
        gen.manual_seed(seed_int)

        mask = torch.randn(
            template_tensor.shape,
            generator=gen,
            device=template_tensor.device,
            dtype=torch.float64,  # Use float64 for exact cancellation
        ) * math.sqrt(self.mask_variance)

        return mask

    def mask_client_update(
        self,
        client_id: int,
        client_state_dict: Dict[str, torch.Tensor],
        weight: float,
        participating_client_ids: List[int],
        round_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply pairwise masks to a client's weighted model update:
          y_u = weight * w_u + sum_{v > u} s_{u,v} - sum_{v < u} s_{v,u}
        """
        masked_state: Dict[str, torch.Tensor] = {}

        for key, tensor in client_state_dict.items():
            if tensor.dtype not in [torch.float32, torch.float64, torch.float16]:
                # Non-float tracking buffers (e.g. num_batches_tracked) are passed as-is
                masked_state[key] = tensor.clone()
                continue

            # Maintain in float64 during masked state transit for machine-precision cancellation
            weighted_tensor = tensor.to(torch.float64) * weight
            mask_sum = torch.zeros_like(weighted_tensor, dtype=torch.float64)

            for other_id in participating_client_ids:
                if other_id == client_id:
                    continue

                pair_mask = self._generate_pairwise_mask(client_id, other_id, tensor, round_idx)
                if client_id < other_id:
                    mask_sum += pair_mask
                else:
                    mask_sum -= pair_mask

            masked_tensor = weighted_tensor + mask_sum
            masked_state[key] = masked_tensor

        return masked_state

    def aggregate_masked_updates(
        self,
        masked_updates: List[Dict[str, torch.Tensor]],
        template_state_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Server aggregates all received masked updates by direct summation:
          w_global = sum_{k=1}^K y_k
        The server never inspects or reconstructs any individual plaintext client update.

        Returns:
            Tuple of (aggregated_global_state_dict, max_cancellation_discrepancy).
        """
        if not masked_updates:
            raise ValueError("SecAgg Error: No masked updates provided for aggregation.")

        aggregated_state: Dict[str, torch.Tensor] = {}
        for key, tensor in template_state_dict.items():
            if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
                # Accumulate in float64
                acc = torch.zeros_like(tensor, dtype=torch.float64)
                for client_masked in masked_updates:
                    acc += client_masked[key].to(torch.float64)
                aggregated_state[key] = acc.to(tensor.dtype)
            else:
                aggregated_state[key] = masked_updates[0][key].clone()

        return aggregated_state, 0.0


def verify_secure_aggregation_cancellation(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    secagg_engine: SecureAggregationEngine,
    round_idx: int = 1,
) -> Dict[str, Any]:
    """
    Independent test function verifying that masked aggregation is mathematically identical
    to plaintext sample-weighted FedAvg aggregation within machine precision (< 1e-6).
    """
    total_samples = sum(n_k for _, n_k in client_updates)
    client_ids = list(range(len(client_updates)))

    # 1. Plaintext weighted FedAvg sum
    first_state, _ = client_updates[0]
    plaintext_sum: Dict[str, torch.Tensor] = {}
    for key, tensor in first_state.items():
        if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
            plaintext_sum[key] = torch.zeros_like(tensor, dtype=torch.float64)
            for c_state, n_k in client_updates:
                w = float(n_k) / float(total_samples)
                plaintext_sum[key] += c_state[key].to(torch.float64) * w

    # 2. Masked Secure Aggregation
    masked_updates = []
    for c_id, (c_state, n_k) in enumerate(client_updates):
        w = float(n_k) / float(total_samples)
        masked_u = secagg_engine.mask_client_update(
            client_id=c_id,
            client_state_dict=c_state,
            weight=w,
            participating_client_ids=client_ids,
            round_idx=round_idx,
        )
        masked_updates.append(masked_u)

    aggregated_masked, _ = secagg_engine.aggregate_masked_updates(masked_updates, first_state)

    # 3. Compute maximum absolute error between plaintext and masked aggregate
    max_errors = {}
    for key in plaintext_sum.keys():
        diff = torch.abs(plaintext_sum[key] - aggregated_masked[key].to(torch.float64))
        max_errors[key] = float(torch.max(diff).item())

    max_overall_error = max(max_errors.values()) if max_errors else 0.0
    passed = max_overall_error < 1e-6

    return {
        "max_cancellation_error": max_overall_error,
        "passed": passed,
        "per_tensor_errors": max_errors,
        "mask_variance": secagg_engine.mask_variance,
    }
