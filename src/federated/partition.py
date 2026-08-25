"""
Data partitioning module for Federated Learning simulations.
Provides deterministic IID (stratified) and Non-IID (Dirichlet) partitioning
with comprehensive validation checks to guarantee zero sample duplication or loss,
and calculates information-theoretic heterogeneity metrics (entropy, TVD, dominant class, client size variance).
"""

import numpy as np
from typing import Dict, List, Any, Optional


def create_iid_partition(
    y_train: np.ndarray,
    num_clients: int = 10,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Create a deterministic, stratified IID partition of training indices across clients.

    Every client receives an approximately equal number of samples, with the global
    class proportions preserved within each local client dataset.

    Args:
        y_train: 1D array of class labels for the training set (length N).
        num_clients: Number of participating clients (default: 10).
        seed: Random seed for deterministic shuffling.

    Returns:
        Dictionary mapping client_id (0..num_clients-1) to list of integer indices in y_train.
    """
    rng = np.random.default_rng(seed)
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    unique_classes = np.unique(y_train)

    for c in unique_classes:
        # Extract indices belonging to class c
        class_idx = np.where(y_train == c)[0]
        rng.shuffle(class_idx)

        # Distribute class c indices as evenly as possible across clients
        chunks = np.array_split(class_idx, num_clients)
        for i in range(num_clients):
            client_indices[i].extend(chunks[i].tolist())

    # Deterministically shuffle each client's sample order
    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def create_dirichlet_partition(
    y_train: np.ndarray,
    num_clients: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Create a deterministic Non-IID partition using Dirichlet(alpha) label skew.

    For each class c in {0..num_classes-1}:
      Sample proportion vector q_c ~ Dirichlet(alpha * 1_K)
      Allocate class c samples to the K clients according to q_c.

    Args:
        y_train: 1D array of class labels for the training set (length N).
        num_clients: Number of participating clients (K = 10).
        alpha: Dirichlet concentration parameter (lower = stronger non-IID heterogeneity).
        seed: Random seed for deterministic generation.

    Returns:
        Dictionary mapping client_id (0..num_clients-1) to list of integer indices.
    """
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y_train)

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    # Group sample indices by class
    class_indices = {c: np.where(y_train == c)[0] for c in unique_classes}
    for c in unique_classes:
        rng.shuffle(class_indices[c])

    # Sample Dirichlet proportions per class
    for c in unique_classes:
        idx_c = class_indices[c]
        n_c = len(idx_c)

        # Sample proportions from Dirichlet(alpha * 1_K)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Scale to integer counts
        counts = (proportions * n_c).astype(int)
        remainder = n_c - counts.sum()

        # Distribute any rounding remainder to clients with largest fractional remainder
        if remainder > 0:
            fractional = proportions * n_c - counts
            rem_indices = np.argsort(fractional)[::-1][:remainder]
            counts[rem_indices] += 1

        # Slice class indices and assign to clients
        split_pts = np.cumsum(counts)[:-1]
        splits = np.split(idx_c, split_pts)
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())

    # Deterministically shuffle each client's sample order
    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def validate_partition(
    partition: Dict[int, List[int]],
    y_train: np.ndarray,
    total_expected: int = 8062,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Perform strict automated assertions on the generated partition.
    Guarantees no sample duplication, no sample loss, and complete data integrity.
    Computes Shannon entropy, Total Variation Distance (TVD), dominant class, and sample variance.

    Raises:
        AssertionError if any validation check fails.
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(5)]

    num_clients = len(partition)
    num_classes = len(class_names)
    all_indices = []

    for client_id, indices in partition.items():
        assert len(indices) > 0, f"Validation Failed: Client {client_id} has 0 samples."
        all_indices.extend(indices)

    total_assigned = len(all_indices)
    unique_assigned = len(set(all_indices))

    # Assertion 1: Total sample count
    assert total_assigned == total_expected, (
        f"Validation Failed: Expected {total_expected} total samples, got {total_assigned}."
    )

    # Assertion 2: No duplicates across clients
    assert unique_assigned == total_assigned, (
        f"Validation Failed: Duplicate samples detected across clients. Total: {total_assigned}, Unique: {unique_assigned}."
    )

    # Assertion 3: Complete coverage of indices [0..total_expected-1]
    assert set(all_indices) == set(range(total_expected)), (
        "Validation Failed: Partition indices do not match the expected range [0..total_expected-1]."
    )

    # Global class distribution (prior)
    global_class_counts = np.bincount(y_train, minlength=num_classes)
    global_class_proportions = global_class_counts / total_expected

    # Compute per-client class distribution table and heterogeneity metrics
    client_stats = {}
    client_entropies = []
    client_tvds = []
    sample_counts = []

    for client_id, indices in partition.items():
        y_client = y_train[indices]
        n_k = len(indices)
        sample_counts.append(n_k)
        counts = [int((y_client == c).sum()) for c in range(num_classes)]
        proportions = [counts[c] / max(n_k, 1) for c in range(num_classes)]
        percentages = [round(p * 100.0, 2) for p in proportions]

        # 1. Shannon Entropy (in bits, base 2): H_k = -sum p_c * log2(p_c)
        entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
        client_entropies.append(entropy)

        # 2. Total Variation Distance (TVD) from global prior: TVD_k = 0.5 * sum |p_kc - p_global_c|
        tvd = 0.5 * sum(abs(proportions[c] - global_class_proportions[c]) for c in range(num_classes))
        client_tvds.append(tvd)

        # 3. Dominant Class
        dom_idx = int(np.argmax(counts))
        dom_class = class_names[dom_idx]
        dom_pct = percentages[dom_idx]

        client_stats[f"client_{client_id}"] = {
            "client_id": client_id,
            "sample_count": n_k,
            "sample_percentage": round((n_k / total_expected) * 100.0, 2),
            "class_counts": {c: counts[c] for c in range(num_classes)},
            "named_class_counts": {class_names[c]: counts[c] for c in range(num_classes)},
            "class_percentages": {class_names[c]: percentages[c] for c in range(num_classes)},
            "dominant_class": dom_class,
            "dominant_class_percentage": dom_pct,
            "entropy_bits": round(float(entropy), 4),
            "tvd_from_global_prior": round(float(tvd), 4),
        }

    # Max possible entropy for 5 uniform classes = log2(5) ≈ 2.3219 bits
    max_entropy = np.log2(num_classes)

    heterogeneity_summary = {
        "min_client_size": int(np.min(sample_counts)),
        "max_client_size": int(np.max(sample_counts)),
        "mean_client_size": round(float(np.mean(sample_counts)), 2),
        "std_client_size": round(float(np.std(sample_counts)), 2),
        "var_client_size": round(float(np.var(sample_counts)), 2),
        "mean_client_entropy_bits": round(float(np.mean(client_entropies)), 4),
        "min_client_entropy_bits": round(float(np.min(client_entropies)), 4),
        "max_possible_entropy_bits": round(float(max_entropy), 4),
        "mean_tvd_from_global_prior": round(float(np.mean(client_tvds)), 4),
        "max_tvd_from_global_prior": round(float(np.max(client_tvds)), 4),
    }

    return {
        "status": "VALID",
        "num_clients": num_clients,
        "total_samples": total_assigned,
        "unique_samples": unique_assigned,
        "heterogeneity_summary": heterogeneity_summary,
        "client_stats": client_stats,
    }
