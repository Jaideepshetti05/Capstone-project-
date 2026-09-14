"""
Federated Learning package for CICMalDroid 2020 Malware Detection.
Provides modular components for client partitioning, local training,
server aggregation (FedAvg), validation tracking, and communication cost analysis.
"""

from .partition import create_iid_partition, create_dirichlet_partition, validate_partition
from .client import FederatedClient
from .fedavg import aggregate_fedavg
from .robust_aggregation import aggregate_coordinate_median, aggregate_trimmed_mean, aggregate_robust
from .evaluation import evaluate_model
from .experiment import FederatedExperiment
from .utils import set_seed, get_system_metadata, compute_communication_cost

__all__ = [
    "create_iid_partition",
    "create_dirichlet_partition",
    "validate_partition",
    "FederatedClient",
    "aggregate_fedavg",
    "aggregate_coordinate_median",
    "aggregate_trimmed_mean",
    "aggregate_robust",
    "evaluate_model",
    "FederatedExperiment",
    "set_seed",
    "get_system_metadata",
    "compute_communication_cost",
]

