"""
PyTorch Multi-Layer Perceptron (MLP) for Multi-Class Malware Classification.
Designed for tabular system call and binder frequency features (443 inputs, 5 classes).
Compatible with centralized training and federated learning aggregation (FedAvg).
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MalwareMLP(nn.Module):
    """
    Multi-Layer Perceptron architecture for CICMalDroid malware detection.
    
    Structure:
      Input (in_features=443) -> [Linear -> BatchNorm1d -> ReLU -> Dropout] * N -> Linear(num_classes=5)
    """

    def __init__(
        self,
        in_features: int = 443,
        num_classes: int = 5,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = in_features

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            # Slightly lower dropout on later layers
            p = dropout_rate if i < len(hidden_dims) - 1 else dropout_rate / 2.0
            layers.append(nn.Dropout(p))
            prev_dim = h_dim

        # Final classification head (logits output)
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized class logits.
        """
        return self.net(x)

    def get_config(self) -> dict:
        """
        Return architectural configuration dictionary.
        """
        return {
            "in_features": self.in_features,
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
