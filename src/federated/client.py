"""
Federated Client module for local training on partitioned decentralized data shards.
Supports:
  1. Vanilla FedAvg local training (AdamW).
  2. FedProx proximal regularization: L_k(w) + (mu / 2) ||w - w_global||^2.
  3. Differential Privacy (DP-SGD): Per-sample gradient clipping & Gaussian noise injection.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Tuple, Optional
import numpy as np

from src.models.mlp import MalwareMLP
from .dp_client import DPClientTrainer


class FederatedClient:
    """
    Simulated Federated Client holding local training data.
    
    Guarantees:
      - Local training uses only private data shard.
      - No access to validation or test data.
      - Communicates only model parameters and sample counts to server.
      - Supports FedProx proximal regularization and DP-SGD per-sample clipping/noise.
    """

    def __init__(
        self,
        client_id: int,
        X_local: np.ndarray,
        y_local: np.ndarray,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.num_samples = len(X_local)
        self.device = torch.device(device)

        # Convert local data to PyTorch tensors
        self.X_local = torch.tensor(X_local, dtype=torch.float32)
        self.y_local = torch.tensor(y_local, dtype=torch.long)
        self.dataset = TensorDataset(self.X_local, self.y_local)

        # Local model template
        in_features = X_local.shape[1]
        self.local_model = MalwareMLP(
            in_features=in_features,
            num_classes=5,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.2,
        ).to(self.device)

    def train(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        config: Dict[str, Any],
    ) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, float]]:
        """
        Execute local training for specified local epochs.
        Supports standard SGD/AdamW, FedProx (mu > 0), and DP-SGD (dp_enabled=True).

        Args:
            global_state_dict: Global model parameters received from server.
            config: Training hyperparameter dictionary (lr, weight_decay, batch_size, local_epochs, mu, dp_enabled, clip_norm, noise_multiplier).

        Returns:
            Tuple of (updated_state_dict, num_samples, training_metrics_dict).
        """
        # 1. Load global parameters into local model
        self.local_model.load_state_dict(copy.deepcopy(global_state_dict))
        self.local_model.train()

        batch_size = config.get("batch_size", 64)
        local_epochs = config.get("local_epochs", 2)
        lr = config.get("learning_rate", 1e-3)
        weight_decay = config.get("weight_decay", 1e-4)
        mu = float(config.get("mu", 0.0))  # FedProx proximal coefficient
        dp_enabled = bool(config.get("dp_enabled", False))
        clip_norm = float(config.get("clip_norm", 1.0))
        noise_multiplier = float(config.get("noise_multiplier", 1.0))

        # Store frozen global parameter reference for proximal term calculation
        global_param_refs = {}
        if mu > 0.0:
            for name, param in self.local_model.named_parameters():
                if param.requires_grad:
                    global_param_refs[name] = param.detach().clone().to(self.device)

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.local_model.parameters(), lr=lr, weight_decay=weight_decay)

        # -------------------------------------------------------------
        # Branch A: Differential Privacy (DP-SGD) Local Training
        # -------------------------------------------------------------
        if dp_enabled:
            dp_trainer = DPClientTrainer(
                clip_norm=clip_norm,
                noise_multiplier=noise_multiplier,
                device=str(self.device),
            )
            dp_stats = []
            for epoch in range(local_epochs):
                stat = dp_trainer.train_epoch_dpsgd(
                    model=self.local_model,
                    dataloader=loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    mu=mu,
                    global_param_refs=global_param_refs if mu > 0.0 else None,
                )
                dp_stats.append(stat)

            updated_state = {
                k: v.cpu().detach().clone() for k, v in self.local_model.state_dict().items()
            }
            metrics = {
                "final_loss": round(dp_stats[-1]["epoch_loss"], 4),
                "clipping_fraction": round(dp_stats[-1]["clipping_fraction"], 4),
                "mean_grad_norm": round(dp_stats[-1]["mean_grad_norm"], 4),
                "noise_std": round(dp_stats[-1]["noise_std"], 6),
            }
            return updated_state, self.num_samples, metrics

        # -------------------------------------------------------------
        # Branch B: Standard FedAvg / FedProx Local Training
        # -------------------------------------------------------------
        epoch_losses = []
        epoch_ce_losses = []
        epoch_prox_losses = []
        initial_loss = None

        for epoch in range(local_epochs):
            running_loss = 0.0
            running_ce_loss = 0.0
            running_prox_loss = 0.0
            total_batches = 0

            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.local_model(bx)
                ce_loss = criterion(logits, by)

                # FedProx Proximal Regularization Term: (mu / 2) * sum ||w - w_global||^2
                if mu > 0.0:
                    prox_term = torch.tensor(0.0, device=self.device)
                    for name, param in self.local_model.named_parameters():
                        if param.requires_grad and name in global_param_refs:
                            prox_term = prox_term + torch.sum((param - global_param_refs[name]) ** 2)
                    total_loss = ce_loss + (mu / 2.0) * prox_term
                    prox_val = float(((mu / 2.0) * prox_term).item())
                else:
                    total_loss = ce_loss
                    prox_val = 0.0

                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                running_ce_loss += ce_loss.item()
                running_prox_loss += prox_val
                total_batches += 1

                if initial_loss is None:
                    initial_loss = total_loss.item()

            avg_loss = running_loss / max(total_batches, 1)
            avg_ce = running_ce_loss / max(total_batches, 1)
            avg_prox = running_prox_loss / max(total_batches, 1)
            epoch_losses.append(avg_loss)
            epoch_ce_losses.append(avg_ce)
            epoch_prox_losses.append(avg_prox)

        # Extract updated state dict on CPU
        updated_state = {
            k: v.cpu().detach().clone() for k, v in self.local_model.state_dict().items()
        }

        metrics = {
            "initial_loss": round(float(initial_loss if initial_loss is not None else 0.0), 4),
            "final_loss": round(float(epoch_losses[-1]), 4),
            "final_ce_loss": round(float(epoch_ce_losses[-1]), 4),
            "final_prox_loss": round(float(epoch_prox_losses[-1]), 6),
            "loss_decrease": round(float((initial_loss or 0.0) - epoch_losses[-1]), 4),
        }

        return updated_state, self.num_samples, metrics
