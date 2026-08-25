"""
Differential Privacy Local Training Engine (DP-SGD).
Implements mathematically rigorous per-sample gradient clipping and Gaussian noise injection.

References:
  - Abadi, M. et al. (2016). "Deep Learning with Differential Privacy." ACM CCS.
  - McMahan, H. B. et al. (2018). "Learning Differentially Private Recurrent Language Models." ICLR.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List, Optional
import numpy as np


class DPClientTrainer:
    """
    Client-side DP-SGD Trainer with Per-Sample Gradient Clipping and Gaussian Noise Addition.
    Computes exact per-sample gradients via single forward pass and per-sample autograd.
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        device: str = "cpu",
    ):
        if clip_norm <= 0.0:
            raise ValueError(f"Clipping norm C must be strictly positive, got {clip_norm}")
        if noise_multiplier < 0.0:
            raise ValueError(f"Noise multiplier sigma must be non-negative, got {noise_multiplier}")

        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.device = torch.device(device)

    def train_epoch_dpsgd(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        mu: float = 0.0,
        global_param_refs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Execute one local epoch of DP-SGD:
        1. Forward pass on mini-batch computing per-sample cross-entropy losses.
        2. Per-sample backward pass via autograd computing exact per-sample gradient vectors g_i.
        3. Clip each sample's gradient norm to max C: g_i <- g_i / max(1, ||g_i||_2 / C).
        4. Sum clipped gradients and inject calibrated Gaussian noise: N(0, (sigma * C)^2 * I).
        5. Divide by batch size B and execute optimizer step.
        """
        model.train()
        total_loss = 0.0
        total_batches = 0
        total_samples = 0
        total_clipped_samples = 0
        sum_grad_norms = 0.0

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        ce_per_sample = nn.CrossEntropyLoss(reduction="none")

        for bx, by in dataloader:
            bx, by = bx.to(self.device), by.to(self.device)
            batch_size = bx.size(0)
            if batch_size == 0:
                continue

            total_samples += batch_size
            total_batches += 1

            # 1. Mini-batch forward pass
            optimizer.zero_grad()
            logits = model(bx)
            per_sample_loss = ce_per_sample(logits, by)

            # Proximal penalty if FedProx is active (mu > 0)
            if mu > 0.0 and global_param_refs is not None:
                prox_term = torch.tensor(0.0, device=self.device)
                for name, param in model.named_parameters():
                    if param.requires_grad and name in global_param_refs:
                        prox_term = prox_term + torch.sum((param - global_param_refs[name]) ** 2)
                per_sample_loss = per_sample_loss + (mu / 2.0) * prox_term

            total_loss += float(per_sample_loss.mean().item())

            # Accumulator for clipped gradients for this batch
            clipped_grad_accum = [torch.zeros_like(p) for p in trainable_params]

            # 2. Per-sample gradient computation and L2 clipping
            for i in range(batch_size):
                retain_graph = (i < batch_size - 1)
                sample_grads = torch.autograd.grad(
                    per_sample_loss[i],
                    trainable_params,
                    retain_graph=retain_graph,
                )

                # Compute L2 norm of the per-sample gradient
                sample_norm_sq = torch.tensor(0.0, device=self.device)
                for g in sample_grads:
                    sample_norm_sq = sample_norm_sq + torch.sum(g ** 2)
                sample_norm = math.sqrt(float(sample_norm_sq.item()))
                sum_grad_norms += sample_norm

                # L2 norm clipping factor
                clip_factor = min(1.0, self.clip_norm / max(sample_norm, 1e-12))
                if sample_norm > self.clip_norm:
                    total_clipped_samples += 1

                for idx, g in enumerate(sample_grads):
                    clipped_grad_accum[idx] += g * clip_factor

            # 3. Add Gaussian noise & assign to parameter gradients
            optimizer.zero_grad()
            for idx, p in enumerate(trainable_params):
                if self.noise_multiplier > 0.0:
                    noise_std = self.noise_multiplier * self.clip_norm
                    noise = torch.randn_like(p, device=self.device) * noise_std
                    p.grad = (clipped_grad_accum[idx] + noise) / float(batch_size)
                else:
                    p.grad = clipped_grad_accum[idx] / float(batch_size)

            # 4. Optimizer Step
            optimizer.step()

        mean_loss = total_loss / max(total_batches, 1)
        clipping_fraction = total_clipped_samples / max(total_samples, 1)
        mean_grad_norm = sum_grad_norms / max(total_samples, 1)

        return {
            "epoch_loss": float(mean_loss),
            "clipping_fraction": float(clipping_fraction),
            "mean_grad_norm": float(mean_grad_norm),
            "noise_std": float((self.noise_multiplier * self.clip_norm) / max(dataloader.batch_size, 1)),
        }
