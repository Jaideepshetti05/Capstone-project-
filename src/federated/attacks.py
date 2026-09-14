"""
Poisoning Attack Simulation Suite for Federated Learning.

Provides threat models to evaluate Byzantine resilience:
1. Label-Flipping Attack (Data Poisoning):
   - Malicious clients flip malware class labels (0..3) to Benign (4) (Targeted Evasion),
     or apply cyclic permutation: y -> (y + 1) % num_classes.
2. Model Weight Poisoning Attack (Model Poisoning):
   - Gaussian Noise Injection: w_mal = w + N(0, sigma_atk^2 * I)
   - Sign Inversion: w_mal = w_global - scale * (w_local - w_global)
   - Scaled Extreme Bias: w_mal = scale * w_local
"""

import copy
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class LabelFlipAttack:
    """
    Data poisoning attack executed locally on malicious client datasets.
    """

    def __init__(
        self,
        attack_type: str = "malware_to_benign",
        source_class: Optional[int] = None,
        target_class: Optional[int] = 4,  # Default: Benign
        num_classes: int = 5,
    ):
        self.attack_type = attack_type
        self.source_class = source_class
        self.target_class = target_class
        self.num_classes = num_classes

    def apply(self, y: np.ndarray) -> np.ndarray:
        """
        Poison label array y.
        """
        y_poisoned = y.copy()

        if self.attack_type == "malware_to_benign":
            # Targeted Evasion: Flip all malware (0, 1, 2, 3) to Benign (4)
            benign_label = self.target_class if self.target_class is not None else 4
            malware_mask = (y_poisoned != benign_label)
            y_poisoned[malware_mask] = benign_label

        elif self.attack_type == "cyclic":
            # Cyclic permutation: y -> (y + 1) % C
            y_poisoned = (y_poisoned + 1) % self.num_classes

        elif self.attack_type == "targeted":
            # Flip specific source_class to target_class
            if self.source_class is not None and self.target_class is not None:
                mask = (y_poisoned == self.source_class)
                y_poisoned[mask] = self.target_class

        return y_poisoned


class WeightPoisonAttack:
    """
    Model poisoning attack executed on client parameter updates before transmission.
    """

    def __init__(
        self,
        attack_type: str = "gaussian_noise",
        noise_std: float = 2.0,
        scale_factor: float = 10.0,
        seed: Optional[int] = None,
    ):
        self.attack_type = attack_type
        self.noise_std = noise_std
        self.scale_factor = scale_factor
        self.seed = seed
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def apply(
        self,
        local_state_dict: Dict[str, torch.Tensor],
        global_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Mutate client model state dictionary according to the attack strategy.
        """
        poisoned_state: Dict[str, torch.Tensor] = {}

        for key, param in local_state_dict.items():
            if param.dtype not in [torch.float32, torch.float64, torch.float16]:
                poisoned_state[key] = param.clone()
                continue

            if self.attack_type == "gaussian_noise":
                # Add high-magnitude zero-mean Gaussian noise to weights
                noise = torch.randn(
                    param.shape,
                    generator=self.rng,
                    dtype=param.dtype,
                    device=param.device,
                ) * self.noise_std
                poisoned_state[key] = param + noise

            elif self.attack_type == "sign_inversion":
                # Invert the parameter update direction: w_mal = w_global - scale * delta
                if global_state_dict is not None and key in global_state_dict:
                    g_param = global_state_dict[key].to(param.device)
                    delta = param - g_param
                    poisoned_state[key] = g_param - self.scale_factor * delta
                else:
                    poisoned_state[key] = -self.scale_factor * param

            elif self.attack_type == "scaling":
                # Scale weights by large factor to overpower honest clients in FedAvg
                poisoned_state[key] = param * self.scale_factor

            else:
                poisoned_state[key] = param.clone()

        return poisoned_state
