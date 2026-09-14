"""
Adaptive Gradient Clipping Controller for Differentially Private Federated Learning.
Implements bounded quantile-based gradient clipping adaptation.

References:
  - Andrew, G. et al. (2021). "Differentially Private Learning with Adaptive Clipping." NeurIPS.
  - McMahan, H. B. et al. (2018). "Learning Differentially Private Recurrent Language Models." ICLR.
"""

import math
from typing import List, Dict, Any, Optional, Union
import numpy as np


class AdaptiveClippingController:
    """
    Adaptive Clipping Controller using Quantile-Based Gradient Norm Adaptation.

    The clipping threshold C is updated based on observed gradient clipping fractions:
      - Target quantile gamma (e.g., 0.90) defines the desired fraction of unclipped samples.
      - Target clipping fraction is target_clip_frac = 1.0 - gamma (e.g., 0.10).
      - Update rule: C_{t+1} = C_t * exp( eta * (empirical_clip_frac - target_clip_frac) )
      - Strictly bounded within [C_min, C_max].
    """

    def __init__(
        self,
        initial_C: float = 1.0,
        target_quantile: float = 0.90,
        learning_rate: float = 0.1,
        C_min: float = 0.1,
        C_max: float = 10.0,
    ):
        if initial_C <= 0.0:
            raise ValueError(f"initial_C must be positive, got {initial_C}")
        if C_min <= 0.0:
            raise ValueError(f"C_min must be positive, got {C_min}")
        if C_max < C_min:
            raise ValueError(f"C_max ({C_max}) cannot be less than C_min ({C_min})")
        if not (0.0 < target_quantile < 1.0):
            raise ValueError(f"target_quantile must be in (0, 1), got {target_quantile}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        self.C_min = float(C_min)
        self.C_max = float(C_max)
        self.target_quantile = float(target_quantile)
        self.target_clip_frac = 1.0 - self.target_quantile
        self.learning_rate = float(learning_rate)
        self.initial_C = float(np.clip(initial_C, self.C_min, self.C_max))
        self.current_C = self.initial_C

        self.history: List[Dict[str, Any]] = []

    def get_current_C(self) -> float:
        """Return active clipping threshold."""
        return float(self.current_C)

    def update_from_clipping_fraction(
        self,
        clipping_fraction: float,
        round_idx: Optional[int] = None,
    ) -> float:
        """
        Update clipping threshold C based on empirical clipping fraction across participating clients.

        Args:
            clipping_fraction: Empirical fraction of gradients that exceeded C in the current round.
            round_idx: Optional round index for tracking history.

        Returns:
            Updated clipping threshold C_{t+1}.
        """
        if math.isnan(clipping_fraction) or math.isinf(clipping_fraction):
            # Guard against NaN/Inf; retain current C
            clipping_fraction = self.target_clip_frac

        clipping_fraction = float(np.clip(clipping_fraction, 0.0, 1.0))
        error = clipping_fraction - self.target_clip_frac

        # Multiplicative update in log space
        log_update = self.learning_rate * error
        # Clamp log_update to prevent numerical overflow in exp()
        log_update = float(np.clip(log_update, -5.0, 5.0))
        new_C = self.current_C * math.exp(log_update)

        # Enforce strict bounds [C_min, C_max]
        new_C = float(np.clip(new_C, self.C_min, self.C_max))

        record = {
            "round": round_idx,
            "C_before": round(self.current_C, 6),
            "clipping_fraction": round(clipping_fraction, 6),
            "target_clip_fraction": round(self.target_clip_frac, 6),
            "error": round(error, 6),
            "C_after": round(new_C, 6),
        }
        self.history.append(record)
        self.current_C = new_C
        return float(self.current_C)

    def update_from_norms(
        self,
        gradient_norms: Union[List[float], np.ndarray],
        round_idx: Optional[int] = None,
    ) -> float:
        """
        Update clipping threshold directly from a collection of raw gradient norms.
        """
        if gradient_norms is None or len(gradient_norms) == 0:
            return self.current_C

        arr = np.asarray(gradient_norms, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return self.current_C

        clip_frac = float(np.mean(arr > self.current_C))
        return self.update_from_clipping_fraction(clip_frac, round_idx=round_idx)

    @staticmethod
    def compute_norm_statistics(
        gradient_norms: Union[List[float], np.ndarray],
        clipping_threshold: float,
    ) -> Dict[str, float]:
        """
        Compute descriptive distribution percentiles and statistics on observed gradient norms.
        """
        if gradient_norms is None or len(gradient_norms) == 0:
            return {
                "mean_gradient_norm": 0.0,
                "median_gradient_norm": 0.0,
                "p75_gradient_norm": 0.0,
                "p90_gradient_norm": 0.0,
                "p95_gradient_norm": 0.0,
                "clipping_fraction": 0.0,
                "clipping_threshold_C": float(clipping_threshold),
                "sample_count": 0,
            }

        arr = np.asarray(gradient_norms, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return {
                "mean_gradient_norm": 0.0,
                "median_gradient_norm": 0.0,
                "p75_gradient_norm": 0.0,
                "p90_gradient_norm": 0.0,
                "p95_gradient_norm": 0.0,
                "clipping_fraction": 0.0,
                "clipping_threshold_C": float(clipping_threshold),
                "sample_count": 0,
            }

        c_val = float(clipping_threshold)
        return {
            "mean_gradient_norm": float(np.mean(arr)),
            "median_gradient_norm": float(np.median(arr)),
            "p75_gradient_norm": float(np.percentile(arr, 75)),
            "p90_gradient_norm": float(np.percentile(arr, 90)),
            "p95_gradient_norm": float(np.percentile(arr, 95)),
            "clipping_fraction": float(np.mean(arr > c_val)),
            "clipping_threshold_C": c_val,
            "sample_count": int(len(arr)),
        }

    def get_state(self) -> Dict[str, Any]:
        """Return state representation for serialization and inspection."""
        return {
            "current_C": round(self.current_C, 6),
            "initial_C": self.initial_C,
            "target_quantile": self.target_quantile,
            "target_clip_fraction": self.target_clip_frac,
            "learning_rate": self.learning_rate,
            "C_min": self.C_min,
            "C_max": self.C_max,
            "num_updates": len(self.history),
            "history": self.history,
        }

    def reset(self) -> None:
        """Reset controller to initial state."""
        self.current_C = self.initial_C
        self.history = []
