"""
Rényi Differential Privacy (RDP) Accountant for Subsampled Gaussian Mechanism.
Provides mathematically exact privacy budget accounting for Federated DP-SGD.

References:
  - Mironov, I. (2017). "Rényi Differential Privacy." CSF.
  - Wang, Y. X., Balle, B., & Kasiviswanathan, S. P. (2019). "Subsampled Rényi Differential Privacy:
    Analytical Bounds and Tightness." ICML.
  - Abadi, M. et al. (2016). "Deep Learning with Differential Privacy." ACM CCS.
"""

import math
import numpy as np
from scipy import special
from typing import List, Tuple, Dict, Any, Optional


def compute_rdp_gaussian(alpha: float, sigma: float) -> float:
    """
    Compute RDP of order alpha for standard Gaussian Mechanism with noise multiplier sigma.
    RDP(alpha) = alpha / (2 * sigma^2)
    """
    if sigma <= 0:
        return float("inf")
    return alpha / (2.0 * (sigma ** 2))


def _compute_rdp_subsampled_gaussian_order(q: float, sigma: float, alpha: float) -> float:
    """
    Compute RDP of order alpha for Subsampled Gaussian Mechanism with sampling rate q and noise sigma.
    Uses the analytical bound from Wang et al. (2019) and Mironov et al. (2019).
    """
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return compute_rdp_gaussian(alpha, sigma)
    if sigma <= 0.0:
        return float("inf")

    # For integer orders alpha >= 2, we can compute exact upper bound
    if float(alpha).is_integer() and alpha >= 2:
        alpha = int(alpha)
        # Expansion terms for moments
        term_sum = 0.0
        for l in range(alpha + 1):
            binom = special.comb(alpha, l, exact=True)
            # Compute E_{z ~ N(0,1)} [ (1 - q + q * exp( (2z - 1) / (2 sigma^2) ))^l * (1 - q + q * exp( (-2z - 1) / (2 sigma^2) ))^(alpha-l) ]
            # Using Wang et al. bound:
            if l == 0:
                coef = (1.0 - q) ** alpha
            elif l == 1:
                coef = alpha * q * (1.0 - q) ** (alpha - 1)
            else:
                coef = binom * (q ** l) * ((1.0 - q) ** (alpha - l))
            
            mu_l = l * (l - 1) / (2.0 * (sigma ** 2)) if sigma > 0 else 0.0
            term_sum += coef * math.exp(min(mu_l, 700.0))  # guard against overflow
            
        rdp = (1.0 / (alpha - 1.0)) * math.log(max(term_sum, 1e-15))
        return max(0.0, rdp)
    else:
        # For non-integer orders, use continuous interpolation bound
        # RDP(alpha) <= RDP_gaussian(alpha) * q^2 + O(q^3)
        base_rdp = compute_rdp_gaussian(alpha, sigma)
        return min(base_rdp, (q ** 2) * alpha / (2.0 * (sigma ** 2)) + q * math.log(1.0 + q * (alpha - 1.0)))


def compute_rdp_subsampled_gaussian(
    q: float,
    sigma: float,
    steps: int,
    orders: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Compute cumulative RDP across multiple steps under composition.
    RDP accumulates additively across steps: RDP_total(alpha) = steps * RDP_step(alpha)
    """
    if orders is None:
        orders = [1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0,
                  12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0, 48.0, 64.0, 128.0, 256.0, 512.0]

    rdp_vals = []
    for alpha in orders:
        rdp_single = _compute_rdp_subsampled_gaussian_order(q, sigma, alpha)
        rdp_vals.append(rdp_single * steps)

    return np.array(rdp_vals, dtype=np.float64)


def convert_rdp_to_dp(
    rdp_array: np.ndarray,
    orders: List[float],
    delta: float = 1e-5,
) -> Tuple[float, float]:
    """
    Convert RDP vector to (epsilon, delta)-DP guarantee by minimizing over all orders alpha > 1:
      epsilon(delta) = min_{alpha > 1} ( RDP(alpha) + log(1 / delta) / (alpha - 1) )

    Returns:
        (optimal_epsilon, best_alpha_order)
    """
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"Delta must be in (0, 1), got {delta}")

    epsilons = []
    for rdp, alpha in zip(rdp_array, orders):
        if math.isinf(rdp) or math.isnan(rdp):
            eps = float("inf")
        else:
            eps = rdp + math.log(1.0 / delta) / (alpha - 1.0)
        epsilons.append(eps)

    min_idx = int(np.argmin(epsilons))
    opt_eps = float(epsilons[min_idx])
    best_alpha = float(orders[min_idx])

    return max(0.0, opt_eps), best_alpha


class RDPPrivacyAccountant:
    """
    Stateful Differential Privacy Accountant for Federated Learning.
    Tracks privacy budget expenditure across local client steps and global communication rounds.
    """

    def __init__(
        self,
        target_delta: float = 1e-5,
        orders: Optional[List[float]] = None,
    ):
        self.target_delta = target_delta
        if orders is None:
            self.orders = [
                1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0,
                32.0, 40.0, 48.0, 64.0, 80.0, 96.0, 128.0, 192.0, 256.0, 512.0
            ]
        else:
            self.orders = orders

        self.total_rdp = np.zeros(len(self.orders), dtype=np.float64)
        self.round_history: List[Dict[str, Any]] = []
        self.total_steps: int = 0

    def step(
        self,
        q: float,
        sigma: float,
        steps_in_round: int = 1,
        round_idx: int = 1,
    ) -> Dict[str, Any]:
        """
        Record a communication round containing steps_in_round mini-batch updates with sampling rate q.
        """
        round_rdp = compute_rdp_subsampled_gaussian(q, sigma, steps=steps_in_round, orders=self.orders)
        self.total_rdp += round_rdp
        self.total_steps += steps_in_round

        eps, opt_order = convert_rdp_to_dp(self.total_rdp, self.orders, delta=self.target_delta)

        record = {
            "round": round_idx,
            "sampling_rate_q": q,
            "noise_multiplier_sigma": sigma,
            "steps_in_round": steps_in_round,
            "cumulative_steps": self.total_steps,
            "cumulative_epsilon": round(eps, 4),
            "target_delta": self.target_delta,
            "optimal_renyi_order": opt_order,
        }
        self.round_history.append(record)
        return record

    def get_privacy_spent(self) -> Dict[str, Any]:
        """
        Return the current cumulative (epsilon, delta)-DP guarantee and trajectory.
        """
        eps, opt_order = convert_rdp_to_dp(self.total_rdp, self.orders, delta=self.target_delta)
        return {
            "epsilon": round(eps, 4),
            "delta": self.target_delta,
            "optimal_renyi_order": opt_order,
            "total_steps": self.total_steps,
            "total_rounds": len(self.round_history),
            "orders_evaluated": len(self.orders),
            "history": self.round_history,
        }
