"""
Global hyperparameters for learning rates, discount factor, and model-based weighting.
"""

__all__ = ['ALPHA', 'GAMMA', 'K', 'ALPHA_MB']

ALPHA: float = 0.01      # TD learning rate (model-free)
GAMMA: float = 0.93      # Discount factor
K: float = 0.50          # Model-based weighting in V_NET = k*V_MB + (1-k)*V_TD
ALPHA_MB: float = 0.50   # Reward function learning rate (model-based)
