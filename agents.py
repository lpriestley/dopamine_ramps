"""
Reinforcement learning agents: standard TD and dual-process (model-based + model-free).
"""
import numpy as np

__all__ = [
    'TDTab', 'TDFeature', 'TDFeatureMikhael',
    'DualTab', 'DualFeature', 'DualNaiveTab', 'DualGrid'
]

# ------------------------------------------------------------
# Trackworld
# ------------------------------------------------------------
class TDTab:
    """Standard TD learning agent with tabular state representation.
    
    Learns state values V(s) using temporal difference updates.
    Designed for deterministic 1D track environments with a single action.
    
    Attributes:
        V: Value estimates for each state.
        alpha: Learning rate.
        gamma: Discount factor.
    """
    
    def __init__(self, n_states: int, n_actions: int = 1,
                 alpha: float = 0.1, gamma: float = 0.9) -> None:
        assert n_actions == 1, "TDTab only supports 1 action"
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.V = np.zeros(n_states)

    def reset(self) -> None:
        self.V = np.zeros(self.n_states)
    
    def select_action(self, state: int) -> int:
        return 1
    
    def update(self, s: int, r: float, s_prime: int, end: bool) -> float:
        """
            V(s) ← V(s) + α [r + γ V(s') - V(s)]
        """
        if end:
            delta = r - self.V[s]
        else:
            delta = r + self.gamma * self.V[s_prime] - self.V[s]
        self.V[s] += self.alpha * delta
        return delta
    
class TDFeature:
    """TD learning agent with feature-based state representation.
    
    Learns state values as V(s) = φ(s)·w.
    Supports separate phi matrices for current (phi_t) and next (phi_t_prime)
    states to model state uncertainty.
    
    Attributes:
        phi_t: Feature matrix for current state.
        phi_t_prime: Feature matrix for next state.
        w: Weight vector for value function approximation.
        alpha: Learning rate.
        gamma: Discount factor.
    """
    
    def __init__(self, phi: np.ndarray, phi_t_prime: np.ndarray | None = None,
                 n_actions: int = 1, alpha: float = 0.1,
                 gamma: float = 0.9) -> None:
        assert n_actions == 1, "TDFeature only supports 1 action"
        self.phi_t = phi
        self.phi_t_prime = phi_t_prime if phi_t_prime is not None else phi
        self.phi = self.phi_t  # Backward compatibility: default phi is phi_t
        self.w = np.zeros(len(self.phi_t[0]))
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

    def reset(self) -> None:
        self.w = np.zeros(len(self.phi_t[0]))
    
    def select_action(self, state: int) -> int:
        return 1
    
    def value(self, state: int) -> float:
        """Compute V(s) using phi_t (current state representation)."""
        return np.dot(self.phi_t[state], self.w)
    
    def value_t_prime(self, state: int) -> float:
        """Compute V(s') using phi_t_prime (next state representation)."""
        return np.dot(self.phi_t_prime[state], self.w)
    
    def update(self, s: int, r: float, s_prime: int, end: bool) -> float:
        """
        TD update: w ← w + α [r + γ V(s') - V(s)] ∇V(s)
        
        Uses phi_t for V(s) and phi_t_prime for V(s'),
        with gradient computed from phi_t[s].
        """
        v_s = self.value(s)  # V(s) using phi_t
        if end:
            delta = r - v_s
        else:
            v_s_prime = self.value_t_prime(s_prime)  # V(s') using phi_t_prime
            delta = r + self.gamma * v_s_prime - v_s
        self.w += self.alpha * delta * self.phi_t[s]
        return delta

class TDFeatureMikhael:
    """TD agent with dynamic state uncertainty (Mikhael et al., 2022).
    
    Variant of TDFeature where phi_t and phi_t_prime are always distinct.
    Allows modeling of dynamic state uncertainty.
    
    Attributes:
        phi_t: Feature matrix for current state.
        phi_t_prime: Feature matrix for next state.
        w: Weight vector for value function approximation.
    """
    
    def __init__(self, phi: np.ndarray, phi_t_prime: np.ndarray,
                 n_actions: int = 1, alpha: float = 0.1,
                 gamma: float = 0.9) -> None:
        assert n_actions == 1, "TDFeatureMikhael only supports 1 action"
        self.phi_t = phi  # phi matrix for s
        self.phi_t_prime = phi_t_prime 
        self.w = np.zeros(len(self.phi_t[0]))
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

    def reset(self) -> None:
        self.w = np.zeros(len(self.phi_t[0]))
    
    def select_action(self, state: int) -> int:
        return 1
    
    def value_t(self, state: int) -> float:
        """Compute V(s) using phi_t (current state representation)."""
        return np.dot(self.phi_t[state], self.w)
    
    def value_t_prime(self, state: int) -> float:
        """Compute V(s') using phi_t_prime (next state representation)."""
        return np.dot(self.phi_t_prime[state], self.w)
    
    def update(self, s: int, r: float, s_prime: int, end: bool) -> float:
        v_s = self.value_t(s) 
        if end:
            delta = r - v_s
        else:
            v_s_prime = self.value_t_prime(s_prime)
            delta = r + self.gamma * v_s_prime - v_s
        self.w += self.alpha * delta * self.phi_t[s]
        return delta

class DualTab:
    """Dual-process agent with tabular state representation.
    
    Update target combines model-based (V_MB) and model-free (V_TD) value estimates.
    V_NET = k * V_MB + (1-k) * V_TD.
    
    Attributes:
        V_TD: Model-free value estimates.
        V_MB: Model-based value estimates.
        R: Learned reward function.
        k: Weighting parameter.
    """
    
    def __init__(self, n_states: int | None = None, n_actions: int = 1,
                 goal_state: int | None = None,
                 init_R: float | np.ndarray | None = None,
                 alpha_mb: float = 0.1, alpha: float = 0.1, gamma: float = 0.9,
                 k: float = 0.50) -> None:
        assert n_actions == 1, "DualTab only supports 1 action"
        self.n_states = n_states
        self.n_actions = n_actions
        self.goal_state = goal_state
        self.alpha_mb = alpha_mb
        self.alpha = alpha
        self.gamma = gamma
        self.k = k
        if init_R is None:
            self.R = np.zeros(n_states, dtype=float)
        elif np.isscalar(init_R):
            self.R = np.full(n_states, init_R, dtype=float)
        else:
            self.R = np.array(init_R, dtype=float)
        self.init_R = self.R.copy()
        self.V_TD = np.zeros(n_states)
        self.V_MB = np.zeros(n_states)
        self.compute_mb_val(n_states, goal_state)

    def reset(self) -> None:
        self.V_TD = np.zeros(self.n_states)
        self.V_MB = np.zeros(self.n_states)
        self.R = self.init_R.copy()
        self.compute_mb_val(self.n_states, self.goal_state)
    
    def select_action(self, state: int) -> int:
        return 1
    
    def compute_mb_val(self, n_states: int, goal_state: int) -> None:
        for s in range(n_states):
            d = abs(goal_state - s)
            self.V_MB[s] = self.R[goal_state] * self.gamma**d
    
    def update(self, s: int, r: float, s_prime: int, end: bool) -> float:
        """
        V_NET = k * V_MB + (1 - k) * V_TD
        V_TD(s) ← V_TD(s) + α [r + γ V_NET(s') - V_TD(s)]
        R(s) ← R(s) + alpha_mb * [r - R(s)]
        """
        self.R[s] += self.alpha_mb * (r - self.R[s])
        self.compute_mb_val(self.n_states, self.goal_state)
        
        if end:
            delta = r - self.V_TD[s]
        else:
            v_net = self.k * self.V_MB[s_prime] + (1 - self.k) * self.V_TD[s_prime]
            delta = r + self.gamma * v_net - self.V_TD[s]
        
        self.V_TD[s] += self.alpha * delta
        return delta
    
class DualFeature:
    """Dual-process agent with feature-based state representation.
    
    Update target combines model-based and model-free value estimates.
    V_NET = k * V_MB + (1-k) * V_TD.
    
    Attributes:
        w_TD: Weight vector for model-free values.
        w_MB: Weight vector for model-based values.
        R: Learned reward function.
        k: Weighting parameter.
    """
    
    def __init__(self, phi_TD: np.ndarray, phi_MB: np.ndarray, n_actions: int = 1,
                 goal_state: int | None = None,
                 init_R: float | np.ndarray | None = None,
                 alpha_mb: float = 0.1, alpha: float = 0.1, gamma: float = 0.9,
                 k: float = 0.50) -> None:
        assert n_actions == 1, "DualFeature only supports 1 action"
        self.phi_TD = phi_TD
        self.phi_MB = phi_MB
        self.goal_state = goal_state
        self.n_states = len(phi_MB)
        self.w_TD = np.zeros(len(self.phi_TD[0]))
        self.w_MB = np.zeros(len(self.phi_MB[0]))
        self.n_actions = n_actions
        self.alpha_mb = alpha_mb
        self.alpha = alpha
        self.gamma = gamma
        self.k = k
        if init_R is None:
            self.R = np.zeros(self.n_states, dtype=float)
        elif np.isscalar(init_R):
            self.R = np.full(self.n_states, init_R, dtype=float)
        else:
            self.R = np.array(init_R, dtype=float)
        self.init_R = self.R.copy()
        self.compute_w_MB(phi_MB, goal_state)
        
    def reset(self) -> None:
        self.w_TD = np.zeros(len(self.phi_TD[0]))
        self.w_MB = np.zeros(len(self.phi_MB[0]))
        self.R = self.init_R.copy()
        self.compute_w_MB(self.phi_MB, self.goal_state)

    def select_action(self, state: int) -> int:
        return 1
    
    def compute_w_MB(self, phi_MB: np.ndarray, goal_state: int) -> None:
        for s in range(len(phi_MB)):
            d = abs(goal_state - s)
            self.w_MB[s] = self.R[goal_state] * self.gamma**d
    
    def td_value(self, state: int) -> float:
        return np.dot(self.phi_TD[state], self.w_TD)
    
    def mb_value(self, state: int) -> float:
        return np.dot(self.phi_MB[state], self.w_MB)
    
    def update(self, s: int, r: float, s_prime: int, end: bool) -> float:
        """
        w_TD ← w_TD + α [r + γ V_NET(s') - V_TD(s)] ∇V_TD(s)
        R(s) ← R(s) + alpha_mb * [r - R(s)]
        """
        self.R[s] += self.alpha_mb * (r - self.R[s])
        self.compute_w_MB(self.phi_MB, self.goal_state)
        
        v_td_s = self.td_value(s)
        if end:
            delta = r - v_td_s
        else:
            v_td_s_prime = self.td_value(s_prime)
            v_mb_s_prime = self.mb_value(s_prime)
            v_net_s_prime = self.k * v_mb_s_prime + (1 - self.k) * v_td_s_prime
            delta = r + self.gamma * v_net_s_prime - v_td_s
        self.w_TD += self.alpha * delta * self.phi_TD[s]
        return delta

class DualNaiveTab:
    """Alternative dual-process agent with tabular state representation.
    
    Uses V_NET in RPE update target and prediction instead of target alone.
    V_TD(s) ← V_TD(s) + α [r + γ V_NET(s') - V_NET(s)].
    
    Attributes:
        V_TD: Model-free value estimates.
        V_MB: Model-based value estimates.
        R: Learned reward function.
        k: Weighting parameter.
    """
    
    def __init__(self, n_states: int | None = None, n_actions: int = 1,
                 goal_state: int | None = None,
                 init_R: float | np.ndarray | None = None,
                 alpha_mb: float = 0.1, alpha: float = 0.1, gamma: float = 0.9,
                 k: float = 0.50) -> None:
        assert n_actions == 1, "DualNaiveTab only supports 1 action"
        self.n_states = n_states
        self.n_actions = n_actions
        self.goal_state = goal_state
        self.alpha_mb = alpha_mb
        self.alpha = alpha
        self.gamma = gamma
        self.k = k
        if init_R is None:
            self.R = np.zeros(n_states, dtype=float)
        elif np.isscalar(init_R):
            self.R = np.full(n_states, init_R, dtype=float)
        else:
            self.R = np.array(init_R, dtype=float)
        self.init_R = self.R.copy()
        self.V_TD = np.zeros(n_states)
        self.V_MB = np.zeros(n_states)
        self.compute_mb_val(n_states, goal_state)

    def reset(self) -> None:
        self.V_TD = np.zeros(self.n_states)
        self.V_MB = np.zeros(self.n_states)
        self.R = self.init_R.copy()
        self.compute_mb_val(self.n_states, self.goal_state)
    
    def select_action(self, state: int) -> int:
        return 1
    
    def compute_mb_val(self, n_states: int, goal_state: int) -> None:
        for s in range(n_states):
            d = abs(goal_state - s)
            self.V_MB[s] = self.R[goal_state] * self.gamma**d
    
    def update(self, s: int, r: float, s_prime: int, end: bool) -> float:
        """
            V_NET = k * V_MB + (1 - k) * V_TD
            V_TD(s) ← V_TD(s) + α [r + γ V_NET(s') - V_NET(s)]
            R(s) ← R(s) + alpha_mb * [r - R(s)]  (reward function learning)
        """
        self.R[s] += self.alpha_mb * (r - self.R[s])
        self.compute_mb_val(self.n_states, self.goal_state)
        
        if end:
            v_net = self.k * self.V_MB[s] + (1 - self.k) * self.V_TD[s]
            delta = r - v_net
        else:
            v_net_s_prime = self.k * self.V_MB[s_prime] + (1 - self.k) * self.V_TD[s_prime]
            v_net_s = self.k * self.V_MB[s] + (1 - self.k) * self.V_TD[s]
            delta = r + self.gamma * v_net_s_prime - v_net_s
        self.V_TD[s] += self.alpha * delta
        return delta
# ------------------------------------------------------------
# Gridworld
# ------------------------------------------------------------
class DualGrid:
    """Dual-process agent for 2D gridworld environments.
    
    Update target combines model-based and model-free value estimates.
    V_NET = k * V_MB + (1-k) * V_TD.
    
    Attributes:
        V_TD: Model-free value estimates (2D array).
        V_MB: Model-based value estimates (2D array).
        R: Learned reward function (2D array).
        k: Weighting parameter.
    """
    
    def __init__(self, grid: tuple[int, int] = (8, 8), n_actions: int = 4,
                 goal_state: tuple[int, int] = (7, 7),
                 init_R: float | np.ndarray | None = None,
                 alpha_mb: float = 0.25, alpha: float = 0.1, gamma: float = 0.9,
                 k: float = 0.50) -> None:
        self.height, self.width = grid
        self.goal_state = goal_state
        self.n_actions = n_actions
        self.alpha_mb = alpha_mb
        self.alpha = alpha
        self.gamma = gamma
        self.k = k
        if init_R is None:
            self.R = np.zeros((self.height, self.width), dtype=float)
        elif np.isscalar(init_R):
            self.R = np.full((self.height, self.width), init_R, dtype=float)
        else:
            self.R = np.array(init_R, dtype=float)
            if self.R.shape != (self.height, self.width):
                raise ValueError(
                    f"init_R shape {self.R.shape} must match grid ({self.height}, {self.width})"
                )
        self.init_R = self.R.copy()
        self.V_TD = np.zeros((self.height, self.width))
        self.V_MB = np.zeros((self.height, self.width))
        self.compute_mb_val()

    def reset(self) -> None:
        self.V_TD = np.zeros((self.height, self.width))
        self.V_MB = np.zeros((self.height, self.width))
        self.R = self.init_R.copy()
        self.compute_mb_val()
    
    def compute_mb_val(self) -> None:
        height, width = self.height, self.width
        goal_row, goal_col = self.goal_state

        for row in range(height):
            for col in range(width):
                d = abs(row - goal_row) + abs(col - goal_col)

                self.V_MB[row, col] = self.R[goal_row, goal_col] * self.gamma**d

    def action_to_int(self, action: str) -> int:
        assert action in ['up', 'down', 'left', 'right'], "Invalid action"
        if action == 'up':
            return 0
        elif action == 'down':
            return 1
        elif action == 'left':
            return 2
        elif action == 'right':
            return 3
    
    def update(self, s: tuple[int, int], r: float,
               s_prime: tuple[int, int], end: bool) -> float:
        """
        V_NET = k * V_MB + (1 - k) * V_TD
        V_TD(s) ← V_TD(s) + α [r + γ V_NET(s') - V_TD(s)]
        R(s) ← R(s) + alpha_mb * [r - R(s)]
        """
        self.R[s] += self.alpha_mb * (r - self.R[s])
        self.compute_mb_val()
        
        if end:
            delta = r - self.V_TD[s]
        else:
            v_net = self.k * self.V_MB[s_prime] + (1 - self.k) * self.V_TD[s_prime]
            delta = r + self.gamma * v_net - self.V_TD[s]
        self.V_TD[s] += self.alpha * delta
        return delta
