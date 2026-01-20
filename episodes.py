"""
Episode runners for executing agent-environment interactions. 
"""
import numpy as np

__all__ = [
    'TrackEpisode', 'WheelEpisode', 'TeleportEpisode', 'PauseEpisode',
    'KrauszGridEpisode'
]

# ------------------------------------------------------------
# Track
# ------------------------------------------------------------
class TrackEpisode:
    """Episode runner for standard track navigation.
    
    Executes agent-environment interaction loop until goal or max_steps.
    
    Attributes:
        env: Track environment instance.
        agent: Learning agent instance.
        max_steps: Maximum steps per episode.
    """
    
    def __init__(self, environment, agent, max_steps: int = 100) -> None:
        self.env = environment
        self.agent = agent
        self.max_steps = max_steps

    def run(self) -> tuple[list, list, list, list, list]:
        s = self.env.reset()
        end = False
        steps: list[int] = []
        states: list[int] = []
        actions: list[int] = []
        deltas: list[float] = []
        rewards: list[float] = []
        t = 0

        while not end and t < self.max_steps:
            states.append(s)
            a = self.agent.select_action(s)
            s_prime, r, end = self.env.step(a)
            delta_t = self.agent.update(s, r, s_prime, end)
            rewards += [r]
            steps += [t]
            t += 1
            actions += [a]
            deltas += [delta_t]
            if not end:
                s = s_prime
        return states, steps, actions, deltas, rewards
    
# ------------------------------------------------------------
# Wheel
# ------------------------------------------------------------
class WheelEpisode:
    """Episode runner for running wheel environment.
    
    Executes agent-environment interaction loop until goal or max_steps.
    
    Attributes:
        env: Wheel environment instance.
        agent: Learning agent instance.
        max_steps: Maximum steps per episode.
    """
    
    def __init__(self, environment, agent, max_steps: int = 100) -> None:
        self.env = environment
        self.agent = agent
        self.max_steps = max_steps
        
    def run(self) -> tuple[list, list, list, list, list]:
        s = self.env.reset()
        end = False
        steps: list[int] = []
        states: list[int] = []
        actions: list[int] = []
        deltas: list[float] = []
        rewards: list[float] = []
        t = 0
    
        while not end and t < self.max_steps:
            states.append(s)
            a = self.agent.select_action(s)
            s_prime, r, end = self.env.step(a)
            delta_t = self.agent.update(s, r, s_prime, end)
            rewards += [r]
            steps += [t]
            t += 1
            actions += [a]
            deltas += [delta_t]
            if not end:
                s = s_prime
        return states, steps, actions, deltas, rewards

# ------------------------------------------------------------
# Teleport
# ------------------------------------------------------------
class TeleportEpisode:
    """Episode runner for track with teleportation (Kim et al., 2020).
    
    Agent is teleported forward at trigger state.
    
    Attributes:
        env: Track environment instance.
        agent: Learning agent instance.
        teleport_destination: State to teleport to.
        teleport_start: State that triggers teleportation.
    """
    
    def __init__(self, environment, teleport_destination: int, teleport_distance: int,
                 agent, max_steps: int = 100) -> None:
        self.env = environment
        self.teleport_destination = teleport_destination
        self.teleport_start = teleport_destination - teleport_distance
        self.agent = agent
        self.max_steps = max_steps

    def run(self) -> tuple[list, list, list, list, list, int]:
        s = self.env.reset()
        end = False
        steps: list[int] = []
        states: list[int] = []
        actions: list[int | None] = []
        deltas: list[float] = []
        rewards: list[float] = []
        t = 0

        while not end and t < self.max_steps:
            states.append(s)
            if s == self.teleport_start:
                a = None  # No action during teleport
                s_prime, r, end = self.env.teleport(self.teleport_destination)
            else:
                a = self.agent.select_action(s)
                s_prime, r, end = self.env.step(a)
            delta_t = self.agent.update(s, r, s_prime, end)
            rewards += [r]
            steps += [t]
            t += 1
            actions += [a]
            deltas += [delta_t]
            if not end:
                s = s_prime
        return states, steps, actions, deltas, rewards, self.teleport_start
    
# ------------------------------------------------------------
# Pause
# ------------------------------------------------------------
class PauseEpisode:
    """Episode runner for track with pause (Kim et al., 2020).
    
    Agent pauses at trigger state for fixed duration.
    
    Attributes:
        env: Track environment instance.
        agent: Learning agent instance.
        pause_location: State where pause occurs.
        pause_duration: Number of timesteps to pause.
    """
    
    def __init__(self, environment, pause_location: int, pause_duration: int,
                 agent, max_steps: int = 1000) -> None:
        self.env = environment
        self.pause_location = pause_location
        self.pause_duration = pause_duration
        self.agent = agent
        self.max_steps = max_steps

    def run(self) -> tuple[list, list, list, list, list, int]:
        s = self.env.reset()
        end = False
        steps: list[int] = []
        states: list[int] = []
        actions: list[int | None] = []
        deltas: list[float] = []
        rewards: list[float] = []
        t = 0

        while not end:
            states.append(s)
            if s == self.pause_location:
                t_pause = t
                original_k = self.agent.k
                original_alpha = self.agent.alpha
                original_alpha_mb = self.agent.alpha_mb
                self.agent.k = 0
                self.agent.alpha = 0
                self.agent.alpha_mb = 0
                while t < t_pause + self.pause_duration:
                    s_prime, r, end = self.env.pause(s)
                    delta_t = self.agent.update(s, r, s_prime, end)
                    rewards += [r]
                    steps += [t]
                    t += 1
                    actions += [None]
                    deltas += [delta_t]
                self.agent.k = original_k
                self.agent.alpha = original_alpha
                self.agent.alpha_mb = original_alpha_mb
                a = self.agent.select_action(s)
                s_prime, r, end = self.env.step(a)
            else:
                a = self.agent.select_action(s)
                s_prime, r, end = self.env.step(a)
            delta_t = self.agent.update(s, r, s_prime, end)
            rewards += [r]
            steps += [t]
            t += 1
            actions += [a]
            deltas += [delta_t]
            if not end:
                s = s_prime
        return states, steps, actions, deltas, rewards, self.pause_location
    
# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------
class KrauszGridEpisode:
    """Episode runner for gridworld navigation (Krausz et al., 2023).
    
    Agent navigates grid with fixed action policy based on start state.
    
    Attributes:
        env: KrauszGrid environment instance.
        agent: Learning agent instance.
        max_steps: Maximum steps per episode.
        start_state: (row, col) starting position.
    """
    
    def __init__(self, environment, agent, max_steps: int = 100,
                 start_state: tuple[int, int] = (0, 0)) -> None:
        self.env = environment
        self.agent = agent
        self.max_steps = max_steps
        self.start_state = start_state
        self.env.set_reward_function(self.env.p_reward, self.env.goal_state)
        self.env.set_goal_state(self.env.goal_state)

    def run(self) -> tuple[list, list, list, list, list]:
        s = self.env.reset(start_state=self.start_state)
        end = False
        steps: list[int] = []
        states: list[tuple[int, int]] = []
        actions: list[int] = []
        deltas: list[float] = []
        rewards: list[float] = []
        t = 0

        while not end and t < self.max_steps:
            states.append(s)
            if self.start_state == (9, 0):
                action = 'up'
            elif self.start_state == (0, 9):
                action = 'left'
            a = self.agent.action_to_int(action)
            s_prime, r, end = self.env.step(a)
            delta_t = self.agent.update(s, r, s_prime, end)
            rewards += [r]
            steps += [t]
            t += 1
            actions += [a]
            deltas += [delta_t]
            if not end:
                s = s_prime
        return states, steps, actions, deltas, rewards
    