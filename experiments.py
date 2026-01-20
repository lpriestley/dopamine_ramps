"""
Experiment runners that simulate paradigms from Guru et al., Kim et al., Mikhael et al., and Krausz et al.
"""
import numpy as np
from environments import (
    Track, DarkTrack, Wheel, KrauszGrid,
    StaticUncertaintyTrack, DynamicUncertaintyTrack
)
from episodes import TrackEpisode, WheelEpisode, TeleportEpisode, PauseEpisode, KrauszGridEpisode
from agents import TDTab, DualTab, TDFeature, DualFeature, DualNaiveTab, DualGrid, TDFeatureMikhael
from config import ALPHA, ALPHA_MB, GAMMA, K
import plotting as plt

__all__ = [
    'dual_process_demo', 'compare_architectures',
    'guru_track', 'guru_wheel',
    'kim_distance', 'kim_location', 'kim_speed',
    'krausz_grid',
    'compare_uncertainty_assumptions', 'mikhael_track'
]

# ------------------------------------------------------------
# Demos
# ------------------------------------------------------------

def dual_process_demo(n_episodes: int = 100, n_states: int = 20,
                      goal_state: int = 19, reward_value: float = 1.0) -> None:
    """Demonstrate dual-process learning on a linear track.
    
    Trains a DualTab agent and plots learned value functions and RPEs.
    """
    env = Track(n_states=n_states, goal_state=goal_state, reward_value=reward_value)
    agent = DualTab(
        n_states, n_actions=1, goal_state=goal_state, init_R=reward_value,
        alpha_mb=ALPHA_MB, alpha=ALPHA, gamma=0.85, k=K
    )
    ep = TrackEpisode(env, agent)

    exp_data = {
        'episodes': [],
    }

    for ep_n in range(n_episodes):
        states, steps, actions, deltas, rewards = ep.run()
        ep_data = {
            'ep': ep_n,
            'states': np.array(states),
            'steps': np.array(steps),
            'actions': np.array(actions),
            'deltas': np.array(deltas),
            'rewards': np.array(rewards),
            'V_TD': np.array(agent.V_TD.copy()),
            'V_MB': np.array(agent.V_MB.copy()),
        }
        exp_data['episodes'].append(ep_data)

    v_td = exp_data['episodes'][-1]['V_TD']
    v_mb = exp_data['episodes'][-1]['V_MB']
    rpe = exp_data['episodes'][-1]['deltas']
    plt.plot_demo_v(v_td, v_mb)
    plt.plot_demo_rpe(rpe)

def compare_architectures(n_episodes: int = int(5e3), n_states: int = 10,
                          goal_state: int = 9, reward_value: float = 1.0) -> None:
    """Compare dual-process architectures against standard TD.
    
    Trains normative dual-process, alternative dual-process, and standard TD agents.
    Plots value error convergence over episodes.
    """
    env = Track(n_states=n_states, goal_state=goal_state, reward_value=reward_value)
    normative_agent = DualTab(
        n_states, n_actions=1, goal_state=goal_state, init_R=reward_value,
        alpha_mb=ALPHA_MB, alpha=ALPHA, gamma=GAMMA, k=K
    )
    naive_agent = DualNaiveTab(
        n_states, n_actions=1, goal_state=goal_state, init_R=reward_value,
        alpha_mb=ALPHA_MB, alpha=ALPHA, gamma=GAMMA, k=K
    )
    td_agent = TDTab(n_states, n_actions=1, alpha=ALPHA, gamma=GAMMA)
    normative_ep = TrackEpisode(env, normative_agent)
    naive_ep = TrackEpisode(env, naive_agent)
    td_ep = TrackEpisode(env, td_agent)
    exp_data = {
        'normative_episodes': [],
        'naive_episodes': [],
        'td_episodes': [],
    }

    for ep_n in range(n_episodes):
        (states_normative, steps_normative, actions_normative,
         deltas_normative, rewards_normative) = normative_ep.run()
        states_naive, steps_naive, actions_naive, deltas_naive, rewards_naive = naive_ep.run()
        states_td, steps_td, actions_td, deltas_td, rewards_td = td_ep.run()
        normative_ep_data = {
            'ep': ep_n,
            'states': np.array(states_normative),
            'steps': np.array(steps_normative),
            'actions': np.array(actions_normative),
            'deltas': np.array(deltas_normative),
            'rewards': np.array(rewards_normative),
            'V_TD': np.array(normative_agent.V_TD.copy()),
            'V_MB': np.array(normative_agent.V_MB.copy()),
        }
        naive_ep_data = {
            'ep': ep_n,
            'states': np.array(states_naive),
            'steps': np.array(steps_naive),
            'actions': np.array(actions_naive),
            'deltas': np.array(deltas_naive),
            'rewards': np.array(rewards_naive),
            'V_TD': np.array(naive_agent.V_TD.copy()),
            'V_MB': np.array(naive_agent.V_MB.copy()),
        }
        td_ep_data = {
            'ep': ep_n,
            'states': np.array(states_td),
            'steps': np.array(steps_td),
            'actions': np.array(actions_td),
            'deltas': np.array(deltas_td),
            'rewards': np.array(rewards_td),
            'V_TD': np.array(td_agent.V.copy()),
        }
        exp_data['normative_episodes'].append(normative_ep_data)
        exp_data['naive_episodes'].append(naive_ep_data)
        exp_data['td_episodes'].append(td_ep_data)
    plt.plot_compare_architectures(exp_data)

# ------------------------------------------------------------
# Guru et al., bioRxiv, 2020
# ------------------------------------------------------------

def guru_track(n_episodes: int = 100, n_sessions: int = 18, n_states: int = 35,
               sessions_to_plot: list[int] | None = None, goal_state: int = 34,
               reward_values: list[float] | None = None,
               swap_session: int | None = 17) -> None:
    """Simulate linear track experiment (Guru et al., 2020).
    
    Trains agents on tracks with different reward magnitudes.
    Plots RPE slopes across sessions and reward value swap effects.
    """
    if sessions_to_plot is None:
        sessions_to_plot = [0, 4, 8, 17]
    if reward_values is None:
        reward_values = [1.0, 2.0]
    
    exp_data = {}
    mean_rpe = {}
    rpe_slope = {}
    
    for rw_val in reward_values:
        exp_data[rw_val] = {}
        mean_rpe[rw_val] = {}
        rpe_slope[rw_val] = {}
    
    envs = {}
    agents = {}
    episodes = {}
    
    for rw_val in reward_values:
        envs[rw_val] = Track(n_states=n_states, goal_state=goal_state, reward_value=rw_val)
        agents[rw_val] = DualTab(
            n_states, n_actions=1, goal_state=goal_state,
            alpha_mb=ALPHA_MB, alpha=0.005, gamma=GAMMA, k=K
        )
        episodes[rw_val] = TrackEpisode(envs[rw_val], agents[rw_val])
    
    active_rewards = {rw_val: rw_val for rw_val in reward_values}
    
    def swap_reward_values():
        """Swap reward values between the two tracks."""
        rw_vals = list(reward_values)
        (active_rewards[rw_vals[0]],
         active_rewards[rw_vals[1]]) = (active_rewards[rw_vals[1]],
                                        active_rewards[rw_vals[0]])
        for original_rw in reward_values:
            new_rw = active_rewards[original_rw]
            envs[original_rw].reward_value = new_rw
            envs[original_rw].set_goal_state(goal_state)
        for original_rw in reward_values:
            agents[original_rw].compute_mb_val(n_states, goal_state)

    for s_n in range(n_sessions):
        if swap_session is not None and s_n == swap_session:
            swap_reward_values()
        
        for original_rw in reward_values:
            active_rw = active_rewards[original_rw]
            
            if s_n not in exp_data[active_rw]:
                exp_data[active_rw][s_n] = {}
            
            session_rpes = []
            env = envs[original_rw]
            agent = agents[original_rw]
            ep = episodes[original_rw]
            
            for ep_n in range(n_episodes):
                exp_data[active_rw][s_n][ep_n] = {}
                exp_data[active_rw][s_n][ep_n]['V_MB'] = np.array(agent.V_MB.copy())
                states, steps, actions, deltas, rewards = ep.run()
                exp_data[active_rw][s_n][ep_n]['states'] = np.array(states)
                exp_data[active_rw][s_n][ep_n]['steps'] = np.array(steps)
                exp_data[active_rw][s_n][ep_n]['actions'] = np.array(actions)
                exp_data[active_rw][s_n][ep_n]['deltas'] = np.array(deltas)
                exp_data[active_rw][s_n][ep_n]['rewards'] = np.array(rewards)
                exp_data[active_rw][s_n][ep_n]['V_TD'] = np.array(agent.V_TD.copy())
                session_rpes.append(deltas) 

            session_rpes = np.array(session_rpes)
            mean_rpe[active_rw][s_n] = np.mean(session_rpes, axis=0)
            rise =  mean_rpe[active_rw][s_n][-1] - mean_rpe[active_rw][s_n][0]
            run = len(mean_rpe[active_rw][s_n]) - 1
            rpe_slope[active_rw][s_n] = rise / run

    exp_data_subset = {}
    rw_val_subset = 2.0
    if rw_val_subset in exp_data and 0 in exp_data[rw_val_subset]:
        exp_data_subset[rw_val_subset] = {}
        exp_data_subset[rw_val_subset][0] = {}
        for ep_n in range(min(3, n_episodes)):  # First three episodes
            if ep_n in exp_data[rw_val_subset][0]:
                exp_data_subset[rw_val_subset][0][ep_n] = exp_data[rw_val_subset][0][ep_n]
    
    plt.plot_guru_track_rpe(
        mean_rpe, reward_vals=[2.0], sessions_to_plot=sessions_to_plot, colors='red'
    )
    plt.plot_guru_track_rpe(
        mean_rpe, reward_vals=[1.0], sessions_to_plot=sessions_to_plot, colors='black'
    )
    plt.plot_guru_track_rpe_slope(rpe_slope)
    
    if len(exp_data_subset) > 0 and rw_val_subset in exp_data_subset:
        plt.plot_guru_track_rpe_with_vmb(
            exp_data_subset, reward_vals=[rw_val_subset], save_fig=True, colors='red'
        )
    
    rw_val = 1.0
    first_session_last_ep = exp_data[rw_val][0][n_episodes-1]
    plt.plot_guru_v(first_session_last_ep['V_TD'], first_session_last_ep['V_MB'], 
                save_fig=True, filename=f"figs/guru_track_v_first_session.pdf")
    
    final_session_last_ep = exp_data[rw_val][4][n_episodes-1]
    plt.plot_guru_v(final_session_last_ep['V_TD'], final_session_last_ep['V_MB'], 
                save_fig=True, filename=f"figs/guru_track_v_final_session.pdf")

def guru_wheel(n_episodes: int = 100, n_sessions: int = 10, n_states: int = 10,
               goal_state: int = 9, reward_value: float = 2.0) -> None:
    """Simulate running wheel experiment (Guru et al., 2020).
    
    Trains DualFeature agent on wheel environment.
    Plots RPE evolution across sessions.
    """
    exp_data = {}
    mean_rpe = {}
    rpe_slope = {}

    env = Wheel(n_states=n_states, goal_state=goal_state, reward_value=reward_value)
    agent = DualFeature(
        env.phi_TD, env.phi_MB, n_actions=1, goal_state=goal_state,
        init_R=reward_value, alpha_mb=ALPHA_MB, alpha=ALPHA, gamma=GAMMA, k=K
    )
    ep = WheelEpisode(env, agent)

    for s_n in range(n_sessions):
        exp_data[s_n] = {}

        for ep_n in range(n_episodes):
            exp_data[s_n][ep_n] = {}
            states, steps, actions, deltas, rewards = ep.run()
            exp_data[s_n][ep_n]['states'] = np.array(states)
            exp_data[s_n][ep_n]['steps'] = np.array(steps)
            exp_data[s_n][ep_n]['actions'] = np.array(actions)
            exp_data[s_n][ep_n]['deltas'] = np.array(deltas) 
            exp_data[s_n][ep_n]['delta_slope'] = (deltas[-1] - deltas[0]) / (len(deltas) - 1)
            exp_data[s_n][ep_n]['rewards'] = np.array(rewards)
            exp_data[s_n][ep_n]['w_TD'] = np.array(agent.w_TD)
            exp_data[s_n][ep_n]['w_MB'] = np.array(agent.w_MB)
            exp_data[s_n][ep_n]['V_TD'] = np.dot(env.phi_TD, agent.w_TD)
            exp_data[s_n][ep_n]['V_MB'] = np.dot(env.phi_MB, agent.w_MB)
        
        mean_rpe[s_n] = np.mean(
            [exp_data[s_n][ep_n]['deltas'] for ep_n in range(n_episodes)], axis=0
        )
        rpe_slope[s_n] = np.mean(
            [exp_data[s_n][ep_n]['delta_slope'] for ep_n in range(n_episodes)]
        )

    plt.plot_guru_wheel_rpe(mean_rpe)
    plt.plot_guru_wheel_rpe_slope(rpe_slope)

# ------------------------------------------------------------
# Kim et al., Cell, 2020
# ------------------------------------------------------------

def kim_distance(n_episodes: int = 200, n_states: int = 32, goal_state: int = 31,
                 reward_value: float = 1.0, teleport_distances: list[int] | None = None,
                 teleport_destination: int = 23) -> None:
    """Simulate teleport distance experiment (Kim et al., 2020).
    
    Tests RPE responses to teleports of varying distances.
    """
    if teleport_distances is None:
        teleport_distances = [0, 1, 2, 10]
    env = Track(n_states=n_states, goal_state=goal_state, reward_value=reward_value)
    agent = DualTab(
        n_states, n_actions=1, goal_state=goal_state, init_R=reward_value,
        alpha_mb=ALPHA_MB, alpha=ALPHA, gamma=GAMMA, k=K
    )
    train_ep = TrackEpisode(env, agent)
    for ep_n in range(n_episodes):
        states, steps, actions, deltas, rewards = train_ep.run()
    
    exp_data = {
        'teleport_condition': [],
    }
    agent.alpha = 0.0
    agent.alpha_mb = 0.0
    for d in teleport_distances:
        env.teleport_destination = teleport_destination
        if d == 0:  # pause condition
            test_ep = PauseEpisode(env, teleport_destination, 65, agent)
        else:  # teleport conditions
            test_ep = TeleportEpisode(env, teleport_destination, d, agent)
        states, steps, actions, deltas, rewards, teleport_start = test_ep.run()
        ep_data = {
            'teleport_distance': d,
            'teleport_destination': teleport_destination,
            'teleport_start': teleport_start,
            'states': np.array(states),
            'steps': np.array(steps),
            'actions': np.array(actions),
            'deltas': np.array(deltas),
            'rewards': np.array(rewards),
            'V_TD': np.array(agent.V_TD.copy()),
            'V_MB': np.array(agent.V_MB.copy()),
        }
        exp_data['teleport_condition'].append(ep_data)
    plt.plot_kim_distance(exp_data)

def kim_location(n_episodes: int = 200, n_states: int = 32, goal_state: int = 31,
                 reward_value: float = 1.0, teleport_loc: list[int] | None = None,
                 teleport_dist: int = 10) -> None:
    """Simulate teleport location experiment (Kim et al., 2020).
    
    Tests RPE responses to teleports at different track positions.
    """
    if teleport_loc is None:
        teleport_loc = [0, 2, 8, 14]
    env = Track(n_states=n_states, goal_state=goal_state, reward_value=reward_value)
    agent = DualTab(
        n_states, n_actions=1, goal_state=goal_state, init_R=reward_value,
        alpha_mb=ALPHA_MB, alpha=ALPHA, gamma=GAMMA, k=K
    )
    train_ep = TrackEpisode(env, agent)
    for ep_n in range(n_episodes):
        states, steps, actions, deltas, rewards = train_ep.run()
    
    exp_data = {
        'teleport_condition': [],
    }
    agent.alpha = 0.0
    agent.alpha_mb = 0.0
    for l in teleport_loc:
        if l == 0:  # standard condition
            test_ep = TrackEpisode(env, agent)
            states, steps, actions, deltas, rewards = test_ep.run()
        else:  # teleport conditions
            teleport_destination = l + teleport_dist
            env.teleport_destination = teleport_destination
            test_ep = TeleportEpisode(env, teleport_destination, teleport_dist, agent)
            states, steps, actions, deltas, rewards, teleport_start = test_ep.run()
        ep_data = {
            'teleport_location': l,
            'states': np.array(states),
            'steps': np.array(steps),
            'actions': np.array(actions),
            'deltas': np.array(deltas),
            'rewards': np.array(rewards),
            'V_TD': np.array(agent.V_TD.copy()),
            'V_MB': np.array(agent.V_MB.copy()),
        }
        exp_data['teleport_condition'].append(ep_data)
    plt.plot_kim_location(exp_data)

def kim_speed(n_episodes: int = 1000, n_states: int = 41, goal_state: int = 40,
              reward_value: float = 1.0, step_sizes: list[int] | None = None) -> None:
    """Simulate traversal speed experiment (Kim et al., 2020).
    
    Tests RPE responses at different movement speeds.
    """
    if step_sizes is None:
        step_sizes = [1, 2, 4]
    env = Track(n_states=n_states, goal_state=goal_state, reward_value=reward_value)
    agent = DualTab(
        n_states, n_actions=1, goal_state=goal_state, init_R=reward_value,
        alpha_mb=ALPHA_MB, alpha=ALPHA, gamma=GAMMA, k=K
    )
    train_ep = TrackEpisode(env, agent)
    for ep_n in range(n_episodes):
        states, steps, actions, deltas, rewards = train_ep.run()
    
    exp_data = {
        'step_size': [],
    }
    agent.alpha = 0.0
    agent.alpha_mb = 0.0
    for s in step_sizes:
        env.step_size = s
        test_ep = TrackEpisode(env, agent)
        states, steps, actions, deltas, rewards = test_ep.run()
        ep_data = {
            'step_size': s,
            'states': np.array(states),
            'steps': np.array(steps),
            'actions': np.array(actions),
            'deltas': np.array(deltas),
            'rewards': np.array(rewards),
            'V_TD': np.array(agent.V_TD.copy()),
            'V_MB': np.array(agent.V_MB.copy()),
        }
        exp_data['step_size'].append(ep_data)
    plt.plot_kim_speed(exp_data)

# ------------------------------------------------------------
# Mikhael et al., Current Biology, 2021
# ------------------------------------------------------------

def compare_uncertainty_assumptions(n_episodes: int = 1000, n_states: int = 50,
                                    goal_state: int = 49,
                                    reward_value: float = 1.0) -> None:
    """Compare static vs. dynamic state uncertainty (Mikhael et al., 2022).
    
    Tests value learning under constant vs. increasing uncertainty.
    """
    # Condition 1: phi_t == phi_t_prime (constant kernel)
    c_width = 0.1
    env_c = StaticUncertaintyTrack(
        n_states=n_states, goal_state=goal_state,
        reward_value=reward_value, s_width=c_width
    )
    agent_c = TDFeature(env_c.phi_t, n_actions=1, alpha=0.1, gamma=0.9)
    
    # Condition 2: phi_t < phi_t_prime (different constant kernels, phi_t smaller)
    s_width = 0.1   # Small kernel for phi_t
    l_width = 3.0  # Large kernel for phi_t_prime
    env_d = DynamicUncertaintyTrack(
        n_states=n_states, goal_state=goal_state,
        reward_value=reward_value, s_width=s_width, l_width=l_width
    )
    agent_d = TDFeatureMikhael(env_d.phi_t, env_d.phi_t_prime, n_actions=1, alpha=0.1, gamma=0.9)
    
    # Run condition 1
    ep_c = TrackEpisode(env_c, agent_c)
    data_c = {
        'episodes': []
    }
    
    for ep_n in range(n_episodes):
        states, steps, actions, deltas, rewards = ep_c.run()
        V = np.dot(env_c.phi_t, agent_c.w)
        ep_data = {
            'ep': ep_n,
            'states': np.array(states),
            'steps': np.array(steps),
            'actions': np.array(actions),
            'deltas': np.array(deltas),
            'rewards': np.array(rewards),
            'V': np.array(V),
        }
        data_c['episodes'].append(ep_data)

    data_d = {
        'episodes': []
    }
    ep_d = TrackEpisode(env_c, agent_d)
    for ep_n in range(n_episodes):
        states, steps, actions, deltas, rewards = ep_d.run()
        V = np.dot(env_d.phi_t, agent_d.w)
        ep_data = {
            'ep': ep_n,
            'states': np.array(states),
            'steps': np.array(steps),
            'actions': np.array(actions),
            'deltas': np.array(deltas),
            'rewards': np.array(rewards),
            'V': np.array(V),
        }
        data_d['episodes'].append(ep_data)
    
    exp_data = {
        'exp_c': data_c,
        'exp_d': data_d,
    }
    
    plt.plot_compare_mikhael(exp_data)
    plt.plot_compare_mikhael_error(exp_data)
    plt.plot_blurred_v()

def mikhael_track(n_training_episodes: int = 150, n_states: int = 87,
                  goal_state: int = 86, reward_value: float = 1.0) -> None:
    """Simulate darkening track experiment (Mikhael et al., 2022).
    
    Tests RPE responses under varying lighting and speed conditions.
    """
    dark_s = 4/87 * n_states
    dark_e = 24/87 * n_states
    bright_s = 4/87 * n_states
    bright_e = 4/87 * n_states

    dark_env = DarkTrack(
        n_states=n_states, goal_state=goal_state, reward_value=reward_value,
        sigma_s=dark_s, sigma_e=dark_e, method='exponential', plot_kernels=True
    )
    bright_env = DarkTrack(
        n_states=n_states, goal_state=goal_state, reward_value=reward_value,
        sigma_s=bright_s, sigma_e=bright_e, method='constant'
    )

    agent = DualFeature(
        dark_env.phi_t, dark_env.phi_t, n_actions=1, goal_state=goal_state,
        init_R=reward_value, alpha_mb=ALPHA_MB, alpha=ALPHA, gamma=GAMMA, k=K
    )
    dark_ep = TrackEpisode(dark_env, agent)
    bright_ep = TrackEpisode(bright_env, agent)

    # ========== TRAINING PHASE ==========
    bright_env.step_size = 1
    agent.phi_MB = bright_env.phi_t
    agent.phi_TD = bright_env.phi_t
    
    for ep_n in range(n_training_episodes):
        states, steps, actions, deltas, rewards = bright_ep.run()
    
    trained_w_TD = agent.w_TD.copy()
    trained_w_MB = agent.w_MB.copy()
    trained_R = agent.R.copy()
    
    # ========== TEST PHASE ==========
    original_alpha = agent.alpha
    original_alpha_mb = agent.alpha_mb
    agent.alpha = 0.0
    agent.alpha_mb = 0.0
    
    test_rpes = {}
    test_conditions = [
        (False, False),  # Bright, slow
        (False, True),   # Bright, fast
        (True, False),   # Dim, slow
        (True, True)     # Dim, fast
    ]
    
    for is_dark, is_fast in test_conditions:
        agent.w_TD = trained_w_TD.copy()
        agent.w_MB = trained_w_MB.copy()
        agent.R = trained_R.copy()
        
        if is_fast:
            dark_env.step_size = 2
            bright_env.step_size = 2
        else:
            dark_env.step_size = 1
            bright_env.step_size = 1
        
        if is_dark:
            agent.phi_MB = dark_env.phi_t
            agent.phi_TD = dark_env.phi_t
            states, steps, actions, deltas, rewards = dark_ep.run()
        else:
            agent.phi_MB = bright_env.phi_t
            agent.phi_TD = bright_env.phi_t
            states, steps, actions, deltas, rewards = bright_ep.run()
        
        test_rpes[(is_dark, is_fast)] = deltas
    
    agent.alpha = original_alpha
    agent.alpha_mb = original_alpha_mb
    
    # ========== VISUALIZATION ==========
    plt.plot_mikhael(test_rpes)

# ------------------------------------------------------------
# Krausz et al., Neuron, 2023
# ------------------------------------------------------------

def krausz_grid(n_training_episodes: int = 500, grid: tuple[int, int] = (10, 10),
                goal_state: tuple[int, int] = (0, 0), p_reward: float = 0.50) -> None:
    """Simulate gridworld experiment (Krausz et al., 2023).
    
    Trains agent then runs 3-trial test: omission, reward, probabilistic.
    """
    env = KrauszGrid(grid=grid, goal=goal_state, p_reward=p_reward)
    agent = DualGrid(
        grid=grid, n_actions=4, goal_state=goal_state, init_R=0,
        alpha_mb=0.10, alpha=ALPHA, gamma=0.85, k=K
    )
    ep = KrauszGridEpisode(env, agent)
    
    # ========== TRAINING PHASE ==========
    for ep_n in range(n_training_episodes):
        ep.start_state = (9, 0) if ep_n % 2 == 0 else (0, 9)
        states, timesteps, actions, rpes, rewards = ep.run()
    
    # ========== TEST PHASE ==========
    test_data = {
        'trial_1': {},
        'trial_2': {},
        'trial_3': {},
    }
    
    original_p_reward = env.p_reward
    
    # Trial 1: Omission
    ep.start_state = (9, 0)
    env.p_reward = 0.0
    states, timesteps, actions, rpes, rewards = ep.run()
    env.p_reward = original_p_reward
    test_data['trial_1'] = {
        'states': np.array(states),
        'timesteps': np.array(timesteps),
        'actions': np.array(actions),
        'rpes': np.array(rpes),
        'rewards': np.array(rewards),
        'V_TD': np.array(agent.V_TD.copy()),
        'V_MB': np.array(agent.V_MB.copy()),
    }
    
    # Trial 2: Reward
    ep.start_state = (0, 9)
    env.p_reward = 1.0
    states, timesteps, actions, rpes, rewards = ep.run()
    env.p_reward = original_p_reward
    test_data['trial_2'] = {
        'states': np.array(states),
        'timesteps': np.array(timesteps),
        'actions': np.array(actions),
        'rpes': np.array(rpes),
        'rewards': np.array(rewards),
        'V_TD': np.array(agent.V_TD.copy()),
        'V_MB': np.array(agent.V_MB.copy()),
    }
    
    # Trial 3: Probabilistic
    ep.start_state = (9, 0)
    states, timesteps, actions, rpes, rewards = ep.run()
    test_data['trial_3'] = {
        'states': np.array(states),
        'timesteps': np.array(timesteps),
        'actions': np.array(actions),
        'rpes': np.array(rpes),
        'rewards': np.array(rewards),
        'V_TD': np.array(agent.V_TD.copy()),
        'V_MB': np.array(agent.V_MB.copy()),
    }
    
    # ========== VISUALIZATION ==========
    v_mb_trial_1 = test_data['trial_1']['V_MB']
    states_trial_1 = test_data['trial_1']['states']
    v_mb_trial_2 = test_data['trial_2']['V_MB']
    states_trial_2 = test_data['trial_2']['states']
    
    v_min = 0.15
    v_max = 0.60
    
    rpe_trial_2 = test_data['trial_2']['rpes']
    rpe_trial_3 = test_data['trial_3']['rpes']
    
    min_len = min(len(rpe_trial_2), len(rpe_trial_3))
    rpe_trial_2_truncated = rpe_trial_2[:min_len]
    rpe_trial_3_truncated = rpe_trial_3[:min_len]
    
    plt.plot_krausz_v_mb(
        v_mb_trial_1, states=states_trial_1, vmin=v_min, vmax=v_max, 
        save_fig=True, filename="figs/krausz_v_mb_trial_1.pdf"
    )
    plt.plot_krausz_v_mb(
        v_mb_trial_2, states=states_trial_2, vmin=v_min, vmax=v_max, 
        save_fig=True, filename="figs/krausz_v_mb_trial_2.pdf"
    )
    plt.plot_rpe_krausz(
        rpe_trial_2_truncated, rpe_trial_3_truncated, 
        save_fig=True, filename="figs/krausz_rpe_trials_2_3.pdf"
    )


