"""
Visualisation functions.
"""
import numpy as np
import matplotlib.pyplot as plt
from config import *

__all__ = [
    'preprocess_rpes',
    'plot_demo_v', 'plot_demo_rpe', 'plot_compare_architectures',
    'plot_guru_track_rpe', 'plot_guru_track_rpe_with_vmb', 'plot_guru_track_rpe_slope', 'plot_guru_v',
    'plot_guru_wheel_rpe', 'plot_guru_wheel_rpe_slope',
    'plot_kim_distance', 'plot_kim_location', 'plot_kim_speed',
    'plot_mikhael', 'plot_blurred_v', 'plot_compare_mikhael', 'plot_compare_mikhael_error', 'plot_kernels',
    'plot_krausz_v_mb', 'plot_rpe_krausz'
]

# Global plotting parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
FONT_SIZE_LABEL = 16
FONT_SIZE_TICK = 12
FONT_SIZE_AXIS = 12
FONT_SIZE_LEGEND = 12
LINEWIDTH = 3
SPINE_WIDTH = 2


def preprocess_rpes(rpes, censor_final=True, scale_negative=True):
    """Preprocess RPEs by censoring final state and scaling negative values.
    
    Args:
        rpes: Array of reward prediction errors.
        censor_final: If True, exclude the final state RPE.
        scale_negative: If True, divide negative RPEs by 5.
    
    Returns:
        Preprocessed RPE array.
    """
    rpes = np.array(rpes)
    if censor_final:
        rpes = rpes[:-1]
    if scale_negative:
        rpes = np.where(rpes < 0, rpes / 5, rpes)
    return rpes


# ------------------------------------------------------------
# Experiment: Demo
# ------------------------------------------------------------

def plot_demo_v(v_td, v_mb, save_fig=True, filename="figs/demo_v.pdf"):
    """Plot learned value functions (V_TD, V_MB, V_NET) across states (Fig. 1D)."""
    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.plot(v_td[0:-1], color='pink', linewidth=2, label='TD')
    ax.plot(v_mb[0:-1], color='darkred', linewidth=2, label='MB')
    ax.plot(K*v_mb[0:-1] + (1-K)*v_td[0:-1], color='red', linewidth=LINEWIDTH, linestyle='--', label='NET')
    ax.axvline(x=len(v_td)-2, color='black', linestyle=':', linewidth=1, alpha=1)
    
    ax.set_xlabel("State", fontsize=FONT_SIZE_LABEL)
    ax.set_xticks([0, len(v_td)-2])
    ax.set_xticklabels(['Start', 'Goal'], fontsize=FONT_SIZE_TICK)
    
    ax.set_ylabel("Value", fontsize=FONT_SIZE_LABEL)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, 0.5, 1], fontsize=FONT_SIZE_TICK)
    
    ax.legend(title='Component', title_fontsize=FONT_SIZE_LEGEND, loc='upper left', fontsize=FONT_SIZE_LEGEND, frameon=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_demo_rpe(early_rpe, mid_rpe, late_rpe, save_fig=True, filename="figs/demo_standard_td_rpe.pdf"):
    """Plot demo RPEs at early, mid, and late training (three lines, grayscale)."""
    fig, ax = plt.subplots(figsize=(3, 3))
    colors = ['0.55', '0.35', '0.15']
    labels = ['Early', 'Mid', 'Late']
    for rpe, color, label in zip([early_rpe, mid_rpe, late_rpe], colors, labels):
        rpe = preprocess_rpes(rpe)
        ax.plot(rpe, color=color, linewidth=LINEWIDTH, label=label)
    ax.axvline(x=len(preprocess_rpes(early_rpe)) - 1, color='black', linestyle=':', linewidth=1, alpha=1)
    ax.set_xlabel("State", fontsize=FONT_SIZE_LABEL)
    ax.set_xticks([0, len(preprocess_rpes(early_rpe)) - 1])
    ax.set_xticklabels(['Start', 'Goal'], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(-0.025, 0.38)
    ax.set_ylabel("RPE", fontsize=FONT_SIZE_LABEL)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(['0.0', '0.1', '0.2', '0.3'], fontsize=FONT_SIZE_TICK)
    ax.legend(
        title='Training-stage',
        title_fontsize=FONT_SIZE_LEGEND,
        loc='center left',
        bbox_to_anchor=(0.0, 0.6),
        fontsize=FONT_SIZE_LEGEND,
        frameon=False,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    plt.tight_layout()
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)


def plot_compare_architectures(exp_data, save_fig=True, filename="figs/compare_agents.pdf"):
    """Plot value error convergence for different agent architectures (Fig. 1C)."""
    fig, ax = plt.subplots(figsize=(3.5, 3))

    cmap = plt.colormaps['magma']
    colors = {
        'naive': cmap(0.2),
        'normative': cmap(0.5),
        'td': cmap(0.8),
    }
    n_states = exp_data['naive_episodes'][0]['states'].shape[0]
    states = np.arange(n_states)
    true_values = GAMMA ** (n_states - 1 - states)
    
    for agent_type in ['normative', 'td', 'naive']:
        episodes_key = f'{agent_type}_episodes'
        if episodes_key not in exp_data:
            continue
            
        episodes = exp_data[episodes_key]
        n_episodes = len(episodes)
        
        avg_differences = []
        for ep in episodes:
            v_td = np.array(ep['V_TD'])
            v_mb = np.array(true_values)
            avg_diff = np.mean(v_mb - v_td)
            avg_differences.append(avg_diff)
        
        episodes_x = np.log10(np.arange(1, n_episodes + 1))
        
        if agent_type == 'naive':
            label = 'Dual \nprocess (Alt.)'
        elif agent_type == 'normative':
            label = 'Dual\nprocess'
        elif agent_type == 'td':
            label = 'Std.\nTD'
            
        ax.plot(episodes_x, avg_differences,
                color=colors[agent_type],
                linewidth=2,
                label=label)
    
    ax.set_xlabel("log(Trial) [N]", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Value error", fontsize=FONT_SIZE_LABEL)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([0, 1, 2, 3], fontsize=FONT_SIZE_TICK)
    
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8], fontsize=FONT_SIZE_TICK)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    
    ax.legend(title="Model", title_fontsize=FONT_SIZE_LEGEND, alignment='left', loc='lower left', fontsize=FONT_SIZE_LEGEND, frameon=False)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

# ------------------------------------------------------------
# Experiment: Guru et al., bioRxiv, 2020 
# ------------------------------------------------------------
def plot_guru_track_rpe(rpes, reward_vals=[2.0, 1.0], sessions_to_plot=None, save_fig=True, filename_prefix="figs/guru_track_rpe", colors=None):
    """Plot mean RPE across states for selected sessions (Fig. 2D)."""
    if colors is None:
        cmap = {2.0: 'red', 1.0: 'blue'}
    elif isinstance(colors, str):
        cmap = {rw_val: colors for rw_val in reward_vals}
    elif isinstance(colors, dict):
        cmap = colors
    else:
        raise ValueError("colors must be None, a string, or a dictionary")
    
    if sessions_to_plot is None:
        sessions_to_plot = list(range(len(rpes[reward_vals[0]])))
    
    n_cols = len(sessions_to_plot)
    n_rows = 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.7, 3.5), squeeze=True, sharey=True)
    
    if n_cols == 1:
        axes = [axes]
    
    for rw_val in reward_vals:
        for i, s_n in enumerate(sessions_to_plot):
            ax = axes[i]
            s_rpe = preprocess_rpes(rpes[rw_val][s_n])
            line_color = cmap.get(rw_val, 'red')
            ax.plot(s_rpe, color=line_color, linewidth=LINEWIDTH+1)

            ax.set_title(f'Session {s_n+1}', fontsize=FONT_SIZE_TICK, loc='left')

            ax.set_yticks([-0.5, 0, 0.5, 1])
            ax.set_xticks([0, len(s_rpe)-1]) 
            if i == 0:
                ax.set_xticklabels(['Start', 'Goal'], fontsize=FONT_SIZE_TICK)
            else:
                ax.set_xticklabels(['', ''])
            
            ax.set_yticklabels([-0.5, 0.0, 0.5, 1.0], fontsize=FONT_SIZE_TICK)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(width=0)
            ax.spines['left'].set_linewidth(SPINE_WIDTH)
            ax.spines['bottom'].set_linewidth(SPINE_WIDTH)
    
    fig.text(0.5, 0.05, 'State', ha='center', va='center', fontsize=FONT_SIZE_LABEL)
    fig.text(0.05, 0.5, 'Mean RPE', ha='center', va='center', rotation='vertical', fontsize=FONT_SIZE_LABEL)
    plt.subplots_adjust(wspace=0.3, hspace=0.15)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    if save_fig:
        rw_val_str = "_".join([str(rw_val) for rw_val in reward_vals])
        filename = f"{filename_prefix}_rw{rw_val_str}_combined.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_guru_track_rpe_with_vmb(exp_data_subset, reward_vals=[2.0, 1.0], save_fig=True, filename_prefix="figs/guru_track_rpe_vmb", colors=None):
    """Plot RPEs with V_MB heatmaps inset for early trials (Fig. 2E)."""
    if colors is None:
        cmap = {2.0: 'red', 1.0: 'blue'}
    elif isinstance(colors, str):
        cmap = {rw_val: colors for rw_val in reward_vals}
    elif isinstance(colors, dict):
        cmap = colors
    else:
        raise ValueError("colors must be None, a string, or a dictionary")
    
    episodes_to_plot = []
    for rw_val in reward_vals:
        if rw_val in exp_data_subset:
            for s_n in sorted(exp_data_subset[rw_val].keys()):
                for ep_n in sorted(exp_data_subset[rw_val][s_n].keys()):
                    episodes_to_plot.append((rw_val, s_n, ep_n))
    
    if len(episodes_to_plot) == 0:
        raise ValueError("exp_data_subset contains no episodes")
    
    n_cols = len(episodes_to_plot)
    n_rows = 1
    
    all_v_mb = []
    for rw_val, s_n, ep_n in episodes_to_plot:
        all_v_mb.append(exp_data_subset[rw_val][s_n][ep_n]['V_MB'])
    
    all_v_mb = np.array(all_v_mb)
    vmin = np.min(all_v_mb)
    vmax = np.max(all_v_mb)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.7, 4.5), squeeze=True, sharey=True)
    
    if n_cols == 1:
        axes = [axes]
    
    im = None 
    for i, (rw_val, s_n, ep_n) in enumerate(episodes_to_plot):
        ax = axes[i]
        
        episode_data = exp_data_subset[rw_val][s_n][ep_n]
        rpe = preprocess_rpes(episode_data['deltas'])
        v_mb = episode_data['V_MB']
        
        if i == 0:
            line_color = 'black'
        else:
            line_color = cmap.get(rw_val, 'red')
        
        ax.plot(rpe, color=line_color, linewidth=LINEWIDTH+1)
        
        inset_ax = ax.inset_axes([0.15, 0.90, 0.70, 0.05])
        
        v_mb_2d = v_mb.reshape(1, -1)
        
        im = inset_ax.imshow(
            v_mb_2d,
            aspect='auto',
            cmap='Reds',
            vmin=0,
            vmax=2,
            interpolation='nearest',
        )
        
        n_states = len(v_mb)
        inset_ax.set_xticks([0, n_states - 1])
        inset_ax.set_xticklabels(['Start', 'Goal'], fontsize=10)
        inset_ax.set_yticks([])
        inset_ax.spines['top'].set_visible(True)
        inset_ax.spines['right'].set_visible(True)
        inset_ax.spines['bottom'].set_visible(True)
        inset_ax.spines['left'].set_visible(True)
        for spine in inset_ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5)
        
        ax.set_yticks([-0.5, 0, 0.5, 1])
        ax.set_xticks([0, len(rpe)-1]) 
        if i == 0:
            ax.set_xticklabels(['Start', 'Goal'], fontsize=FONT_SIZE_TICK)
        else:
            ax.set_xticklabels(['', ''])
        
        ax.set_yticklabels([-0.5, 0.0, 0.5, 1.0], fontsize=FONT_SIZE_TICK)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(width=0)
        ax.spines['left'].set_linewidth(SPINE_WIDTH)
        ax.spines['bottom'].set_linewidth(SPINE_WIDTH)
    
    if im is not None:
        cbar_ax = fig.add_axes([0.90, 0.25, 0.04, 0.5])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.ax.set_title('V$_{MB}$', fontsize=FONT_SIZE_TICK, pad=10)
        cbar.ax.set_yticks([0, 1, 2])
        cbar.ax.tick_params(labelsize=FONT_SIZE_TICK-2)
        for spine in cbar_ax.spines.values():
            spine.set_linewidth(1.5)
    
    fig.text(0.5, 0.05, 'State', ha='center', va='center', fontsize=FONT_SIZE_LABEL)
    fig.text(0.05, 0.5, 'RPE', ha='center', va='center', rotation='vertical', fontsize=FONT_SIZE_LABEL)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.15, right=0.88)
    plt.tight_layout(rect=[0.05, 0.05, 0.88, 0.95])
    
    if save_fig:
        rw_val_str = "_".join([str(rw_val) for rw_val in reward_vals])
        filename = f"{filename_prefix}_rw{rw_val_str}_combined.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_guru_track_rpe_slope(rpe_slopes, reward_vals=[2.0, 1.0], save_fig=True, filename="figs/guru_track_rpe_slope.pdf"):
    """Plot RPE slope evolution across sessions for different reward values (NA)."""
    assert reward_vals == [2.0, 1.0], "Only reward values [1.0, 2.0] are supported"
    cmap = {1.0: 'black', 2.0: 'red'}

    fig, ax = plt.subplots(figsize=(3, 3))

    for rw_val in reward_vals:
        n_sessions = len(rpe_slopes[rw_val])
        x = range(n_sessions)
        y = [rpe_slopes[rw_val][s_n] for s_n in range(n_sessions)]
        ax.plot(x, y, color=cmap.get(rw_val, 'black'), marker='o', markersize=10, linestyle='-', label=f'Reward = {rw_val}')

    ax.set_xlabel("Session (N)", fontsize=15)
    ax.set_ylabel("RPE Slope", fontsize=15)
    
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_guru_v(v_td, v_mb, save_fig=True, filename="figs/demo_v.pdf"):
    """Plot learned value functions (V_TD, V_MB, V_NET) for Guru experiment (Fig. 2C)."""
    fig, ax = plt.subplots(figsize=(3, 3))
    
    ax.plot(v_td[0:-1], color='pink', linewidth=LINEWIDTH, label='V$_{TD}$')
    ax.plot(v_mb[0:-1], color='darkred', linewidth=LINEWIDTH, label='V$_{MB}$')
    ax.plot(K*v_mb[0:-1] + (1-K)*v_td[0:-1], color='red', linewidth=LINEWIDTH, linestyle = '--', label='V$_{NET}$')
    ax.axvline(x=len(v_td)-2, color='black', linestyle=':', linewidth=1, alpha = 0.5)
    
    ax.set_xlabel("State", fontsize=FONT_SIZE_LABEL)
    ax.set_xticks([0, len(v_td)-2])
    ax.set_xticklabels(['Start', 'Goal'], fontsize=FONT_SIZE_TICK)
    
    ax.set_ylabel("Value", fontsize=FONT_SIZE_LABEL)
    ax.set_yticks([0,  1.0])
    ax.set_yticklabels(['Low', 'High'], fontsize=FONT_SIZE_TICK)
    
    ax.legend(loc='upper left', fontsize=FONT_SIZE_LEGEND, frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_guru_wheel_rpe(rpes, save_fig=True, filename_prefix="figs/guru_wheel_rpe"):
    """Plot mean RPE across positions for running wheel sessions (NA)."""
    n_sessions = len(rpes)
    
    fig, axes = plt.subplots(1, n_sessions, figsize=(3*n_sessions, 6), squeeze=True, sharex=True, sharey=True)
    
    for s_n in range(n_sessions):
        ax = axes[s_n]
        s_rpe = rpes[s_n]
        ax.plot(s_rpe, color='red', linewidth=6)
        ax.set_title(f'Session {s_n+1}', fontsize=32, loc='center')

        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([0, len(s_rpe)-1]) 
        ax.set_xticklabels(['Start', 'Goal'], fontsize=28)
        ax.set_yticklabels([0, 0.5, 1], fontsize=28)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.text(0.5, -0.01, 'Position', ha='center', va='center', fontsize=42)
    fig.text(-0.01, 0.5, 'Mean RPE', ha='center', va='center', rotation='vertical', fontsize=42)
   
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_fig:
        filename = f"{filename_prefix}_combined.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_guru_wheel_rpe_slope(rpe_slopes, save_fig=True, filename="figs/guru_wheel_rpe_slope.pdf"):
    """Plot RPE slope evolution across running wheel sessions (NA)."""
    fig, ax = plt.subplots(figsize=(3, 3))

    n_sessions = len(rpe_slopes)
    x = range(n_sessions)
    y = [rpe_slopes[s_n] for s_n in range(n_sessions)]
    ax.plot(x, y, color='red', marker='o', markersize=10, linestyle='-')

    ax.set_xlabel("Session (N)", fontsize=15)
    ax.set_ylabel("RPE Slope", fontsize=15)
    
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

# ------------------------------------------------------------
# Experiment: Kim et al., Cell, 2020 
# ------------------------------------------------------------
def plot_kim_distance(exp_data, save_fig=True, filename="figs/kim_distance.pdf"):
    """Plot RPE responses to teleports of varying distances (Fig. 4C-ii)."""
    fig, ax = plt.subplots(figsize=(3.8, 3))
    colors = {0: 'gold', 1: 'black', 2: 'orange', 10: 'red'}
    conditions = {0: 'Pause', 1: 'Std', 2: 'Short', 10: 'Long'}

    for d in exp_data['teleport_condition']:
        teleport_distance = d['teleport_distance']
        teleport_start = d['teleport_start'] - 1
        deltas = preprocess_rpes(d['deltas'])
        time = (d['steps'][0:len(deltas)] - teleport_start)/4.5
        ax.plot(time, deltas, color=colors[teleport_distance], linewidth=LINEWIDTH, label=conditions[teleport_distance])
    ax.axvline(x=0, color='red', linewidth=1, alpha=0.5)
    ax.axvline(x=2, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Time (a.u.) [0=teleport]", fontsize=FONT_SIZE_LABEL)
    ax.set_xlim(-2, 4)
    ax.set_xticks([-2, 0, 2, 4])
    ax.set_xticklabels([-2, 0, 2, 4], fontsize=FONT_SIZE_TICK)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_linewidth(SPINE_WIDTH)
    ax.spines['left'].set_linewidth(SPINE_WIDTH)

    ax.set_ylabel("RPE", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(bottom=-0.05)
    ax.set_yticks([0.0, 0.1, 0.2])
    ax.set_yticklabels([0.0, 0.1, 0.2], fontsize=FONT_SIZE_TICK)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_kim_location(exp_data, save_fig=True, filename="figs/kim_location.pdf"):
    """Plot RPE responses to teleports at different track locations (Fig. 4C-i)."""
    fig, ax = plt.subplots(figsize=(3.8, 3))
    colors = {0: 'black',  2: 'red',  8: 'orange',  14: 'gold'}
    conditions = {0: 'Std', 2: 'T1', 8: 'T2', 14: 'T3'}

    temporal_scale = 3.5

    # Small vertical offsets to avoid overplotting
    offsets = {0: 0.0, 2: 0.005, 8: 0.010, 14: 0.015}
    
    for d in exp_data['teleport_condition']:
        teleport_location = d['teleport_location']
        label = conditions[teleport_location] if conditions[teleport_location] != 'Std' else None
        offset = offsets[teleport_location]
        deltas = preprocess_rpes(d['deltas'])
        ax.plot(d['steps'][0:len(deltas)]/temporal_scale, deltas + offset, color=colors[teleport_location], linewidth=LINEWIDTH, label=label)
        if conditions[teleport_location] != 'Std':
            ax.axvline(x=(teleport_location-1)/temporal_scale, color=colors[teleport_location], linewidth=1, alpha=0.5)
    ax.axvline(x=21/temporal_scale, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=30/temporal_scale, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel("Time (a.u.) [0=trial start]", fontsize=FONT_SIZE_LABEL)
    ax.set_xticks([0, 4, 8])
    ax.set_xticklabels([0, 4, 8], fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("RPE", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(bottom=-0.05)
    ax.set_yticks([0.0, 0.10, 0.20])
    ax.set_yticklabels([0.0, 0.10, 0.20], fontsize=FONT_SIZE_TICK)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_linewidth(SPINE_WIDTH)
    ax.spines['left'].set_linewidth(SPINE_WIDTH)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_kim_speed(exp_data, save_fig=True, filename="figs/kim_speed.pdf"):
    """Plot RPE responses at different movement speeds (Fig. 4C-iii)."""
    fig, ax = plt.subplots(figsize=(3.8, 3))
    colors = {1: 'red', 2: 'black', 4: 'gold'}
    conditions = {1: 'Slow', 2: 'Med', 4: 'Fast'}
    rw_time = {1: 16, 2: 8, 4: 4}

    for d in exp_data['step_size']:
        step_size = d['step_size']
        deltas = preprocess_rpes(d['deltas'])
        ax.plot((d['steps'][0:len(deltas)] + 1)/2.5, deltas, color=colors[step_size], linewidth=LINEWIDTH, label=conditions[step_size])
        ax.axvline(x=rw_time[step_size], color=colors[step_size], linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Time (a.u.) [0=trial start]", fontsize=FONT_SIZE_LABEL)
    ax.set_xticks([0, 8, 16])
    ax.set_xticklabels([0, 8, 16], fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("RPE", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(bottom=-0.05)
    ax.set_yticks([0.0, 0.10, 0.20, 0.30])
    ax.set_yticklabels([0.0, 0.10, 0.20, 0.30], fontsize=FONT_SIZE_TICK)
    plt.tight_layout()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_linewidth(SPINE_WIDTH)
    ax.spines['left'].set_linewidth(SPINE_WIDTH)
    
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

# ------------------------------------------------------------
# Experiment: Mikhael et al., Current Biology, 2021 
# ------------------------------------------------------------
def plot_mikhael(mean_rpes, save_fig=True, filename="figs/mikhael.pdf"):
    """Plot RPE responses under varying lighting and speed conditions (Fig. 5C)."""
    fig, ax = plt.subplots(figsize=(5.25, 3))
    colors = {
        (False, False): 'black',     # Bright, slow
        (False, True): 'red',        # Bright, fast
        (True, False): 'grey',       # Dim, slow
        (True, True): 'orange'       # Dim, fast
    }
    conditions = {
        (False, False): "Const., Std",
        (False, True): "Const., Fast",
        (True, False): "Dark, Std",
        (True, True): "Dark, Fast"
    }

    preprocessed_rpes = {k: preprocess_rpes(v) for k, v in mean_rpes.items()}
    n_states = max(len(deltas) for deltas in preprocessed_rpes.values())
    
    for condition in preprocessed_rpes.keys():
        deltas = preprocessed_rpes[condition]
        time = np.arange(len(deltas))/n_states * 10
        color = colors[condition]
        label = conditions[condition]
        ax.plot(time, deltas, color=color, linewidth=LINEWIDTH)

    ax.axvline(x=5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=10, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel("Time (a.u.) [0 = trial-start]", fontsize=FONT_SIZE_LABEL)
    ax.set_xlim(-1, 11)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels([0, 5, 10], fontsize=FONT_SIZE_TICK)
    max_rpe = max(max(deltas) for deltas in preprocessed_rpes.values())
    max_rpe = round(max_rpe, 1)
    ax.set_ylim(-0.035, max_rpe+0.035)
    ax.set_yticks(np.linspace(0, max_rpe, 3).round(2))
    ax.set_yticklabels(np.linspace(0, max_rpe, 3).round(2), fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("RPE(a.u.)", fontsize=FONT_SIZE_LABEL)
    
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(SPINE_WIDTH)
    ax.spines['bottom'].set_linewidth(SPINE_WIDTH)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_blurred_v(discount_rate=0.9, kernel_center=35, kernel_std=5.0, n_states=50, save_fig=True, filename="figs/exponential_value_uncertainty.pdf"):
    """Plot exponential value function with Gaussian uncertainty illustration (Fig. 5D)."""
    fig, ax = plt.subplots(figsize=(4.5, 3))
    
    states = np.arange(n_states)
    
    values = discount_rate ** (n_states - 1 - states)
    
    ax.plot(states, values, color='black', linewidth=2)

    kernel_range = np.linspace(kernel_center - 3*kernel_std, kernel_center + 3*kernel_std, 100)
    gaussian_kernel = np.exp(-0.5 * ((kernel_range - kernel_center) / kernel_std) ** 2)
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    
    kernel_values = np.interp(kernel_range, states, values)
    
    for i in range(len(kernel_range) - 1):
        x_coords = [kernel_range[i], kernel_range[i+1], kernel_range[i+1], kernel_range[i]]
        y_coords = [0, 0, kernel_values[i+1], kernel_values[i]]
        alpha = gaussian_kernel[i]/max(gaussian_kernel) 
        ax.fill(x_coords, y_coords, color='red', alpha=0.50*alpha, edgecolor='none')
    
    weighted_values = kernel_values * gaussian_kernel
    prob_weighted_avg = np.sum(weighted_values)
    
    closest_state = int(np.round(kernel_center))
    closest_state = np.clip(closest_state, 0, n_states - 1)
    
    ax.plot(kernel_center, prob_weighted_avg+0.05, 'o', markersize=8, color = 'red', label='V$_{x}$(s)')
    ax.plot(kernel_center, values[kernel_center], 'o', markersize=8, markerfacecolor = 'white', markeredgecolor = 'red', label='V(s)')
    
    ax.axvline(x=kernel_center, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel("State", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Value", fontsize=FONT_SIZE_LABEL)
    
    ax.set_xticks([0, n_states-1])
    ax.set_xticklabels(['Start', 'Goal'], fontsize=FONT_SIZE_TICK)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, 0.5, 1], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(-0.20, 1.05)
    
    ax.legend(loc = 'upper left', fontsize=15, frameon=False, markerscale=1.5, handletextpad=0.1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_compare_mikhael(exp_data, reward_value=1.0, save_fig=True, filename="figs/compare_mikhael.pdf"):
    """Plot learned values under static vs. dynamic uncertainty assumptions (NA)."""
    c_v = exp_data['exp_c']['episodes'][-1]['V']
    d_v = exp_data['exp_d']['episodes'][-1]['V']
    gamma = 0.9
    n_states = len(c_v)
    states = np.arange(n_states)
    goal_state = n_states - 1
    distances = np.abs(goal_state - states)
    true_v = reward_value * (gamma ** distances)
    
    fig, ax = plt.subplots(figsize=(4.5, 3))
    
    ax.plot(states, c_v, color='red', linewidth=LINEWIDTH, label='Constant')
    ax.plot(states, d_v, color='lightblue', linewidth=LINEWIDTH, label='Dynamic')
    ax.plot(states, true_v, color='black', linewidth=LINEWIDTH, linestyle='--', label='True')

    ax.set_xlabel("State", fontsize=FONT_SIZE_LABEL)
    ax.set_xticks([0, goal_state])
    ax.set_xticklabels(['Start', 'Goal'], fontsize=FONT_SIZE_TICK)
    
    ax.set_ylabel("Value", fontsize=FONT_SIZE_LABEL)
    
    y_max = max(np.max(c_v), np.max(d_v), np.max(true_v))
    y_ticks = np.linspace(0, y_max, 5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.2f}" for y in y_ticks], fontsize=FONT_SIZE_TICK)
    
    ax.legend(loc='upper left', fontsize=FONT_SIZE_LEGEND, frameon=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_compare_mikhael_error(exp_data, reward_value=1.0, save_fig=True, filename="figs/compare_mikhael_error.pdf"):
    """Plot value estimation errors under static vs. dynamic uncertainty (Fig. 5E)."""
    c_v = exp_data['exp_c']['episodes'][-1]['V']
    d_v = exp_data['exp_d']['episodes'][-1]['V']
    gamma = 0.9
    n_states = len(c_v)
    states = np.arange(n_states)
    goal_state = n_states - 1
    distances = np.abs(goal_state - states)
    true_v = reward_value * (gamma ** distances)
    
    c_error = np.round(c_v - true_v, 5)
    d_error = np.round(d_v - true_v, 5)
    
    fig, ax = plt.subplots(figsize=(4.5, 3))
    
    ax.plot(states, c_error, color='black', linewidth=LINEWIDTH, linestyle='--', label=r'$\sigma_{t+1} \approx \sigma_t$')
    ax.plot(states, d_error, color='red', linewidth=LINEWIDTH, label=r'$\sigma_{t+1} > \sigma_t$')
    
    ax.set_xlabel("State", fontsize=FONT_SIZE_LABEL)
    ax.set_xticks([0, goal_state])
    ax.set_xticklabels(['Start', 'Goal'], fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Value Error", fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel("State", fontsize=FONT_SIZE_LABEL)
    
    y_ticks = (0, 0.05, 0.10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.2f}" for y in y_ticks], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(-0.025, 0.10)
    
    
    legend = ax.legend(title='State Uncertainty', loc='upper left', fontsize=10, 
                       title_fontsize=FONT_SIZE_TICK, frameon=False)
    legend.get_title().set_fontsize(FONT_SIZE_TICK)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_kernels(kernels, save_fig=True, filename="figs/kernels.pdf"):
    """Plot Gaussian kernel widths across track states (NA)."""
    n_states = len(kernels)
    kernels_normalized = np.array(kernels) / n_states
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(kernels_normalized, color='black', linewidth=LINEWIDTH)
    ax.set_xlabel("State", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Kernel Width (norm.)", fontsize=FONT_SIZE_LABEL)
    ax.set_xticks([0, n_states-1])
    ax.set_xticklabels(['Start', 'Goal'], fontsize=FONT_SIZE_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    plt.tight_layout()
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

# ------------------------------------------------------------
# Experiment: Krausz et al., Neuron, 2023 
# ------------------------------------------------------------
def plot_krausz_v_mb(v_mb, states=None, vmin=None, vmax=None, line_color='red', line_style='-', arrow_size=1.0, arrow_style='->', save_fig=True, filename="figs/krausz_v_mb.pdf"):
    """Plot gridworld value function as a heatmap with optional trajectory overlay (Fig. 4C)."""
    fig, ax = plt.subplots(figsize=(3, 3))
    
    im = ax.imshow(v_mb, cmap='Oranges', interpolation='nearest', vmin=vmin, vmax=vmax)
    
    ax.set_xticks(np.arange(-0.5, v_mb.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, v_mb.shape[0], 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=1)
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Value', fontsize=FONT_SIZE_TICK)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(SPINE_WIDTH)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.tick_params(left=False, bottom=False, top=False, right=False, 
                   labelleft=False, labelbottom=False, labeltop=False, labelright=False,
                   length=0, width=0)
    
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)

def plot_rpe_krausz(trace_t, trace_t_plus_1, save_fig=True, filename="figs/krausz_rpe.pdf"):
    """Plot comparison of RPE traces across consecutive trials (Fig. 4D)."""
    fig, ax = plt.subplots(figsize=(4, 3))
    
    trace_t = preprocess_rpes(trace_t)
    trace_t_plus_1 = preprocess_rpes(trace_t_plus_1)
    
    ax.plot(trace_t, color='black', linestyle='--', linewidth=LINEWIDTH, label='t-1')
    ax.plot(trace_t_plus_1, color='red', linestyle='-', linewidth=LINEWIDTH, label='t')
    
    ax.axvline(x=len(trace_t)-1, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel("State", fontsize=FONT_SIZE_LABEL)
    ax.set_xticks([0, len(trace_t)-1])
    ax.set_xticklabels(['Start', 'Goal'], fontsize=FONT_SIZE_TICK)
    
    ax.set_ylabel("RPE", fontsize=FONT_SIZE_LABEL)
    
    
    ax.legend(title='Trial', alignment='left', title_fontsize=FONT_SIZE_LEGEND, bbox_to_anchor=(0.45, 0.6), fontsize=FONT_SIZE_LEGEND, frameon=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.1)