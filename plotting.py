"""
Plotting Utilities
==================
Visualization functions for comparing agent performance:
  - Reward curves with smoothing
  - Success rate comparison
  - Q-value heatmaps
  - Episode length comparison
  - Side-by-side policy comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Optional


# ── Styling ──────────────────────────────────────────────────────────── #

STYLE = {
    "q_learning": {"color": "#2ECC71", "label": "Q-Learning"},
    "random":     {"color": "#E74C3C", "label": "Random Policy"},
    "dqn":        {"color": "#3498DB", "label": "DQN"},
}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── Smoothing helper ─────────────────────────────────────────────────── #

def smooth(data: list, window: int = 20) -> np.ndarray:
    """Moving average smoothing."""
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


# ── Main comparison plot ─────────────────────────────────────────────── #

def plot_comparison(
    histories:   Dict[str, dict],
    save_path:   Optional[str] = None,
    smooth_window: int = 30,
) -> plt.Figure:
    """
    Compare reward curves from multiple agents.

    histories: {"q_learning": {"rewards": [...], ...}, "random": {...}}
    """
    n_agents = len(histories)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Robot Navigation: Agent Comparison", fontsize=16,
                 fontweight="bold", y=0.98)

    ax_reward, ax_smooth, ax_success, ax_length = axes.flatten()

    # ── Raw reward curves ── #
    for key, hist in histories.items():
        s = STYLE.get(key, {"color": "#888", "label": key})
        ax_reward.plot(hist["rewards"], alpha=0.3, color=s["color"])
        ax_reward.plot(smooth(hist["rewards"], smooth_window),
                       color=s["color"], linewidth=2, label=s["label"])

    ax_reward.set_title("Episode Reward (raw + smoothed)", fontweight="bold")
    ax_reward.set_xlabel("Episode"); ax_reward.set_ylabel("Total Reward")
    ax_reward.legend(fontsize=10)

    # ── Smoothed only (cleaner view) ── #
    for key, hist in histories.items():
        s = STYLE.get(key, {"color": "#888", "label": key})
        sm = smooth(hist["rewards"], smooth_window)
        ax_smooth.plot(sm, color=s["color"], linewidth=2.5, label=s["label"])
        ax_smooth.fill_between(range(len(sm)), sm, alpha=0.12, color=s["color"])

    ax_smooth.set_title(f"Smoothed Reward (window={smooth_window})", fontweight="bold")
    ax_smooth.set_xlabel("Episode"); ax_smooth.set_ylabel("Avg Reward")
    ax_smooth.legend(fontsize=10)

    # ── Rolling success rate ── #
    win = max(50, len(list(histories.values())[0]["rewards"]) // 10)
    for key, hist in histories.items():
        s = STYLE.get(key, {"color": "#888", "label": key})
        rewards = hist["rewards"]
        # Success ≈ reward > 0 (got to goal)
        success_flags = [1 if r > 0 else 0 for r in rewards]
        roll_sr = smooth(success_flags, win) * 100
        ax_success.plot(roll_sr, color=s["color"], linewidth=2.5, label=s["label"])

    ax_success.set_title("Rolling Success Rate (%)", fontweight="bold")
    ax_success.set_xlabel("Episode"); ax_success.set_ylabel("Success Rate (%)")
    ax_success.set_ylim(0, 105)
    ax_success.legend(fontsize=10)

    # ── Episode length ── #
    for key, hist in histories.items():
        if "lengths" in hist:
            s = STYLE.get(key, {"color": "#888", "label": key})
            sm = smooth(hist["lengths"], smooth_window)
            ax_length.plot(sm, color=s["color"], linewidth=2.5, label=s["label"])

    ax_length.set_title(f"Episode Length (smoothed)", fontweight="bold")
    ax_length.set_xlabel("Episode"); ax_length.set_ylabel("Steps")
    ax_length.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    return fig


# ── Q-value heatmap ─────────────────────────────────────────────────── #

def plot_q_heatmap(
    q_table:   np.ndarray,
    grid_size: int,
    env,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the maximum Q-value for each cell and the optimal policy direction.
    """
    max_q  = np.max(q_table, axis=1).reshape(grid_size, grid_size)
    policy = np.argmax(q_table, axis=1).reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Q-Table Analysis", fontsize=15, fontweight="bold")

    # Heatmap of max Q-values
    im = axes[0].imshow(max_q, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=axes[0], shrink=0.8)
    axes[0].set_title("Max Q-Value per Cell", fontweight="bold")
    axes[0].set_xlabel("Column"); axes[0].set_ylabel("Row")

    # Overlay obstacles
    for r in range(grid_size):
        for c in range(grid_size):
            if env.base_grid[r, c] == 1:   # obstacle
                axes[0].add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                                 color="black", alpha=0.7))
    axes[0].plot(*reversed(env.goal_pos), "g*", markersize=18, label="Goal")
    axes[0].plot(*reversed(env.start_pos), "bs", markersize=12, label="Start")
    axes[0].legend(fontsize=9)

    # Policy direction arrows
    arrow_map = {0: (0, -0.35), 1: (0, 0.35), 2: (-0.35, 0), 3: (0.35, 0)}
    axes[1].imshow(max_q, cmap="RdYlGn", alpha=0.5, aspect="auto")
    axes[1].set_title("Optimal Policy (greedy arrows)", fontweight="bold")

    for r in range(grid_size):
        for c in range(grid_size):
            if env.base_grid[r, c] == 1:
                axes[1].add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                                 color="black", alpha=0.7))
            elif (r, c) != env.goal_pos:
                a = policy[r, c]
                dx, dy = arrow_map[a]
                axes[1].annotate("", xy=(c + dx, r + dy),
                                  xytext=(c, r),
                                  arrowprops=dict(arrowstyle="->",
                                                  color="#1A252F", lw=1.5))

    axes[1].plot(*reversed(env.goal_pos), "g*", markersize=18)
    axes[1].plot(*reversed(env.start_pos), "bs", markersize=12)

    for ax in axes:
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    return fig


# ── Summary stats table ─────────────────────────────────────────────── #

def print_comparison_table(results: Dict[str, dict]):
    """Pretty-print a comparison table of evaluation metrics."""
    print(f"\n{'═'*70}")
    print(f"{'AGENT COMPARISON SUMMARY':^70}")
    print(f"{'═'*70}")
    print(f"{'Agent':<18} {'Avg Reward':>12} {'Std Reward':>12} "
          f"{'Avg Length':>12} {'Success %':>10}")
    print(f"{'─'*70}")
    for name, res in results.items():
        label = STYLE.get(name, {"label": name})["label"]
        print(f"{label:<18} "
              f"{res.get('mean_reward',0):>12.2f} "
              f"{res.get('std_reward',0):>12.2f} "
              f"{res.get('mean_length',0):>12.1f} "
              f"{res.get('success_rate',0)*100:>10.1f}%")
    print(f"{'═'*70}\n")
