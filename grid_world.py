"""
GridWorld Environment - Custom OpenAI Gym Environment
======================================================
A robot navigation simulation where an agent must reach a goal
while avoiding obstacles on a configurable N x N grid.

REWARD FUNCTION DESIGN:
- +100  : Reaching the goal → strongly shapes goal-seeking behavior
- -100  : Hitting an obstacle → strongly discourages dangerous paths
- -1    : Each step taken → encourages shortest-path finding (efficiency)
- -0.5  : Moving into a wall → discourages wall-bumping
These rewards guide the robot to find the *shortest safe path* to the goal.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import io

# Cell type constants
EMPTY    = 0
OBSTACLE = 1
GOAL     = 2
ROBOT    = 3
VISITED  = 4

# Action constants
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
ACTION_NAMES = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}


class GridWorldEnv(gym.Env):
    """
    Custom Grid World Environment for Robot Navigation.

    The robot starts at a fixed position and must navigate to the goal
    while avoiding obstacles. The grid is fully observable.

    Observation: flattened grid state (integer array)
    Action space: Discrete(4) — UP, DOWN, LEFT, RIGHT
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 8,
        obstacle_density: float = 0.15,
        render_mode: str = None,
        custom_grid: np.ndarray = None,
        start_pos: tuple = (0, 0),
        goal_pos: tuple = None,
        max_steps: int = 200,
        seed: int = 42,
    ):
        super().__init__()

        self.grid_size       = grid_size
        self.obstacle_density = obstacle_density
        self.render_mode     = render_mode
        self.max_steps       = max_steps
        self.custom_grid     = custom_grid
        self.start_pos       = start_pos
        self.goal_pos        = goal_pos if goal_pos else (grid_size - 1, grid_size - 1)
        self._np_random      = np.random.default_rng(seed)

        # Action space: 4 discrete moves
        self.action_space = spaces.Discrete(4)

        # Observation: grid_size x grid_size integer values (0-4)
        self.observation_space = spaces.Box(
            low=0, high=4,
            shape=(grid_size * grid_size,),
            dtype=np.int32
        )

        # Reward structure (clearly documented)
        self.REWARD_GOAL     =  100.0   # Reach goal
        self.REWARD_OBSTACLE = -100.0   # Hit obstacle
        self.REWARD_STEP     =   -1.0   # Each step (efficiency pressure)
        self.REWARD_WALL     =   -0.5   # Bump into wall

        # State tracking
        self.grid        = None
        self.robot_pos   = None
        self.step_count  = 0
        self.path_taken  = []
        self.visited     = set()

        self._build_grid()

    # ------------------------------------------------------------------ #
    # GRID CONSTRUCTION                                                    #
    # ------------------------------------------------------------------ #

    def _build_grid(self):
        """Build the base grid layout (obstacles + goal). Called once."""
        if self.custom_grid is not None:
            self.base_grid = self.custom_grid.copy()
        else:
            self.base_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            # Randomly place obstacles
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if (r, c) == self.start_pos or (r, c) == self.goal_pos:
                        continue
                    if self._np_random.random() < self.obstacle_density:
                        self.base_grid[r, c] = OBSTACLE

        # Ensure start and goal are clear
        self.base_grid[self.start_pos] = EMPTY
        self.base_grid[self.goal_pos]  = GOAL

    def _get_observation(self) -> np.ndarray:
        """Return flattened grid with robot position marked."""
        obs = self.grid.copy()
        obs[self.robot_pos] = ROBOT
        return obs.flatten().astype(np.int32)

    # ------------------------------------------------------------------ #
    # CORE GYM API                                                         #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        self.grid       = self.base_grid.copy()
        self.robot_pos  = self.start_pos
        self.step_count = 0
        self.path_taken = [self.start_pos]
        self.visited    = {self.start_pos}
        return self._get_observation(), {}

    def step(self, action: int):
        """Execute one action and return (obs, reward, terminated, truncated, info)."""
        self.step_count += 1

        # Compute new position
        r, c = self.robot_pos
        if   action == UP:    nr, nc = r - 1, c
        elif action == DOWN:  nr, nc = r + 1, c
        elif action == LEFT:  nr, nc = r,     c - 1
        elif action == RIGHT: nr, nc = r,     c + 1
        else: raise ValueError(f"Invalid action: {action}")

        terminated = False
        truncated  = False
        reward     = self.REWARD_STEP  # default: step penalty

        # Check wall collision
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            reward = self.REWARD_WALL
            nr, nc = r, c  # stay in place

        # Check obstacle collision
        elif self.base_grid[nr, nc] == OBSTACLE:
            reward     = self.REWARD_OBSTACLE
            terminated = True

        # Check goal reached
        elif (nr, nc) == self.goal_pos:
            reward     = self.REWARD_GOAL
            terminated = True
            self.robot_pos = (nr, nc)

        # Normal move
        else:
            self.robot_pos = (nr, nc)

        self.path_taken.append(self.robot_pos)
        self.visited.add(self.robot_pos)

        # Truncate if max steps exceeded
        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "step": self.step_count,
            "position": self.robot_pos,
            "goal": self.goal_pos,
            "reached_goal": (self.robot_pos == self.goal_pos),
            "path_length": len(self.path_taken),
        }

        return self._get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # STATE HELPERS                                                        #
    # ------------------------------------------------------------------ #

    def get_state_index(self) -> int:
        """Return a single integer state ID for Q-table lookup."""
        r, c = self.robot_pos
        return r * self.grid_size + c

    @property
    def n_states(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def n_actions(self) -> int:
        return self.action_space.n

    # ------------------------------------------------------------------ #
    # RENDERING                                                            #
    # ------------------------------------------------------------------ #

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode in ("human", "rgb_array"):
            return self._render_rgb()

    def _render_ansi(self) -> str:
        symbols = {EMPTY: ".", OBSTACLE: "█", GOAL: "G", ROBOT: "R", VISITED: "·"}
        grid_view = self.grid.copy()
        for pos in self.visited:
            if grid_view[pos] == EMPTY:
                grid_view[pos] = VISITED
        grid_view[self.robot_pos] = ROBOT
        grid_view[self.goal_pos]  = GOAL
        rows = []
        for r in range(self.grid_size):
            rows.append(" ".join(symbols.get(grid_view[r, c], "?")
                                  for c in range(self.grid_size)))
        return "\n".join(rows)

    def _render_rgb(self):
        """Render colored grid using matplotlib."""
        fig, ax = plt.subplots(figsize=(6, 6))
        display = self.grid.copy()
        for pos in self.visited:
            if display[pos] == EMPTY:
                display[pos] = VISITED
        display[self.robot_pos] = ROBOT

        colors = ["#F8F9FA", "#2C3E50", "#27AE60", "#E74C3C", "#AED6F1"]
        cmap   = ListedColormap(colors)
        ax.imshow(display, cmap=cmap, vmin=0, vmax=4)

        # Grid lines
        for x in range(self.grid_size + 1):
            ax.axhline(x - 0.5, color="gray", linewidth=0.5, alpha=0.3)
            ax.axvline(x - 0.5, color="gray", linewidth=0.5, alpha=0.3)

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Step {self.step_count} | Pos: {self.robot_pos}", fontsize=12)

        patches = [
            mpatches.Patch(color=colors[0], label="Empty"),
            mpatches.Patch(color=colors[1], label="Obstacle"),
            mpatches.Patch(color=colors[2], label="Goal"),
            mpatches.Patch(color=colors[3], label="Robot"),
            mpatches.Patch(color=colors[4], label="Visited"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=7,
                  bbox_to_anchor=(1.35, 1.0))

        if self.render_mode == "human":
            plt.tight_layout(); plt.show()
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)
            return buf

    def visualize_path(self, q_table=None, title="Robot Path", save_path=None):
        """Visualize the grid with the path taken, and optionally the Q-policy."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Color the grid
        display = self.base_grid.copy().astype(float)
        ax.imshow(display, cmap="RdYlGn_r", alpha=0.3, vmin=0, vmax=1)

        # Draw grid lines
        for x in range(self.grid_size + 1):
            ax.axhline(x - 0.5, color="gray", linewidth=0.8, alpha=0.4)
            ax.axvline(x - 0.5, color="gray", linewidth=0.8, alpha=0.4)

        # Color cells
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.base_grid[r, c] == OBSTACLE:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               color="#2C3E50", zorder=2))
                elif (r, c) == self.goal_pos:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               color="#27AE60", alpha=0.8, zorder=2))
                    ax.text(c, r, "G", ha="center", va="center",
                            color="white", fontsize=14, fontweight="bold", zorder=3)

        # Draw Q-policy arrows
        if q_table is not None:
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if self.base_grid[r, c] == OBSTACLE: continue
                    if (r, c) == self.goal_pos: continue
                    s = r * self.grid_size + c
                    a = np.argmax(q_table[s])
                    dr = [-0.3, 0.3,  0,    0  ][a]
                    dc = [ 0,   0,   -0.3,  0.3][a]
                    ax.annotate("", xy=(c + dc, r + dr),
                                xytext=(c, r),
                                arrowprops=dict(arrowstyle="->",
                                                color="#3498DB", lw=1.5),
                                zorder=4)

        # Draw the path
        if len(self.path_taken) > 1:
            path_r = [p[0] for p in self.path_taken]
            path_c = [p[1] for p in self.path_taken]
            ax.plot(path_c, path_r, "o-", color="#E74C3C",
                    linewidth=2.5, markersize=6, zorder=5, label="Robot path")
            # Start marker
            ax.plot(path_c[0], path_r[0], "s", color="#9B59B6",
                    markersize=12, zorder=6, label="Start")

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()
        return fig
