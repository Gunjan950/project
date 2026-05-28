"""
Q-Learning Agent
================
Tabular Q-Learning with epsilon-greedy exploration.

The Q-table maps (state, action) → expected cumulative reward.
Updated via: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) − Q(s,a)]

Key hyperparameters:
  α (learning_rate)   : how fast the agent updates its beliefs
  γ (gamma)           : discount factor — how much future rewards matter
  ε (epsilon)         : exploration rate (decays over training)
"""

import numpy as np
import json
import os
from typing import Tuple


class QLearningAgent:
    """
    Tabular Q-Learning agent for discrete state/action spaces.

    Learns a Q-table Q[s, a] representing the expected return
    when taking action a from state s.
    """

    def __init__(
        self,
        n_states:      int,
        n_actions:     int,
        learning_rate: float = 0.1,
        gamma:         float = 0.99,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_states      = n_states
        self.n_actions     = n_actions
        self.lr            = learning_rate
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: shape (n_states, n_actions), initialized to zeros
        self.q_table = np.zeros((n_states, n_actions))

        # Training metrics
        self.episode_rewards  = []
        self.episode_lengths  = []
        self.epsilon_history  = []

    # ------------------------------------------------------------------ #
    # ACTION SELECTION                                                     #
    # ------------------------------------------------------------------ #

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        - With probability ε: explore (random action)
        - With probability 1-ε: exploit (greedy Q-table action)
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)   # explore
        return int(np.argmax(self.q_table[state]))      # exploit

    # ------------------------------------------------------------------ #
    # Q-TABLE UPDATE                                                       #
    # ------------------------------------------------------------------ #

    def update(
        self,
        state:      int,
        action:     int,
        reward:     float,
        next_state: int,
        done:       bool,
    ) -> float:
        """
        Bellman update for Q-Learning.
        Returns the TD error (useful for monitoring convergence).
        """
        current_q  = self.q_table[state, action]
        target_q   = reward + (0 if done else self.gamma * np.max(self.q_table[next_state]))
        td_error   = target_q - current_q
        self.q_table[state, action] += self.lr * td_error
        return td_error

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    # TRAINING LOOP                                                        #
    # ------------------------------------------------------------------ #

    def train(
        self,
        env,
        n_episodes:    int = 1000,
        verbose:       bool = True,
        verbose_every: int  = 100,
    ) -> dict:
        """
        Full training loop. Returns training history.
        """
        print(f"\n{'='*50}")
        print("  Q-LEARNING TRAINING")
        print(f"  Episodes: {n_episodes} | lr={self.lr} | γ={self.gamma}")
        print(f"  ε: {self.epsilon:.2f} → {self.epsilon_min}")
        print(f"{'='*50}")

        successes = 0

        for episode in range(n_episodes):
            obs, _ = env.reset()
            state  = env.get_state_index()
            total_reward = 0
            steps        = 0

            while True:
                action             = self.select_action(state, training=True)
                obs, reward, term, trunc, info = env.step(action)
                next_state         = env.get_state_index()

                self.update(state, action, reward, next_state, term or trunc)

                state        = next_state
                total_reward += reward
                steps        += 1

                if term or trunc:
                    break

            self.decay_epsilon()
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.epsilon)

            if info.get("reached_goal"):
                successes += 1

            if verbose and (episode + 1) % verbose_every == 0:
                avg_r   = np.mean(self.episode_rewards[-verbose_every:])
                avg_len = np.mean(self.episode_lengths[-verbose_every:])
                sr      = successes / (episode + 1) * 100
                print(f"  Ep {episode+1:>5}/{n_episodes} | "
                      f"AvgReward: {avg_r:>8.1f} | "
                      f"AvgLen: {avg_len:>5.1f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Success: {sr:.1f}%")

        print(f"\n  ✓ Training complete. "
              f"Final success rate: {successes/n_episodes*100:.1f}%\n")

        return {
            "rewards":  self.episode_rewards,
            "lengths":  self.episode_lengths,
            "epsilons": self.epsilon_history,
            "final_success_rate": successes / n_episodes,
        }

    # ------------------------------------------------------------------ #
    # EVALUATION                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(self, env, n_episodes: int = 100) -> dict:
        """Run greedy policy and return evaluation metrics."""
        rewards  = []
        lengths  = []
        successes = 0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            state  = env.get_state_index()
            total_r = 0
            steps   = 0

            while True:
                action = self.select_action(state, training=False)
                obs, reward, term, trunc, info = env.step(action)
                state   = env.get_state_index()
                total_r += reward
                steps   += 1
                if term or trunc:
                    break

            rewards.append(total_r)
            lengths.append(steps)
            if info.get("reached_goal"):
                successes += 1

        return {
            "mean_reward":   np.mean(rewards),
            "std_reward":    np.std(rewards),
            "mean_length":   np.mean(lengths),
            "success_rate":  successes / n_episodes,
            "n_episodes":    n_episodes,
        }

    # ------------------------------------------------------------------ #
    # SAVE / LOAD                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        """Export Q-table and hyperparameters to JSON."""
        data = {
            "q_table":       self.q_table.tolist(),
            "n_states":      self.n_states,
            "n_actions":     self.n_actions,
            "learning_rate": self.lr,
            "gamma":         self.gamma,
            "epsilon":       self.epsilon,
            "episode_rewards": self.episode_rewards[-100:],  # last 100 for reference
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Q-table saved → {path}")

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        """Load a saved Q-Learning agent."""
        with open(path) as f:
            data = json.load(f)
        agent = cls(
            n_states      = data["n_states"],
            n_actions     = data["n_actions"],
            learning_rate = data["learning_rate"],
            gamma         = data["gamma"],
            epsilon       = 0.0,  # greedy at load time
        )
        agent.q_table        = np.array(data["q_table"])
        agent.episode_rewards = data.get("episode_rewards", [])
        print(f"  ✓ Q-table loaded ← {path}")
        return agent
