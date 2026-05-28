"""
Random Policy Agent (Baseline)
===============================
A purely random agent that selects actions uniformly at random.
Used as a baseline to demonstrate the improvement gained by Q-Learning.

Expected behavior: very low success rate, high variance in rewards,
long episode lengths that rarely reach the goal.
"""

import numpy as np


class RandomAgent:
    """
    Random policy baseline — selects actions uniformly at random.
    No learning, no memory, no strategy.
    """

    def __init__(self, n_actions: int):
        self.n_actions       = n_actions
        self.episode_rewards = []
        self.episode_lengths = []

    def select_action(self, state=None, training: bool = True) -> int:
        """Always pick a random action."""
        return np.random.randint(self.n_actions)

    def run(self, env, n_episodes: int = 1000, verbose: bool = True) -> dict:
        """
        Run the random policy for n_episodes.
        Returns a dict of training metrics (mirrors Q-Learning API).
        """
        print(f"\n{'='*50}")
        print("  RANDOM POLICY BASELINE")
        print(f"  Episodes: {n_episodes}")
        print(f"{'='*50}")

        successes = 0

        for episode in range(n_episodes):
            obs, _   = env.reset()
            total_r  = 0
            steps    = 0

            while True:
                action = self.select_action()
                obs, reward, term, trunc, info = env.step(action)
                total_r += reward
                steps   += 1
                if term or trunc:
                    break

            self.episode_rewards.append(total_r)
            self.episode_lengths.append(steps)
            if info.get("reached_goal"):
                successes += 1

        sr = successes / n_episodes * 100
        avg_r = np.mean(self.episode_rewards)
        print(f"  Done. AvgReward: {avg_r:.1f} | Success rate: {sr:.1f}%\n")

        return {
            "rewards":  self.episode_rewards,
            "lengths":  self.episode_lengths,
            "epsilons": [1.0] * n_episodes,  # always exploring
            "final_success_rate": successes / n_episodes,
        }
