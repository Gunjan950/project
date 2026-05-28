"""
Deep Q-Network (DQN) Agent — Bonus Implementation
===================================================
Replaces the Q-table with a neural network: Q(s, a; θ)
Uses two key stabilization techniques:
  1. Experience Replay Buffer — breaks temporal correlations in training data
  2. Target Network — provides stable training targets (copied every C steps)

Architecture: 3-layer MLP
  Input  → 128 neurons (ReLU) → 64 neurons (ReLU) → n_actions (linear)

Difference vs Q-Learning:
  - Q-Learning: Q(s) stored in a table, exact lookup
  - DQN:        Q(s) approximated by neural network, generalises to unseen states
"""

import numpy as np
import json
import os
from collections import deque

# TensorFlow optional — skip gracefully if unavailable
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[DQN] TensorFlow not installed. Install: pip install tensorflow")


class ReplayBuffer:
    """Fixed-size circular replay buffer storing (s, a, r, s', done) tuples."""

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch   = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for the GridWorld environment.

    Works on the flattened grid observation (grid_size^2 floats).
    Uses an MLP to approximate the Q-function.
    """

    def __init__(
        self,
        state_size:        int,
        n_actions:         int,
        learning_rate:     float = 0.001,
        gamma:             float = 0.99,
        epsilon:           float = 1.0,
        epsilon_min:       float = 0.01,
        epsilon_decay:     float = 0.995,
        batch_size:        int   = 64,
        memory_capacity:   int   = 10_000,
        target_update_freq: int  = 50,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for DQN. pip install tensorflow")

        self.state_size         = state_size
        self.n_actions          = n_actions
        self.lr                 = learning_rate
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq

        self.memory         = ReplayBuffer(memory_capacity)
        self.train_step_count = 0

        # Build online and target networks
        self.online_net = self._build_network("online")
        self.target_net = self._build_network("target")
        self._sync_target()

        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.losses          = []

    def _build_network(self, name: str):
        """Build a 3-layer MLP: Input → 128 → 64 → n_actions."""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64,  activation="relu"),
            keras.layers.Dense(self.n_actions, activation="linear"),
        ], name=f"dqn_{name}")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss="mse",
        )
        return model

    def _sync_target(self):
        """Copy weights from online net → target net."""
        self.target_net.set_weights(self.online_net.get_weights())

    # ------------------------------------------------------------------ #
    # ACTION SELECTION                                                     #
    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.online_net.predict(
            state.reshape(1, -1).astype(np.float32), verbose=0
        )
        return int(np.argmax(q_values[0]))

    # ------------------------------------------------------------------ #
    # LEARNING STEP                                                        #
    # ------------------------------------------------------------------ #

    def learn(self) -> float:
        """Sample a minibatch from replay buffer and update online network."""
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Target Q-values using target network
        next_q   = self.target_net.predict(next_states, verbose=0)
        targets  = rewards + (1 - dones) * self.gamma * np.max(next_q, axis=1)

        # Current Q-values
        current_q = self.online_net.predict(states, verbose=0)
        for i, a in enumerate(actions):
            current_q[i, a] = targets[i]

        # Gradient update
        history = self.online_net.fit(states, current_q,
                                       batch_size=self.batch_size,
                                       epochs=1, verbose=0)
        loss = history.history["loss"][0]

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self._sync_target()

        return loss

    # ------------------------------------------------------------------ #
    # TRAINING LOOP                                                        #
    # ------------------------------------------------------------------ #

    def train(
        self,
        env,
        n_episodes:    int  = 500,
        verbose:       bool = True,
        verbose_every: int  = 50,
    ) -> dict:
        print(f"\n{'='*50}")
        print("  DQN TRAINING")
        print(f"  Episodes: {n_episodes} | Network: 128→64→{self.n_actions}")
        print(f"  Batch size: {self.batch_size} | Memory: {self.memory.buffer.maxlen}")
        print(f"{'='*50}")

        successes = 0

        for episode in range(n_episodes):
            obs, _    = env.reset()
            state     = obs.astype(np.float32)
            total_r   = 0.0
            steps     = 0
            ep_losses = []

            while True:
                action            = self.select_action(state, training=True)
                next_obs, reward, term, trunc, info = env.step(action)
                next_state        = next_obs.astype(np.float32)

                self.memory.push(state, action, reward, next_state, float(term or trunc))
                loss   = self.learn()
                if loss: ep_losses.append(loss)

                state    = next_state
                total_r += reward
                steps   += 1

                if term or trunc:
                    break

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episode_rewards.append(total_r)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.epsilon)
            self.losses.append(np.mean(ep_losses) if ep_losses else 0.0)

            if info.get("reached_goal"):
                successes += 1

            if verbose and (episode + 1) % verbose_every == 0:
                avg_r  = np.mean(self.episode_rewards[-verbose_every:])
                avg_l  = np.mean(self.episode_lengths[-verbose_every:])
                avg_ls = np.mean(self.losses[-verbose_every:])
                sr     = successes / (episode + 1) * 100
                print(f"  Ep {episode+1:>4}/{n_episodes} | "
                      f"AvgR: {avg_r:>8.1f} | Steps: {avg_l:>5.1f} | "
                      f"Loss: {avg_ls:.4f} | ε: {self.epsilon:.3f} | "
                      f"Success: {sr:.1f}%")

        print(f"\n  ✓ DQN training complete. "
              f"Final success: {successes/n_episodes*100:.1f}%\n")

        return {
            "rewards":  self.episode_rewards,
            "lengths":  self.episode_lengths,
            "epsilons": self.epsilon_history,
            "losses":   self.losses,
            "final_success_rate": successes / n_episodes,
        }

    # ------------------------------------------------------------------ #
    # SAVE / LOAD                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.online_net.save(os.path.join(path, "dqn_online.keras"))
        meta = {
            "state_size": self.state_size,
            "n_actions":  self.n_actions,
            "epsilon":    self.epsilon,
            "gamma":      self.gamma,
        }
        with open(os.path.join(path, "dqn_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  ✓ DQN saved → {path}/")

    @classmethod
    def load(cls, path: str) -> "DQNAgent":
        with open(os.path.join(path, "dqn_meta.json")) as f:
            meta = json.load(f)
        agent = cls(state_size=meta["state_size"], n_actions=meta["n_actions"],
                    gamma=meta["gamma"], epsilon=0.0)
        agent.online_net = keras.models.load_model(
            os.path.join(path, "dqn_online.keras"))
        agent._sync_target()
        print(f"  ✓ DQN loaded ← {path}/")
        return agent
