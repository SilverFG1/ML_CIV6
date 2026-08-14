"""Simple Q-learning agent for the ML_CIV6 environment."""

import json
import random
from pathlib import Path

import numpy as np


class QLearningAgent:
    """Tabular Q-learning over discretized observations."""

    def __init__(
        self,
        action_count,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        state_precision=8,
        initial_q=-100.0,
        rng=None,
    ):
        self.action_count = int(action_count)
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.state_precision = int(state_precision)
        self.initial_q = float(initial_q)
        self.rng = rng or random.Random()
        self.q_table = {}

    def choose_action(
        self,
        observation,
        explore=True,
        fallback_action=None,
        guided_exploration_rate=0.0,
    ):
        """Choose the best known action, with optional epsilon exploration."""

        if explore and self.rng.random() < self.epsilon:
            if fallback_action is not None and self.rng.random() < guided_exploration_rate:
                return int(fallback_action)
            return self.rng.randrange(self.action_count)

        q_values = self._q_values(observation)
        if fallback_action is not None and np.allclose(q_values, self.initial_q):
            return int(fallback_action)

        best_value = np.max(q_values)
        best_actions = np.flatnonzero(q_values == best_value)
        return int(self.rng.choice(best_actions.tolist()))

    def learn(self, observation, action, reward, next_observation, done):
        """Update one Q value from an environment transition."""

        q_values = self._q_values(observation)
        next_q_values = self._q_values(next_observation)

        current_value = q_values[action]
        future_value = 0.0 if done else np.max(next_q_values)
        target_value = reward + self.discount_factor * future_value

        q_values[action] = current_value + self.learning_rate * (target_value - current_value)

    def finish_episode(self):
        """Decay exploration after an episode."""

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Persist the learned Q-table as JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "action_count": self.action_count,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "state_precision": self.state_precision,
            "initial_q": self.initial_q,
            "q_table": [
                {"state": list(state), "values": values.tolist()}
                for state, values in self.q_table.items()
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path, rng=None):
        """Load an agent previously saved with save()."""

        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        agent = cls(
            payload["action_count"],
            learning_rate=payload.get("learning_rate", 0.1),
            discount_factor=payload.get("discount_factor", 0.95),
            epsilon=payload.get("epsilon", 0.05),
            epsilon_decay=payload.get("epsilon_decay", 0.995),
            epsilon_min=payload.get("epsilon_min", 0.05),
            state_precision=payload.get("state_precision", 8),
            initial_q=payload.get("initial_q", -100.0),
            rng=rng,
        )
        agent.q_table = {
            tuple(entry["state"]): np.array(entry["values"], dtype=float)
            for entry in payload.get("q_table", [])
        }
        return agent

    def _q_values(self, observation):
        state = self._state_key(observation)
        if state not in self.q_table:
            self.q_table[state] = np.full(self.action_count, self.initial_q, dtype=float)
        return self.q_table[state]

    def _state_key(self, observation):
        clipped = np.clip(np.asarray(observation, dtype=float), -1.0, 1.0)
        return tuple(np.rint(clipped * self.state_precision).astype(int).tolist())
