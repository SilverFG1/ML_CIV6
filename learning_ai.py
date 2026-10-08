"""Validated tabular Q-learning with legal actions and resumable checkpoints."""

import json
import math
import numbers
import os
import random
import tempfile
from pathlib import Path

import numpy as np


CHECKPOINT_VERSION = 2


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _finite(value, name, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _tuples(value):
    """Restore the nested tuples required by random.Random.setstate()."""
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def _reject_nonfinite_json(value):
    raise ValueError(f"nonfinite JSON number {value}")


def _validate_metadata(value, location="training_state", ancestors=None):
    """Reject invalid JSON metadata, including numbers that overflow on parsing."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must contain only finite numbers")
        return
    if not isinstance(value, (dict, list, tuple)):
        raise ValueError(f"{location} must contain only JSON values")
    ancestors = set() if ancestors is None else ancestors
    if id(value) in ancestors:
        raise ValueError(f"{location} must not contain circular references")
    ancestors.add(id(value))
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{location} keys must be strings")
                _validate_metadata(item, f"{location}.{key}", ancestors)
        else:
            for index, item in enumerate(value):
                _validate_metadata(item, f"{location}[{index}]", ancestors)
    finally:
        ancestors.remove(id(value))


class QLearningAgent:
    """Tabular Q-learning over normalized, discretized observations.

    Legal-action arguments contain action indices. Evaluation can read an unseen
    state without adding it to the Q-table. Greedy ties use the agent's RNG, so
    an evaluator should use a separate RNG or restore its state afterward.
    """

    def __init__(
        self,
        action_count,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        state_precision=100,
        initial_q=-100.0,
        rng=None,
        observation_schema=None,
        observation_size=None,
    ):
        self.action_count = _integer(action_count, "action_count", 1)
        self.learning_rate = _finite(learning_rate, "learning_rate", 0.0, 1.0)
        self.discount_factor = _finite(discount_factor, "discount_factor", 0.0, 1.0)
        self.epsilon = _finite(epsilon, "epsilon", 0.0, 1.0)
        self.epsilon_decay = _finite(epsilon_decay, "epsilon_decay", 0.0, 1.0)
        self.epsilon_min = _finite(epsilon_min, "epsilon_min", 0.0, 1.0)
        self.state_precision = _integer(state_precision, "state_precision", 1)
        self.initial_q = _finite(initial_q, "initial_q")
        self.rng = rng if rng is not None else random.Random()
        if not isinstance(self.rng, random.Random):
            raise ValueError("rng must be a random.Random instance")
        self.q_table = {}
        self.last_action_source = None
        self.episodes_trained = 0
        self.observation_schema = observation_schema
        self.observation_size = observation_size
        self.training_state = {}
        self._validate_configuration()

    def _validate_configuration(self):
        """Also check settings changed by a command-line runner after loading."""
        _integer(self.action_count, "action_count", 1)
        _integer(self.state_precision, "state_precision", 1)
        for name in ("learning_rate", "discount_factor", "epsilon", "epsilon_decay", "epsilon_min"):
            _finite(getattr(self, name), name, 0.0, 1.0)
        _finite(self.initial_q, "initial_q")
        _integer(self.episodes_trained, "episodes_trained")
        if self.observation_size is not None:
            _integer(self.observation_size, "observation_size", 1)
        if self.observation_schema is not None and (
            not isinstance(self.observation_schema, str) or not self.observation_schema.strip()
        ):
            raise ValueError("observation_schema must be a nonempty string or None")
        if not isinstance(self.training_state, dict):
            raise ValueError("training_state must be a dictionary")

    def _action(self, action, name="action"):
        action = _integer(action, name)
        if action >= self.action_count:
            raise ValueError(f"{name} must be below action_count ({self.action_count})")
        return action

    def _valid_actions(self, actions, allow_empty=False):
        if actions is None:
            return list(range(self.action_count))
        try:
            allowed = sorted({self._action(action, "valid action") for action in actions})
        except TypeError as exc:
            raise ValueError("valid_actions must be an iterable of action indices") from exc
        if not allowed and not allow_empty:
            raise ValueError("valid_actions must contain at least one action")
        return allowed

    def choose_action(
        self,
        observation,
        explore=True,
        fallback_action=None,
        guided_exploration_rate=0.0,
        valid_actions=None,
    ):
        """Choose an allowed action using exploration, guidance, or Q-values."""
        self._validate_configuration()
        if not isinstance(explore, (bool, np.bool_)):
            raise ValueError("explore must be a boolean")
        guided_rate = _finite(guided_exploration_rate, "guided_exploration_rate", 0.0, 1.0)
        allowed = self._valid_actions(valid_actions)
        fallback = None if fallback_action is None else self._action(fallback_action, "fallback_action")
        if fallback not in allowed:
            fallback = None
        state = self._state_key(observation)
        if explore and self.rng.random() < self.epsilon:
            if fallback is not None and self.rng.random() < guided_rate:
                self.last_action_source = "guided"
                return fallback
            self.last_action_source = "random"
            return int(self.rng.choice(allowed))

        q_values = self._values_for_state(state, create=bool(explore))
        legal_values = q_values[allowed]
        if fallback is not None and np.all(legal_values == self.initial_q):
            self.last_action_source = "guided"
            return fallback
        best_value = np.max(legal_values)
        best_actions = [action for action in allowed if q_values[action] == best_value]
        self.last_action_source = "policy"
        return int(self.rng.choice(best_actions))

    def learn(self, observation, action, reward, next_observation, done, next_valid_actions=None):
        """Update an action from a transition, bootstrapping only legal actions."""
        self._validate_configuration()
        action = self._action(action)
        reward = _finite(reward, "reward")
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError("done must be a boolean")
        state = self._state_key(observation)
        next_state = self._state_key(next_observation)
        if len(state) != len(next_state):
            raise ValueError("next_observation must have the same size as observation")
        allowed = self._valid_actions(next_valid_actions, allow_empty=bool(done))
        q_values = self._values_for_state(state)
        future_value = 0.0
        if not done:
            future_value = float(np.max(self._values_for_state(next_state)[allowed]))
        current_value = q_values[action]
        target_value = reward + self.discount_factor * future_value
        updated_value = current_value + self.learning_rate * (target_value - current_value)
        if not math.isfinite(updated_value):
            raise ValueError("Q-learning update produced a nonfinite value")
        q_values[action] = updated_value

    def finish_episode(self):
        """Record completed training and decay exploration."""
        self._validate_configuration()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1

    def _values_for_state(self, state, create=True):
        if state not in self.q_table:
            values = np.full(self.action_count, self.initial_q, dtype=float)
            if create:
                if self.observation_size is None:
                    self.observation_size = len(state)
                self.q_table[state] = values
            return values
        return self.q_table[state]

    def _q_values(self, observation):
        """Return a training row; retained for existing callers."""
        return self._values_for_state(self._state_key(observation))

    def _state_key(self, observation):
        try:
            vector = np.asarray(observation, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("observation must be a numeric vector") from exc
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("observation must be a nonempty one-dimensional vector")
        if not np.all(np.isfinite(vector)):
            raise ValueError("observation must contain only finite values")
        if self.observation_size is not None and vector.size != self.observation_size:
            raise ValueError(f"observation has {vector.size} values; expected {self.observation_size}")
        clipped = np.clip(vector, -1.0, 1.0)
        return tuple(np.rint(clipped * self.state_precision).astype(int).tolist())

    def _checkpoint_payload(self):
        self._validate_configuration()
        _validate_metadata(self.training_state)
        entries = []
        observed_size = self.observation_size
        for state, values in self.q_table.items():
            if not isinstance(state, tuple) or not state:
                raise ValueError("Q-table states must be nonempty tuples")
            checked_state = [_integer(value, "state value", -self.state_precision) for value in state]
            if any(abs(value) > self.state_precision for value in checked_state):
                raise ValueError("Q-table state value exceeds state_precision")
            if observed_size is None:
                observed_size = len(state)
            if len(state) != observed_size:
                raise ValueError("Q-table state dimensions do not match observation_size")
            try:
                checked_values = np.asarray(values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Q-table values must be numeric") from exc
            if checked_values.shape != (self.action_count,) or not np.all(np.isfinite(checked_values)):
                raise ValueError("Q-table values must have action_count finite values")
            entries.append({"state": checked_state, "values": checked_values.tolist()})
        payload = {
            "format_version": CHECKPOINT_VERSION,
            "action_count": self.action_count,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "state_precision": self.state_precision,
            "initial_q": self.initial_q,
            "episodes_trained": self.episodes_trained,
            "observation_schema": self.observation_schema,
            "observation_size": observed_size,
            "training_state": self.training_state,
            "rng_state": self.rng.getstate(),
            "q_table": entries,
        }
        return payload

    def save(self, path):
        """Atomically replace a versioned JSON checkpoint, including RNG state."""
        try:
            serialized = json.dumps(self._checkpoint_payload(), indent=2, allow_nan=False)
        except (TypeError, OverflowError) as exc:
            raise ValueError(f"Checkpoint metadata is not valid JSON: {exc}") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
                suffix=".tmp", delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(cls, path, rng=None):
        """Load and validate a versioned checkpoint or a legacy Q-table.

        Legacy checkpoints have no observation schema; the runner must establish
        compatibility before using one with a changed environment.
        """
        try:
            payload = json.loads(
                Path(path).read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json
            )
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"Invalid checkpoint JSON: {exc}") from exc
        try:
            return cls._load_payload(payload, rng)
        except (KeyError, TypeError, IndexError, OverflowError, ValueError) as exc:
            raise ValueError(f"Invalid checkpoint: {exc}") from exc

    @classmethod
    def _load_payload(cls, payload, rng):
        if not isinstance(payload, dict):
            raise ValueError("root must be a JSON object")
        version = payload.get("format_version", 1)
        if not isinstance(version, int) or isinstance(version, bool) or version not in (1, CHECKPOINT_VERSION):
            raise ValueError(f"unsupported format_version {version!r}")
        if version == CHECKPOINT_VERSION:
            required = (
                "action_count", "learning_rate", "discount_factor", "epsilon", "epsilon_decay",
                "epsilon_min", "state_precision", "initial_q", "episodes_trained",
                "observation_schema", "observation_size", "training_state", "rng_state", "q_table",
            )
            missing = [name for name in required if name not in payload]
            if missing:
                raise ValueError(f"missing required fields: {', '.join(missing)}")
        agent = cls(
            payload["action_count"], learning_rate=payload.get("learning_rate", 0.1),
            discount_factor=payload.get("discount_factor", 0.95), epsilon=payload.get("epsilon", 0.05),
            epsilon_decay=payload.get("epsilon_decay", 0.995), epsilon_min=payload.get("epsilon_min", 0.05),
            state_precision=payload.get("state_precision", 8), initial_q=payload.get("initial_q", -100.0),
            rng=rng, observation_schema=payload.get("observation_schema"),
            observation_size=payload.get("observation_size"),
        )
        agent.episodes_trained = _integer(payload.get("episodes_trained", 0), "episodes_trained")
        agent.training_state = payload.get("training_state", {})
        rows = payload.get("q_table", [])
        if not isinstance(rows, list):
            raise ValueError("q_table must be a list")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("state"), list):
                raise ValueError("each Q-table row must contain a state list and values list")
            state = tuple(row["state"])
            if state in agent.q_table:
                raise ValueError("duplicate Q-table state")
            agent.q_table[state] = np.asarray(row["values"], dtype=float)
        agent._checkpoint_payload()
        if agent.observation_size is None and agent.q_table:
            agent.observation_size = len(next(iter(agent.q_table)))
        if version == CHECKPOINT_VERSION:
            restored_state = _tuples(payload["rng_state"])
            probe = random.Random()
            probe.setstate(restored_state)
            agent.rng.setstate(restored_state)
        return agent
