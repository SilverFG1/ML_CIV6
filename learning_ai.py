"""Validated tabular Q-learning agents with legal actions and resumable checkpoints.

TacticalQAgent is the squad learner trained by run_game.py. QLearningAgent is
the original joint-action Q-table over raw observations, kept as a baseline.
EmpireQAgent plays the empire game (empire.py), where every city, unit and the
research queue decide each turn.
"""

import collections
import json
import math
import numbers
import os
import random
import tempfile
from pathlib import Path

import numpy as np

from empire import DECISION_KINDS
from game import (COMMANDS, TACTICAL_FEATURE_LIMITS, TACTICAL_FEATURES, TACTICAL_FIRST_TACTIC,
                  TACTICAL_HEADER, TACTICAL_UNIT_WIDTH, TACTICS)


CHECKPOINT_VERSION = 2
TACTICAL_CHECKPOINT_VERSION = 3
EMPIRE_CHECKPOINT_VERSION = 1
# Longest loop of consecutive transitions the tactical learner recognizes.
LOOP_LIMIT = 4
# A unit's value for a tactic is the sum of one table entry per group: the full
# feature set for detail plus coarser subsets that generalize to situations
# rarely seen in full (tile coding).
TILE_GROUPS = (
    TACTICAL_FEATURES,
    ("unit_type", "distance", "risk", "city_ranged", "city_heals"),
    ("unit_type", "distance", "allies_adjacent", "allies_away", "city_ranged", "city_heals"),
    ("unit_type", "distance", "risk", "city_health", "walls", "ram_ready", "city_ranged", "city_heals"),
    ("unit_type", "distance", "health", "city_ranged", "city_heals"),
)

# Tile groups of the empire learner, by decision kind. Each kind's first group
# is every feature; the others are coarser subsets that share what is learned
# between situations that differ only in the features they leave out.
EMPIRE_TILE_GROUPS = {
    "research": (("stage", "technologies"), ("technologies", "cities", "coastal", "separated"),
                 ("technologies", "threat", "balance")),
    "city": (("idle", "stage", "population", "housing", "sites", "settlers", "cities"),
             ("idle", "threat", "defenders", "walls", "capital", "army"),
             ("idle", "production", "builders", "food", "amenities", "population"),
             ("idle", "stage", "coastal", "cities", "army")),
    "settler": (("here", "best", "distance"), ("stage", "cities", "threat", "escort")),
    "builder": (("here", "targets", "distance"), ("stage", "charges", "threat")),
    "land": (("unit_type", "health", "contact", "odds"),
             ("unit_type", "threat", "post", "territory", "escort"),
             ("stage", "unit_type", "enemy_city", "balance", "health")),
    "naval": (("unit_type", "health", "contact", "odds"), ("stage", "threat", "enemy_city", "balance")),
    "trader": (),
}
EMPIRE_TILE_GROUPS = {kind: (DECISION_KINDS[kind].features,) + groups for kind, groups in EMPIRE_TILE_GROUPS.items()}


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


def _save_payload(payload, path):
    """Atomically replace a JSON checkpoint; a failed write keeps the old file."""
    try:
        serialized = json.dumps(payload, indent=2, allow_nan=False)
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


def _read_checkpoint(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid checkpoint JSON: {exc}") from exc


def _restore(cls, payload, rng):
    try:
        return cls._load_payload(payload, rng)
    except (KeyError, TypeError, IndexError, OverflowError, ValueError) as exc:
        raise ValueError(f"Invalid checkpoint: {exc}") from exc


def _restore_rng(agent, rng_state):
    """Validate a saved random.Random state before touching the agent's RNG."""
    restored_state = _tuples(rng_state)
    probe = random.Random()
    probe.setstate(restored_state)
    agent.rng.setstate(restored_state)


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

    agent_kind = "joint"
    observation_mode = "raw"

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

    def finish_episode(self, anneal=True):
        """Record completed training and decay exploration.

        anneal is accepted for a common interface; this learner keeps a
        constant learning rate.
        """
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
        _save_payload(self._checkpoint_payload(), path)

    @classmethod
    def load(cls, path, rng=None):
        """Load and validate a versioned checkpoint or a legacy Q-table.

        Legacy checkpoints have no observation schema; the runner must establish
        compatibility before using one with a changed environment.
        """
        return _restore(cls, _read_checkpoint(path), rng)

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
            _restore_rng(agent, payload["rng_state"])
        return agent


class TacticalQAgent:
    """Factored Q-learning for a squad, trained on Game tactical observations.

    A joint action's value is the sum, over living units, of each unit's value
    for the tactic its command performs (a value-decomposition network in
    tabular form). The squad is still evaluated jointly over legal actions,
    while units of the same type share what they learn. Each unit's value is
    the sum of tile-coded entries over TILE_GROUPS, so situations never seen in
    full still inherit estimates from coarser feature subsets. Updates shape
    the reward with the potential ``shaping * total health fraction`` of living
    units (zero at terminal states): healing is credited and damage charged
    when they happen, and potential-based shaping leaves the optimal policy
    unchanged. After each annealed episode the learning rate shrinks by
    learning_rate_decay toward learning_rate_min (a rate already below the
    floor is left alone), so the greedy policy settles. The runner anneals
    only once a curriculum has reached its final scenario, since a changing
    task still needs full-sized updates.

    The interface matches QLearningAgent. A joint action index decodes to one
    command per unit in itertools.product order, with each unit's commands a
    prefix of game.COMMANDS: unit_command_counts holds 7 for units without
    SHOOT and 8 for ranged units.
    """

    agent_kind = "tactical"
    observation_mode = "tactical"

    def __init__(
        self,
        unit_command_counts,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        initial_q=0.0,
        shaping=0.5,
        learning_rate_decay=0.998,
        learning_rate_min=0.01,
        rng=None,
        observation_schema=None,
    ):
        if isinstance(unit_command_counts, (str, bytes)):
            raise ValueError("unit_command_counts must be a sequence of command counts")
        try:
            counts = tuple(_integer(count, "unit command count", len(COMMANDS) - 1)
                           for count in unit_command_counts)
        except TypeError as exc:
            raise ValueError("unit_command_counts must be a sequence of command counts") from exc
        if not counts or any(count > len(COMMANDS) for count in counts):
            raise ValueError(f"unit command counts must be {len(COMMANDS) - 1} or {len(COMMANDS)}")
        self.unit_command_counts = counts
        self.unit_count = len(counts)
        self.action_count = math.prod(counts)
        self.observation_size = self.unit_count * TACTICAL_UNIT_WIDTH
        self.learning_rate = _finite(learning_rate, "learning_rate", 0.0, 1.0)
        self.discount_factor = _finite(discount_factor, "discount_factor", 0.0, 1.0)
        self.epsilon = _finite(epsilon, "epsilon", 0.0, 1.0)
        self.epsilon_decay = _finite(epsilon_decay, "epsilon_decay", 0.0, 1.0)
        self.epsilon_min = _finite(epsilon_min, "epsilon_min", 0.0, 1.0)
        self.initial_q = _finite(initial_q, "initial_q")
        self.shaping = _finite(shaping, "shaping", 0.0, 10.0)
        self.learning_rate_decay = _finite(learning_rate_decay, "learning_rate_decay", 0.0, 1.0)
        self.learning_rate_min = _finite(learning_rate_min, "learning_rate_min", 0.0, 1.0)
        self.rng = rng if rng is not None else random.Random()
        if not isinstance(self.rng, random.Random):
            raise ValueError("rng must be a random.Random instance")
        self.observation_schema = observation_schema
        self.q_table = {}
        self.last_action_source = None
        self.episodes_trained = 0
        self.training_state = {}
        # Within-episode record of greedy decisions and recent transitions,
        # used to value loops the policy would repeat forever.
        self._decision = None
        self._trace = []
        self._groups = tuple(frozenset(TACTICAL_FEATURES.index(name) for name in group)
                             for group in TILE_GROUPS)
        # Row j holds the command index each unit performs in joint action j.
        self._commands = np.indices(counts).reshape(self.unit_count, -1).T
        self._limits = np.asarray(TACTICAL_FEATURE_LIMITS)
        self._validate_configuration()

    def _validate_configuration(self):
        """Also check settings changed by a command-line runner after loading."""
        for name in ("learning_rate", "discount_factor", "epsilon", "epsilon_decay", "epsilon_min",
                     "learning_rate_decay", "learning_rate_min"):
            _finite(getattr(self, name), name, 0.0, 1.0)
        _finite(self.initial_q, "initial_q")
        _finite(self.shaping, "shaping", 0.0, 10.0)
        _integer(self.episodes_trained, "episodes_trained")
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

    def _parse(self, observation):
        """Validate a tactical observation and split it into per-unit parts."""
        try:
            vector = np.asarray(observation, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("observation must be a numeric vector") from exc
        if vector.ndim != 1 or vector.size != self.observation_size:
            raise ValueError(f"observation must be a vector of {self.observation_size} tactical values")
        if not np.all(np.isfinite(vector)):
            raise ValueError("observation must contain only finite values")
        blocks = vector.reshape(self.unit_count, TACTICAL_UNIT_WIDTH)
        alive, health, fractions = blocks[:, 0], blocks[:, 1], blocks[:, 1:len(TACTICAL_HEADER)]
        features = blocks[:, len(TACTICAL_HEADER):TACTICAL_FIRST_TACTIC]
        tactics = blocks[:, TACTICAL_FIRST_TACTIC:]
        if not np.all((alive == 0) | (alive == 1)) or np.any((fractions < 0) | (fractions > 1)):
            raise ValueError("observation alive flags and fractions are invalid")
        if (np.any(features != np.round(features)) or np.any(features < 0)
                or np.any(features > self._limits)):
            raise ValueError("observation tactical features are out of range")
        if np.any(tactics != np.round(tactics)) or np.any(tactics < -1) or np.any(tactics >= len(TACTICS)):
            raise ValueError("observation tactics are out of range")
        return alive.astype(bool), health, features.astype(int), tactics.astype(int)

    def _tile_keys(self, features):
        values = [int(value) for value in features]
        return [(group, *(value if index in members else -1 for index, value in enumerate(values)))
                for group, members in enumerate(self._groups)]

    def _unit_values(self, features):
        """Return a unit's value for every tactic without inserting table rows."""
        total = np.zeros(len(TACTICS))
        for key in self._tile_keys(features):
            row = self.q_table.get(key)
            total += row if row is not None else self.initial_q / len(self._groups)
        return total

    def _joint_values(self, parsed, actions):
        alive, _, features, tactics = parsed
        commands = self._commands[actions]
        total = np.zeros(len(actions))
        for unit in np.flatnonzero(alive):
            chosen = tactics[unit, commands[:, unit]]
            if np.any(chosen < 0):
                raise ValueError("an allowed action uses a command the observation marks unavailable")
            total += self._unit_values(features[unit])[chosen]
        return total

    def _unseen(self, parsed):
        alive, _, features, _ = parsed
        return not any(key in self.q_table for unit in np.flatnonzero(alive)
                       for key in self._tile_keys(features[unit]))

    def _potential(self, parsed):
        alive, health, _, _ = parsed
        return self.shaping * float(health[alive].sum())

    def choose_action(
        self,
        observation,
        explore=True,
        fallback_action=None,
        guided_exploration_rate=0.0,
        valid_actions=None,
    ):
        """Choose an allowed action using exploration, guidance, or joint values.

        Guidance is used for a share of exploratory steps and whenever no living
        unit's situation has been learned at all.
        """
        self._validate_configuration()
        if not isinstance(explore, (bool, np.bool_)):
            raise ValueError("explore must be a boolean")
        guided_rate = _finite(guided_exploration_rate, "guided_exploration_rate", 0.0, 1.0)
        allowed = self._valid_actions(valid_actions)
        fallback = None if fallback_action is None else self._action(fallback_action, "fallback_action")
        if fallback not in allowed:
            fallback = None
        parsed = self._parse(observation)
        if explore and self.rng.random() < self.epsilon:
            if fallback is not None and self.rng.random() < guided_rate:
                self.last_action_source = "guided"
                return fallback
            self.last_action_source = "random"
            return int(self.rng.choice(allowed))
        if fallback is not None and self._unseen(parsed):
            self.last_action_source = "guided"
            return fallback
        values = self._joint_values(parsed, allowed)
        best_value = np.max(values)
        best_actions = [action for action, value in zip(allowed, values) if value == best_value]
        self.last_action_source = "policy"
        action = int(self.rng.choice(best_actions))
        if explore:
            self._decision = (np.asarray(observation, dtype=float).copy(), action)
        return action

    def learn(self, observation, action, reward, next_observation, done, next_valid_actions=None):
        """Move the taken joint action's value toward a shaped one-step target.

        A non-terminal next state, including one cut off by a turn limit,
        bootstraps from its best legal joint action, except that a closed loop
        is valued as repeating forever (see _record). The joint value moves
        learning_rate of the way to the target, spread over the tile entries
        of every living unit's chosen tactic.
        """
        self._validate_configuration()
        action = self._action(action)
        reward = _finite(reward, "reward")
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError("done must be a boolean")
        parsed = self._parse(observation)
        next_parsed = self._parse(next_observation)
        allowed = self._valid_actions(next_valid_actions, allow_empty=bool(done))
        alive, _, features, tactics = parsed
        commands = self._commands[action]
        updates = collections.Counter()
        current = 0.0
        for unit in np.flatnonzero(alive):
            tactic = int(tactics[unit, commands[unit]])
            if tactic < 0:
                raise ValueError("action uses a command the observation marks unavailable")
            current += self._unit_values(features[unit])[tactic]
            updates.update((key, tactic) for key in self._tile_keys(features[unit]))
        next_potential = 0.0 if done else self._potential(next_parsed)
        target = reward + self.discount_factor * next_potential - self._potential(parsed)
        loop = None if done else self._record(observation, action, target, next_observation)
        if loop is not None:
            target = loop
        elif not done:
            target += self.discount_factor * float(np.max(self._joint_values(next_parsed, allowed)))
        if not updates:
            return
        # A normalized step moves the joint value exactly learning_rate of the way
        # to the target. Units in the same situation share entries, so an entry
        # counted c times has gradient c; an unnormalized step would grow with
        # the square of the squad size and diverge for larger squads.
        step = self.learning_rate * (target - current) / sum(count * count for count in updates.values())
        if not math.isfinite(step):
            raise ValueError("Q-learning update produced a nonfinite value")
        for (key, tactic), count in updates.items():
            row = self.q_table.get(key)
            if row is None:
                row = self.q_table[key] = np.full(len(TACTICS), self.initial_q / len(self._groups))
            row[tactic] += step * count

    def _record(self, observation, action, shaped_reward, next_observation):
        """Track this episode's transitions and value a loop that just closed.

        A deterministic loop repeats forever once it closes, so its value is
        exact for the policy that formed it, and bootstrapping from the inflated
        estimates inside a stall would decay by only (1 - discount**k) per lap.
        A single step that changes nothing is valued this way whatever chose it:
        the result is exact when staying is optimal and otherwise underestimates
        an action the policy should not take. A longer loop is used only when
        the agent's own greedy choices formed every step, so the closing action
        is not blamed for an exploratory detour. Returns None without a loop.
        """
        observation = np.asarray(observation, dtype=float)
        next_observation = np.asarray(next_observation, dtype=float)
        greedy = (self._decision is not None and self._decision[1] == action
                  and np.array_equal(self._decision[0], observation))
        self._decision = None
        if self._trace and not np.array_equal(self._trace[-1][3], observation):
            self._trace = []  # not a continuation of the recorded episode
        self._trace = (self._trace + [(observation, shaped_reward, greedy, next_observation)])[-LOOP_LIMIT:]
        if self.discount_factor >= 1:
            return None
        for length in range(1, len(self._trace) + 1):
            start = self._trace[-length]
            if not np.array_equal(start[0], next_observation):
                continue
            if length > 1 and not all(step[2] for step in self._trace[-length:]):
                return None
            rewards = [step[1] for step in self._trace[-length:]]
            # Value of the loop from the state after this step, which starts it again.
            cycle = sum(self.discount_factor ** index * value for index, value in enumerate(rewards))
            return shaped_reward + self.discount_factor * cycle / (1 - self.discount_factor ** length)
        return None

    def finish_episode(self, anneal=True):
        """Record completed training, decay exploration and, with anneal, the learning rate."""
        self._decision = None
        self._trace = []
        self._validate_configuration()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if anneal:
            self.learning_rate = max(min(self.learning_rate, self.learning_rate_min),
                                     self.learning_rate * self.learning_rate_decay)
        self.episodes_trained += 1

    def _checkpoint_payload(self):
        self._validate_configuration()
        _validate_metadata(self.training_state)
        entries = []
        for key, values in self.q_table.items():
            self._validate_key(key)
            checked = np.asarray(values, dtype=float)
            if checked.shape != (len(TACTICS),) or not np.all(np.isfinite(checked)):
                raise ValueError(f"Q-table values must have {len(TACTICS)} finite values")
            entries.append({"state": list(key), "values": checked.tolist()})
        return {
            "format_version": TACTICAL_CHECKPOINT_VERSION,
            "agent": self.agent_kind,
            "unit_command_counts": list(self.unit_command_counts),
            "action_count": self.action_count,
            "features": list(TACTICAL_FEATURES),
            "tactics": list(TACTICS),
            "tile_groups": [list(group) for group in TILE_GROUPS],
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "initial_q": self.initial_q,
            "shaping": self.shaping,
            "learning_rate_decay": self.learning_rate_decay,
            "learning_rate_min": self.learning_rate_min,
            "episodes_trained": self.episodes_trained,
            "observation_schema": self.observation_schema,
            "observation_size": self.observation_size,
            "training_state": self.training_state,
            "rng_state": self.rng.getstate(),
            "q_table": entries,
        }

    def _validate_key(self, key):
        if (not isinstance(key, tuple) or len(key) != 1 + len(TACTICAL_FEATURES)
                or any(isinstance(value, bool) or not isinstance(value, int) for value in key)):
            raise ValueError("Q-table states must be tuples of a group index and feature values")
        group = key[0]
        if not 0 <= group < len(self._groups):
            raise ValueError("Q-table state has an unknown tile group")
        for index, value in enumerate(key[1:]):
            if index in self._groups[group]:
                if not 0 <= value <= TACTICAL_FEATURE_LIMITS[index]:
                    raise ValueError("Q-table state feature is out of range")
            elif value != -1:
                raise ValueError("Q-table state sets a feature outside its tile group")

    def save(self, path):
        """Atomically replace a versioned JSON checkpoint, including RNG state."""
        _save_payload(self._checkpoint_payload(), path)

    @classmethod
    def load(cls, path, rng=None):
        """Load and validate a tactical checkpoint."""
        return _restore(cls, _read_checkpoint(path), rng)

    @classmethod
    def _load_payload(cls, payload, rng):
        if not isinstance(payload, dict):
            raise ValueError("root must be a JSON object")
        version = payload.get("format_version")
        if isinstance(version, bool) or version != TACTICAL_CHECKPOINT_VERSION or payload.get("agent") != cls.agent_kind:
            raise ValueError("not a tactical agent checkpoint")
        required = (
            "unit_command_counts", "action_count", "features", "tactics", "tile_groups", "learning_rate",
            "discount_factor", "epsilon", "epsilon_decay", "epsilon_min", "initial_q", "shaping",
            "learning_rate_decay", "learning_rate_min", "episodes_trained", "observation_schema", "observation_size", "training_state", "rng_state",
            "q_table",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        if (payload["features"] != list(TACTICAL_FEATURES) or payload["tactics"] != list(TACTICS)
                or payload["tile_groups"] != [list(group) for group in TILE_GROUPS]):
            raise ValueError("checkpoint uses a different tactical encoding; retrain with --fresh")
        if not isinstance(payload["unit_command_counts"], list):
            raise ValueError("unit_command_counts must be a list")
        agent = cls(
            payload["unit_command_counts"], learning_rate=payload["learning_rate"],
            discount_factor=payload["discount_factor"], epsilon=payload["epsilon"],
            epsilon_decay=payload["epsilon_decay"], epsilon_min=payload["epsilon_min"],
            initial_q=payload["initial_q"], shaping=payload["shaping"],
            learning_rate_decay=payload["learning_rate_decay"],
            learning_rate_min=payload["learning_rate_min"], rng=rng,
            observation_schema=payload["observation_schema"],
        )
        if payload["action_count"] != agent.action_count or payload["observation_size"] != agent.observation_size:
            raise ValueError("action_count and observation_size do not match the unit command counts")
        agent.episodes_trained = _integer(payload["episodes_trained"], "episodes_trained")
        agent.training_state = payload["training_state"]
        rows = payload["q_table"]
        if not isinstance(rows, list):
            raise ValueError("q_table must be a list")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("state"), list):
                raise ValueError("each Q-table row must contain a state list and values list")
            key = tuple(row["state"])
            if key in agent.q_table:
                raise ValueError("duplicate Q-table state")
            agent.q_table[key] = np.asarray(row["values"], dtype=float)
        agent._checkpoint_payload()
        _restore_rng(agent, payload["rng_state"])
        return agent


class EmpireQAgent:
    """Tabular Q-learning for a whole civilization in the empire game.

    An observation is a tuple of empire.Decision records: one for the research
    queue, each city and each unit, with integer features and the legal option
    indices. An action is one option per decision. Every decision estimates
    the civilization's discounted return, as independent learners sharing one
    target would: the turn's reward plus the discounted mean, over the next
    turn's decisions, of each decision's best legal value. A decision's own
    entity is left out of that mean when it decides again next turn, so a unit
    that holds still cannot bootstrap from its own unchanged estimate and keep
    an overestimate alive. A sum of per-entity values, as in the tactical
    learner's value decomposition, does not suit a set of entities that
    changes size: it would credit training any unit with one more term and
    blame a builder for spending its last charge or a settler for founding a
    city.

    A decision's value for option o is V(f) + A(f, o): a situation value V that
    every visit updates and an advantage A that only choosing o updates, each a
    sum of tile-coded table rows over EMPIRE_TILE_GROUPS[kind]. The value level
    therefore follows the situation even for options seldom taken, and options
    are ranked by advantage alone. A new row starts V at its share of the
    target that created it and every advantage at -untried_penalty (shared over
    the groups), so the learner keeps to options it has evidence for, such as
    the heuristic's in demonstrations, until its own exploration shows another
    option is better.

    Thirty or more entities share one reward, so a single unit's choice barely
    moves its target and learned advantages are noisy. Demonstrations
    therefore also teach with a large-margin loss (DQfD, Hester et al. 2018):
    on a demonstrated decision, every other option valued within
    demonstration_margin of the demonstrated one has its advantage pushed
    down by learning_rate of the shortfall. The demonstrated option keeps its
    temporal-difference estimate. The greedy policy follows the demonstrations
    unless the learner's own experience values another option by more than
    the margin.

    Exploration is per decision: each explores with probability epsilon, using
    the heuristic's option for a guided share of those, so a turn mixes
    exploratory and greedy choices instead of randomizing a whole empire at
    once. A decision whose situation has never been learned at all follows the
    heuristic when one is supplied. Each update moves V and the chosen
    advantage learning_rate of the way to their targets, and the learning rate
    anneals as in TacticalQAgent.
    """

    agent_kind = "empire"
    observation_mode = "empire"
    # run_game.run_episode tells learn() which transitions are demonstrations.
    learns_from_demonstrations = True

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.97,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        initial_q=0.0,
        learning_rate_decay=0.998,
        learning_rate_min=0.01,
        untried_penalty=0.5,
        demonstration_margin=1.0,
        rng=None,
        observation_schema=None,
    ):
        self.learning_rate = _finite(learning_rate, "learning_rate", 0.0, 1.0)
        self.discount_factor = _finite(discount_factor, "discount_factor", 0.0, 1.0)
        self.epsilon = _finite(epsilon, "epsilon", 0.0, 1.0)
        self.epsilon_decay = _finite(epsilon_decay, "epsilon_decay", 0.0, 1.0)
        self.epsilon_min = _finite(epsilon_min, "epsilon_min", 0.0, 1.0)
        self.initial_q = _finite(initial_q, "initial_q")
        self.learning_rate_decay = _finite(learning_rate_decay, "learning_rate_decay", 0.0, 1.0)
        self.learning_rate_min = _finite(learning_rate_min, "learning_rate_min", 0.0, 1.0)
        self.untried_penalty = _finite(untried_penalty, "untried_penalty", 0.0, 100.0)
        self.demonstration_margin = _finite(demonstration_margin, "demonstration_margin", 0.0, 100.0)
        self.rng = rng if rng is not None else random.Random()
        if not isinstance(self.rng, random.Random):
            raise ValueError("rng must be a random.Random instance")
        self.observation_schema = observation_schema
        self.observation_size = None
        self.action_count = None
        # Each row holds the situation value V followed by one advantage per option.
        self.q_table = {}
        self.last_action_source = None
        self.episodes_trained = 0
        self.training_state = {}
        self._groups = {kind: tuple(frozenset(DECISION_KINDS[kind].features.index(name) for name in group)
                                    for group in groups) for kind, groups in EMPIRE_TILE_GROUPS.items()}
        self._validate_configuration()

    def _validate_configuration(self):
        for name in ("learning_rate", "discount_factor", "epsilon", "epsilon_decay", "epsilon_min",
                     "learning_rate_decay", "learning_rate_min"):
            _finite(getattr(self, name), name, 0.0, 1.0)
        _finite(self.initial_q, "initial_q")
        _finite(self.untried_penalty, "untried_penalty", 0.0, 100.0)
        _finite(self.demonstration_margin, "demonstration_margin", 0.0, 100.0)
        _integer(self.episodes_trained, "episodes_trained")
        if self.observation_schema is not None and (
            not isinstance(self.observation_schema, str) or not self.observation_schema.strip()
        ):
            raise ValueError("observation_schema must be a nonempty string or None")
        if not isinstance(self.training_state, dict):
            raise ValueError("training_state must be a dictionary")

    def _parse(self, observation, valid_actions=None):
        """Validate decisions and return (kind, features, legal options) for each."""
        return [(kind, features, legal) for kind, _, features, legal in self._decode(observation, valid_actions)]

    def _decode(self, observation, valid_actions=None):
        """Validate decisions and return (kind, entity, features, legal options) for each."""
        if isinstance(observation, (str, bytes)):
            raise ValueError("observation must be a sequence of decisions")
        try:
            decisions = list(observation)
        except TypeError as exc:
            raise ValueError("observation must be a sequence of decisions") from exc
        if valid_actions is not None:
            valid_actions = list(valid_actions)
            if len(valid_actions) != len(decisions):
                raise ValueError("valid_actions must hold one option list per decision")
        parsed = []
        for index, decision in enumerate(decisions):
            try:
                kind, entity, features, valid = decision
            except (TypeError, ValueError) as exc:
                raise ValueError("each decision must hold a kind, entity, features and valid options") from exc
            spec = DECISION_KINDS.get(kind) if isinstance(kind, str) else None
            if spec is None:
                raise ValueError(f"unknown decision kind {kind!r}")
            features = tuple(features)
            if len(features) != len(spec.features) or any(
                    isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= value <= limit
                    for value, limit in zip(features, spec.limits)):
                raise ValueError(f"{kind} decision features are out of range")
            legal = valid if valid_actions is None else valid_actions[index]
            legal = tuple(sorted({self._option(kind, option) for option in legal}))
            if not legal:
                raise ValueError(f"{kind} decision has no legal option")
            parsed.append((kind, entity, tuple(int(value) for value in features), legal))
        return parsed

    @staticmethod
    def _option(kind, option):
        option = _integer(option, f"{kind} option")
        if option >= len(DECISION_KINDS[kind].options):
            raise ValueError(f"{kind} option {option} is out of range")
        return option

    def _tile_keys(self, kind, features):
        return [(kind, group, *(value if index in members else -1 for index, value in enumerate(features)))
                for group, members in enumerate(self._groups[kind])]

    def _parts(self, kind, features):
        """The situation value and the advantage of every option, summed over tile groups.

        Rows not yet in the table count as the initial value with untried advantages.
        """
        groups = len(self._groups[kind])
        value, advantages = 0.0, np.zeros(len(DECISION_KINDS[kind].options))
        for key in self._tile_keys(kind, features):
            row = self.q_table.get(key)
            if row is None:
                value += self.initial_q / groups
                advantages -= self.untried_penalty / groups
            else:
                value += row[0]
                advantages += row[1:]
        return value, advantages

    def _values(self, kind, features):
        value, advantages = self._parts(kind, features)
        return value + advantages

    def _unseen(self, kind, features):
        return not any(key in self.q_table for key in self._tile_keys(kind, features))

    def _best(self, kind, features, legal):
        values = self._values(kind, features)
        best = max(values[option] for option in legal)
        return [option for option in legal if values[option] == best], best

    def choose_action(
        self,
        observation,
        explore=True,
        fallback_action=None,
        guided_exploration_rate=0.0,
        valid_actions=None,
    ):
        """Choose one legal option per decision by exploration, guidance or greedy value."""
        self._validate_configuration()
        if not isinstance(explore, (bool, np.bool_)):
            raise ValueError("explore must be a boolean")
        guided_rate = _finite(guided_exploration_rate, "guided_exploration_rate", 0.0, 1.0)
        parsed = self._parse(observation, valid_actions)
        if fallback_action is not None:
            fallback_action = list(fallback_action)
            if len(fallback_action) != len(parsed):
                raise ValueError("fallback_action must hold one option per decision")
        sources = set()
        action = []
        for index, (kind, features, legal) in enumerate(parsed):
            fallback = None
            if fallback_action is not None and fallback_action[index] in legal:
                fallback = fallback_action[index]
            if explore and self.rng.random() < self.epsilon:
                if fallback is not None and self.rng.random() < guided_rate:
                    sources.add("guided")
                    action.append(fallback)
                else:
                    sources.add("random")
                    action.append(int(self.rng.choice(legal)))
            elif fallback is not None and self._unseen(kind, features):
                sources.add("guided")
                action.append(fallback)
            else:
                best, _ = self._best(kind, features, legal)
                action.append(int(self.rng.choice(best)))
        self.last_action_source = "random" if "random" in sources else "guided" if sources else "policy"
        return tuple(action)

    def value(self, observation):
        """The mean, over decisions, of each decision's best legal value (zero without decisions)."""
        parsed = self._parse(observation)
        if not parsed:
            return 0.0
        return float(sum(self._best(kind, features, legal)[1] for kind, features, legal in parsed) / len(parsed))

    def learn(self, observation, action, reward, next_observation, done, next_valid_actions=None,
              demonstration=False):
        """Move every decision's value for its chosen option toward the shared one-step target.

        The target bootstraps from the next turn's decisions unless the game
        ended. A turn cut off by a turn limit bootstraps too. With
        demonstration, the large-margin loss also applies to every decision.
        """
        self._validate_configuration()
        if not isinstance(demonstration, (bool, np.bool_)):
            raise ValueError("demonstration must be a boolean")
        reward = _finite(reward, "reward")
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError("done must be a boolean")
        decisions = self._decode(observation)
        if isinstance(action, (str, bytes)) or len(action) != len(decisions):
            raise ValueError("action must hold one option per decision")
        following = self._decode(next_observation, None if done or next_valid_actions is None else next_valid_actions)
        chosen, legal_options = [], {}
        for (kind, entity, features, legal), option in zip(decisions, action):
            option = self._option(kind, option)
            if option not in legal:
                raise ValueError(f"option {option} is not legal for the {kind} decision")
            chosen.append((kind, entity, features, option))
            legal_options[(kind, entity)] = legal
        successors, total = {}, 0.0
        if not done:
            for kind, entity, features, legal in following:
                best = self._best(kind, features, legal)[1]
                successors[(kind, entity)] = best
                total += best
        rate = self.learning_rate
        for kind, entity, features, option in chosen:
            target = reward
            if successors:
                own = successors.get((kind, entity))
                if own is not None and len(successors) > 1:
                    target += self.discount_factor * (total - own) / (len(successors) - 1)
                else:
                    target += self.discount_factor * total / len(successors)
            if not math.isfinite(target):
                raise ValueError("Q-learning update produced a nonfinite value")
            keys = self._tile_keys(kind, features)
            groups = len(keys)
            fresh = [key for key in keys if key not in self.q_table]
            if fresh:
                # New rows bring the situation value to the target and mark every option untried.
                known = sum(self.q_table[key][0] for key in keys if key in self.q_table)
                for key in fresh:
                    row = np.full(1 + len(DECISION_KINDS[kind].options), -self.untried_penalty / groups)
                    row[0] = (target - known) / len(fresh)
                    self.q_table[key] = row
            value, advantages = self._parts(kind, features)
            value_step = rate * (target - value) / groups
            advantage_step = rate * (target - value - advantages[option]) / groups
            for key in keys:
                row = self.q_table[key]
                row[0] += value_step
                row[1 + option] += advantage_step
            if demonstration and self.demonstration_margin > 0:
                advantages = self._parts(kind, features)[1]
                for other in legal_options[(kind, entity)]:
                    shortfall = advantages[other] + self.demonstration_margin - advantages[option]
                    if other != option and shortfall > 0:
                        for key in keys:
                            self.q_table[key][1 + other] -= rate * shortfall / groups

    def finish_episode(self, anneal=True):
        """Record completed training, decay exploration and, with anneal, the learning rate."""
        self._validate_configuration()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if anneal:
            self.learning_rate = max(min(self.learning_rate, self.learning_rate_min),
                                     self.learning_rate * self.learning_rate_decay)
        self.episodes_trained += 1

    def _validate_key(self, key):
        if not isinstance(key, tuple) or len(key) < 2 or key[0] not in DECISION_KINDS:
            raise ValueError("Q-table states must start with a decision kind")
        kind, group, features = key[0], key[1], key[2:]
        spec = DECISION_KINDS[kind]
        if isinstance(group, bool) or not isinstance(group, int) or not 0 <= group < len(self._groups[kind]):
            raise ValueError("Q-table state has an unknown tile group")
        if len(features) != len(spec.features):
            raise ValueError("Q-table state has the wrong number of features")
        for index, (value, limit) in enumerate(zip(features, spec.limits)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("Q-table state features must be integers")
            if index in self._groups[kind][group]:
                if not 0 <= value <= limit:
                    raise ValueError("Q-table state feature is out of range")
            elif value != -1:
                raise ValueError("Q-table state sets a feature outside its tile group")

    @staticmethod
    def encoding():
        """The decision kinds and tile groups a checkpoint must match."""
        return {kind: {"features": list(spec.features), "limits": list(spec.limits), "options": list(spec.options),
                       "tile_groups": [list(group) for group in EMPIRE_TILE_GROUPS[kind]]}
                for kind, spec in DECISION_KINDS.items()}

    def _checkpoint_payload(self):
        self._validate_configuration()
        _validate_metadata(self.training_state)
        entries = []
        for key, values in self.q_table.items():
            self._validate_key(key)
            checked = np.asarray(values, dtype=float)
            if checked.shape != (1 + len(DECISION_KINDS[key[0]].options),) or not np.all(np.isfinite(checked)):
                raise ValueError("Q-table rows must hold a finite value and one finite advantage per option")
            entries.append({"state": list(key), "values": checked.tolist()})
        return {
            "format_version": EMPIRE_CHECKPOINT_VERSION,
            "agent": self.agent_kind,
            "encoding": self.encoding(),
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "initial_q": self.initial_q,
            "learning_rate_decay": self.learning_rate_decay,
            "learning_rate_min": self.learning_rate_min,
            "untried_penalty": self.untried_penalty,
            "demonstration_margin": self.demonstration_margin,
            "episodes_trained": self.episodes_trained,
            "observation_schema": self.observation_schema,
            "training_state": self.training_state,
            "rng_state": self.rng.getstate(),
            "q_table": entries,
        }

    def save(self, path):
        """Atomically replace a versioned JSON checkpoint, including RNG state."""
        _save_payload(self._checkpoint_payload(), path)

    @classmethod
    def load(cls, path, rng=None):
        """Load and validate an empire checkpoint."""
        return _restore(cls, _read_checkpoint(path), rng)

    @classmethod
    def _load_payload(cls, payload, rng):
        if not isinstance(payload, dict):
            raise ValueError("root must be a JSON object")
        version = payload.get("format_version")
        if isinstance(version, bool) or version != EMPIRE_CHECKPOINT_VERSION or payload.get("agent") != cls.agent_kind:
            raise ValueError("not an empire agent checkpoint")
        required = ("encoding", "learning_rate", "discount_factor", "epsilon", "epsilon_decay", "epsilon_min",
                    "initial_q", "learning_rate_decay", "learning_rate_min", "untried_penalty",
                    "demonstration_margin", "episodes_trained",
                    "observation_schema", "training_state", "rng_state", "q_table")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        if payload["encoding"] != cls.encoding():
            raise ValueError("checkpoint uses a different empire encoding; retrain with --fresh")
        agent = cls(
            learning_rate=payload["learning_rate"], discount_factor=payload["discount_factor"],
            epsilon=payload["epsilon"], epsilon_decay=payload["epsilon_decay"], epsilon_min=payload["epsilon_min"],
            initial_q=payload["initial_q"], learning_rate_decay=payload["learning_rate_decay"],
            learning_rate_min=payload["learning_rate_min"], untried_penalty=payload["untried_penalty"],
            demonstration_margin=payload["demonstration_margin"], rng=rng,
            observation_schema=payload["observation_schema"],
        )
        agent.episodes_trained = _integer(payload["episodes_trained"], "episodes_trained")
        agent.training_state = payload["training_state"]
        rows = payload["q_table"]
        if not isinstance(rows, list):
            raise ValueError("q_table must be a list")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("state"), list):
                raise ValueError("each Q-table row must contain a state list and values list")
            key = tuple(row["state"])
            if key in agent.q_table:
                raise ValueError("duplicate Q-table state")
            agent.q_table[key] = np.asarray(row["values"], dtype=float)
        agent._checkpoint_payload()
        _restore_rng(agent, payload["rng_state"])
        return agent


AGENT_TYPES = {"tactical": TacticalQAgent, "joint": QLearningAgent, "empire": EmpireQAgent}


def load_agent(path, rng=None):
    """Load a checkpoint of either agent, choosing the class from its contents.

    Joint-agent checkpoints predate the agent field, so its absence means joint.
    """
    payload = _read_checkpoint(path)
    kind = payload.get("agent", "joint") if isinstance(payload, dict) else "joint"
    if kind not in AGENT_TYPES:
        raise ValueError(f"Invalid checkpoint: unknown agent {kind!r}")
    return _restore(AGENT_TYPES[kind], payload, rng)
