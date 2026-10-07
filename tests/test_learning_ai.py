"""Regression checks for legal-action learning and checkpoint continuation."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from learning_ai import QLearningAgent


class LearningAgentTests(unittest.TestCase):
    def make_agent(self, **kwargs):
        options = dict(action_count=4, initial_q=0, rng=random.Random(13))
        options.update(kwargs)
        return QLearningAgent(**options)

    def test_default_precision_preserves_nearby_positions(self):
        agent = self.make_agent()
        self.assertEqual(agent.state_precision, 100)
        self.assertNotEqual(agent._state_key([0.01]), agent._state_key([0.02]))

    def test_random_exploration_obeys_legal_action_mask(self):
        agent = self.make_agent(epsilon=1)
        actions = [agent.choose_action([0], valid_actions=[1, 3]) for _ in range(100)]
        self.assertEqual(set(actions), {1, 3})
        self.assertEqual(agent.last_action_source, "random")

    def test_greedy_selection_excludes_larger_illegal_q_value(self):
        agent = self.make_agent(epsilon=0)
        agent._q_values([0])[:] = [1000, 2, 100, 3]
        self.assertEqual(agent.choose_action([0], False, valid_actions=[1, 3]), 3)
        self.assertEqual(agent.last_action_source, "policy")

    def test_guidance_is_masked_and_reported(self):
        agent = self.make_agent(epsilon=1)
        self.assertEqual(
            agent.choose_action([0], True, 3, 1, valid_actions=[1, 3]), 3
        )
        self.assertEqual(agent.last_action_source, "guided")
        self.assertEqual(
            agent.choose_action([0], True, 0, 1, valid_actions=[3]), 3
        )
        self.assertEqual(agent.last_action_source, "random")

    def test_unseen_state_guidance_applies_to_legal_values_only(self):
        agent = self.make_agent(epsilon=0)
        agent._q_values([0])[:] = [10, 0, 10, 0]
        self.assertEqual(agent.choose_action([0], False, 1, valid_actions=[1, 3]), 1)
        self.assertEqual(agent.last_action_source, "guided")

    def test_evaluation_does_not_insert_states_or_infer_dimensions(self):
        agent = self.make_agent()
        for _ in range(10):
            self.assertIn(agent.choose_action([0.2, 0.3], False, valid_actions=[1, 3]), [1, 3])
        self.assertEqual(agent.q_table, {})
        self.assertIsNone(agent.observation_size)

    def test_evaluation_does_not_change_existing_values(self):
        agent = self.make_agent()
        agent._q_values([0, 1])[:] = [4, 3, 2, 1]
        before = {state: row.copy() for state, row in agent.q_table.items()}
        agent.choose_action([0, 1], False)
        agent.choose_action([1, 1], False)
        self.assertEqual(set(before), set(agent.q_table))
        for state, values in before.items():
            np.testing.assert_array_equal(values, agent.q_table[state])

    def test_bootstrap_uses_only_legal_next_actions(self):
        agent = self.make_agent(learning_rate=1, discount_factor=0.5)
        agent._q_values([1])[:] = [100, 2, 200, 4]
        agent.learn([0], 2, 3, [1], False, next_valid_actions=[1, 3])
        self.assertEqual(agent._q_values([0])[2], 5)

    def test_terminal_update_has_no_bootstrap_or_next_state_insertion(self):
        agent = self.make_agent(learning_rate=0.5, initial_q=10)
        agent.learn([0], 1, -4, [1], True, next_valid_actions=[])
        self.assertEqual(agent._q_values([0])[1], 3)
        self.assertNotIn(agent._state_key([1]), agent.q_table)

    def test_finish_episode_tracks_progress_and_minimum_epsilon(self):
        agent = self.make_agent(epsilon=0.2, epsilon_decay=0.1, epsilon_min=0.05)
        agent.finish_episode()
        self.assertEqual(agent.episodes_trained, 1)
        self.assertEqual(agent.epsilon, 0.05)

    def test_invalid_configuration_rejected(self):
        cases = [
            {"action_count": 0}, {"action_count": 2.2}, {"action_count": True},
            {"state_precision": 0}, {"learning_rate": -0.1}, {"discount_factor": 1.1},
            {"epsilon": float("nan")}, {"epsilon_decay": float("inf")},
            {"epsilon_min": -1}, {"initial_q": float("nan")},
            {"observation_size": 0}, {"observation_schema": ""},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_agent(**kwargs)

    def test_invalid_observations_rejected(self):
        agent = self.make_agent(observation_size=2)
        for observation in ([], [[0, 1]], [0], [0, float("nan")], [0, float("inf")], ["bad", 1]):
            with self.subTest(observation=observation), self.assertRaisesRegex(ValueError, "observation"):
                agent.choose_action(observation, False)
        self.assertEqual(agent.q_table, {})

    def test_invalid_actions_and_masks_rejected(self):
        agent = self.make_agent()
        for mask in ([], [-1], [4], [1.5], [True], 2):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                agent.choose_action([0], valid_actions=mask)
        for action in (-1, 4, 0.1, True):
            with self.subTest(action=action), self.assertRaises(ValueError):
                agent.learn([0], action, 1, [1], False)
        with self.assertRaises(ValueError):
            agent.choose_action([0], fallback_action=4)
        with self.assertRaises(ValueError):
            agent.choose_action([0], guided_exploration_rate=2)

    def test_invalid_transition_rejected_before_inserting_states(self):
        agent = self.make_agent()
        for reward in (float("nan"), float("inf"), "3", True):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                agent.learn([0], 0, reward, [1], False)
        with self.assertRaisesRegex(ValueError, "same size"):
            agent.learn([0], 0, 1, [0, 1], False)
        with self.assertRaisesRegex(ValueError, "at least one"):
            agent.learn([0], 0, 1, [1], False, next_valid_actions=[])
        self.assertEqual(agent.q_table, {})

    def test_schema_dimensions_metadata_and_rng_round_trip(self):
        agent = self.make_agent(observation_schema="civ-v2", observation_size=2)
        agent.learn([0, 1], 2, 8, [1, 1], True)
        agent.finish_episode()
        agent.training_state = {"seed": 7, "history": [{"win": True}]}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "nested" / "model.json"
            agent.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["format_version"], 2)
            restored = QLearningAgent.load(path, rng=random.Random(999))
        self.assertEqual(restored.observation_schema, "civ-v2")
        self.assertEqual(restored.observation_size, 2)
        self.assertEqual(restored.episodes_trained, 1)
        self.assertEqual(restored.training_state, agent.training_state)
        self.assertEqual(restored.rng.getstate(), agent.rng.getstate())
        np.testing.assert_array_equal(restored._q_values([0, 1]), agent._q_values([0, 1]))

    def test_training_resume_matches_uninterrupted_training(self):
        uninterrupted = self.make_agent(epsilon=0.8, epsilon_decay=0.9)
        interrupted = self.make_agent(epsilon=0.8, epsilon_decay=0.9)

        def train(agent, start, stop):
            actions = []
            for episode in range(start, stop):
                observation = [episode % 2]
                next_observation = [(episode + 1) % 2]
                action = agent.choose_action(observation, True, 1, 0.3, valid_actions=[0, 1, 3])
                actions.append(action)
                agent.learn(observation, action, action - 1, next_observation, episode % 3 == 0,
                            next_valid_actions=[0, 3])
                agent.finish_episode()
            return actions

        expected = train(uninterrupted, 0, 30)
        actual = train(interrupted, 0, 12)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "model.json"
            interrupted.save(path)
            resumed = QLearningAgent.load(path)
            actual.extend(train(resumed, 12, 30))
        self.assertEqual(expected, actual)
        self.assertEqual(uninterrupted.epsilon, resumed.epsilon)
        self.assertEqual(uninterrupted.episodes_trained, resumed.episodes_trained)
        self.assertEqual(uninterrupted.rng.getstate(), resumed.rng.getstate())
        for state, row in uninterrupted.q_table.items():
            np.testing.assert_array_equal(row, resumed.q_table[state])

    def test_legacy_checkpoint_loads_with_unspecified_schema(self):
        legacy = {"action_count": 4, "q_table": [{"state": [0, 8], "values": [0, 1, 2, 3]}]}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "legacy.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            restored = QLearningAgent.load(path)
        self.assertIsNone(restored.observation_schema)
        self.assertEqual(restored.observation_size, 2)
        self.assertEqual(restored.state_precision, 8)

    def test_corrupt_checkpoint_fields_rejected(self):
        agent = self.make_agent(observation_schema="civ-v2", observation_size=1)
        agent._q_values([0])
        payload = agent._checkpoint_payload()
        bad_payloads = [
            [], {**payload, "format_version": 3}, {**payload, "format_version": 2.0},
            {**payload, "rng_state": []}, {**payload, "training_state": []},
            {**payload, "training_state": {"bad": float("nan")}},
            {**payload, "episodes_trained": -1}, {**payload, "observation_schema": 2},
            {**payload, "q_table": [{"state": [0], "values": [1, 2]}]},
            {**payload, "q_table": [{"state": [0], "values": [1, 2, 3, float("nan")]}]},
            {**payload, "q_table": [{"state": [0, 0], "values": [1, 2, 3, 4]}]},
            {**payload, "q_table": [{"state": [101], "values": [1, 2, 3, 4]}]},
            {**payload, "q_table": [{"state": [False], "values": [1, 2, 3, 4]}]},
            {**payload, "q_table": payload["q_table"] * 2},
        ]
        missing_rng = payload.copy()
        missing_rng.pop("rng_state")
        bad_payloads.append(missing_rng)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "bad.json"
            for index, bad_payload in enumerate(bad_payloads):
                path.write_text(json.dumps(bad_payload), encoding="utf-8")
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, "Invalid checkpoint"):
                    QLearningAgent.load(path)
            path.write_text("{invalid json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint JSON"):
                QLearningAgent.load(path)

    def test_failed_load_does_not_change_supplied_rng(self):
        agent = self.make_agent()
        payload = agent._checkpoint_payload()
        payload["rng_state"] = []
        rng = random.Random(3)
        before = rng.getstate()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                QLearningAgent.load(path, rng=rng)
        self.assertEqual(before, rng.getstate())

    def test_failed_atomic_replace_preserves_previous_model_and_removes_temp(self):
        agent = self.make_agent()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "model.json"
            agent.save(path)
            before = path.read_bytes()
            agent.finish_episode()
            with patch("learning_ai.os.replace", side_effect=OSError("simulated disk failure")):
                with self.assertRaisesRegex(OSError, "simulated disk failure"):
                    agent.save(path)
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_invalid_save_does_not_replace_checkpoint(self):
        agent = self.make_agent()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "model.json"
            agent.save(path)
            before = path.read_bytes()
            agent.training_state = {"bad": float("nan")}
            with self.assertRaises(ValueError):
                agent.save(path)
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()

