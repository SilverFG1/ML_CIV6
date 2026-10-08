"""Runner regressions for fair evaluation, reporting, and exact continuation."""

import csv
import contextlib
import io
import json
import random
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np

import game
import run_game
from learning_ai import QLearningAgent


class ShortEpisode:
    """A changing legal mask with a configurable terminal outcome."""
    render = False

    def __init__(self, outcome="loss", terminal=True):
        self.outcome = outcome
        self.terminal = terminal
        self.last_info = {}
        self.turn = 0

    def reset(self, seed=None):
        self.turn = 0
        return np.array([0.0])

    def valid_actions(self):
        return [1] if self.turn == 0 else [2]

    def step(self, action):
        if action not in self.valid_actions():
            raise AssertionError("runner supplied an illegal action")
        self.turn += 1
        done = self.terminal and self.turn == 2
        self.last_info = {"outcome": self.outcome if done else "ongoing"}
        return np.array([self.turn / 10]), 1.0, done

    def guided_action(self, rng):
        return self.valid_actions()[0]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def args(self, *extra):
        return run_game.parse_args([
            "--model-path", str(self.root / "model.json"),
            "--metrics-dir", str(self.root / "metrics"),
            "--episodes", "2", "--max-steps", "4", "--eval-episodes", "2", "--log-every", "0",
            *extra,
        ])

    def make_agent(self, args):
        return run_game.build_agent(args, random.Random(999))

    def assert_agents_equal(self, first, second):
        self.assertEqual(first.episodes_trained, second.episodes_trained)
        self.assertEqual(first.epsilon, second.epsilon)
        self.assertEqual(first.rng.getstate(), second.rng.getstate())
        self.assertEqual(first.training_state, second.training_state)
        self.assertEqual(set(first.q_table), set(second.q_table))
        for key in first.q_table:
            np.testing.assert_array_equal(first.q_table[key], second.q_table[key])

    def test_terminal_loss_is_not_reported_as_a_win(self):
        agent = QLearningAgent(3, rng=random.Random(7))
        result = run_game.run_episode(ShortEpisode("loss"), agent, 0, 8, train=True)
        self.assertEqual(result.outcome, "loss")
        self.assertFalse(result.won)
        self.assertEqual(tuple(result), (2.0, False, 2))

    def test_terminal_win_and_guidance_counts(self):
        agent = QLearningAgent(3, epsilon=1, rng=random.Random(7))
        result = run_game.run_episode(ShortEpisode("win"), agent, 0, 8, train=True,
                                      use_guidance=True, guided_exploration_rate=1)
        self.assertTrue(result.won)
        self.assertEqual(result.guidance_actions, 2)
        self.assertEqual(result.guidance_actions + result.random_actions + result.policy_actions, result.steps)

    def test_learning_masks_next_state_and_ends_timeout_trajectory(self):
        agent = QLearningAgent(3, rng=random.Random(7))
        with patch.object(agent, "learn", wraps=agent.learn) as learn:
            result = run_game.run_episode(ShortEpisode(terminal=False), agent, 0, 2, train=True)
        self.assertEqual(result.outcome, "timeout")
        self.assertFalse(result.won)
        self.assertEqual(learn.call_args_list[0].kwargs["next_valid_actions"], [2])
        self.assertFalse(learn.call_args_list[0].args[4])
        self.assertEqual(learn.call_args_list[1].kwargs["next_valid_actions"], [])
        self.assertTrue(learn.call_args_list[1].args[4])

    def test_default_learned_evaluation_uses_no_guidance(self):
        args = self.args("--no-save")
        agent = self.make_agent(args)
        summaries = run_game.compare_policies(agent, args, verbose=False)
        self.assertEqual(set(summaries), {"learned", "heuristic", "random"})
        self.assertEqual(sum(row["guidance_actions"] for row in summaries["learned"]["episodes_detail"]), 0)
        self.assertGreater(sum(row["guidance_actions"] for row in summaries["heuristic"]["episodes_detail"]), 0)

    def test_eval_preserves_agent_and_all_policies_share_fixed_seeds(self):
        args = self.args("--no-save", "--eval-guidance")
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        before_rng = agent.rng
        before_state = agent.rng.getstate()
        before_table = {key: value.copy() for key, value in agent.q_table.items()}
        before_progress = (agent.episodes_trained, agent.epsilon, agent.observation_size, agent.last_action_source)
        summaries = run_game.compare_policies(agent, args, verbose=False)
        self.assertIs(agent.rng, before_rng)
        self.assertEqual(agent.rng.getstate(), before_state)
        self.assertEqual(before_progress,
                         (agent.episodes_trained, agent.epsilon, agent.observation_size, agent.last_action_source))
        self.assertEqual(set(before_table), set(agent.q_table))
        for key in before_table:
            np.testing.assert_array_equal(before_table[key], agent.q_table[key])
        for summary in summaries.values():
            self.assertEqual([row["seed"] for row in summary["episodes_detail"]],
                             [args.eval_seed, args.eval_seed + 1])
            self.assertEqual(summary["wins"] + summary["losses"] + summary["timeouts"], 2)
        reversed_results = run_game.compare_policies(agent, args, policies=list(reversed(summaries)), verbose=False)
        self.assertEqual(summaries, reversed_results)

    def test_curriculum_absolute_offsets_and_difficulty_presets(self):
        for difficulty, strength in run_game.DIFFICULTY_STRENGTHS.items():
            args = self.args("--difficulty", difficulty)
            self.assertEqual(run_game.config_from_args(args).city_strength, strength)
        args = self.args("--difficulty", "hard", "--city-strength", "23")
        self.assertEqual(run_game.config_from_args(args).city_strength, 23)
        args = self.args("--curriculum-episodes", "3")
        self.make_agent(args)
        self.assertEqual([run_game.episode_config(args, index).city_strength for index in range(5)],
                         [18, 23, 28, 28, 28])

    def test_split_resume_matches_uninterrupted_curriculum_training(self):
        full_args = self.args("--episodes", "4", "--curriculum-episodes", "4", "--seed", "23", "--no-save")
        full = self.make_agent(full_args)
        run_game.train_agent(full, full_args)
        split_args = self.args("--curriculum-episodes", "4", "--seed", "23", "--checkpoint-every", "1")
        split = self.make_agent(split_args)
        run_game.train_agent(split, split_args)
        # Evaluating the saved run must not consume continuation randomness.
        run_game.compare_policies(split, split_args, verbose=False)
        split.save(split_args.model_path)
        resume_args = self.args("--no-save")
        resumed = self.make_agent(resume_args)
        self.assertEqual(resume_args.seed, 23)
        self.assertEqual(resume_args.curriculum_episodes, 4)
        run_game.train_agent(resumed, resume_args)
        self.assert_agents_equal(full, resumed)
        self.assertEqual([row["episode"] for row in resumed.training_state["metrics"]], [1, 2, 3, 4])
        self.assertEqual([row["seed"] for row in resumed.training_state["metrics"]], [23, 24, 25, 26])

    def test_resume_restores_scenario_omitted_cli_flags(self):
        args = self.args("--width", "5", "--height", "6", "--unit-count", "2", "--difficulty", "easy",
                         "--wall-hp", "40", "--forest-density", "0.2", "--no-city-healing", "--unit-strengths", "18", "30")
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        resumed_args = self.args()
        resumed = self.make_agent(resumed_args)
        self.assertEqual(resumed_args.config, args.config)
        self.assertEqual(resumed.action_count, 49)
        self.assertEqual(resumed.observation_schema, agent.observation_schema)

    def test_resume_rejects_precision_and_scenario_changes(self):
        args = self.args()
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        for options in (("--state-precision", "8"), ("--width", "6"), ("--seed", "19"), ("--max-steps", "8")):
            with self.assertRaisesRegex(ValueError, "fresh"):
                self.make_agent(self.args(*options))

    def test_runner_rejects_legacy_observation_checkpoint(self):
        legacy = QLearningAgent(343, state_precision=8)
        legacy.save(self.root / "model.json")
        with self.assertRaisesRegex(ValueError, "observation format"):
            self.make_agent(self.args())
        fresh = self.make_agent(self.args("--fresh"))
        self.assertIsNotNone(fresh.observation_schema)
        self.assertEqual(fresh.state_precision, 100)

    def test_periodic_saves_track_absolute_episode_and_no_save_is_respected(self):
        args = self.args("--episodes", "3", "--checkpoint-every", "2")
        agent = self.make_agent(args)
        with patch.object(agent, "save", wraps=agent.save) as save:
            run_game.train_agent(agent, args)
        self.assertEqual(save.call_count, 1)
        saved = QLearningAgent.load(args.model_path)
        self.assertEqual(saved.training_state["checkpoint_episode"], 2)
        self.assertEqual(saved.episodes_trained, 2)
        other = self.args("--fresh", "--no-save", "--checkpoint-every", "1")
        no_save = self.make_agent(other)
        with patch.object(no_save, "save") as save:
            run_game.train_agent(no_save, other)
        save.assert_not_called()

    def test_best_checkpoint_uses_validation_seeds_and_preserves_rng(self):
        args = self.args("--save-best", "--validation-episodes", "2")
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        rng_state = agent.rng.getstate()
        with patch.object(run_game, "compare_policies", wraps=run_game.compare_policies) as compare:
            run_game._save_checkpoint(agent, args, agent.training_state["metrics"])
        self.assertEqual(compare.call_args.kwargs["seed"], args.validation_seed)
        self.assertEqual(compare.call_args.kwargs["policies"], ["learned"])
        self.assertEqual(agent.rng.getstate(), rng_state)
        self.assertTrue((self.root / "model.best.json").exists())
        self.assertTrue(args.model_path.exists())
        self.assertIn("best_validation_score", agent.training_state)

    def test_metrics_export_has_training_evaluation_and_valid_svg(self):
        args = self.args("--no-save")
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        evaluations = run_game.compare_policies(agent, args, verbose=False)
        rows = agent.training_state["metrics"]
        run_game.write_metrics(args.metrics_dir, rows, evaluations, args.config)
        payload = json.loads((args.metrics_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["training"], rows)
        self.assertEqual(payload["evaluation"], evaluations)
        with (args.metrics_dir / "training.csv").open(encoding="utf-8", newline="") as handle:
            exported = list(csv.DictReader(handle))
        self.assertEqual(len(exported), 2)
        self.assertEqual(set(exported[0]), set(run_game.METRIC_FIELDS))
        svg = ET.parse(args.metrics_dir / "learning_curve.svg")
        self.assertEqual(svg.getroot().tag, "{http://www.w3.org/2000/svg}svg")
        self.assertEqual(len(svg.findall(".//{http://www.w3.org/2000/svg}polyline")), 4)

    def test_cli_rejects_invalid_ranges(self):
        for options in (("--episodes", "-1"), ("--max-steps", "0"), ("--guided-exploration", "1.1"),
                        ("--learning-rate", "0"), ("--curriculum-episodes", "-1")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_game.parse_args(list(options))

    def test_best_validation_settings_restore_and_reject_changes(self):
        args = self.args('--validation-seed', '3000007', '--validation-episodes', '3')
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        restored_args = self.args()
        self.make_agent(restored_args)
        self.assertEqual(restored_args.validation_seed, 3000007)
        self.assertEqual(restored_args.validation_episodes, 3)
        for flags in [('--validation-seed', '9'), ('--validation-episodes', '8')]:
            with self.assertRaisesRegex(ValueError, 'fresh'):
                self.make_agent(self.args(*flags))

    def test_mixed_unit_types_resume_and_evaluate(self):
        args = self.args('--unit-count', '2', '--unit-types', 'warrior', 'archer')
        agent = self.make_agent(args)
        self.assertEqual(agent.action_count, 56)
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        resume_args = self.args()
        resumed = self.make_agent(resume_args)
        self.assertEqual(resume_args.config.unit_types, ('warrior', 'archer'))
        self.assertEqual(resumed.action_count, 56)
        evaluations = run_game.compare_policies(resumed, resume_args, verbose=False)
        self.assertEqual(set(evaluations), {'learned', 'heuristic', 'random'})

    def test_runner_rejects_corrupt_progress_metadata(self):
        args = self.args()
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        original = json.loads(args.model_path.read_text(encoding='utf-8'))
        for name, value in [('seed', 'bad'), ('max_steps', 0), ('episodes_completed', 999),
                            ('metrics', {}), ('no_guidance', 'false'),
                            ('best_validation_score', [2, 1])]:
            payload = json.loads(json.dumps(original))
            payload['training_state'][name] = value
            args.model_path.write_text(json.dumps(payload), encoding='utf-8')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Checkpoint'):
                self.make_agent(self.args())

    def test_curriculum_preset_and_single_episode_target(self):
        args = self.args('--difficulty', 'curriculum')
        self.make_agent(args)
        self.assertEqual(args.curriculum_episodes, 1000)
        self.assertEqual(run_game.episode_config(args, 0).city_strength, 18)
        self.assertEqual(run_game.episode_config(args, 999).city_strength, 38)
        single = self.args('--difficulty', 'hard', '--curriculum-episodes', '1')
        self.make_agent(single)
        self.assertEqual(run_game.episode_config(single, 0).city_strength, 38)


if __name__ == "__main__":
    unittest.main()
