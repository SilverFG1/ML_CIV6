"""Runner regressions for fair evaluation, reporting, and exact continuation."""

import csv
import contextlib
import io
import json
import random
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

import empire
import game
import run_game
from learning_ai import EmpireQAgent, QLearningAgent, TacticalQAgent, load_agent


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

    def test_learning_masks_next_state_and_bootstraps_timeout(self):
        agent = QLearningAgent(3, rng=random.Random(7))
        with patch.object(agent, "learn", wraps=agent.learn) as learn:
            result = run_game.run_episode(ShortEpisode(terminal=False), agent, 0, 2, train=True)
        self.assertEqual(result.outcome, "timeout")
        self.assertFalse(result.won)
        self.assertEqual(learn.call_args_list[0].kwargs["next_valid_actions"], [2])
        self.assertFalse(learn.call_args_list[0].args[4])
        # The turn limit is not a game state: the cut-off transition still bootstraps.
        self.assertEqual(learn.call_args_list[1].kwargs["next_valid_actions"], [2])
        self.assertFalse(learn.call_args_list[1].args[4])

    def test_true_terminal_transition_does_not_bootstrap(self):
        agent = QLearningAgent(3, rng=random.Random(7))
        with patch.object(agent, "learn", wraps=agent.learn) as learn:
            run_game.run_episode(ShortEpisode("win"), agent, 0, 2, train=True)
        self.assertEqual(learn.call_args_list[-1].kwargs["next_valid_actions"], [])
        self.assertTrue(learn.call_args_list[-1].args[4])

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
        self.assertEqual(fresh.agent_kind, "tactical")
        joint = self.make_agent(self.args("--fresh", "--agent", "joint"))
        self.assertEqual(joint.state_precision, 100)

    def test_periodic_saves_track_absolute_episode_and_no_save_is_respected(self):
        args = self.args("--episodes", "3", "--checkpoint-every", "2")
        agent = self.make_agent(args)
        with patch.object(agent, "save", wraps=agent.save) as save:
            run_game.train_agent(agent, args)
        self.assertEqual(save.call_count, 1)
        saved = load_agent(args.model_path)
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
                        ("--learning-rate", "0"), ("--curriculum-episodes", "-1"), ("--demonstrations", "1.5")):
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

    def test_spearman_and_slinger_resume_and_evaluate(self):
        args = self.args('--unit-count', '2', '--unit-types', 'spearman', 'slinger')
        agent = self.make_agent(args)
        self.assertEqual(agent.action_count, 56)
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        resume_args = self.args()
        resumed = self.make_agent(resume_args)
        self.assertEqual(resume_args.config.unit_types, ('spearman', 'slinger'))
        evaluations = run_game.compare_policies(resumed, resume_args, verbose=False)
        self.assertEqual(set(evaluations), {'learned', 'heuristic', 'random'})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args('--unit-count', '1', '--unit-types', 'cavalry')

    def test_heavy_chariot_and_battering_ram_train_and_evaluate(self):
        args = self.args('--unit-count', '2', '--unit-types', 'heavy_chariot', 'battering_ram', '--wall-hp', '20')
        agent = self.make_agent(args)
        self.assertEqual(agent.action_count, 49)
        run_game.train_agent(agent, args)
        evaluations = run_game.compare_policies(agent, args, verbose=False)
        self.assertEqual(set(evaluations), {'learned', 'heuristic', 'random'})

    def test_rivers_resume_evaluate_and_reject_changes(self):
        args = self.args('--river-count', '2', '--forest-density', '0.1')
        agent = self.make_agent(args)
        # The joint learner's raw observation marks the option in its schema; the tactical
        # learner's egocentric format does not change.
        raw = game.Game(config=args.config, observation='raw')
        self.assertTrue(raw.observation_schema.endswith(':rivers'))
        self.assertTrue(agent.observation_schema.startswith('ml_civ6.tactical.v1:'))
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        resume_args = self.args()
        resumed = self.make_agent(resume_args)
        self.assertEqual(resume_args.config.river_count, 2)
        self.assertEqual(resumed.observation_size, agent.observation_size)
        evaluations = run_game.compare_policies(resumed, resume_args, verbose=False)
        self.assertEqual(set(evaluations), {'learned', 'heuristic', 'random'})
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.make_agent(self.args('--river-count', '1'))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_game.parse_args(['--river-count', '11'])

    def test_ancient_walls_resume_and_reject_changes(self):
        args = self.args('--walls', 'ancient', '--unit-count', '2', '--unit-types', 'warrior', 'archer')
        agent = self.make_agent(args)
        self.assertEqual((args.config.wall_hp, args.config.ranged_strength), (100, 15))
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        resume_args = self.args()
        self.make_agent(resume_args)
        self.assertEqual(resume_args.config, args.config)
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.make_agent(self.args('--walls', 'none'))
        custom = self.args('--walls', 'ancient', '--wall-hp', '60', '--fresh')
        self.make_agent(custom)
        self.assertEqual((custom.config.wall_hp, custom.config.ranged_strength), (60, 15))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_game.parse_args(['--walls', 'medieval'])

    def test_terrain_options_resume_and_reject_changes(self):
        args = self.args('--land-types', 'plains', 'desert', '--hills-density', '0.2', '--water-density', '0.1')
        agent = self.make_agent(args)
        # The joint learner's raw observation marks the option in its schema; the tactical
        # learner's egocentric format does not change.
        raw = game.Game(config=args.config, observation='raw')
        self.assertTrue(raw.observation_schema.endswith(':terrain'))
        self.assertTrue(agent.observation_schema.startswith('ml_civ6.tactical.v1:'))
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        resume_args = self.args()
        self.make_agent(resume_args)
        self.assertEqual(resume_args.config, args.config)
        self.assertEqual(resume_args.config.land_types, ('plains', 'desert'))
        for flags in [('--land-types', 'plains'), ('--hills-density', '0.3'), ('--water-density', '0')]:
            with self.assertRaisesRegex(ValueError, 'fresh'):
                self.make_agent(self.args(*flags))
        for flags in [('--land-types', 'lava'), ('--hills-density', '2')]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_game.parse_args(list(flags))

    def test_buildings_resume_and_reject_changes(self):
        args = self.args('--buildings', 'monument', 'granary')
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        resume_args = self.args()
        self.make_agent(resume_args)
        self.assertEqual(resume_args.config.buildings, ('monument', 'granary'))
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.make_agent(self.args('--buildings', 'monument'))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_game.parse_args(['--buildings', 'library'])

    def test_economy_resumes_and_rejects_changes(self):
        args = self.args('--economy', '--city-population', '2', '--build-queue', 'ancient_walls', 'monument')
        agent = self.make_agent(args)
        # The joint learner's raw observation marks the option in its schema; the tactical
        # learner's egocentric format does not change.
        raw = game.Game(config=args.config, observation='raw')
        self.assertTrue(raw.observation_schema.endswith(':economy'))
        self.assertTrue(agent.observation_schema.startswith('ml_civ6.tactical.v1:'))
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        resume_args = self.args()
        self.make_agent(resume_args)
        self.assertEqual(resume_args.config, args.config)
        self.assertEqual(resume_args.config.build_queue, ('ancient_walls', 'monument'))
        for flags in [('--city-population', '3'), ('--build-queue', 'monument')]:
            with self.assertRaisesRegex(ValueError, 'fresh'):
                self.make_agent(self.args(*flags))
        with self.assertRaisesRegex(ValueError, 'need the city economy'):
            self.make_agent(self.args('--city-population', '2', '--fresh'))
        for flags in [('--build-queue', 'palace'), ('--city-population', '0')]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_game.parse_args(list(flags))

    def test_checkpoint_without_river_setting_resumes_without_rivers(self):
        args = self.args()
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        del agent.training_state['config']['river_count']
        agent.save(args.model_path)
        resume_args = self.args()
        self.make_agent(resume_args)
        self.assertEqual(resume_args.config.river_count, 0)

    def test_runner_rejects_corrupt_progress_metadata(self):
        args = self.args()
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        original = json.loads(args.model_path.read_text(encoding='utf-8'))
        for name, value in [('seed', 'bad'), ('max_steps', 0), ('episodes_completed', 999),
                            ('metrics', {}), ('no_guidance', 'false'),
                            ('best_validation_score', [2, 1]), ('demonstrations', 2)]:
            payload = json.loads(json.dumps(original))
            payload['training_state'][name] = value
            args.model_path.write_text(json.dumps(payload), encoding='utf-8')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Checkpoint'):
                self.make_agent(self.args())
        steps = original['training_state']['metrics'][0]['steps']
        # A demonstration is played entirely by the heuristic.
        for changes in ({'demonstration': 'yes'},
                        {'demonstration': True, 'guidance_actions': 0, 'random_actions': steps, 'policy_actions': 0}):
            payload = json.loads(json.dumps(original))
            payload['training_state']['metrics'][0].update(changes)
            args.model_path.write_text(json.dumps(payload), encoding='utf-8')
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'Checkpoint'):
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


    def test_new_models_use_the_tactical_learner_and_checkpoints_keep_theirs(self):
        args = self.args()
        agent = self.make_agent(args)
        self.assertIsInstance(agent, TacticalQAgent)
        self.assertEqual((args.agent, args.observation), ("tactical", "tactical"))
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        resumed_args = self.args()
        self.assertIsInstance(self.make_agent(resumed_args), TacticalQAgent)
        self.assertEqual(resumed_args.agent, "tactical")
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.make_agent(self.args("--agent", "joint"))
        with self.assertRaisesRegex(ValueError, "joint learner"):
            self.make_agent(self.args("--state-precision", "8"))

    def test_joint_learner_remains_available_as_a_baseline(self):
        args = self.args("--agent", "joint")
        agent = self.make_agent(args)
        self.assertIsInstance(agent, QLearningAgent)
        self.assertEqual(args.observation, "raw")
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        resumed_args = self.args()
        resumed = self.make_agent(resumed_args)
        self.assertIsInstance(resumed, QLearningAgent)
        self.assertEqual(set(run_game.compare_policies(resumed, resumed_args, verbose=False)),
                         {"learned", "heuristic", "random"})
        with self.assertRaisesRegex(ValueError, "tactical learner"):
            self.make_agent(self.args("--shaping", "1"))

    def test_tactical_settings_restore_and_accept_overrides(self):
        args = self.args("--shaping", "0.25", "--learning-rate-decay", "0.99", "--learning-rate-min", "0.02")
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        restored = self.make_agent(self.args())
        self.assertEqual((restored.shaping, restored.learning_rate_decay, restored.learning_rate_min),
                         (0.25, 0.99, 0.02))
        self.assertAlmostEqual(restored.learning_rate, 0.1 * 0.99 ** 2)
        self.assertEqual(self.make_agent(self.args("--shaping", "0.75")).shaping, 0.75)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--learning-rate-decay", "1.5")

    def test_learning_rate_anneals_only_once_the_curriculum_is_complete(self):
        args = self.args("--no-save", "--episodes", "4", "--curriculum-episodes", "3", "--learning-rate-decay", "0.5")
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        # Episodes 0 and 1 still ramp the city's strength; episodes 2 and 3 use the final one.
        self.assertAlmostEqual(agent.learning_rate, 0.1 * 0.5 ** 2)
        steady = self.args("--fresh", "--no-save", "--episodes", "3", "--learning-rate-decay", "0.5")
        agent = self.make_agent(steady)
        run_game.train_agent(agent, steady)
        self.assertAlmostEqual(agent.learning_rate, 0.1 * 0.5 ** 3)

    def test_demonstrations_are_drawn_from_each_episode_seed_at_the_configured_share(self):
        args = self.args("--demonstrations", "0.4")
        self.make_agent(args)
        chosen = [run_game.is_demonstration(args, episode) for episode in range(1000)]
        self.assertTrue(350 <= sum(chosen) <= 450)  # 400 expected
        # A draw depends only on the episode, not on the draws before it.
        self.assertEqual(chosen[150:], [run_game.is_demonstration(args, episode) for episode in range(150, 1000)])
        other_seed = self.args("--seed", "8", "--demonstrations", "0.4")
        self.make_agent(other_seed)
        self.assertNotEqual(chosen, [run_game.is_demonstration(other_seed, episode) for episode in range(1000)])
        for share, expected in (("0", False), ("1", True)):
            extreme = self.args("--demonstrations", share)
            self.make_agent(extreme)
            self.assertEqual({run_game.is_demonstration(extreme, episode) for episode in range(100)}, {expected})

    def test_demonstrations_are_heuristic_episodes_that_train_the_learner(self):
        args = self.args("--no-save", "--episodes", "3")
        agent = self.make_agent(args)
        with patch.object(run_game, "is_demonstration", side_effect=lambda _, episode: episode != 1), \
                patch.object(agent, "choose_action", wraps=agent.choose_action) as choose, \
                patch.object(agent, "learn", wraps=agent.learn) as learn:
            rewards, _ = run_game.train_agent(agent, args)
        rows = agent.training_state["metrics"]
        self.assertEqual([row["demonstration"] for row in rows], [True, False, True])
        for row in (rows[0], rows[2]):
            self.assertEqual((row["guidance_actions"], row["random_actions"], row["policy_actions"]),
                             (row["steps"], 0, 0))
        # Every move trains the learner, but it chooses moves only in its own episode.
        self.assertEqual(learn.call_count, sum(row["steps"] for row in rows))
        self.assertEqual(choose.call_count, rows[1]["steps"])
        self.assertEqual(rewards, [rows[1]["reward"]])
        self.assertEqual(agent.episodes_trained, 3)
        summary = run_game.compare_policies(agent, args, policies=["learned"], verbose=False)["learned"]
        self.assertEqual(sum(row["guidance_actions"] for row in summary["episodes_detail"]), 0)

    def test_demonstrations_pause_while_they_score_below_the_learners_own_episodes(self):
        ahead = run_game.demonstrations_ahead
        evidence = run_game.DEMONSTRATION_EVIDENCE
        self.assertTrue(ahead({True: deque(), False: deque([5.0] * evidence)}))
        self.assertTrue(ahead({True: deque([2.0, 0.0] * evidence), False: deque([0.5] * evidence)}))
        self.assertFalse(ahead({True: deque([-26.0] * evidence), False: deque([-4.0] * evidence)}))
        # Too few demonstrations to judge: an unlucky start does not stop them.
        self.assertTrue(ahead({True: deque([-26.0] * (evidence - 1)), False: deque([-4.0] * evidence)}))
        args = self.args("--no-save", "--episodes", "4", "--seed", "23", "--demonstrations", "0.5")
        agent = self.make_agent(args)
        self.assertEqual([run_game.is_demonstration(args, episode) for episode in range(4)],
                         [True, False, True, True])
        play = run_game.run_episode

        def demonstrations_score_worse(*arguments, **options):
            result = play(*arguments, **options)
            result.reward = -50.0 if options["policy"] == "heuristic" else 0.0
            return result

        with patch.object(run_game, "run_episode", side_effect=demonstrations_score_worse), \
                patch.object(run_game, "DEMONSTRATION_EVIDENCE", 1):
            run_game.train_agent(agent, args)
        # Episodes 2 and 3 drew demonstrations, but the first one scored below the learner's episode.
        self.assertEqual([row["demonstration"] for row in agent.training_state["metrics"]],
                         [True, False, False, False])

    def test_split_resume_with_demonstrations_matches_uninterrupted_training(self):
        options = ("--seed", "23", "--demonstrations", "0.5")
        # With the gate judging from one episode of each, the resumed half must rebuild
        # its comparison from the recorded history to pause the same demonstrations.
        with patch.object(run_game, "DEMONSTRATION_EVIDENCE", 1):
            full_args = self.args("--episodes", "4", "--no-save", *options)
            full = self.make_agent(full_args)
            run_game.train_agent(full, full_args)
            self.assertEqual([run_game.is_demonstration(full_args, episode) for episode in range(4)],
                             [True, False, True, True])
            # The first demonstration scored below the learner's first episode, so later ones pause.
            self.assertEqual([row["demonstration"] for row in full.training_state["metrics"]],
                             [True, False, False, False])
            split_args = self.args("--checkpoint-every", "1", *options)
            run_game.train_agent(self.make_agent(split_args), split_args)
            resume_args = self.args("--no-save")
            resumed = self.make_agent(resume_args)
            self.assertEqual(resume_args.demonstrations, 0.5)
            run_game.train_agent(resumed, resume_args)
        self.assert_agents_equal(full, resumed)

    def test_demonstration_settings_restore_and_older_checkpoints_continue_without(self):
        args = self.args("--demonstrations", "0.2")
        agent = self.make_agent(args)
        run_game.train_agent(agent, args)
        agent.save(args.model_path)
        restored_args = self.args()
        self.make_agent(restored_args)
        self.assertEqual(restored_args.demonstrations, 0.2)
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.make_agent(self.args("--demonstrations", "0.3"))
        # Checkpoints saved before demonstrations existed have no share or history field for them.
        payload = json.loads(args.model_path.read_text(encoding="utf-8"))
        del payload["training_state"]["demonstrations"]
        for row in payload["training_state"]["metrics"]:
            del row["demonstration"]
        args.model_path.write_text(json.dumps(payload), encoding="utf-8")
        older_args = self.args()
        older = self.make_agent(older_args)
        self.assertEqual(older_args.demonstrations, 0.0)
        run_game.train_agent(older, older_args)
        self.assertEqual([row["demonstration"] for row in older.training_state["metrics"]], [False] * 4)

    def test_demonstrations_need_guidance_and_are_off_for_the_joint_learner(self):
        tactical = self.args()
        self.make_agent(tactical)
        self.assertEqual(tactical.demonstrations, run_game.DEMONSTRATION_DEFAULTS["tactical"])
        self.assertGreater(tactical.demonstrations, 0)
        unguided = self.args("--no-guidance")
        self.make_agent(unguided)
        self.assertEqual(unguided.demonstrations, 0.0)
        with self.assertRaisesRegex(ValueError, "no-guidance"):
            self.make_agent(self.args("--no-guidance", "--demonstrations", "0.3"))
        joint = self.args("--agent", "joint")
        self.make_agent(joint)
        self.assertEqual(joint.demonstrations, 0.0)

    def test_learning_curves_leave_demonstrations_out_of_performance_panels(self):
        rows = [dict(episode=index + 1, reward=float(index), outcome="win", steps=5, q_table_size=index,
                     demonstration=index % 2 == 0) for index in range(6)]
        svg = ET.fromstring(run_game._curve_svg(rows))
        lines = svg.findall(".//{http://www.w3.org/2000/svg}polyline")
        self.assertEqual([len(line.get("points").split()) for line in lines], [3, 3, 3, 6])
        self.assertIn("without demonstrations", ET.tostring(svg, encoding="unicode"))
        only_demonstrations = ET.fromstring(run_game._curve_svg(rows[:1]))
        self.assertEqual(len(only_demonstrations.findall(".//{http://www.w3.org/2000/svg}polyline")), 1)

    def test_tactical_learner_wins_held_out_games_without_guidance(self):
        """Regression check that learning works: a short schedule on a small map."""
        args = self.args("--fresh", "--no-save", "--width", "6", "--height", "6", "--unit-count", "2",
                         "--episodes", "200", "--epsilon-decay", "0.98", "--max-steps", "30", "--seed", "3",
                         "--eval-episodes", "20", "--eval-policies", "learned", "random")
        agent = self.make_agent(args)
        with contextlib.redirect_stdout(io.StringIO()):
            run_game.train_agent(agent, args)
        summaries = run_game.compare_policies(agent, args, verbose=False)
        self.assertEqual(sum(row["guidance_actions"] for row in summaries["learned"]["episodes_detail"]), 0)
        self.assertGreaterEqual(summaries["learned"]["win_rate"], 0.9)
        self.assertEqual(summaries["random"]["win_rate"], 0.0)


class EmpireRunnerTests(unittest.TestCase):
    """--mode empire trains, resumes, evaluates and validates like the siege runner."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def args(self, *extra):
        return run_game.parse_args([
            "--model-path", str(self.root / "model.json"), "--metrics-dir", str(self.root / "metrics"),
            "--episodes", "2", "--eval-episodes", "1", "--log-every", "0", *extra,
        ])

    def empire_args(self, *extra):
        return self.args("--mode", "empire", "--turn-limit", "12", "--width", "10", "--height", "9", *extra)

    def test_new_empire_models_use_the_empire_learner_and_its_settings(self):
        args = self.empire_args("--difficulty", "hard", "--map-type", "islands")
        agent = run_game.build_agent(args, random.Random(1))
        self.assertIsInstance(agent, EmpireQAgent)
        self.assertEqual((args.mode, args.agent, args.max_steps, args.demonstrations), ("empire", "empire", 12, 0.3))
        self.assertEqual((args.config.rival_bonus, args.config.map_type, args.config.difficulty), (1.25, "islands",
                                                                                                    "hard"))
        self.assertEqual(agent.discount_factor, 0.97)
        self.assertEqual(agent.observation_schema, empire.OBSERVATION_SCHEMA)

    def test_flags_from_the_other_mode_and_mismatched_learners_are_rejected(self):
        for extra, message in [(("--unit-count", "2"), "unit-count"), (("--economy",), "economy"),
                               (("--agent", "tactical"), "empire learner"), (("--shaping", "1"), "shaping"),
                               (("--state-precision", "5"), "joint")]:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                run_game.build_agent(self.empire_args("--fresh", *extra), random.Random(1))
        for extra in [("--rivals", "0"), ("--speed", "quick"), ("--mode", "siege", "--agent", "empire")]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                run_game.build_agent(self.args("--fresh", *extra), random.Random(1))
        implied = self.args("--fresh", "--agent", "empire")
        run_game.build_agent(implied, random.Random(1))
        self.assertEqual(implied.mode, "empire")

    def test_empire_training_resumes_exactly_and_rejects_scenario_changes(self):
        full_args = self.empire_args("--episodes", "4", "--seed", "5", "--no-save")
        full = run_game.build_agent(full_args, random.Random(9))
        run_game.train_agent(full, full_args)
        split_args = self.empire_args("--seed", "5", "--checkpoint-every", "1")
        split = run_game.build_agent(split_args, random.Random(9))
        run_game.train_agent(split, split_args)
        run_game.compare_policies(split, split_args, verbose=False)
        split.save(split_args.model_path)
        resume_args = self.args("--no-save")
        resumed = run_game.build_agent(resume_args, random.Random(3))
        self.assertEqual((resume_args.mode, resume_args.config, resume_args.max_steps),
                         ("empire", split_args.config, 12))
        run_game.train_agent(resumed, resume_args)
        self.assertEqual(full.rng.getstate(), resumed.rng.getstate())
        self.assertEqual(full.training_state["metrics"], resumed.training_state["metrics"])
        self.assertEqual(set(full.q_table), set(resumed.q_table))
        for key in full.q_table:
            np.testing.assert_array_equal(full.q_table[key], resumed.q_table[key])
        for extra in [("--map-type", "islands"), ("--rivals", "0"), ("--mode", "siege"), ("--speed", "standard")]:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "fresh"):
                run_game.build_agent(self.args(*extra), random.Random(1))

    def test_empire_demonstrations_are_played_by_the_heuristic(self):
        args = self.empire_args("--no-save", "--episodes", "2")
        agent = run_game.build_agent(args, random.Random(1))
        with patch.object(run_game, "is_demonstration", side_effect=lambda _, episode: episode == 0), \
                patch.object(agent, "learn", wraps=agent.learn) as learn:
            run_game.train_agent(agent, args)
        demonstration, own = agent.training_state["metrics"]
        flags = [call.kwargs["demonstration"] for call in learn.call_args_list]
        self.assertEqual(flags, [True] * demonstration["steps"] + [False] * own["steps"])
        self.assertTrue(demonstration["demonstration"])
        self.assertEqual(demonstration["guidance_actions"], demonstration["steps"])
        self.assertFalse(own["demonstration"])
        self.assertEqual(own["city_strength"], 1.0)  # The column records the rival's bonus in empire mode.

    def test_empire_evaluation_reports_score_cities_and_research(self):
        args = self.empire_args("--no-save", "--episodes", "0", "--eval-episodes", "2")
        agent = run_game.build_agent(args, random.Random(1))
        summaries = run_game.compare_policies(agent, args, verbose=False)
        self.assertEqual(set(summaries), {"learned", "heuristic", "random"})
        for summary in summaries.values():
            for name in ("average_score", "average_rival_score", "average_cities", "average_technologies"):
                self.assertIn(name, summary)
            self.assertEqual(summary["episodes"], 2)
        self.assertGreater(summaries["heuristic"]["average_cities"], 0)
        run_game.write_metrics(args.metrics_dir, [], evaluations=summaries, config=args.config)
        payload = json.loads((args.metrics_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["config"]["map_type"], "continents")

    def test_empire_curriculum_ramps_the_rival_bonus(self):
        args = self.empire_args("--no-save", "--difficulty", "curriculum", "--curriculum-episodes", "3")
        run_game.build_agent(args, random.Random(1))
        self.assertEqual([run_game.episode_config(args, episode).rival_bonus for episode in range(4)],
                         [0.5, 0.875, 1.25, 1.25])
        self.assertTrue(run_game.curriculum_complete(args, 2))

    def test_cli_trains_and_evaluates_an_empire(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_game.main(["--mode", "empire", "--turn-limit", "10", "--width", "10", "--height", "9",
                           "--episodes", "1", "--eval-episodes", "1", "--model-path", str(self.root / "cli.json"),
                           "--metrics-dir", str(self.root / "cli"), "--log-every", "1"])
        text = output.getvalue()
        self.assertIn("learned: wins", text)
        self.assertIn("score", text)
        self.assertIsInstance(load_agent(self.root / "cli.json"), EmpireQAgent)


if __name__ == "__main__":
    unittest.main()
