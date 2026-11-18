"""The empire learner: decision parsing, exploration, the mean-value update and checkpoints."""

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from empire import DECISION_KINDS, OPTION_INDEX, Decision, EmpireConfig, EmpireGame
from learning_ai import EMPIRE_TILE_GROUPS, EmpireQAgent, TacticalQAgent, load_agent

LAND = OPTION_INDEX['land']
CITY = OPTION_INDEX['city']


def land(features=(3, 0, 3, 0, 0, 0, 3, 1, 0, 1, 0), valid=('fortify', 'garrison', 'siege'), entity=1):
    return Decision('land', entity, features, tuple(LAND[name] for name in valid))


def city(features=(3, 1, 1, 1, 2, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1), valid=('warrior', 'builder', 'settler'),
         entity=2):
    return Decision('city', entity, features, tuple(CITY[name] for name in valid))


class EmpireAgentTests(unittest.TestCase):
    def agent(self, **kwargs):
        options = dict(epsilon=0.0, rng=random.Random(4))
        options.update(kwargs)
        return EmpireQAgent(**options)

    def test_tile_groups_cover_known_features(self):
        for kind, groups in EMPIRE_TILE_GROUPS.items():
            self.assertEqual(groups[0], DECISION_KINDS[kind].features)
            for group in groups:
                self.assertTrue(set(group) <= set(DECISION_KINDS[kind].features))

    def test_greedy_choices_respect_legal_options(self):
        agent = self.agent()
        observation = (land(), city())
        key = agent._tile_keys('land', observation[0].features)[1]
        agent.q_table[key] = np.zeros(1 + len(DECISION_KINDS['land'].options))
        agent.q_table[key][1 + LAND['hunt']] = 9.0  # Best, but not legal here.
        agent.q_table[key][1 + LAND['siege']] = 1.0
        action = agent.choose_action(observation, explore=False)
        self.assertEqual(action[0], LAND['siege'])
        self.assertIn(action[1], observation[1].valid)
        self.assertEqual(agent.last_action_source, 'policy')
        narrowed = agent.choose_action(observation, explore=False,
                                       valid_actions=[(LAND['fortify'],), observation[1].valid])
        self.assertEqual(narrowed[0], LAND['fortify'])

    def test_exploration_is_per_decision_and_guidance_fills_unseen_situations(self):
        observation = tuple(land(entity=index) for index in range(40))
        fallback = tuple(LAND['garrison'] for _ in observation)
        explorer = self.agent(epsilon=0.5)
        action = explorer.choose_action(observation, explore=True, valid_actions=None)
        self.assertEqual(explorer.last_action_source, 'random')
        self.assertGreater(len(set(action)), 1)
        guided = self.agent(epsilon=1.0)
        action = guided.choose_action(observation, fallback_action=fallback, guided_exploration_rate=1.0)
        self.assertEqual((action, guided.last_action_source), (fallback, 'guided'))
        unseen = self.agent()
        self.assertEqual(unseen.choose_action(observation, explore=False, fallback_action=fallback), fallback)
        self.assertEqual(unseen.last_action_source, 'guided')

    def test_new_rows_start_at_the_target_with_untried_options_below_it(self):
        agent = self.agent(learning_rate=0.5, untried_penalty=0.5)
        observation = (land(), city())
        action = (LAND['siege'], CITY['settler'])
        agent.learn(observation, action, 2.0, observation, done=True, next_valid_actions=[])
        groups = len(EMPIRE_TILE_GROUPS['land'])
        for key in agent._tile_keys('land', land().features):
            self.assertAlmostEqual(agent.q_table[key][0], 2.0 / groups)
        values = agent._values('land', land().features)
        # The chosen option's advantage moves half way from -0.5 toward 0; the others stay untried.
        self.assertAlmostEqual(values[LAND['siege']], 2.0 - .25)
        self.assertAlmostEqual(values[LAND['fortify']], 2.0 - .5)
        self.assertEqual(agent.choose_action(observation, explore=False), action)

    def test_value_follows_every_visit_and_advantage_only_the_chosen_option(self):
        agent = self.agent(learning_rate=0.5, discount_factor=0.9, untried_penalty=0.0)
        observation = (land(), land(entity=5))
        agent.learn(observation, (LAND['siege'], LAND['siege']), 1.0, observation, done=True)
        value, advantages = agent._parts('land', land().features)
        self.assertAlmostEqual(value, 1.0)
        np.testing.assert_allclose(advantages, 0.0)
        # Both decisions share every table row, so each update applies twice.
        agent.learn(observation, (LAND['fortify'], LAND['fortify']), -1.0, observation, done=False)
        target = -1.0 + 0.9 * 1.0
        expected_value, expected_advantage = 1.0, 0.0
        for _ in range(2):
            value_step = .5 * (target - expected_value)
            expected_advantage += .5 * (target - expected_value - expected_advantage)
            expected_value += value_step
        value, advantages = agent._parts('land', land().features)
        self.assertAlmostEqual(value, expected_value)
        self.assertAlmostEqual(advantages[LAND['fortify']], expected_advantage)
        self.assertAlmostEqual(advantages[LAND['siege']], 0.0)
        self.assertAlmostEqual(agent.value(observation), expected_value)  # Siege is now the best option.

    def test_an_entity_does_not_bootstrap_from_its_own_next_decision(self):
        agent = self.agent(learning_rate=0.5, discount_factor=0.9, untried_penalty=0.0)
        holding = land(features=(3, 0, 3, 0, 0, 0, 3, 1, 0, 1, 0), entity=1)
        other = land(features=(3, 1, 3, 0, 0, 0, 3, 1, 0, 1, 0), entity=2)
        agent.learn((holding,), (LAND['fortify'],), 8.0, (holding,), done=True)  # The unit's own value: 8.
        agent.learn((other,), (LAND['fortify'],), 2.0, (other,), done=True)
        own_before = agent._parts('land', holding.features)[0]
        # Next turn both decide again; the holding unit's target uses only the other unit's value.
        agent.learn((holding,), (LAND['fortify'],), 0.0, (holding, other), done=False)
        other_value = max(agent._values('land', other.features)[option] for option in other.valid)
        self.assertAlmostEqual(agent._parts('land', holding.features)[0],
                               own_before + .5 * (0.9 * other_value - own_before))
        # Alone, a decision has nothing else to bootstrap from and uses its own successor.
        alone = self.agent(learning_rate=0.5, discount_factor=0.9, untried_penalty=0.0)
        alone.learn((holding,), (LAND['fortify'],), 8.0, (holding,), done=True)
        alone.learn((holding,), (LAND['fortify'],), 0.0, (holding,), done=False)
        self.assertAlmostEqual(alone._parts('land', holding.features)[0], 8.0 + .5 * (0.9 * 8.0 - 8.0))

    def test_demonstrations_keep_other_options_a_margin_below_the_demonstrated_one(self):
        agent = self.agent(learning_rate=0.5, untried_penalty=0.0, demonstration_margin=1.0)
        observation, garrison = (land(),), (LAND['garrison'],)
        agent.learn(observation, garrison, 1.0, (), done=True)
        before = agent._parts('land', land().features)[1]
        self.assertAlmostEqual(before[LAND['fortify']], before[LAND['garrison']])
        agent.learn(observation, garrison, 1.0, (), done=True, demonstration=True)
        advantages = agent._parts('land', land().features)[1]
        # Legal rivals within the margin move half way down to it; illegal options are untouched.
        for option in (LAND['fortify'], LAND['siege']):
            self.assertAlmostEqual(advantages[option], advantages[LAND['garrison']] - .5)
        self.assertAlmostEqual(advantages[LAND['hunt']], before[LAND['hunt']])
        for _ in range(20):
            agent.learn(observation, garrison, 1.0, (), done=True, demonstration=True)
        values = agent._values('land', land().features)
        self.assertLess(values[LAND['siege']], values[LAND['garrison']] - .99)
        self.assertEqual(agent.choose_action(observation, explore=False), garrison)
        with self.assertRaises(ValueError):
            agent.learn(observation, garrison, 1.0, (), done=True, demonstration='yes')

    def test_terminal_updates_do_not_bootstrap(self):
        observation, action = (land(),), (LAND['fortify'],)
        results = []
        for following in ((), (land(entity=9),)):
            agent = self.agent(learning_rate=0.5)
            agent.learn(observation, action, 3.0, observation, done=True)
            key = agent._tile_keys('land', land(entity=9).features)[0]
            agent.q_table[key][0] += 100.0  # A large next-state value must not leak into a terminal target.
            agent.learn(observation, action, 1.0, following, done=True)
            results.append(agent._values('land', land().features)[LAND['fortify']])
        self.assertAlmostEqual(results[0], results[1])
        agent = self.agent(learning_rate=0.5)
        agent.learn(observation, action, 1.0, (), done=False)  # A turn with no further decisions.
        self.assertAlmostEqual(agent._parts('land', land().features)[0], 1.0)

    def test_invalid_decisions_and_actions_are_rejected(self):
        agent = self.agent()
        bad = [
            (Decision('trade', 1, (0,), (0,)),),
            (Decision('land', 1, (9,) * 11, (0,)),),
            (Decision('land', 1, (3,) * 10, (0,)),),
            (Decision('land', 1, land().features, ()),),
            (Decision('land', 1, land().features, (99,)),),
            'abc',
        ]
        for observation in bad:
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                agent.choose_action(observation, explore=False)
        with self.assertRaises(ValueError):
            agent.learn((land(),), (LAND['hunt'],), 0.0, (land(),), False)  # Not a legal option.
        with self.assertRaises(ValueError):
            agent.learn((land(),), (), 0.0, (land(),), False)
        with self.assertRaises(ValueError):
            agent.choose_action((land(),), explore=False, fallback_action=(0, 0))

    def test_epsilon_and_learning_rate_anneal(self):
        agent = self.agent(epsilon=1.0, epsilon_decay=.5, epsilon_min=.3, learning_rate=.1,
                           learning_rate_decay=.5, learning_rate_min=.04)
        agent.finish_episode()
        self.assertEqual((agent.epsilon, agent.learning_rate, agent.episodes_trained), (.5, .05, 1))
        agent.finish_episode(anneal=False)
        self.assertEqual((agent.epsilon, agent.learning_rate), (.3, .05))
        agent.finish_episode()
        self.assertEqual(agent.learning_rate, .04)

    def test_checkpoints_round_trip_and_reject_other_encodings(self):
        agent = self.agent(epsilon=.4)
        env = EmpireGame(EmpireConfig(turn_limit=15), seed=2)
        observation = env.reset(seed=2)
        done = False
        while not done:
            action = agent.choose_action(observation, explore=True, valid_actions=env.valid_actions())
            following, reward, done = env.step(action)
            agent.learn(observation, action, reward, following, done, next_valid_actions=env.valid_actions())
            observation = following
        agent.finish_episode()
        agent.training_state = {'note': [1, 2]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empire.json'
            agent.save(path)
            loaded = load_agent(path)
            self.assertIsInstance(loaded, EmpireQAgent)
            self.assertEqual(set(loaded.q_table), set(agent.q_table))
            for key in agent.q_table:
                np.testing.assert_array_equal(loaded.q_table[key], agent.q_table[key])
            self.assertEqual((loaded.epsilon, loaded.rng.getstate(), loaded.training_state),
                             (agent.epsilon, agent.rng.getstate(), agent.training_state))
            payload = json.loads(path.read_text())
            for change in ({'encoding': {}}, {'format_version': 9}, {'agent': 'tactical'},
                           {'q_table': [{'state': ['land', 0] + [0] * 11, 'values': [0.0] * 6}] * 2},
                           {'q_table': [{'state': ['land', 1] + [0] * 11, 'values': [0.0] * 6}]},
                           {'q_table': [{'state': ['land', 0] + [0] * 11, 'values': [0.0] * 5}]},
                           {'learning_rate': 2}):
                broken = {**payload, **change}
                path.write_text(json.dumps(broken))
                with self.subTest(change=list(change)), self.assertRaises(ValueError):
                    load_agent(path)
            with self.assertRaises(ValueError):
                TacticalQAgent.load(Path(directory) / 'empire.json')


if __name__ == '__main__':
    unittest.main()
