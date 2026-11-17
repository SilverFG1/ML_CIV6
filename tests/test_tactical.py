"""Tactical observations, combat odds, the siege heuristic and the tactical learner."""

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

import game
from game import (COMMANDS, TACTICAL_FEATURES, TACTICAL_FIRST_TACTIC, TACTICAL_HEADER, TACTICAL_UNIT_WIDTH,
                  TACTICS, Game, GameConfig, attack, damage_at_least)
from learning_ai import QLearningAgent, TacticalQAgent, load_agent

T = {name: index for index, name in enumerate(TACTICS)}
FIRST_FEATURE = len(TACTICAL_HEADER)


def blocks(observation, units):
    return np.asarray(observation).reshape(units, TACTICAL_UNIT_WIDTH)


def features(block):
    return dict(zip(TACTICAL_FEATURES, block[FIRST_FEATURE:TACTICAL_FIRST_TACTIC]))


def tactics(block):
    return {command: (TACTICS[int(value)] if value >= 0 else None)
            for command, value in zip(COMMANDS, block[TACTICAL_FIRST_TACTIC:])}


def unit_block(health=1.0, alive=1, unit_features=(0, 1, 3, 0, 4, 0, 0, 0, 0, 0, 1),
               unit_tactics=('ATTACK', 'ADVANCE', 'SIDESTEP', 'RETREAT', 'RETREAT', 'SIDESTEP', 'WAIT', None)):
    """One hand-built per-unit block with the city at full health and no walls."""
    return [alive, health, 1.0, 0.0, *unit_features,
            *(T[name] if name is not None else -1 for name in unit_tactics)]


class TacticalEnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs):
        if 'unit_types' in kwargs:
            kwargs.setdefault('unit_count', len(kwargs['unit_types']))
        env = Game(config=GameConfig(**kwargs), seed=7, observation='tactical')
        env.reset(seed=7)
        return env

    def place(self, env, positions):
        for unit, position in zip(env.units, positions):
            unit.x, unit.y = position

    def test_death_odds_match_simulated_retaliation(self):
        env = self.make_env(unit_count=1, unit_types=('heavy_chariot',), city_heals=False)
        for unit_type, strength, hp, terrain in [('warrior', 20, 40, 'GRASSLAND'), ('warrior', 20, 30, 'FOREST'),
                                                 ('heavy_chariot', 28, 35, 'GRASSLAND')]:
            with self.subTest(unit_type=unit_type, hp=hp, terrain=terrain):
                unit = env.units[0]
                unit.unit_type, unit.strength, unit.hp = unit_type, strength, hp
                self.place(env, [(3, 4)])
                env.map.grid[(3, 4)].set_terrain(terrain)
                rng = random.Random(11)
                deaths = sum(attack(unit, env.city, rng=rng)[1] >= hp for _ in range(20000)) / 20000
                self.assertAlmostEqual(env.melee_death_probability(unit), deaths, delta=.02)
        self.assertEqual(damage_at_least(0, 1.0), 1.0)
        self.assertEqual(damage_at_least(37, 1.0), 0.0)
        self.assertGreater(damage_at_least(36, 1.0), 0.0)

    def test_capture_odds_require_breaking_walls_and_surviving(self):
        env = self.make_env(unit_count=1, city_heals=False)
        self.place(env, [(3, 4)])
        env.city.hp = 1
        self.assertEqual(env.capture_probability(env.units[0]), 1.0)
        env.city.wall_hp = env.city.wall_hp_max = 30  # half wall damage: 61 needed, at most 26 dealt
        self.assertEqual(env.capture_probability(env.units[0]), 0.0)
        env.city.wall_hp = 0
        env.units[0].hp = 1
        self.assertEqual(env.capture_probability(env.units[0]), 0.0)

    def test_tactical_layout_features_and_tactics(self):
        env = self.make_env(unit_types=('warrior', 'archer'), wall_hp=40)
        self.place(env, [(3, 4), (6, 4)])
        observation = env.tactical_observation()
        self.assertEqual(observation.shape, (2 * TACTICAL_UNIT_WIDTH,))
        warrior, archer = blocks(observation, 2)
        self.assertEqual(list(warrior[:FIRST_FEATURE]), [1, 1, 1, 1])
        self.assertEqual(features(warrior), dict(
            unit_type=0, distance=1, health=3, risk=0, city_health=4, walls=2, allies_adjacent=0,
            allies_away=1, ram_ready=0, city_ranged=0, city_heals=1))
        self.assertEqual(tactics(warrior)['E'], 'ATTACK')
        self.assertEqual(tactics(warrior)['SPACE'], 'WAIT')
        self.assertIsNone(tactics(warrior)['SHOOT'])
        self.assertEqual(features(archer)['distance'], 2)
        self.assertEqual(tactics(archer)['SHOOT'], 'SHOOT')
        self.assertEqual(tactics(archer)['W'], 'ADVANCE')
        self.assertEqual(tactics(archer)['E'], 'RETREAT')
        env.units[0].hp = 50
        warrior = blocks(env.tactical_observation(), 2)[0]
        self.assertEqual(tactics(warrior)['SPACE'], 'HEAL')
        self.assertEqual((features(warrior)['health'], warrior[1]), (1, .5))
        self.assertGreater(features(warrior)['risk'], 0)
        env.units[1].death_unit()
        archer = blocks(env.tactical_observation(), 2)[1]
        self.assertTrue(np.all(archer[:TACTICAL_FIRST_TACTIC] == 0))
        self.assertTrue(np.all(archer[TACTICAL_FIRST_TACTIC:] == -1))

    def test_tactical_observation_ignores_absolute_position_and_map_size(self):
        large = self.make_env(unit_count=1)
        small = self.make_env(unit_count=1, width=6, height=6, city_position=(2, 2))
        self.place(large, [(3, 4)])
        self.place(small, [(1, 2)])
        np.testing.assert_array_equal(large.tactical_observation(), small.tactical_observation())
        self.assertEqual(large.observation_schema, small.observation_schema)

    def test_allies_and_ram_features(self):
        env = self.make_env(unit_types=('warrior', 'spearman', 'battering_ram', 'warrior'), wall_hp=20)
        self.place(env, [(3, 4), (5, 4), (4, 3), (0, 0)])
        first = features(blocks(env.tactical_observation(), 4)[0])
        self.assertEqual((first['allies_adjacent'], first['allies_away'], first['ram_ready']), (2, 1, 1))
        self.assertIsNone(tactics(blocks(env.tactical_observation(), 4)[2])['SW'])  # rams cannot attack

    def test_routes_go_around_allies_blocking_the_ring(self):
        env = self.make_env(unit_count=3)
        self.place(env, [(3, 3), (3, 4), (2, 4)])
        blocked = blocks(env.tactical_observation(), 3)[2]
        self.assertEqual(env.step_distances()[(2, 4)], 2)
        self.assertEqual(features(blocked)['distance'], 3)
        self.assertEqual(tactics(blocked)['SE'], 'ADVANCE')  # toward the free ring tile
        self.assertEqual(tactics(blocked)['E'], 'RETREAT')  # into an ally

    def test_step_distances_count_turns_not_forest_cost(self):
        env = self.make_env(unit_count=1)
        env.map.grid[(3, 4)].set_terrain('FOREST')
        env.map.grid[(2, 4)].set_terrain('MOUNTAIN')
        steps = env.step_distances()
        self.assertEqual(steps[(3, 4)], 1)
        self.assertNotIn((2, 4), steps)
        self.assertEqual(steps[(1, 4)], 4)  # around the mountain

    def test_observation_modes_and_schema(self):
        env = Game(config=GameConfig(unit_count=2, unit_types=('warrior', 'archer')), observation='tactical')
        self.assertEqual(env.reset(seed=3).shape, (2 * TACTICAL_UNIT_WIDTH,))
        observation, _, _ = env.step(env.encode_action(['SPACE', 'SPACE']))
        self.assertEqual(observation.shape, (2 * TACTICAL_UNIT_WIDTH,))
        self.assertEqual(env.observation_schema, 'ml_civ6.tactical.v1:warrior,archer')
        self.assertEqual(env.get_observation().shape, (7 + 8 * 2 + 64,))
        with self.assertRaises(ValueError):
            Game(observation='pixels')


class SiegeHeuristicTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = Game(config=GameConfig(**kwargs), seed=7)
        env.reset(seed=7)
        return env

    def choose(self, env, positions):
        for unit, position in zip(env.units, positions):
            unit.x, unit.y = position
        return env.actions[env.guided_action(random.Random(1))]

    def test_waits_for_an_encirclement_before_attacking(self):
        env = self.make_env()
        self.assertEqual(self.choose(env, [(3, 4), (0, 0), (7, 7)])[0], 'SPACE')

    def test_encircled_squad_attacks_together_when_safe(self):
        env = self.make_env()
        self.assertEqual(self.choose(env, [(3, 4), (5, 4), (4, 3)]), ('E', 'W', 'SW'))

    def test_heals_instead_of_risking_death_while_encircled(self):
        env = self.make_env()
        env.units[0].hp = 40
        self.assertEqual(self.choose(env, [(3, 4), (5, 4), (4, 3)]), ('SPACE', 'W', 'SW'))

    def test_takes_a_near_certain_capture(self):
        env = self.make_env()
        env.city.hp = 5
        self.assertEqual(self.choose(env, [(3, 4), (0, 0), (7, 7)])[0], 'E')

    def test_stays_outside_a_ranged_city_until_the_squad_closes_up(self):
        env = self.make_env(ranged_strength=25)
        command = self.choose(env, [(4, 1), (0, 7), (7, 7)])[0]
        dx, dy = game.constants.movement_delta(command, 1)
        self.assertGreaterEqual(env.step_distances()[(4 + dx, 1 + dy)], 3)

    def test_wins_most_hard_sieges(self):
        config = GameConfig(city_strength=38)
        wins = 0
        for seed in range(20):
            env = Game(config=config, seed=seed)
            env.reset(seed=seed)
            rng = random.Random(seed)
            for _ in range(80):
                _, _, done = env.step(env.guided_action(rng))
                if done:
                    break
            wins += env.last_info['outcome'] == 'win'
        self.assertGreaterEqual(wins, 18)


class TacticalAgentTests(unittest.TestCase):
    def make_agent(self, counts=(7,), **kwargs):
        options = dict(rng=random.Random(5), observation_schema='test', learning_rate_decay=1.0)
        options.update(kwargs)
        return TacticalQAgent(list(counts), **options)

    def test_joint_actions_decode_like_the_environment(self):
        env = Game(config=GameConfig(unit_count=3, unit_types=('warrior', 'archer', 'warrior')))
        agent = TacticalQAgent([len(directions) for directions in env.unit_directions])
        self.assertEqual(agent.action_count, env.action_count)
        for action in range(env.action_count):
            self.assertEqual(tuple(COMMANDS[index] for index in agent._commands[action]), env.actions[action])

    def test_greedy_choice_sums_unit_values_over_legal_actions(self):
        agent = self.make_agent(counts=(7, 7), epsilon=0)
        first = unit_block(unit_features=(0, 1, 3, 0, 4, 0, 1, 0, 0, 0, 1))
        second = unit_block(unit_features=(1, 1, 3, 0, 4, 0, 1, 0, 0, 0, 1),  # a mountain lies to the NW
                            unit_tactics=('ATTACK', 'ADVANCE', 'SIDESTEP', 'RETREAT', 'RETREAT', None, 'WAIT', None))
        observation = first + second
        for block, attack_value in ((first, 5), (second, 1)):
            key = agent._tile_keys(np.asarray(block[FIRST_FEATURE:TACTICAL_FIRST_TACTIC], dtype=int))[0]
            agent.q_table[key] = np.zeros(len(TACTICS))
            agent.q_table[key][T['ATTACK']] = attack_value
        attack_both, wait_attack, wait_wait = 0 * 7 + 0, 6 * 7 + 0, 6 * 7 + 6
        self.assertEqual(agent.choose_action(observation, False, valid_actions=[attack_both, wait_attack]), attack_both)
        # The best joint action (value 6) is illegal here, so the best legal one (value 1) wins.
        self.assertEqual(agent.choose_action(observation, False, valid_actions=[wait_attack, wait_wait]), wait_attack)
        self.assertEqual(agent.last_action_source, 'policy')
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            agent.choose_action(observation, False, valid_actions=[0 * 7 + 5])  # (ATTACK, NW)

    def test_update_moves_joint_value_by_learning_rate_times_shaped_error(self):
        agent = self.make_agent(learning_rate=.5, discount_factor=.9, shaping=.5)
        before, after = unit_block(health=.6), unit_block(health=.5)
        agent.learn(before, 0, 1.0, after, False, next_valid_actions=[6])
        # target = 1 + 0.9 * 0.5 * 0.5 - 0.5 * 0.6 + 0.9 * max(0) = 0.925
        values = agent._unit_values(np.asarray(before[FIRST_FEATURE:TACTICAL_FIRST_TACTIC], dtype=int))
        self.assertAlmostEqual(values[T['ATTACK']], .5 * .925)
        self.assertEqual(len(agent.q_table), len(agent._groups))

    def test_terminal_update_drops_bootstrap_and_potential(self):
        agent = self.make_agent(learning_rate=1, discount_factor=.9, shaping=.5)
        before = unit_block(health=.6)
        agent.learn(before, 0, 10.0, unit_block(health=.2), True, next_valid_actions=[])
        values = agent._unit_values(np.asarray(before[FIRST_FEATURE:TACTICAL_FIRST_TACTIC], dtype=int))
        self.assertAlmostEqual(values[T['ATTACK']], 10 - .3)

    def test_a_step_that_changes_nothing_is_valued_as_repeating_forever(self):
        agent = self.make_agent(learning_rate=1, discount_factor=.9, shaping=.5)
        state = unit_block()
        agent.learn(state, 6, -.05, list(state), False, next_valid_actions=[0, 6])
        values = agent._unit_values(np.asarray(state[FIRST_FEATURE:TACTICAL_FIRST_TACTIC], dtype=int))
        # shaped reward -0.05 + 0.9 * 0.5 - 0.5 = -0.1, repeated forever: -1
        self.assertAlmostEqual(values[T['WAIT']], -1.0)

    def test_greedy_loops_are_valued_as_repeating_but_exploratory_loops_bootstrap(self):
        near = unit_block(unit_features=(0, 1, 3, 0, 4, 0, 0, 0, 0, 0, 1))
        far = unit_block(unit_features=(0, 2, 3, 0, 4, 0, 0, 0, 0, 0, 1),
                         unit_tactics=('ADVANCE', 'SIDESTEP', 'SIDESTEP', 'RETREAT', 'RETREAT', 'SIDESTEP', 'WAIT', None))
        retreat, advance = 3, 0
        for greedy in (True, False):
            with self.subTest(greedy=greedy):
                agent = self.make_agent(learning_rate=1, discount_factor=.9, shaping=0, epsilon=0 if greedy else 1)
                agent.choose_action(near, True, valid_actions=[retreat])
                agent.learn(near, retreat, -.1, far, False, next_valid_actions=[advance])
                agent.choose_action(far, True, valid_actions=[advance])
                agent.learn(far, advance, .05, near, False, next_valid_actions=[retreat])
                value = agent._unit_values(np.asarray(far[FIRST_FEATURE:TACTICAL_FIRST_TACTIC], dtype=int))[T['ADVANCE']]
                if greedy:
                    cycle = (-.1 + .9 * .05) / (1 - .9 ** 2)
                    self.assertAlmostEqual(value, .05 + .9 * cycle)
                else:
                    self.assertAlmostEqual(value, .05 + .9 * -.1)

    def test_dead_units_contribute_nothing(self):
        agent = self.make_agent(counts=(7, 7), learning_rate=1, shaping=0)
        dead = unit_block(alive=0, health=0, unit_features=(0,) * 11, unit_tactics=(None,) * 8)
        dead[2:4] = [0, 0]
        agent.learn(unit_block() + dead, 6, 2.0, unit_block(health=.5) + dead, True, next_valid_actions=[])
        self.assertEqual(len(agent.q_table), 5)

    def test_evaluation_reads_without_inserting_and_guides_unseen_states(self):
        agent = self.make_agent()
        state = unit_block()
        self.assertEqual(agent.choose_action(state, False, fallback_action=6, valid_actions=[0, 6]), 6)
        self.assertEqual(agent.last_action_source, 'guided')
        self.assertIn(agent.choose_action(state, False, valid_actions=[0, 6]), [0, 6])
        self.assertEqual(agent.q_table, {})
        agent.learn(state, 0, 1.0, unit_block(health=.5), True, next_valid_actions=[])
        self.assertEqual(agent.choose_action(state, False, fallback_action=6, valid_actions=[0, 6]), 0)
        self.assertEqual(agent.last_action_source, 'policy')

    def test_learning_rate_anneals_to_its_floor(self):
        agent = self.make_agent(learning_rate=.1, learning_rate_decay=.5, learning_rate_min=.02, epsilon=1,
                                epsilon_decay=.5, epsilon_min=.3)
        for _ in range(4):
            agent.finish_episode()
        self.assertEqual((agent.learning_rate, agent.epsilon, agent.episodes_trained), (.02, .3, 4))
        low = self.make_agent(learning_rate=.005, learning_rate_min=.01, learning_rate_decay=.5)
        low.finish_episode()
        self.assertEqual(low.learning_rate, .005)

    def test_invalid_configuration_and_observations_rejected(self):
        for counts in ([], [6], [9], 'abc', [7.5], [True]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                TacticalQAgent(counts)
        for kwargs in ({'shaping': -1}, {'learning_rate_decay': 2}, {'learning_rate_min': float('nan')},
                       {'discount_factor': 1.5}, {'observation_schema': ''}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_agent(**kwargs)
        agent = self.make_agent()
        good = unit_block()
        for bad in (good[:-1], good[:4] + [9] + good[5:], good[:15] + [7] + good[16:],
                    [2] + good[1:], good[:1] + [1.5] + good[2:], good[:4] + [0.5] + good[5:],
                    good[:-1] + [float('nan')]):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'observation'):
                agent.choose_action(bad, False)
        self.assertEqual(agent.q_table, {})

    def test_checkpoint_round_trip_and_dispatch(self):
        agent = self.make_agent(counts=(7, 8), learning_rate=.3)
        archer_tactics = ('ADVANCE', 'SIDESTEP', 'SIDESTEP', 'RETREAT', 'RETREAT', 'SIDESTEP', 'WAIT', 'SHOOT')
        observation = unit_block() + unit_block(unit_features=(4, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1),
                                                unit_tactics=archer_tactics)
        agent.learn(observation, 7, 1.0, observation[:1] + [.5] + observation[2:], False,
                    next_valid_actions=[7, 8])
        agent.finish_episode()
        agent.training_state = {'seed': 3}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / 'model.json'
            agent.save(path)
            payload = json.loads(path.read_text(encoding='utf-8'))
            restored = load_agent(path, rng=random.Random(99))
            self.assertIsInstance(restored, TacticalQAgent)
            joint = QLearningAgent(4, rng=random.Random(1))
            joint.save(Path(folder) / 'joint.json')
            self.assertIsInstance(load_agent(Path(folder) / 'joint.json'), QLearningAgent)
            bad_payloads = [
                {**payload, 'tile_groups': payload['tile_groups'][:-1]},
                {**payload, 'tactics': payload['tactics'][::-1]},
                {**payload, 'action_count': 99},
                {**payload, 'unit_command_counts': [7, 9]},
                {**payload, 'q_table': payload['q_table'] * 2},
                {**payload, 'q_table': [{'state': [0] + [99] * 11, 'values': [0] * len(TACTICS)}]},
                {**payload, 'q_table': [{'state': [1] + [0] * 11, 'values': [0] * len(TACTICS)}]},
                {**payload, 'q_table': [{'state': payload['q_table'][0]['state'], 'values': [0, 1]}]},
                {**payload, 'learning_rate_decay': 3},
                {**payload, 'agent': 'mystery'},
            ]
            for index, bad in enumerate(bad_payloads):
                path.write_text(json.dumps(bad), encoding='utf-8')
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'Invalid checkpoint'):
                    load_agent(path)
        for name in ('learning_rate', 'epsilon', 'shaping', 'learning_rate_decay', 'learning_rate_min',
                     'episodes_trained', 'training_state', 'unit_command_counts', 'observation_schema'):
            self.assertEqual(getattr(restored, name), getattr(agent, name))
        self.assertEqual(restored.rng.getstate(), agent.rng.getstate())
        self.assertEqual(set(restored.q_table), set(agent.q_table))
        for key, row in agent.q_table.items():
            np.testing.assert_array_equal(row, restored.q_table[key])


if __name__ == '__main__':
    unittest.main()
