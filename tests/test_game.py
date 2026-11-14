"""Regression tests for combat outcomes, terrain, masks and independent games."""

import random
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import constants
from class_hex import HexMap
from game import C_Unit, Game, GameConfig, attack, growth_threshold, hex_distance


class GameTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = Game(config=GameConfig(**kwargs), seed=7)
        env.reset(seed=7)
        self.addCleanup(env.close)
        return env

    def place(self, env, positions):
        for unit, pos in zip(env.units, positions):
            unit.x, unit.y = pos

    def wait(self, env):
        return env.encode_action(['SPACE'] * len(env.units))

    def test_headless_does_not_import_pygame(self):
        script = "import game, sys; e=game.Game(); e.reset(seed=1); e.step(e.action_count-1); assert 'pygame' not in sys.modules"
        result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_units_dead_is_terminal_loss(self):
        env = self.make_env()
        for unit in env.units:
            unit.death_unit()
        _, reward, done = env.step(self.wait(env))
        self.assertTrue(done)
        self.assertEqual(env.last_info['outcome'], 'loss')
        self.assertAlmostEqual(reward, -10.05)
        _, next_reward, done = env.step(self.wait(env))
        self.assertEqual(next_reward, 0)
        self.assertTrue(done)

    def test_captured_city_never_heals_or_attacks(self):
        env = self.make_env(ranged_strength=30)
        self.place(env, [(3, 4), (0, 0), (1, 0)])
        env.city.hp = 1
        with patch('game.attack', return_value=(5, 0)):
            _, reward, done = env.step(env.encode_action(['E', 'SPACE', 'SPACE']))
        self.assertTrue(done)
        self.assertEqual(env.last_info['outcome'], 'win')
        self.assertEqual(env.city.hp, 0)
        self.assertFalse(env.city.alive)
        self.assertAlmostEqual(reward, 9.98)
        self.assertEqual(env.units[0].hp, 100)
        self.assertNotIn('city_healed', [e['type'] for e in env.events])
        self.assertEqual(env.step(self.wait(env))[1], 0)

    def test_multiple_attacks_each_record_damage(self):
        env = self.make_env(city_heals=False)
        self.place(env, [(3, 4), (4, 3), (4, 5)])
        with patch('game.attack', return_value=(10, 0)):
            _, reward, _ = env.step(env.encode_action(['E', 'SW', 'NW']))
        events = [e for e in env.events if e['type'] == 'city_damaged']
        self.assertEqual(len(events), 3)
        self.assertEqual(env.city.hp, 70)
        self.assertAlmostEqual(reward, .85)
        self.assertEqual(env.get_rewards(), reward)
        self.assertEqual(env.get_rewards(), reward)

    def test_death_and_damage_survive_later_city_healing(self):
        env = self.make_env()
        self.place(env, [(3, 4), (0, 0), (1, 0)])
        env.units[0].hp = 1
        with patch('game.attack', return_value=(2, 2)):
            _, reward, done = env.step(env.encode_action(['E', 'SPACE', 'SPACE']))
        kinds = [e['type'] for e in env.events]
        self.assertIn('unit_died', kinds)
        self.assertIn('city_damaged', kinds)
        self.assertIn('city_healed', kinds)
        self.assertFalse(done)
        self.assertLess(reward, -5)
        self.assertEqual(env.units[0].status, 'dead')

    def test_no_repeated_death_or_distance_penalty(self):
        env = self.make_env(city_heals=False)
        self.place(env, [(0, 0), (1, 0), (2, 0)])
        env.units[0].death_unit()
        _, reward, _ = env.step(self.wait(env))
        self.assertAlmostEqual(reward, -.05)
        self.assertNotIn('unit_died', [e['type'] for e in env.events])

    def test_healing_caps_at_maximum_and_full_health_has_no_event(self):
        env = self.make_env(unit_count=1)
        env.units[0].hp = 96
        env.step(self.wait(env))
        self.assertEqual(env.units[0].hp, 100)
        self.assertEqual(next(e['amount'] for e in env.events if e['type'] == 'unit_healed'), 4)
        env.step(self.wait(env))
        self.assertNotIn('unit_healed', [e['type'] for e in env.events])

    def test_hex_neighbors_match_movements_for_both_parities(self):
        grid = HexMap(8, 8).grid
        for position in [(3, 2), (3, 3)]:
            neighbors = grid[position].get_neighbors(grid)
            self.assertEqual(len(neighbors), 6)
            for cell in neighbors:
                self.assertEqual(hex_distance(position, cell.index), 1)
                self.assertEqual(hex_distance(cell.index, position), 1)
                self.assertIn(grid[position], cell.get_neighbors(grid))

    def test_mask_excludes_edges_mountains_and_dead_unit_moves(self):
        env = self.make_env(unit_count=2)
        self.place(env, [(0, 0), (7, 7)])
        env.units[1].death_unit()
        env.map.grid[(1, 0)].set_terrain('MOUNTAIN')
        actions = [env.actions[a] for a in env.valid_actions()]
        self.assertEqual(set(a[1] for a in actions), {'SPACE'})
        self.assertNotIn(('W', 'SPACE'), actions)
        self.assertNotIn(('E', 'SPACE'), actions)
        self.assertIn(('SPACE', 'SPACE'), actions)
        self.assertEqual(int(env.action_mask().sum()), len(actions))

    def test_joint_mask_accounts_for_vacated_and_contested_tiles(self):
        env = self.make_env(unit_count=2)
        self.place(env, [(1, 0), (0, 0)])
        actions = [env.actions[a] for a in env.valid_actions()]
        self.assertIn(('E', 'E'), actions)
        self.assertNotIn(('SPACE', 'E'), actions)
        self.assertNotIn(('W', 'SPACE'), actions)
        env.step(env.encode_action(['E', 'E']))
        self.assertEqual([u.position for u in env.units], [(2, 0), (1, 0)])

    def test_walls_absorb_damage_then_overflow_to_city(self):
        env = self.make_env(wall_hp=15, city_heals=False)
        env.events = []
        env.city.take_damage(10)
        self.assertEqual((env.city.wall_hp, env.city.hp), (5, 100))
        env.city.take_damage(10)
        self.assertEqual((env.city.wall_hp, env.city.hp), (0, 95))
        self.assertEqual(sum(e.get('amount', 0) for e in env.events if e['type'] == 'wall_damaged'), 15)

    def test_capture_requires_surviving_attacker(self):
        env = self.make_env(unit_count=1, city_heals=False)
        self.place(env, [(3, 4)])
        env.units[0].hp = 1
        env.city.hp = 1
        with patch('game.attack', return_value=(10, 10)):
            _, _, done = env.step(env.encode_action(['E']))
        self.assertTrue(done)
        self.assertEqual(env.last_info['outcome'], 'loss')
        self.assertEqual(env.city.hp, 1)

    def test_ranged_range_uses_hex_distance_and_no_retaliation(self):
        env = self.make_env(unit_count=2, ranged_strength=20, city_heals=False)
        self.place(env, [(6, 4), (0, 0)])
        with patch('game.attack', return_value=(10, 0)):
            env.step(self.wait(env))
        self.assertEqual(env.units[0].hp, 90)
        self.assertEqual(env.units[1].hp, 100)
        self.assertEqual(env.city.hp, 100)

    def test_forest_provides_defense(self):
        env = self.make_env(unit_count=1, city_heals=False)
        self.place(env, [(3, 4)])
        _, grass_damage = attack(env.units[0], env.city, rng=random.Random(9))
        env.map.grid[(3, 4)].set_terrain('FOREST')
        _, forest_damage = attack(env.units[0], env.city, rng=random.Random(9))
        self.assertLess(forest_damage, grass_damage)

    def test_observations_distinguish_parity_alive_walls_and_terrain(self):
        env = self.make_env(wall_hp=100)
        initial = env.get_observation()
        self.assertEqual(initial.shape, (7 + 8 * 3 + 64,))
        env.units[0].alive = False
        self.assertFalse(np.array_equal(initial, env.get_observation()))
        env.units[0].alive = True
        env.city.wall_hp -= 1
        self.assertFalse(np.array_equal(initial, env.get_observation()))
        env.city.wall_hp += 1
        env.map.grid[(0, 0)].set_terrain('FOREST')
        self.assertFalse(np.array_equal(initial, env.get_observation()))

    def test_isolated_instances_and_reproducible_traces(self):
        first = self.make_env(forest_density=.2, mountain_density=.1)
        second = self.make_env(forest_density=.2, mountain_density=.1)
        unrelated = self.make_env(unit_count=1)
        rng1, rng2 = random.Random(40), random.Random(40)
        for _ in range(20):
            np.testing.assert_array_equal(first.get_observation(), second.get_observation())
            action1 = first.guided_action(rng1)
            action2 = second.guided_action(rng2)
            self.assertEqual(action1, action2)
            unrelated.reset()
            unrelated.step(unrelated.guided_action())
            next1, reward1, done1 = first.step(action1)
            next2, reward2, done2 = second.step(action2)
            np.testing.assert_array_equal(next1, next2)
            self.assertEqual((reward1, done1, first.events), (reward2, done2, second.events))
            if done1:
                break

    def test_generated_maps_have_reachable_distinct_starts(self):
        for count in range(1, 5):
            env = self.make_env(width=6, height=5, unit_count=count,
                                mountain_density=.3, forest_density=.3)
            self.assertEqual(env.action_count, 7 ** count)
            for seed in range(10):
                env.reset(seed=seed)
                starts = [u.position for u in env.units]
                self.assertEqual(len(set(starts)), count)
                self.assertNotIn(env.city.position, starts)
                self.assertTrue(set(starts) <= env.map.reachable(env.city.position))
                self.assertIn(env.guided_action(), env.valid_actions())

    def test_invalid_config_and_action_rejected(self):
        for kwargs in [{'width': 2}, {'unit_count': 0}, {'city_position': (99, 0)},
                       {'wall_hp': -1}, {'forest_density': .9},
                       {'unit_strengths': (20, 30)}, {'city_strength': float('nan')},
                       {'unit_types': ('archer',)}, {'unit_count': 1, 'unit_types': ('cavalry',)},
                       {'unit_count': 2, 'unit_types': ('battering_ram', 'battering_ram')}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GameConfig(**kwargs)
        env = self.make_env()
        for action in [-1, env.action_count, .5, True]:
            with self.assertRaises(ValueError):
                env.step(action)

    def test_archer_has_masked_ranged_action_and_no_melee_retaliation(self):
        env = self.make_env(unit_count=2, unit_types=('warrior', 'archer'), city_heals=False)
        self.assertEqual(env.action_count, 56)
        self.place(env, [(0, 0), (6, 4)])
        action = env.encode_action(['SPACE', 'SHOOT'])
        self.assertIn(action, env.valid_actions())
        with patch('game.attack', return_value=(30, 0)):
            _, _, done = env.step(action)
        self.assertFalse(done)
        self.assertEqual(env.city.hp, 70)
        self.assertEqual(env.units[1].hp, 100)
        env.city.hp = 1
        with patch('game.attack', return_value=(30, 0)):
            env.step(action)
        self.assertEqual(env.city.hp, 1)
        self.assertTrue(env.city.alive)
        env.units[1].x, env.units[1].y = 7, 7
        self.assertNotIn(action, env.valid_actions())

    def test_archer_can_capture_with_a_melee_action(self):
        env = self.make_env(unit_count=1, unit_types=('archer',), city_heals=False)
        self.place(env, [(3, 4)])
        env.city.hp = 1
        with patch('game.attack', return_value=(5, 0)):
            _, _, done = env.step(env.encode_action(['E']))
        self.assertTrue(done)
        self.assertEqual(env.last_info['outcome'], 'win')

    def test_unit_types_use_their_default_strengths(self):
        env = self.make_env(unit_count=4, unit_types=('warrior', 'spearman', 'slinger', 'archer'))
        profiles = [(u.unit_type, u.strength, u.strength_ranged, u.attack_range, u.ranged) for u in env.units]
        self.assertEqual(profiles, [('warrior', 20, 0, 1, False), ('spearman', 25, 0, 1, False),
                                    ('slinger', 5, 15, 1, True), ('archer', 12, 20, 2, True)])
        self.assertEqual(env.action_count, 7 * 7 * 8 * 8)
        self.assertEqual([len(d) for d in env.unit_directions], [7, 7, 8, 8])
        override = self.make_env(unit_count=2, unit_types=('spearman', 'slinger'), unit_strengths=(30, 9))
        self.assertEqual([(u.strength, u.strength_ranged) for u in override.units], [(30, 0), (9, 15)])
        with self.assertRaises(ValueError):
            C_Unit(0, 0, unit_type='cavalry')

    def test_existing_unit_types_keep_their_action_layout(self):
        env = self.make_env(unit_count=2, unit_types=('warrior', 'archer'))
        self.assertEqual(env.observation_schema, 'ml_civ6.v2:8x8:warrior,archer')
        self.assertEqual(env.actions[-1], ('SPACE', 'SHOOT'))

    def test_spearman_is_melee_only_and_hits_harder_than_a_warrior(self):
        env = self.make_env(unit_count=2, unit_types=('warrior', 'spearman'), city_heals=False)
        self.assertEqual(env.action_count, 49)
        self.place(env, [(3, 4), (5, 4)])
        warrior_out, warrior_taken = attack(env.units[0], env.city, rng=random.Random(3))
        spear_out, spear_taken = attack(env.units[1], env.city, rng=random.Random(3))
        self.assertGreater(spear_out, warrior_out)
        self.assertLess(spear_taken, warrior_taken)
        env.city.hp = 1
        with patch('game.attack', return_value=(5, 0)):
            _, _, done = env.step(env.encode_action(['SPACE', 'W']))
        self.assertTrue(done)
        self.assertEqual(env.last_info['outcome'], 'win')

    def test_slinger_shoots_only_adjacent_city_without_retaliation(self):
        env = self.make_env(unit_count=1, unit_types=('slinger',), city_heals=False)
        self.assertEqual(env.action_count, 8)
        shoot = env.encode_action(['SHOOT'])
        self.place(env, [(6, 4)])
        self.assertEqual(hex_distance(env.units[0].position, env.city.position), 2)
        self.assertNotIn(shoot, env.valid_actions())
        self.place(env, [(5, 4)])
        self.assertIn(shoot, env.valid_actions())
        with patch('game.attack', return_value=(30, 0)):
            _, _, done = env.step(shoot)
        self.assertFalse(done)
        self.assertEqual(env.city.hp, 70)
        self.assertEqual(env.units[0].hp, 100)
        self.assertIn('ranged_attack', [e['type'] for e in env.events])
        env.city.hp = 1
        with patch('game.attack', return_value=(30, 0)):
            env.step(shoot)
        self.assertEqual(env.city.hp, 1)
        self.assertTrue(env.city.alive)
        with patch('game.attack', return_value=(5, 0)):
            _, _, done = env.step(env.encode_action(['W']))
        self.assertTrue(done)
        self.assertEqual(env.last_info['outcome'], 'win')

    def test_slinger_ranged_strength_sets_damage_and_melee_is_risky(self):
        env = self.make_env(unit_count=2, unit_types=('slinger', 'archer'), city_heals=False)
        self.place(env, [(5, 4), (6, 4)])
        slinger_shot, _ = attack(env.units[0], env.city, ranged=True, rng=random.Random(5))
        archer_shot, _ = attack(env.units[1], env.city, ranged=True, rng=random.Random(5))
        self.assertLess(slinger_shot, archer_shot)
        _, slinger_melee_taken = attack(env.units[0], env.city, rng=random.Random(5))
        _, archer_melee_taken = attack(env.units[1], env.city, rng=random.Random(5))
        self.assertGreater(slinger_melee_taken, archer_melee_taken)

    def test_heuristic_slinger_shoots_until_city_can_be_captured(self):
        env = self.make_env(unit_count=1, unit_types=('slinger',), city_heals=False)
        self.place(env, [(5, 4)])
        self.assertEqual(env.actions[env.guided_action(random.Random(1))], ('SHOOT',))
        env.city.hp = 1
        self.assertEqual(env.actions[env.guided_action(random.Random(1))], ('W',))

    def test_heavy_chariot_and_battering_ram_profiles(self):
        env = self.make_env(unit_count=2, unit_types=('heavy_chariot', 'battering_ram'))
        profiles = [(u.strength, u.strength_ranged, u.ranged, u.support) for u in env.units]
        self.assertEqual(profiles, [(28, 0, False, False), (0, 0, False, True)])
        self.assertEqual(env.action_count, 49)
        spearman = self.make_env(unit_count=1, unit_types=('spearman',))
        self.place(env, [(3, 4), (0, 0)])
        self.place(spearman, [(3, 4)])
        chariot_out, chariot_taken = attack(env.units[0], env.city, rng=random.Random(3))
        spear_out, spear_taken = attack(spearman.units[0], spearman.city, rng=random.Random(3))
        self.assertGreater(chariot_out, spear_out)
        self.assertLess(chariot_taken, spear_taken)

    def test_battering_ram_cannot_attack_or_capture(self):
        env = self.make_env(unit_count=2, unit_types=('warrior', 'battering_ram'), city_heals=False)
        self.place(env, [(0, 0), (5, 4)])
        actions = [env.actions[a] for a in env.valid_actions()]
        self.assertNotIn('W', {a[1] for a in actions})
        self.assertIn('SPACE', {a[1] for a in actions})
        env.city.hp = 1
        with patch('game.attack', return_value=(50, 0)) as mocked:
            _, _, done = env.step(env.encode_action(['SPACE', 'W']))
        mocked.assert_not_called()
        self.assertFalse(done)
        self.assertEqual(env.city.hp, 1)
        self.assertEqual(env.units[1].position, (5, 4))
        self.assertIn('invalid_move', [e['type'] for e in env.events])

    def test_squad_with_only_a_ram_left_has_lost(self):
        env = self.make_env(unit_count=2, unit_types=('warrior', 'battering_ram'))
        env.units[0].death_unit()
        _, _, done = env.step(self.wait(env))
        self.assertTrue(done)
        self.assertEqual(env.last_info['outcome'], 'loss')

    def melee_wall_damage(self, unit_types, positions, actions, wall_hp=30):
        env = self.make_env(unit_count=len(unit_types), unit_types=unit_types,
                            wall_hp=wall_hp, city_heals=False)
        self.place(env, positions)
        with patch('game.attack', return_value=(20, 0)):
            env.step(env.encode_action(actions))
        return wall_hp - env.city.wall_hp, env.city.hp

    def test_melee_does_half_damage_to_walls_without_an_adjacent_ram(self):
        self.assertEqual(self.melee_wall_damage(('warrior',), [(3, 4)], ['E']), (10, 100))
        ram_too_far = self.melee_wall_damage(('warrior', 'battering_ram'), [(3, 4), (6, 4)], ['E', 'SPACE'])
        self.assertEqual(ram_too_far, (10, 100))
        # Walls that break mid-attack pass the unspent damage to the city at full rate.
        self.assertEqual(self.melee_wall_damage(('warrior',), [(3, 4)], ['E'], wall_hp=4), (4, 88))

    def test_adjacent_ram_gives_melee_and_anti_cavalry_full_wall_damage(self):
        for kind in ('warrior', 'spearman'):
            with self.subTest(kind=kind):
                result = self.melee_wall_damage((kind, 'battering_ram'), [(3, 4), (5, 4)], ['E', 'SPACE'])
                self.assertEqual(result, (20, 100))
        # The ram can move into place earlier in the same turn.
        moved_in = self.melee_wall_damage(('battering_ram', 'warrior'), [(6, 4), (3, 4)], ['W', 'E'])
        self.assertEqual(moved_in, (20, 100))

    def test_ram_does_not_help_heavy_chariots(self):
        result = self.melee_wall_damage(('heavy_chariot', 'battering_ram'), [(3, 4), (5, 4)], ['E', 'SPACE'])
        self.assertEqual(result, (10, 100))

    def test_ranged_attacks_still_do_full_wall_damage(self):
        env = self.make_env(unit_count=1, unit_types=('archer',), wall_hp=30, city_heals=False)
        self.place(env, [(6, 4)])
        with patch('game.attack', return_value=(20, 0)):
            env.step(env.encode_action(['SHOOT']))
        self.assertEqual(env.city.wall_hp, 10)
        with self.assertRaises(ValueError):
            env.city.take_damage(5, wall_multiplier=0)

    def test_heuristic_holds_ram_next_to_city(self):
        env = self.make_env(unit_count=2, unit_types=('warrior', 'battering_ram'), city_heals=False)
        self.place(env, [(0, 0), (5, 4)])
        self.assertEqual(env.actions[env.guided_action(random.Random(1))][1], 'SPACE')


    # City population, worked tiles and growth.
    def place_far(self, env):
        self.place(env, [(0, 0), (2, 0), (0, 2), (2, 2)])

    def test_city_growth_is_off_by_default(self):
        env = self.make_env()
        self.assertEqual(env.city.population, 0)
        self.assertIsNone(env.city_report())
        self.assertEqual(env.worked_tiles(), [])
        self.assertEqual(env.observation_schema, 'ml_civ6.v2:8x8:warrior,warrior,warrior')
        self.assertEqual(len(env.get_observation()), 7 + 8 * 3 + 64)
        for _ in range(20):
            env.step(self.wait(env))
        self.assertEqual(env.city.population, 0)
        self.assertNotIn('city', env.last_info)
        self.assertEqual(set(env.get_current_state()['Ottertopia']), {'health', 'position', 'alive'})

    def test_growth_threshold_follows_civ6(self):
        self.assertEqual([growth_threshold(p) for p in range(1, 7)], [15, 24, 33, 44, 55, 66])

    def test_unblocked_city_grows_and_penalizes_the_attacker(self):
        env = self.make_env(city_population=1)
        self.place_far(env)
        report = env.city_report()
        self.assertEqual((report['food_yield'], report['production_yield'], report['food_surplus'],
                          report['turns_to_grow'], report['blocked_tiles']), (4, 1, 2, 8, []))
        rewards = [env.step(self.wait(env))[1] for _ in range(8)]
        for reward in rewards[:-1]:
            self.assertAlmostEqual(reward, -0.05)
        self.assertAlmostEqual(rewards[-1], -1.05)
        self.assertIn({'type': 'city_grew', 'object': 'Ottertopia', 'population': 2}, env.events)
        self.assertEqual((env.city.population, env.city.food), (2, 1))  # 16 stored, 15 spent
        self.assertEqual(env.last_info['city']['population'], 2)
        self.assertEqual(len(env.worked_tiles()), 2)
        state = env.get_current_state()['Ottertopia']
        self.assertEqual((state['population'], state['growth_threshold']), (2, 24))

    def test_citizens_prefer_woods_skip_mountains_and_avoid_occupied_tiles(self):
        env = self.make_env(city_population=3)
        grid = env.map.grid
        grid[(4, 3)].set_terrain('MOUNTAIN')
        grid[(3, 5)].set_terrain('FOREST')
        grid[(5, 4)].set_terrain('FOREST')
        self.place(env, [(5, 4), (0, 0), (2, 0)])
        self.assertEqual(sorted(env.territory()), [(3, 3), (3, 4), (3, 5), (4, 5), (5, 4)])
        self.assertEqual(env.blocked_tiles(), [(5, 4)])
        self.assertEqual(env.worked_tiles(), [(3, 5), (3, 3), (3, 4)])
        self.assertEqual(env.city_yields(), (8, 2))
        env.units[0].death_unit()
        self.assertEqual(env.blocked_tiles(), [])
        self.assertEqual(env.worked_tiles()[:2], [(5, 4), (3, 5)])

    def test_blockade_starves_city_to_a_sustainable_size(self):
        env = self.make_env(city_population=7)
        self.place(env, [(3, 4), (5, 4), (4, 3)])
        self.assertEqual(env.city_report()['food_surplus'], -6)
        self.assertIsNone(env.city_report()['turns_to_grow'])
        _, reward, _ = env.step(self.wait(env))
        self.assertAlmostEqual(reward, 0.95)
        self.assertIn({'type': 'city_starved', 'object': 'Ottertopia', 'population': 6}, env.events)
        for _ in range(10):
            env.step(self.wait(env))
        # Three free tiles feed four citizens exactly: 2 + 3 * 2 food for 4 * 2 eaten.
        self.assertEqual((env.city.population, env.city.food, env.food_surplus()), (4, 0, 0))

    def test_starving_city_keeps_one_citizen(self):
        env = self.make_env(city_population=2)
        self.place_far(env)
        for pos in env.territory():
            env.map.grid[pos].set_terrain('MOUNTAIN')
        env.step(self.wait(env))
        self.assertEqual(env.city.population, 1)
        for _ in range(5):
            _, reward, _ = env.step(self.wait(env))
            self.assertAlmostEqual(reward, -0.05)
        self.assertEqual((env.city.population, env.city.food), (1, 0))

    def test_captured_city_stops_growing(self):
        env = self.make_env(city_population=1)
        self.place(env, [(3, 4), (0, 0), (2, 0)])
        env.city.hp, env.city.food = 1, 14
        with patch('game.attack', return_value=(5, 0)):
            env.step(env.encode_action(['E', 'SPACE', 'SPACE']))
        self.assertEqual(env.last_info['outcome'], 'win')
        self.assertEqual((env.city.population, env.city.food), (1, 14))

    def test_growth_observation_and_schema(self):
        env = self.make_env(city_population=2)
        base = self.make_env()
        self.assertEqual(env.observation_schema, base.observation_schema + ':city_growth')
        self.assertEqual(len(env.get_observation()), len(base.get_observation()) + 3)
        self.place_far(env)
        np.testing.assert_allclose(env.get_observation()[-3:], [2 / 20, 0, 0.2])
        env.city.food = 13  # just over half of 24, reported in quarters
        self.assertEqual(env.get_observation()[-2], 0.5)

    def test_invalid_city_population_rejected(self):
        for value in (-1, 21, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GameConfig(city_population=value)


    # Builders and settlers.
    def put_civilian(self, env, kind, pos):
        """Move the scenario's first civilian of this kind to pos."""
        civilian = next(c for c in env.civilians if c.kind == kind)
        civilian.x, civilian.y = pos
        return civilian

    def test_civilians_are_off_by_default_and_need_population(self):
        env = self.make_env(city_population=2)
        self.assertEqual(env.civilians, [])
        self.assertIsNone(env.civilian_report())
        self.assertNotIn('civilians', env.observation_schema)
        for kwargs in ({'city_builders': 1}, {'city_settlers': 1}, {'city_population': 1, 'city_builders': 3},
                       {'city_population': 1, 'city_settlers': -1}, {'city_population': 1, 'city_builders': True}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                GameConfig(**kwargs)

    def test_civilian_free_scenarios_keep_their_random_maps(self):
        plain = self.make_env(city_population=2, forest_density=.2)
        with_civilians = self.make_env(city_population=2, forest_density=.2, city_builders=1)
        self.assertEqual([u.position for u in plain.units], [u.position for u in with_civilians.units])

    def test_civilians_start_on_free_territory(self):
        env = self.make_env(city_population=2, city_builders=2, city_settlers=2)
        self.assertEqual([c.name_instance for c in env.civilians], ['Builder', 'Builder 2', 'Settler', 'Settler 2'])
        positions = [c.position for c in env.civilians]
        self.assertEqual(len(set(positions)), 4)
        self.assertFalse(set(positions) & ({u.position for u in env.units} | {env.city.position}))
        self.assertTrue(all(hex_distance(pos, env.city.position) <= 2 for pos in positions))
        self.assertEqual(env.observation_schema,
                         'ml_civ6.v2:8x8:warrior,warrior,warrior:city_growth:civilians=builder,builder,settler,settler')
        self.assertEqual(len(env.get_observation()), len(self.make_env(city_population=2).get_observation()) + 16)

    def test_builder_farms_grassland_until_its_charges_run_out(self):
        env = self.make_env(city_population=3, city_builders=1)
        self.place_far(env)
        env.map.grid[(4, 3)].set_terrain('FOREST')
        builder = self.put_civilian(env, 'builder', (3, 3))
        self.assertEqual(env.food_surplus(), 2)
        env.step(self.wait(env))
        self.assertEqual(env.farms, {(3, 3)})
        self.assertEqual(env.tile_yield((3, 3)), (3, 0))
        self.assertEqual(env.city_report()['farms'], [[3, 3]])
        self.assertEqual(env.food_surplus(), 3)
        for _ in range(6):
            env.step(self.wait(env))
        self.assertEqual(len(env.farms), 3)
        self.assertNotIn((4, 3), env.farms)  # woods are not farmed
        self.assertEqual((builder.alive, builder.status, builder.charges), (False, 'used up', 0))
        # Citizens now work the farms first.
        self.assertTrue(set(env.worked_tiles()) <= env.farms)

    def test_settler_walks_out_and_founds_a_city(self):
        env = self.make_env(city_population=1, city_settlers=1)
        self.place_far(env)
        settler = self.put_civilian(env, 'settler', (5, 4))
        rewards = []
        while settler.alive:
            rewards.append(env.step(self.wait(env))[1])
        self.assertEqual(settler.status, 'settled')
        self.assertEqual(env.founded_cities, [settler.position])
        self.assertGreaterEqual(hex_distance(settler.position, env.city.position), 4)
        self.assertEqual(len(rewards), 4)  # three moves, then founding
        self.assertAlmostEqual(rewards[-1], -5.05)
        self.assertEqual(env.last_info['civilians']['founded_cities'], [list(settler.position)])
        # A second settler must keep its distance from the new city too.
        self.assertTrue(all(hex_distance(pos, settler.position) >= 4 for pos in env.settlement_sites()))

    def test_military_unit_captures_civilian_but_ram_cannot(self):
        env = self.make_env(city_population=1, unit_count=2, unit_types=('warrior', 'battering_ram'),
                            city_builders=1, city_settlers=1)
        self.place(env, [(1, 1), (5, 2)])
        builder = self.put_civilian(env, 'builder', (2, 1))
        settler = self.put_civilian(env, 'settler', (6, 2))
        legal = [env.actions[a] for a in env.valid_actions()]
        self.assertIn(('E', 'SPACE'), legal)
        self.assertNotIn(('SPACE', 'E'), legal)
        _, reward, _ = env.step(env.encode_action(['E', 'SPACE']))
        self.assertAlmostEqual(reward, 1.95)
        self.assertEqual((builder.status, env.units[0].position), ('captured', (2, 1)))
        self.assertIn({'type': 'civilian_captured', 'object': 'Builder', 'civilian': 'builder', 'by': 'Otto'},
                      env.events)
        settler.x, settler.y = 6, 2  # keep the settler next to the ram
        env.units[1].move(1, 0)
        self.assertTrue(settler.alive)
        self.assertEqual(env.units[1].position, (5, 2))
        self.assertEqual(env.events[-1]['type'], 'blocked_move')
        self.assertEqual(env.civilian_report()['captured'], 1)

    def test_capturing_a_settler_is_worth_more(self):
        env = self.make_env(city_population=1, unit_count=1, city_settlers=1)
        self.place(env, [(1, 1)])
        self.put_civilian(env, 'settler', (2, 1))
        _, reward, _ = env.step(env.encode_action(['E']))
        self.assertAlmostEqual(reward, 3.95)

    def test_civilians_walk_around_units(self):
        env = self.make_env(city_population=1, city_builders=1)
        self.place(env, [(4, 3), (0, 0), (2, 0)])
        builder = self.put_civilian(env, 'builder', (5, 3))
        env.map.grid[(5, 4)].set_terrain('MOUNTAIN')
        for pos in [(3, 3), (3, 4), (3, 5), (4, 5)]:
            env.farms.add(pos)
        # The only farm site is occupied by a unit, so the builder waits.
        env.step(self.wait(env))
        self.assertEqual(builder.position, (5, 3))
        env.units[0].x, env.units[0].y = 0, 2
        env.step(self.wait(env))
        self.assertEqual(builder.position, (4, 3))

    def test_heuristic_hunts_nearby_civilians(self):
        env = self.make_env(city_population=1, unit_count=1, city_settlers=1)
        self.place(env, [(1, 1)])
        settler = self.put_civilian(env, 'settler', (3, 1))
        self.assertEqual(env.actions[env.guided_action(random.Random(1))], ('E',))
        env.step(env.encode_action(['E']))
        self.assertEqual(settler.position, (3, 0))  # stepped onto a settling site
        _, reward, _ = env.step(env.guided_action(random.Random(1)))
        self.assertEqual(settler.status, 'captured')  # caught before it could found a city
        self.assertAlmostEqual(reward, 3.95)


    # Ancient-era districts and buildings.
    def test_builds_are_off_by_default(self):
        env = self.make_env(city_population=2)
        self.assertEqual((env.districts, env.buildings), ({}, []))
        self.assertNotIn('built', env.observation_schema)
        self.assertEqual(len(env.get_observation()), 7 + 8 * 3 + 64 + 3)
        report = env.city_report()
        self.assertEqual((report['culture'], report['science'], report['faith']), (0, 0, 0))
        self.assertNotIn('producing', report)

    def test_build_config_validation(self):
        for kwargs, message in [
                ({'city_districts': ('campus',)}, 'population'),
                ({'city_population': 1, 'city_buildings': ('library',)}, 'campus'),
                ({'city_population': 1, 'city_build_queue': ('shrine', 'holy_site')}, 'holy_site'),
                ({'city_population': 3, 'city_districts': ('campus', 'holy_site')}, 'at most 1 district'),
                ({'city_population': 1, 'city_buildings': ('monument',), 'city_build_queue': ('monument',)}, 'once'),
                ({'city_population': 1, 'city_buildings': ('water_mill',)}, 'monument'),
                ({'city_population': 1, 'city_buildings': ['monument']}, 'monument')]:
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, message):
                GameConfig(**kwargs)
        config = GameConfig(city_population=4, city_districts=('campus', 'encampment'), city_buildings=('library',),
                            city_build_queue=('barracks', 'holy_site', 'shrine'))
        self.assertEqual(config.builds[-1], 'shrine')

    def test_district_adjacency_and_yields(self):
        env = self.make_env(city_population=4, city_build_queue=('campus', 'holy_site', 'library', 'shrine'))
        self.place_far(env)
        for pos in [(5, 2), (4, 2)]:
            env.map.grid[pos].set_terrain('MOUNTAIN')
        for pos in [(2, 3), (3, 2)]:
            env.map.grid[pos].set_terrain('FOREST')
        self.assertTrue(env.complete_build('campus'))
        self.assertEqual(env.districts['campus'], (4, 3))  # between two mountains
        self.assertNotIn((4, 3), env.territory())
        self.assertTrue(env.complete_build('holy_site'))
        self.assertEqual(env.districts['holy_site'], (3, 3))
        output = env.city_output()
        # Campus: 2 mountains + 1 for its two district neighbors (city center, holy site).
        # Holy site: 1 mountain + 1 for two woods + 1 for two districts.
        self.assertEqual((output['science'], output['faith']), (3, 3))
        env.complete_build('library')
        env.complete_build('shrine')
        self.assertEqual(env.city_report()['science'], 5)
        self.assertEqual(env.city_report()['faith'], 5)
        self.assertEqual([e['item'] for e in env.events if e['type'] == 'city_built'],
                         ['campus', 'holy_site', 'library', 'shrine'])

    def test_district_avoids_occupied_tiles_and_replaces_farms(self):
        env = self.make_env(city_population=1, city_build_queue=('campus',))
        self.place(env, [(3, 3), (0, 0), (2, 0)])
        env.farms.add((4, 3))
        env.complete_build('campus')
        self.assertNotEqual(env.districts['campus'], (3, 3))
        site = env.districts['campus']
        self.assertEqual(env.map.grid[site].district, 'campus')
        env.farms.add(site)
        self.assertNotIn(site, env.farm_sites())

    def test_buildings_add_food_production_and_culture(self):
        env = self.make_env(city_population=1, city_buildings=('monument', 'granary'))
        self.place_far(env)
        self.assertEqual(env.city_yields(), (5, 1))  # 2 center + 2 grassland + 1 granary
        self.assertEqual(env.city_output()['culture'], 2)
        self.assertEqual(env.food_surplus(), 3)
        self.assertEqual(env.city.wall_hp, 0)
        self.assertFalse(env.city.ranged_combat)

    def test_queue_finishes_walls_carries_overflow_and_city_strikes(self):
        env = self.make_env(city_population=1, city_build_queue=('ancient_walls', 'monument'))
        self.place_far(env)
        for pos in env.territory():
            env.map.grid[pos].set_terrain('FOREST')
        self.assertEqual(env.city_yields()[1], 2)
        env.build_progress['ancient_walls'] = 79
        _, reward, _ = env.step(self.wait(env))
        self.assertAlmostEqual(reward, -1.05)
        self.assertIn({'type': 'city_built', 'object': 'Ottertopia', 'item': 'ancient_walls'}, env.events)
        self.assertEqual((env.city.wall_hp, env.city.wall_hp_max), (100, 100))
        self.assertEqual((env.city.ranged_combat, env.city.strength_ranged), (True, 15))
        report = env.city_report()
        self.assertEqual((report['producing'], report['production_progress'], report['production_cost']),
                         ('monument', 1, 60))
        self.assertEqual(report['completed'], ['ancient_walls'])
        env.units[0].x, env.units[0].y = 3, 4
        with patch('game.attack', return_value=(7, 0)):
            env.step(self.wait(env))
        self.assertEqual(env.units[0].hp, 93)

    def test_queue_skips_districts_over_the_population_limit(self):
        env = self.make_env(city_population=1, city_districts=('campus',), city_build_queue=('holy_site', 'monument'))
        self.assertEqual(env.current_build(), 'monument')
        env.city.population = 4
        self.assertEqual(env.current_build(), 'holy_site')
        env.city.population = 1
        env.build_progress['monument'] = 59
        env.step(self.wait(env))
        self.assertIn('monument', env.buildings)
        self.assertIsNone(env.current_build())

    def test_encampment_two_hexes_out_blocks_units_and_strikes(self):
        env = self.make_env(city_population=1, unit_count=1, city_districts=('encampment',),
                            city_buildings=('barracks',))
        camp = env.districts['encampment']
        self.assertEqual(hex_distance(camp, env.city.position), 2)
        self.assertTrue(env.map.grid[camp].block_path)
        self.assertNotEqual(env.units[0].position, camp)
        self.assertEqual(env.city_yields()[1], 2)  # city center 1 + barracks 1
        self.assertEqual((env.city.ranged_combat, env.city.strength_ranged), (False, 15))
        observation = env.get_observation()
        self.assertEqual(observation[7 + 8 + camp[1] * 8 + camp[0]], 1.0)
        # Stand next to the encampment, out of the city's reach.
        neighbor = next(cell.index for cell in env.map.grid[camp].get_neighbors(env.map.grid)
                        if not cell.block_path and hex_distance(cell.index, env.city.position) == 3)
        self.place(env, [neighbor])
        unit = env.units[0]
        into_camp = next(d for d in constants.DIRECTIONS[:-1]
                         if (unit.x + constants.movement_delta(d, unit.y)[0],
                             unit.y + constants.movement_delta(d, unit.y)[1]) == camp)
        self.assertNotIn(env.encode_action([into_camp]), env.valid_actions())
        with patch('game.attack', return_value=(9, 0)):
            env.step(self.wait(env))
        self.assertIn({'type': 'encampment_strike', 'object': 'Ottertopia', 'target': 'Otto'}, env.events)
        self.assertEqual(unit.hp, 91)

    def test_build_observation_and_schema(self):
        env = self.make_env(city_population=1, city_buildings=('monument',), city_build_queue=('granary',))
        self.assertTrue(env.observation_schema.endswith(':built=monument:queue=granary'))
        base = self.make_env(city_population=1)
        observation = env.get_observation()
        self.assertEqual(len(observation), len(base.get_observation()) + 3)
        np.testing.assert_allclose(observation[-3:], [1, 0, 0])
        env.build_progress['granary'] = 33  # just over half of 65
        np.testing.assert_allclose(env.get_observation()[-3:], [1, 0, .5])

    def test_starting_campus_keeps_random_maps(self):
        plain = self.make_env(city_population=1, forest_density=.2)
        campus = self.make_env(city_population=1, forest_density=.2, city_districts=('campus',))
        self.assertEqual([u.position for u in plain.units], [u.position for u in campus.units])


if __name__ == '__main__':
    unittest.main()
