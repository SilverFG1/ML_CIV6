"""Regression tests for combat outcomes, terrain, masks and independent games."""

import random
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import constants
from class_hex import HexMap
from game import C_Unit, Game, GameConfig, attack, hex_distance


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


if __name__ == '__main__':
    unittest.main()
