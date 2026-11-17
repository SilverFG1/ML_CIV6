"""The defending city's economy: growth, housing, production, borders and blockades."""

import unittest

import numpy as np

from class_hex import hex_distance
from economy import border_cost, growth_multiplier, growth_threshold
from game import BUILD_ITEMS, Game, GameConfig


class EconomyTests(unittest.TestCase):
    def make_env(self, **kwargs):
        options = dict(economy=True, unit_count=1, city_heals=False)
        options.update(kwargs)
        env = Game(config=GameConfig(**options), seed=7)
        env.reset(seed=7)
        self.addCleanup(env.close)
        env.units[0].x, env.units[0].y = 0, 0  # Out of the city's way unless a test moves it.
        return env

    def wait(self, env, turns=1):
        events = []
        for _ in range(turns):
            env.step(env.encode_action(['SPACE'] * len(env.units)))
            events.extend(env.events)
        return events

    def set_ring(self, env, terrains):
        """Set the six tiles around the city (in neighbour order) and return their positions."""
        ring = [cell.index for cell in env.map.grid[env.city.position].get_neighbors(env.map.grid)]
        for position, terrain in zip(ring, terrains):
            env.map.grid[position].set_terrain(terrain)
        return ring

    def test_civ6_growth_housing_and_border_formulas(self):
        self.assertEqual([growth_threshold(n) for n in (1, 2, 3, 4)], [15, 24, 33, 44])
        self.assertEqual([growth_multiplier(n, 5) for n in (3, 4, 5, 9, 10)], [1.0, .5, .25, .25, 0.0])
        self.assertEqual(border_cost(0), 10)
        self.assertAlmostEqual(border_cost(1), 10 + 6 ** 1.3)

    def test_housing_depends_on_water_rivers_and_granary(self):
        env = self.make_env()
        ring = self.set_ring(env, ['GRASSLAND'] * 6)
        self.assertEqual(env.economy.housing(), 2)
        env.map.grid[ring[0]].set_terrain('COAST')
        self.assertEqual(env.economy.housing(), 3)
        env.map.add_river(env.city.position, ring[1])
        self.assertEqual(env.economy.housing(), 5)
        env.complete_build('granary')
        self.assertEqual(env.economy.housing(), 7)

    def test_city_centre_has_minimum_yields_and_citizens_work_the_best_tiles(self):
        env = self.make_env()
        env.map.grid[env.city.position].set_terrain('DESERT')
        self.assertEqual(env.economy.center_yields(), {'food': 2, 'production': 1})
        ring = self.set_ring(env, ['GRASSLAND', 'PLAINS_HILLS', 'MOUNTAIN', 'DESERT', 'COAST', 'TUNDRA'])
        env.map.grid[ring[1]].set_feature('FOREST')
        self.assertEqual(env.economy.worked_tiles(), [ring[1]])  # 1 food, 3 production
        self.assertNotIn(ring[2], env.economy.workable_tiles())  # Mountains cannot be worked.
        self.assertIn(ring[4], env.economy.workable_tiles())  # Water can.
        env.units[0].x, env.units[0].y = ring[1]
        self.assertEqual(env.economy.worked_tiles(), [ring[0]])  # A unit blocks its tile.
        self.assertEqual(env.economy.yields(), {'food': 4, 'production': 1})

    def test_citizens_trade_production_for_food_rather_than_starve(self):
        env = self.make_env(city_population=3)
        ring = self.set_ring(env, ['PLAINS', 'PLAINS', 'PLAINS', 'GRASSLAND', 'GRASSLAND', 'GRASSLAND'])
        for position in ring[:3]:
            env.map.grid[position].set_feature('FOREST')  # 1 food, 2 production: the highest score.
        worked = env.economy.worked_tiles()
        self.assertEqual(len(worked), 3)
        self.assertEqual(env.economy.food_surplus(), 0)
        self.assertEqual(sum(position in ring[3:] for position in worked), 1)
        self.assertNotIn('city_starved', [e['type'] for e in self.wait(env)])

    def test_a_city_that_cannot_feed_itself_shrinks(self):
        env = self.make_env(city_population=2)
        ring = self.set_ring(env, ['DESERT'] * 6)
        events = self.wait(env)
        self.assertEqual(env.economy.population, 1)
        self.assertIn({'type': 'city_starved', 'object': 'Ottertopia', 'population': 1}, events)
        env.map.grid[ring[0]].set_terrain('GRASSLAND')
        self.assertEqual(env.economy.food_surplus(), 2)

    def test_city_grows_and_builds_its_queue_in_order(self):
        env = self.make_env(build_queue=('monument', 'granary'))
        ring = self.set_ring(env, ['GRASSLAND', 'FOREST', 'GRASSLAND', 'GRASSLAND', 'GRASSLAND', 'GRASSLAND'])
        env.map.add_river(env.city.position, ring[2])  # Fresh water: housing 5, so growth is not slowed.
        self.assertEqual(env.economy.yields(), {'food': 4, 'production': 2})
        events = self.wait(env, 8)
        self.assertEqual(env.economy.population, 2)
        self.assertIn('city_grew', [e['type'] for e in events])
        self.assertEqual(env.economy.current_build(), 'monument')
        env.economy.production = 59
        events = self.wait(env)
        self.assertEqual(env.city.buildings, ('monument',))
        self.assertIn({'type': 'building_completed', 'object': 'Ottertopia', 'building': 'monument'}, events)
        self.assertEqual(env.economy.current_build(), 'granary')
        self.assertGreater(env.economy.production, 0)  # Overflow carries over.
        env.economy.production = 64
        self.wait(env)
        self.assertIsNone(env.economy.current_build())
        self.assertEqual(env.economy.production, 0)

    def test_finished_walls_bring_outer_defense_and_a_ranged_strike(self):
        env = self.make_env(build_queue=('ancient_walls',))
        self.assertFalse(env.city.ranged_combat)
        env.economy.production = 79
        self.wait(env)
        city = env.city
        self.assertEqual((city.walls, city.wall_hp, city.wall_hp_max), ('ancient', 100, 100))
        self.assertEqual((city.ranged_combat, city.strength_ranged), (True, 15))
        observation = env.get_observation()
        self.assertAlmostEqual(observation[3], .1)
        self.assertAlmostEqual(observation[5], .15)

    def test_items_the_city_already_has_are_skipped(self):
        env = self.make_env(buildings=('monument',), walls='ancient')
        self.assertEqual(env.economy.build_queue, ('monument', 'granary', 'ancient_walls'))
        self.assertEqual(env.economy.current_build(), 'granary')
        env.complete_build('granary')
        self.assertIsNone(env.economy.current_build())

    def test_culture_claims_the_best_nearby_tile(self):
        env = self.make_env(buildings=('monument',), build_queue=('granary',))
        self.assertEqual(len(env.economy.territory), 7)
        best = env.economy.border_candidates()[0]
        env.map.grid[best].set_terrain('PLAINS_HILLS')
        env.map.grid[best].set_feature('FOREST')
        self.assertEqual(env.economy.border_candidates()[0], best)
        events = self.wait(env, 5)
        self.assertIn(best, env.economy.territory)
        self.assertEqual(len(env.economy.territory), 8)
        self.assertIn({'type': 'border_expanded', 'object': 'Ottertopia', 'tile': list(best)}, events)
        self.assertLessEqual(max(hex_distance(p, env.city.position) for p in env.economy.territory), 3)

    def test_observation_adds_city_features_only_with_the_economy(self):
        plain = Game(config=GameConfig(unit_count=1), seed=7)
        plain.reset(seed=7)
        self.assertIsNone(plain.economy)
        env = self.make_env(city_population=2)
        self.assertEqual(env.observation_schema, 'ml_civ6.v2:8x8:warrior:economy')
        observation = env.get_observation()
        self.assertEqual(observation.shape[0], plain.get_observation().shape[0] + 5 + len(BUILD_ITEMS))
        features = observation[-(5 + len(BUILD_ITEMS)):]
        np.testing.assert_allclose(features[:5], [.1, 0, 0, 0, env.economy.housing() / 20])
        self.assertEqual(features[5:].tolist(), [1.0, 0.0, 0.0])  # Building the monument.

    def test_invalid_economy_settings_are_rejected(self):
        for kwargs in [{'economy': 1}, {'economy': True, 'city_population': 0},
                       {'economy': True, 'city_population': 21}, {'economy': True, 'city_population': True},
                       {'economy': True, 'build_queue': ('palace',)},
                       {'economy': True, 'build_queue': ('granary', 'granary')},
                       {'city_population': 2}, {'build_queue': ('monument',)}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GameConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
