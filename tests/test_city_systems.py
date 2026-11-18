"""The enemy capital's districts, civilians, mood, war weariness, loyalty, governors and pantheon."""

import random
import unittest
from unittest.mock import patch

import constants
from class_hex import hex_distance
from economy import (DISTRICT_SPECS, amenities_needed, amenity_mood, citizen_pressure, district_limit,
                     housing_growth_rate, loyalty_effects)
from game import (BUILD_ITEMS, COMMANDS, TACTIC_INDEX, TACTICAL_FIRST_TACTIC, TACTICAL_UNIT_WIDTH, Game,
                  GameConfig)


class CitySystemsTests(unittest.TestCase):
    def make_env(self, observation='raw', **kwargs):
        options = dict(economy=True, unit_count=1, city_heals=False)
        options.update(kwargs)
        env = Game(config=GameConfig(**options), seed=7, observation=observation)
        env.reset(seed=7)
        self.addCleanup(env.close)
        for unit, position in zip(env.units, [(0, 0), (2, 0), (0, 2), (2, 2)]):
            unit.x, unit.y = position  # Out of the city's way unless a test moves them.
        for civilian in env.civilians:
            civilian.x, civilian.y = 7, 7
        return env

    def wait(self, env, turns=1):
        events, rewards = [], []
        for _ in range(turns):
            rewards.append(env.step(env.encode_action(['SPACE'] * len(env.units)))[1])
            events.extend(env.events)
        return events, rewards

    def ring(self, env):
        return [cell.index for cell in env.map.grid[env.city.position].get_neighbors(env.map.grid)]

    def flatten(self, env, terrain='GRASSLAND'):
        """Make every tile but the city centre the same terrain, with no rivers."""
        for position, cell in env.map.grid.items():
            if position != env.city.position:
                cell.set_terrain(terrain)
        env.map.rivers.clear()

    def direction_to(self, origin, destination):
        for direction in constants.DIRECTIONS[:-1]:
            dx, dy = constants.movement_delta(direction, origin[1])
            if (origin[0] + dx, origin[1] + dy) == tuple(destination):
                return direction
        raise AssertionError(f'{destination} is not next to {origin}')

    # Settings.
    def test_city_systems_are_off_by_default_and_need_the_economy(self):
        plain = Game(config=GameConfig(), seed=7)
        plain.reset(seed=7)
        self.assertEqual(plain.civilians, [])
        self.assertIsNone(plain.economy)
        for kwargs in [{'districts': ('campus',)}, {'city_luxuries': 1}, {'city_war_weariness': 10},
                       {'city_loyalty': 50}, {'foreign_city': (5, 5)}, {'city_governor': 'victor'},
                       {'city_pantheon': 'fertility_rites'}, {'city_faith': 5}, {'city_builders': 1},
                       {'city_settlers': 1}]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'need the city economy'):
                GameConfig(**kwargs)

    def test_invalid_city_system_settings_are_rejected(self):
        for kwargs, message in [
                ({'districts': ('forum',)}, 'districts'), ({'districts': ('campus', 'campus')}, 'districts'),
                ({'districts': ('campus', 'holy_site')}, 'at most 1 district'),
                ({'buildings': ('library',)}, 'campus'), ({'build_queue': ('shrine', 'holy_site')}, 'holy_site'),
                ({'city_luxuries': 9}, 'city_luxuries'), ({'city_luxuries': True}, 'city_luxuries'),
                ({'city_war_weariness': 10001}, 'city_war_weariness'), ({'city_loyalty': 0}, 'city_loyalty'),
                ({'city_loyalty': 101}, 'city_loyalty'), ({'foreign_city': (5, 3)}, 'foreign_city'),
                ({'foreign_city': (5, 10)}, 'foreign_city'), ({'foreign_city': (0, 5)}, 'foreign_city'),
                ({'foreign_city': (5,)}, 'foreign_city'), ({'city_governor': 'ibrahim'}, 'city_governor'),
                ({'city_governor_promotion': 'defense_logistics'}, 'none for this governor'),
                ({'city_governor': 'magnus', 'city_governor_promotion': 'defense_logistics'}, 'provision'),
                ({'city_pantheon': 'god_of_the_sea'}, 'city_pantheon'), ({'city_faith': 501}, 'city_faith'),
                ({'city_builders': 3}, 'city_builders'), ({'city_settlers': -1}, 'city_settlers')]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                GameConfig(economy=True, **kwargs)
        config = GameConfig(economy=True, city_population=4, districts=('campus', 'encampment'),
                            buildings=('library',), build_queue=('barracks', 'holy_site', 'shrine'))
        self.assertEqual(config.districts, ('campus', 'encampment'))

    # Districts and buildings.
    def test_district_limit_follows_population(self):
        self.assertEqual([district_limit(n) for n in (1, 3, 4, 6, 7)], [1, 1, 2, 2, 3])

    def test_districts_take_the_best_adjacency_and_add_their_yields(self):
        env = self.make_env(city_population=4, city_luxuries=1,
                            build_queue=('campus', 'holy_site', 'library', 'shrine'))
        self.flatten(env)
        ring = self.ring(env)
        north = [p for p in env.map.grid if hex_distance(p, ring[0]) == 1 and hex_distance(p, env.city.position) == 2]
        for position in north:
            env.map.grid[position].set_terrain('MOUNTAIN')
        self.assertTrue(env.complete_build('campus'))
        campus = env.economy.districts['campus']
        self.assertEqual(campus, ring[0])
        self.assertEqual(env.economy.adjacency('campus', campus), 3)  # three mountains; one district neighbour
        self.assertNotIn(campus, env.economy.workable_tiles())
        self.assertEqual(env.map.grid[campus].district, 'campus')
        self.assertTrue(env.complete_build('library'))
        yields = env.economy.yields()
        self.assertEqual(yields['science'], env.economy.adjacency('campus', campus) + 2)
        self.assertTrue(env.economy.can_build('holy_site'))  # population 4 allows two districts
        env.economy.population = 3
        self.assertFalse(env.economy.can_build('holy_site'))
        self.assertFalse(env.economy.can_build('shrine'))  # no holy site yet

    def test_districts_avoid_units_water_and_the_centre_ring_for_encampments(self):
        env = self.make_env(city_population=1)
        self.flatten(env)
        ring = self.ring(env)
        for position in ring[1:]:
            env.map.grid[position].set_terrain('COAST')
        env.units[0].x, env.units[0].y = ring[0]
        self.assertIsNone(env.economy.district_site('campus'))  # the only land tile is occupied
        env.units[0].x, env.units[0].y = 0, 0
        self.assertEqual(env.economy.district_site('campus'), ring[0])
        self.assertIsNone(env.economy.district_site('encampment'))  # nothing owned two hexes out yet
        env.economy.claim_tile()
        site = env.economy.district_site('encampment')
        self.assertEqual(hex_distance(site, env.city.position), 2)

    def test_starting_encampment_claims_land_blocks_units_and_strikes(self):
        env = self.make_env(city_population=1, districts=('encampment',), buildings=('barracks',))
        camp = env.economy.districts['encampment']
        self.assertEqual(hex_distance(camp, env.city.position), 2)
        self.assertTrue(env.map.grid[camp].block_path)
        self.assertGreater(env.economy.tiles_acquired, 0)
        self.assertEqual((env.city.ranged_combat, env.city.strength_ranged), (False, 15))
        self.assertEqual(env.city.building_housing, 1)
        self.assertEqual(env.economy.yields()['production'], 2)  # centre 1 + barracks 1, grassland worked
        neighbor = next(cell.index for cell in env.map.grid[camp].get_neighbors(env.map.grid)
                        if not cell.block_path and hex_distance(cell.index, env.city.position) == 3)
        env.units[0].x, env.units[0].y = neighbor
        into_camp = env.encode_action([self.direction_to(neighbor, camp)])
        self.assertNotIn(into_camp, env.valid_actions())
        with patch('game.attack', return_value=(9, 0)):
            events, _ = self.wait(env)
        self.assertIn({'type': 'encampment_strike', 'object': 'Ottertopia', 'target': 'Otto'}, events)
        self.assertEqual(env.units[0].hp, 91)

    def test_an_encampment_never_cuts_a_tile_off(self):
        env = self.make_env(city_population=1)
        self.flatten(env, 'MOUNTAIN')
        corridor = [(4, 3), (4, 2), (4, 1), (4, 0)]
        for position in corridor:
            env.map.grid[position].set_terrain('GRASSLAND')
        env.economy.territory |= set(corridor)
        self.assertIsNone(env.economy.district_site('encampment'))  # (4, 2) would seal off the far end
        env.map.grid[(4, 0)].set_terrain('MOUNTAIN')
        env.map.grid[(4, 1)].set_terrain('MOUNTAIN')
        self.assertEqual(env.economy.district_site('encampment'), (4, 2))  # a dead end is fine

    def test_production_overflows_waits_and_rewards_the_attacker_negatively(self):
        env = self.make_env(build_queue=('ancient_walls', 'campus', 'monument'))
        self.flatten(env, 'PLAINS')  # 1 food, 1 production per worked tile
        env.economy.production = 79
        events, rewards = self.wait(env)
        self.assertIn({'type': 'building_completed', 'object': 'Ottertopia', 'building': 'ancient_walls'}, events)
        self.assertAlmostEqual(rewards[0], -1.05)
        self.assertEqual(env.economy.current_build(), 'campus')
        self.assertEqual(env.economy.production, 1)  # 2 production, 1 needed
        env.economy.production = 53.5
        events, rewards = self.wait(env)
        self.assertIn({'type': 'district_completed', 'object': 'Ottertopia', 'district': 'campus'}, events)
        self.assertEqual(env.economy.progress['monument'], 1.5)
        self.assertEqual(env.economy.completed, ['ancient_walls', 'campus'])

    def test_zoning_commissioner_and_city_patron_goddess_speed_up_districts(self):
        env = self.make_env(city_governor='liang', city_governor_promotion='zoning_commissioner',
                            city_pantheon='city_patron_goddess', build_queue=('campus', 'monument'))
        self.flatten(env, 'DESERT')  # the centre's 1 production only
        env.economy.governor_turns = 5
        env.economy.pantheon_founded = True
        env.economy.production = 53
        self.wait(env)
        self.assertIn('campus', env.economy.districts)
        # 1 production became 1.45 toward the campus; what it did not need goes to the monument.
        self.assertAlmostEqual(env.economy.progress['monument'], round(1 - 1 / 1.45, 4))

    def test_district_buildings_are_drawn_only_from_the_queue_order(self):
        env = self.make_env(city_population=1, districts=('campus',), build_queue=('holy_site', 'monument'))
        self.assertEqual(env.economy.current_build(), 'monument')  # the limit blocks a second district
        env.economy.population = 4
        self.assertEqual(env.economy.current_build(), 'holy_site')

    # Builders and settlers.
    def test_civilians_start_on_free_territory_and_join_the_raw_observation(self):
        options = dict(economy=True, city_population=2, city_builders=2, city_settlers=2)
        env = Game(config=GameConfig(**options), seed=7)
        env.reset(seed=7)
        self.assertEqual([c.name_instance for c in env.civilians], ['Builder', 'Builder 2', 'Settler', 'Settler 2'])
        positions = [c.position for c in env.civilians]
        self.assertEqual(len(set(positions)), 4)
        self.assertFalse(set(positions) & ({u.position for u in env.units} | {env.city.position}))
        self.assertTrue(all(hex_distance(p, env.city.position) <= 2 for p in positions))
        self.assertTrue(env.observation_schema.endswith(':economy:civilians=builder,builder,settler,settler'))
        plain = Game(config=GameConfig(economy=True, city_population=2), seed=7)
        plain.reset(seed=7)
        self.assertEqual(len(env.get_observation()), len(plain.get_observation()) + 16)
        self.assertEqual([u.position for u in env.units], [u.position for u in plain.units])

    def test_builder_farms_flat_open_land_until_its_charges_run_out(self):
        env = self.make_env(city_population=3, city_builders=1, city_luxuries=1)
        self.flatten(env)
        ring = self.ring(env)
        env.map.grid[ring[0]].set_feature('FOREST')
        env.map.grid[ring[1]].set_terrain('PLAINS')
        builder = env.civilians[0]
        builder.x, builder.y = ring[1]
        surplus = env.economy.food_surplus()
        events, _ = self.wait(env)
        self.assertIn({'type': 'farm_built', 'object': 'Builder', 'position': list(ring[1]), 'charges': 2}, events)
        self.assertEqual(env.economy.tile_yields(ring[1]), {'food': 2, 'production': 1})
        self.assertEqual(env.economy.housing(), 2.5)
        self.wait(env, 6)
        self.assertEqual(len(env.economy.farms), 3)
        self.assertNotIn(ring[0], env.economy.farms)  # woods are not farmed
        self.assertEqual((builder.alive, builder.status, builder.charges), (False, 'used up', 0))
        self.assertEqual(env.economy.housing(), 3.5)
        self.assertGreater(env.economy.food_surplus(), surplus)

    def test_settler_walks_out_and_founds_a_city(self):
        env = self.make_env(city_settlers=1)
        settler = env.civilians[0]
        settler.x, settler.y = self.ring(env)[1]
        rewards = []
        while settler.alive:
            rewards.append(self.wait(env)[1][0])
        self.assertEqual(settler.status, 'settled')
        self.assertEqual(env.economy.founded_cities, [settler.position])
        self.assertGreaterEqual(hex_distance(settler.position, env.city.position), 4)
        self.assertAlmostEqual(rewards[-1], -5.05)
        self.assertEqual(env.summary()['cities_founded'], 1)
        self.assertTrue(all(hex_distance(p, settler.position) >= 4 for p in env.settlement_sites()))
        # Its citizen adds to the city's loyalty.
        self.assertGreater(env.economy.loyalty_sources()['citizen_pressure'], 2)

    def test_units_capture_civilians_but_rams_cannot(self):
        env = self.make_env(unit_count=2, unit_types=('warrior', 'battering_ram'), city_builders=1, city_settlers=1)
        builder, settler = env.civilians
        env.units[0].x, env.units[0].y = 1, 1
        env.units[1].x, env.units[1].y = 5, 2
        builder.x, builder.y = 2, 1
        settler.x, settler.y = 6, 2  # three hexes from the city: not yet a place to settle
        legal = [env.actions[a] for a in env.valid_actions()]
        self.assertIn(('E', 'SPACE'), legal)
        self.assertNotIn(('SPACE', 'E'), legal)
        _, reward, _ = env.step(env.encode_action(['E', 'SPACE']))
        self.assertAlmostEqual(reward, -.05 + 2)
        self.assertEqual((builder.status, env.units[0].position), ('captured', (2, 1)))
        self.assertIn({'type': 'civilian_captured', 'object': 'Builder', 'civilian': 'builder', 'by': 'Otto'},
                      env.events)
        settler.x, settler.y = 6, 2
        env.units[1].move(1, 0)
        self.assertTrue(settler.alive)
        self.assertEqual(env.events[-1]['type'], 'blocked_move')
        env.units[0].x, env.units[0].y = 7, 2
        _, reward, _ = env.step(env.encode_action([self.direction_to((7, 2), (6, 2)), 'SPACE']))
        self.assertAlmostEqual(reward, -.05 + 4)
        self.assertEqual(env.summary()['civilians_captured'], 2)

    def test_heuristic_hunts_nearby_civilians_and_the_tactical_encoding_agrees(self):
        env = self.make_env(observation='tactical', city_settlers=1)
        settler = env.civilians[0]
        env.units[0].x, env.units[0].y = 1, 1
        settler.x, settler.y = 2, 1
        self.assertIs(env.hunt_target(env.units[0]), settler)
        self.assertEqual(env.actions[env.guided_action(random.Random(1))], ('E',))
        block = env.observe()[:TACTICAL_UNIT_WIDTH]
        self.assertEqual(block[TACTICAL_FIRST_TACTIC + COMMANDS.index('E')], TACTIC_INDEX['ATTACK'])
        env.units[0].hp = 30
        self.assertIsNone(env.hunt_target(env.units[0]))  # too hurt to give chase

    def test_raiders_walk_around_civilians(self):
        env = self.make_env(city_builders=1, player_capital=True, raid_interval=0)
        self.flatten(env)
        env.capital.x, env.capital.y = 1, 6
        from game import C_Raider
        raider = C_Raider(4, 6, name_instance='Raider 1', strength=20, env=env)
        env.raiders.append(raider)
        steps = env._raider_steps()
        best = min((steps[c.index], c.index) for c in env.map.grid[raider.position].get_neighbors(env.map.grid)
                   if c.index in steps)[1]
        env.civilians[0].x, env.civilians[0].y = best
        env.raider_turn()
        self.assertNotEqual(raider.position, best)

    def test_pantheon_civilian_waits_until_the_pantheon_is_founded(self):
        env = self.make_env(city_pantheon='fertility_rites', city_faith=24, buildings=('monument',))
        builder = env.civilians[-1]
        self.assertEqual((builder.alive, builder.status, builder.name_instance),
                         (False, 'not yet trained', 'Pantheon builder'))
        self.assertTrue(env.observation_schema.endswith(':civilians=builder'))
        env.economy.faith = 25
        events, rewards = self.wait(env)
        self.assertIn({'type': 'pantheon_founded', 'object': 'Ottertopia', 'belief': 'fertility_rites'}, events)
        self.assertTrue(builder.alive)
        self.assertEqual(hex_distance(builder.position, env.city.position), 1)
        self.assertAlmostEqual(rewards[0], -1.05)

    # Growth, housing and rewards.
    def test_housing_growth_rate_handles_half_houses(self):
        for population, housing, rate in [(3, 5, 1.0), (4, 5, .5), (5, 5, .25), (9, 5, .25), (10, 5, 0.0),
                                          (4, 5.5, 1.0), (5, 5.5, .5), (6, 5.5, .25), (11, 5.5, 0.0)]:
            with self.subTest(population=population, housing=housing):
                self.assertEqual(housing_growth_rate(population, housing), rate)

    def test_growth_and_starvation_change_the_reward(self):
        env = self.make_env()
        self.flatten(env)
        env.map.add_river(env.city.position, self.ring(env)[0])
        env.economy.food = 14
        events, rewards = self.wait(env)
        self.assertIn('city_grew', [e['type'] for e in events])
        self.assertAlmostEqual(rewards[0], -1.05)
        hungry = self.make_env(city_population=2)
        self.flatten(hungry, 'DESERT')
        events, rewards = self.wait(hungry)
        self.assertIn('city_starved', [e['type'] for e in events])
        self.assertAlmostEqual(rewards[0], .95)

    # Amenities and war weariness.
    def test_amenities_needed_and_moods_follow_gathering_storm(self):
        self.assertEqual([amenities_needed(n) for n in (1, 2, 3, 4, 5, 6, 11, 12)], [0, 0, 1, 1, 2, 2, 5, 5])
        for balance, mood in [(5, 'ecstatic'), (4, 'happy'), (3, 'happy'), (2, 'content'), (0, 'content'),
                              (-1, 'displeased'), (-3, 'unhappy'), (-5, 'unrest'), (-7, 'revolt')]:
            with self.subTest(balance=balance):
                self.assertEqual(amenity_mood(balance)[0], mood)

    def test_mood_scales_growth_and_non_food_yields(self):
        for luxuries, mood, scale in [(0, 'displeased', .95), (1, 'content', 1.0), (4, 'happy', 1.1),
                                      (6, 'ecstatic', 1.2)]:
            with self.subTest(luxuries=luxuries):
                env = self.make_env(city_population=3, city_luxuries=luxuries, buildings=('monument',))
                self.flatten(env, 'PLAINS')
                env.map.add_river(env.city.position, self.ring(env)[0])  # housing 5: growth not slowed
                self.assertEqual(env.economy.mood()[0], mood)
                yields = env.economy.yields()
                self.assertEqual(yields['food'], 5)  # food is not scaled
                self.assertAlmostEqual(yields['production'], round(4 * scale, 2))
                self.assertAlmostEqual(yields['culture'], round(2 * scale, 2))
        env = self.make_env(city_population=11)
        self.assertEqual(env.economy.mood()[0], 'unrest')
        with patch.object(env.economy, 'food_surplus', return_value=4):
            self.assertEqual(env.economy.growth_surplus(), 0)  # unrest stops growth
        with patch.object(env.economy, 'food_surplus', return_value=-4):
            self.assertEqual(env.economy.growth_surplus(), -4)  # but never a shortfall

    def test_combat_at_the_enemy_capital_adds_war_weariness_which_fades(self):
        env = self.make_env(unit_count=2, unit_types=('warrior', 'archer'), ranged_strength=30,
                            city_war_weariness=1000, city_count=2)
        env.cities[1].x, env.cities[1].y = 1, 1
        env.units[0].x, env.units[0].y = 3, 4
        env.units[1].x, env.units[1].y = 7, 7
        with patch('game.attack', return_value=(5, 0)):
            env.step(env.encode_action(['E', 'SPACE']))  # melee, then the city strikes back
        self.assertEqual(env.economy.war_weariness, 1000 + 16 + 16 - 50)
        env.units[0].x, env.units[0].y = 2, 1  # fighting the other civilization adds nothing
        env.units[1].x, env.units[1].y = 7, 7
        with patch('game.attack', return_value=(5, 0)):
            env.step(env.encode_action(['W', 'SPACE']))
        self.assertEqual(env.economy.war_weariness, 982 - 50)

    def test_war_weariness_costs_amenities_up_to_the_need(self):
        for population, weariness, lost in [(2, 2000, 0), (3, 399, 0), (3, 1000, 1), (7, 1000, 2), (7, 2000, 3)]:
            with self.subTest(population=population, weariness=weariness):
                env = self.make_env(city_population=population, city_luxuries=3, city_war_weariness=weariness)
                self.assertEqual(env.economy.amenities_lost_to_war(), lost)
                self.assertEqual(env.economy.amenities()[0], 3 - lost)

    # Loyalty.
    def test_loyalty_formulas(self):
        self.assertEqual([loyalty_effects(n) for n in (100, 75, 74, 25, 24)],
                         [(0, 0), (0, 0), (-.25, -.25), (-.25, -.25), (-.75, -.5)])
        self.assertEqual(citizen_pressure(3, 0, capital=True), 6)
        self.assertAlmostEqual(citizen_pressure(10, 4), 6)
        self.assertEqual(citizen_pressure(10, 10), 0)

    def test_an_isolated_capital_stays_loyal_and_a_foreign_city_wears_it_down(self):
        env = self.make_env(city_population=3)
        self.wait(env)
        self.assertEqual(env.economy.loyalty, 100)
        env = self.make_env(city_population=3, city_loyalty=80, foreign_city=(10, 4), buildings=('monument',))
        # 6 domestic - 6 foreign, -3 for a displeased city, +1 from the monument.
        self.assertEqual(env.economy.loyalty_sources(), {'citizen_pressure': 0, 'amenities': -3, 'buildings': 1})
        self.wait(env)
        self.assertEqual(env.economy.loyalty, 78)

    def test_low_loyalty_cuts_growth_and_yields(self):
        env = self.make_env(city_population=3, city_luxuries=1)
        self.flatten(env)
        for cell in env.map.grid.values():
            if cell.index != env.city.position:
                cell.set_feature('FOREST')  # 2 food, 1 production
        env.map.add_river(env.city.position, self.ring(env)[0])
        env.economy.loyalty = 50
        self.assertAlmostEqual(env.economy.yields()['production'], 3)  # 4 x 0.75
        self.assertAlmostEqual(env.economy.growth_surplus(), 1.5)  # a surplus of 2 x 0.75
        env.economy.loyalty = 20
        self.assertAlmostEqual(env.economy.yields()['production'], 2)  # 4 x 0.5
        self.assertAlmostEqual(env.economy.growth_surplus(), .5)  # 2 x 0.25

    def test_the_city_declares_independence_at_zero_loyalty(self):
        env = self.make_env(city_population=3, city_loyalty=5, foreign_city=(20, 4))
        _, reward, done = env.step(env.encode_action(['SPACE']))
        self.assertTrue(done)
        self.assertEqual(env.last_info['outcome'], 'free_city')
        self.assertIn('city_declared_independence', [e['type'] for e in env.events])
        self.assertAlmostEqual(reward, -.05 + 5)

    # Governors.
    def test_governor_establishes_then_adds_loyalty(self):
        for governor, turns in [('victor', 3), ('reyna', 5), ('pingala', 5)]:
            with self.subTest(governor=governor):
                env = self.make_env(city_population=3, city_loyalty=60, foreign_city=(12, 4), city_governor=governor)
                self.wait(env, turns - 1)
                self.assertFalse(env.economy.governor_established())
                self.wait(env)
                self.assertTrue(env.economy.governor_established())
                self.assertEqual(env.economy.loyalty_sources()['governor'], 8)

    def test_defense_logistics_lets_a_surrounded_city_heal(self):
        for promotion, healed in [('defense_logistics', True), ('garrison_commander', False), ('', False)]:
            with self.subTest(promotion=promotion):
                env = self.make_env(unit_count=3, city_heals=True, city_governor='victor',
                                    city_governor_promotion=promotion)
                for unit, position in zip(env.units, self.ring(env)):
                    unit.x, unit.y = position
                env.economy.governor_turns = 3
                env.city.hp = 80
                self.wait(env)
                self.assertEqual(env.city.hp > 80, healed)

    def test_magnus_and_pingala_raise_yields(self):
        env = self.make_env(city_governor='magnus', city_governor_promotion='surplus_logistics')
        self.flatten(env)
        env.economy.governor_turns = 5
        self.assertEqual(env.economy.yields()['food'], 4.8)  # (2 + 2) x 1.2
        for promotion, culture, science in [('', 2.3, None), ('connoisseur', 3.45, None), ('researcher', 2.3, 1.15)]:
            with self.subTest(promotion=promotion):
                env = self.make_env(city_governor='pingala', city_governor_promotion=promotion,
                                    buildings=('monument',))
                env.economy.governor_turns = 5
                yields = env.economy.yields()
                self.assertAlmostEqual(yields['culture'], culture)
                self.assertEqual(yields.get('science'), science)

    # Religion and borders.
    def test_faith_founds_the_pantheon_and_its_beliefs_take_effect(self):
        env = self.make_env(city_population=3, city_luxuries=1, city_pantheon='religious_settlements',
                            build_queue=('holy_site', 'shrine'), buildings=('monument',))
        self.flatten(env)
        ring = self.ring(env)
        env.map.grid[ring[0]].set_terrain('MOUNTAIN')
        self.assertTrue(env.complete_build('holy_site'))
        env.complete_build('shrine')
        faith = env.economy.yields()['faith']
        self.assertGreaterEqual(faith, 3)  # a mountain, the shrine's 2
        turns = 0
        while not env.economy.pantheon_founded:
            self.wait(env)
            turns += 1
        self.assertEqual(turns, -(-25 // faith))
        self.assertAlmostEqual(env.economy.border_growth(), round(env.economy.yields()['culture'] * 1.15, 2))
        settler = env.civilians[-1]
        self.assertTrue(settler.alive)

    def test_borders_grow_ring_by_ring_out_to_five_but_work_only_three(self):
        env = self.make_env(width=14, height=14, buildings=('monument',))
        center = env.city.position
        for _ in range(30):
            env.economy.claim_tile()
        distances = [hex_distance(p, center) for p in env.economy.territory]
        self.assertEqual(max(distances), 3)  # 6 + 12 tiles first fill rings one and two
        for _ in range(60):
            env.economy.claim_tile()
        self.assertEqual(max(hex_distance(p, center) for p in env.economy.territory), 5)
        self.assertLessEqual(max(hex_distance(p, center) for p in env.economy.territory_tiles()), 3)
        rings = sorted((hex_distance(p, center) for p in env.economy.territory))
        self.assertEqual(rings, sorted(rings))

    def test_borders_never_take_another_city_or_the_capital(self):
        env = self.make_env(city_count=2, player_capital=True, width=12, height=12)
        for _ in range(90):
            env.economy.claim_tile()
        others = {city.position for city in env.cities[1:]} | {env.capital.position}
        self.assertFalse(env.economy.territory & others)

    def test_raw_observation_reports_mood_loyalty_and_what_is_built(self):
        env = self.make_env(city_population=3, city_war_weariness=800, buildings=('monument',),
                            city_governor='victor', city_pantheon='fertility_rites', city_faith=10)
        features = env.get_observation()[-(10 + 2 * len(BUILD_ITEMS)) - 4:-4]  # before the pantheon civilian
        after = 5 + len(BUILD_ITEMS)
        # Weariness costs the one amenity needed (balance -2), loyalty 100, two amenities' worth of
        # weariness, no governor yet, faith 10 of 25.
        self.assertEqual(features[after:after + 5].tolist(), [-.2, 1.0, .2, 0.0, .4])
        built = features[after + 5:]
        self.assertEqual(built.tolist(), [float(name == 'monument') for name in BUILD_ITEMS])
        self.assertIn(DISTRICT_SPECS['campus'].cost, (54,))


if __name__ == '__main__':
    unittest.main()
