"""The empire game: map, settlers and cities, economy, research, builders, roads, ships and combat."""

import math
import random
import unittest

import empire_rules as R
from class_hex import hex_distance
from empire import (ALL_MOVES, DECISION_KINDS, OPTION_INDEX, City, Decision, EmpireConfig, EmpireGame,
                    combat_damage)
from game import damage_at_least


def make_game(**kwargs):
    options = dict(width=10, height=9, turn_limit=60, river_count=0)
    options.update(kwargs)
    env = EmpireGame(EmpireConfig(**options), seed=7)
    env.reset(seed=7)
    return env


def blank(env, terrain='GRASSLAND'):
    """Clear the map to one terrain with no rivers, resources, units or cities."""
    for cell in env.map.grid.values():
        cell.set_terrain(terrain)
    env.map.rivers = set()
    env.resources, env.improvements, env.roads = {}, {}, set()
    for unit in list(env.units):
        env._remove_unit(unit)
    env.cities, env.city_at, env.tile_owner = [], {}, {}
    env._city_version += 1
    for player in env.players:
        player.cities_founded = 0
    env._decisions = env._build_decisions(0)
    return env


def found(env, owner, position):
    settler = env._add_unit(owner, 'settler', position)
    return env.found_city(settler)


def options(decision):
    return [DECISION_KINDS[decision.kind].options[index] for index in decision.valid]


def act(env, choose):
    """Step with one option name per decision, chosen by choose(decision) (None keeps the first legal option)."""
    action = []
    for decision in env.observe():
        name = choose(decision)
        action.append(decision.valid[0] if name is None else OPTION_INDEX[decision.kind][name])
    return env.step(tuple(action))


def holding(decision):
    """Keep everything still: units fortify or hold, cities build their first option."""
    names = options(decision)
    for name in ('hold', 'fortify'):
        if name in names:
            return name
    return None


class RulesTests(unittest.TestCase):
    def test_amenities_and_moods(self):
        self.assertEqual([R.amenities_required(n) for n in range(1, 8)], [0, 0, 1, 1, 2, 2, 3])
        self.assertEqual([R.amenity_mood(s)[0] for s in (3, 1, 0, -1, -3, -5, -7)],
                         ['ecstatic', 'happy', 'content', 'displeased', 'unhappy', 'unrest', 'revolt'])
        self.assertEqual(R.amenity_mood(-2)[1:], (.85, .95))

    def test_technology_tree_and_costs(self):
        self.assertEqual(R.available_technologies(set()),
                         ('pottery', 'animal_husbandry', 'mining', 'sailing'))
        self.assertIn('masonry', R.available_technologies({'mining'}))
        self.assertNotIn('engineering', R.available_technologies({'mining'}))
        self.assertEqual(R.TECHNOLOGIES['shipbuilding'].cost, 200)
        self.assertEqual([R.unit_cost('settler', n) for n in range(3)], [80, 110, 140])
        env = make_game()
        self.assertEqual(env.config.speed, 'online')
        self.assertEqual((env.item_cost(0, 'warrior'), env.tech_cost('pottery'), env.growth_threshold(1)), (20, 13, 8))
        standard = make_game(speed='standard')
        self.assertEqual((standard.item_cost(0, 'warrior'), standard.tech_cost('pottery')), (40, 25))
        self.assertEqual(env.item_price(0, 'warrior'), 80)

    def test_invalid_configurations_are_rejected(self):
        for kwargs in [{'rivals': 2}, {'height': 12}, {'width': 7}, {'map_type': 'pangaea'}, {'speed': 'epic'},
                       {'land_types': ()}, {'land_types': ('swamp',)}, {'water_density': .5, 'forest_density': .2},
                       {'resource_density': .6}, {'river_count': 11}, {'rival_bonus': 5}, {'turn_limit': 5},
                       {'difficulty': 'deity'}, {'rivals': True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EmpireConfig(**kwargs)
        EmpireConfig(rivals=0, height=12)


class MapTests(unittest.TestCase):
    def test_maps_are_mirrored_and_reproducible(self):
        for map_type in ('continents', 'islands'):
            env = make_game(width=14, height=13, map_type=map_type, river_count=3, resource_density=.3)
            with self.subTest(map_type=map_type):
                for position, cell in env.map.grid.items():
                    twin = env.map.grid[env.mirror(position)]
                    self.assertEqual((cell.terrain_type, cell.feature), (twin.terrain_type, twin.feature))
                    self.assertEqual(env.resources.get(position), env.resources.get(env.mirror(position)))
                for edge in env.map.rivers:
                    self.assertIn(frozenset(env.mirror(p) for p in edge), env.map.rivers)
                self.assertEqual(env.starts[1], env.mirror(env.starts[0]))
                again = make_game(width=14, height=13, map_type=map_type, river_count=3, resource_density=.3)
                self.assertEqual({p: c.terrain_type for p, c in env.map.grid.items()},
                                 {p: c.terrain_type for p, c in again.map.grid.items()})
                self.assertEqual(env.map.rivers, again.map.rivers)
                for resource, position in ((r, p) for p, r in env.resources.items()):
                    self.assertTrue(env._resource_fits(resource, position))

    def test_continents_connect_the_starts_and_islands_separate_them(self):
        for seed in range(5):
            continents = EmpireGame(EmpireConfig(), seed=seed)
            continents.reset(seed=seed)
            islands = EmpireGame(EmpireConfig(map_type='islands'), seed=seed)
            islands.reset(seed=seed)
            with self.subTest(seed=seed):
                self.assertIn(continents.starts[1], continents._bfs([continents.starts[0]], continents._land))
                self.assertNotIn(islands.starts[1], islands._bfs([islands.starts[0]], islands._land))
                middle = islands.config.height // 2
                self.assertTrue(all(islands.map.grid[(x, middle)].is_water for x in range(islands.config.width)))
                self.assertEqual(islands.separated(0), 1)
                self.assertEqual(continents.separated(0), 0)

    def test_each_player_starts_with_a_settler_and_a_warrior(self):
        env = make_game()
        for player, start in enumerate(env.starts):
            self.assertEqual(sorted(u.kind for u in env.player_units(player)), ['settler', 'warrior'])
            self.assertTrue(env.legal_site(player, start))
        self.assertEqual(env.observe(), env._build_decisions(0))
        self.assertEqual([d.kind for d in env.observe()], ['research', 'land', 'settler'])


class CityTests(unittest.TestCase):
    def setUp(self):
        self.env = blank(make_game())

    def test_city_sites_need_room_and_open_land(self):
        env = self.env
        capital = found(env, 0, (2, 2))
        self.assertTrue(capital.capital and capital.has_palace)
        self.assertEqual(capital.name, 'Otterholt')
        self.assertEqual(capital.territory, {(2, 2), *env._neighbors((2, 2))})
        self.assertFalse(env.legal_site(0, (5, 2)))  # Three tiles away is too close.
        self.assertTrue(env.legal_site(0, (6, 2)))
        env.map.grid[(6, 2)].set_terrain('GRASSLAND_MOUNTAIN')
        self.assertFalse(env.legal_site(0, (6, 2)))
        env.map.grid[(6, 3)].set_terrain('COAST')
        self.assertFalse(env.legal_site(0, (6, 3)))
        rival = found(env, 1, (2, 7))
        self.assertFalse(env.legal_site(0, (3, 5)) and (3, 5) in rival.territory)
        border = next(iter(sorted(rival.territory - {rival.position})))
        self.assertFalse(env.legal_site(0, border))
        second = found(env, 0, (7, 5))
        self.assertFalse(second.capital)
        self.assertEqual(second.name, 'Kelpford')

    def test_settlers_need_two_citizens_cost_one_and_get_dearer(self):
        env = self.env
        city = found(env, 0, (2, 2))
        self.assertFalse(env.can_build(city, 'settler'))
        city.population = 2
        self.assertTrue(env.can_build(city, 'settler'))
        first = env.item_cost(0, 'settler')
        city.item, city.production = 'settler', first - .5
        env._end_turn()
        self.assertEqual(city.population, 1)
        self.assertEqual(len(env.player_units(0, 'settler')), 1)
        self.assertEqual(env.item_cost(0, 'settler'), first + 15)  # 30 more, at half cost on online speed.

    def test_housing_counts_water_rivers_granaries_and_improvements(self):
        env = self.env
        city = found(env, 0, (4, 4))
        ring = env._neighbors(city.position)
        self.assertEqual(env.housing(city), 2)
        env.map.grid[ring[0]].set_terrain('COAST')
        self.assertEqual(env.housing(city), 3)
        env.map.add_river(city.position, ring[1])
        self.assertEqual(env.housing(city), 5)
        city.buildings.append('granary')
        env.improvements[ring[2]] = 'farm'
        env.improvements[ring[3]] = 'mine'
        self.assertEqual(env.housing(city), 7.5)

    def test_luxuries_supply_amenities_to_the_neediest_cities(self):
        env = self.env
        cities = [found(env, 0, p) for p in ((1, 1), (5, 1), (1, 5), (5, 5), (8, 7))]
        for index, city in enumerate(cities):
            city.population = 1 + 2 * index  # Need 0, 1, 2, 3, 4 amenities.
        luxury = sorted(cities[0].territory - {cities[0].position})[0]
        env.map.grid[luxury].set_terrain('PLAINS')
        env.resources[luxury] = 'citrus'
        self.assertEqual(env.amenities(0), {c.id: 0 for c in cities})
        env.improvements[luxury] = 'plantation'
        self.assertEqual(env.luxuries(0), ['citrus'])
        self.assertEqual(env.amenities(0), {cities[0].id: 0, cities[1].id: 1, cities[2].id: 1,
                                            cities[3].id: 1, cities[4].id: 1})
        self.assertEqual(env.amenity_surplus(cities[4]), -3)

    def test_unhappiness_slows_growth_and_trims_yields(self):
        env = self.env
        city = found(env, 0, (4, 4))
        content = env.city_yields(city)
        city.population = 7  # Three amenities short: unhappy.
        unhappy = env.city_yields(city)
        mood, growth, multiplier = R.amenity_mood(env.amenity_surplus(city))
        self.assertEqual((mood, growth, multiplier), ('unhappy', .7, .9))
        self.assertAlmostEqual(unhappy['gold'], content['gold'] * .9)
        self.assertEqual(R.amenity_mood(-3)[0], 'unhappy')

    def test_city_yields_include_palace_citizens_and_routes(self):
        env = self.env
        capital = found(env, 0, (2, 2))
        yields = env.city_yields(capital)
        # Grassland centre (2 food, raised to 1 production), one grassland tile, the Palace, half a science.
        self.assertEqual(yields, {'food': 4, 'production': 3, 'science': 2.5, 'gold': 5, 'culture': 1})
        other = found(env, 0, (7, 2))
        env.players[0].routes.append((capital.id, other.id))
        routed = env.city_yields(capital)
        self.assertEqual((routed['food'], routed['production']), (5, 4))
        env.players[1].bonus = 1.25
        rival = found(env, 1, (4, 7))
        self.assertAlmostEqual(env.city_yields(rival)['production'], 1.25 * 3)

    def test_production_buildings_purchases_and_overflow(self):
        env = self.env
        city = found(env, 0, (4, 4))
        city.item, city.production = 'monument', env.item_cost(0, 'monument') + 2 - 3
        env._end_turn()
        self.assertEqual(city.buildings, ['monument'])
        self.assertIsNone(city.item)
        self.assertEqual(city.production, 2)
        env.players[0].known.add('masonry')
        env.players[0].gold = 1000
        city.item = 'ancient_walls'
        env._decisions = env._build_decisions(0)
        decision = next(d for d in env.observe() if d.kind == 'city')
        self.assertEqual(options(decision), ['ancient_walls', 'buy'])
        self.assertEqual(dict(zip(DECISION_KINDS['city'].features, decision.features))['treasury'], 2)
        act(env, lambda d: 'buy' if d.kind == 'city' else holding(d))
        self.assertEqual((city.wall_hp, city.wall_hp_max), (100, 100))
        purchase = next(e for e in env.events if e['type'] == 'item_purchased')
        self.assertEqual(purchase['price'], 4 * env.item_cost(0, 'ancient_walls'))
        self.assertLess(env.players[0].gold, 1000 - purchase['price'] + 20)

    def test_bankruptcy_disbands_a_unit(self):
        env = self.env
        found(env, 0, (4, 4))
        for position in ((1, 1), (2, 1), (3, 1)):
            env._add_unit(0, 'warrior', position)
        env.players[0].gold = -50
        env._end_turn()
        self.assertEqual(len(env.player_units(0, 'warrior')), 2)
        self.assertEqual(env.players[0].gold, 0)

    def test_borders_claim_the_best_unowned_tile(self):
        env = self.env
        city = found(env, 0, (4, 4))
        best = env._border_candidates(city)[0]
        env.map.grid[best].set_terrain('PLAINS_HILLS')
        env.map.grid[best].set_feature('FOREST')
        self.assertEqual(env._border_candidates(city)[0], best)
        city.culture = env.border_cost(0)
        city.item = 'warrior'
        env._end_turn()
        self.assertIn(best, city.territory)
        self.assertIs(env.tile_owner[best], city)

    def test_enemy_units_block_worked_tiles(self):
        env = self.env
        city = found(env, 0, (4, 4))
        best = env.worked_tiles(city)[0]
        env.map.grid[best].set_terrain('PLAINS_HILLS')
        self.assertEqual(env.worked_tiles(city), [best])
        env._add_unit(1, 'warrior', best)
        self.assertNotIn(best, env.worked_tiles(city))


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.env = blank(make_game())

    def test_research_completes_carries_overflow_and_unlocks(self):
        env = self.env
        city = found(env, 0, (4, 4))
        player = env.players[0]
        decision = env._build_decisions(0)[0]
        self.assertEqual(options(decision), list(R.available_technologies(set())))
        env._set_research(0, 'pottery')
        with self.assertRaises(ValueError):
            env._set_research(1, 'irrigation')
        self.assertFalse(env.can_build(city, 'granary'))
        env._research(0, env.tech_cost('pottery') + 3)
        self.assertIn('pottery', player.known)
        self.assertIsNone(player.research)
        self.assertEqual(player.progress['_overflow'], 3)
        self.assertTrue(env.can_build(city, 'granary'))
        self.assertIn('irrigation', env.available_technologies(0))

    def test_boosts_pay_half_of_a_technology(self):
        env = self.env
        env.map.grid[(5, 4)].set_terrain('COAST')
        found(env, 0, (4, 4))
        self.assertIn('sailing', env.players[0].boosted)
        self.assertEqual(env.players[0].progress['sailing'], env.tech_cost('sailing') / 2)
        env._add_unit(1, 'warrior', (4, 2))
        env._meet()
        self.assertIn('writing', env.players[0].boosted)
        self.assertIn('writing', env.players[1].boosted)

    def test_strategic_units_need_an_improved_resource(self):
        env = self.env
        city = found(env, 0, (4, 4))
        player = env.players[0]
        player.known |= {'animal_husbandry', 'horseback_riding'}
        self.assertFalse(env.can_build(city, 'horseman'))
        horses = sorted(city.territory - {city.position})[0]
        env.resources[horses] = 'horses'
        self.assertFalse(env.can_build(city, 'horseman'))
        env.improvements[horses] = 'pasture'
        self.assertTrue(env.can_build(city, 'horseman'))


class BuilderAndRoadTests(unittest.TestCase):
    def setUp(self):
        self.env = blank(make_game())

    def test_improvements_follow_terrain_resources_and_technology(self):
        env = self.env
        city = found(env, 0, (4, 4))
        ring = sorted(city.territory - {city.position})
        env.map.grid[ring[0]].set_terrain('GRASSLAND_HILLS')
        env.map.grid[ring[1]].set_feature('FOREST')
        env.map.grid[ring[2]].set_terrain('DESERT')
        env.resources[ring[3]] = 'cattle'
        self.assertEqual(env.improvable(0, ring[0]), 'farm')
        env.players[0].known.add('mining')
        self.assertEqual(env.improvable(0, ring[0]), 'mine')
        self.assertIsNone(env.improvable(0, ring[1]))
        self.assertIsNone(env.improvable(0, ring[2]))
        self.assertIsNone(env.improvable(0, ring[3]))
        env.players[0].known.add('animal_husbandry')
        self.assertEqual(env.improvable(0, ring[3]), 'pasture')
        self.assertEqual(env.improvable(0, ring[4]), 'farm')
        self.assertIsNone(env.improvable(1, ring[4]))
        self.assertIsNone(env.improvable(0, city.position))

    def test_builders_spend_charges_and_reach_fishing_boats_from_land(self):
        env = self.env
        city = found(env, 0, (4, 4))
        ring = sorted(city.territory - {city.position})
        builder = env._add_unit(0, 'builder', ring[0])
        before = env.city_yields(city)['food']
        env.improve(builder)
        self.assertEqual((env.improvements[ring[0]], builder.charges), ('farm', 2))
        self.assertGreaterEqual(env.city_yields(city)['food'], before)
        water = ring[1]
        env.map.grid[water].set_terrain('COAST')
        env.resources[water] = 'fish'
        builder.charges = 1
        env._relocate(builder, city.position)  # Every tile of the first ring is next to the centre.
        self.assertIsNone(env.builder_work_tile(builder))  # Fishing boats need Sailing.
        env.players[0].known.add('sailing')
        self.assertEqual(env.builder_work_tile(builder), water)
        env.improve(builder)
        self.assertEqual(env.improvements[water], 'fishing_boats')
        self.assertNotIn(builder, env.units)

    def test_trade_routes_lay_roads_and_feed_the_origin(self):
        env = self.env
        origin = found(env, 0, (1, 4))
        destination = found(env, 0, (6, 4))
        env.players[0].known.add('currency')
        self.assertTrue(env.can_build(origin, 'trader'))
        trader = env._add_unit(0, 'trader', origin.position)
        decision = next(d for d in env._build_decisions(0) if d.kind == 'trader')
        self.assertEqual(options(decision), ['route', 'hold'])
        env.establish_route(trader)
        path = env.land_path(0, origin.position, destination.position)
        self.assertTrue(set(path) <= env.roads | {origin.position, destination.position})
        self.assertEqual(env.players[0].routes, [(origin.id, destination.id)])
        self.assertFalse(env.can_build(origin, 'trader'))  # One route per civilization without Markets.
        origin.buildings.append('market')
        self.assertTrue(env.can_build(origin, 'trader'))
        self.assertIn('currency', env.players[0].boosted | env.players[0].known)

    def test_roads_ignore_terrain_and_bridge_rivers_with_engineering(self):
        env = self.env
        warrior = env._add_unit(0, 'warrior', (2, 4))
        env.map.grid[(3, 4)].set_terrain('GRASSLAND_HILLS')
        env.map.grid[(3, 4)].set_feature('FOREST')
        self.assertEqual(env.step_cost(warrior, (2, 4), (3, 4)), 3)
        env.roads |= {(2, 4), (3, 4)}
        self.assertEqual(env.step_cost(warrior, (2, 4), (3, 4)), 1)
        env.map.add_river((2, 4), (3, 4))
        self.assertEqual(env.step_cost(warrior, (2, 4), (3, 4)), ALL_MOVES)
        env.players[0].known.add('engineering')
        self.assertEqual(env.step_cost(warrior, (2, 4), (3, 4)), 1)


class MovementTests(unittest.TestCase):
    def setUp(self):
        self.env = blank(make_game())

    def test_movement_points_terrain_rivers_and_zones_of_control(self):
        env = self.env
        warrior = env._add_unit(0, 'warrior', (1, 4))
        self.assertTrue(env.travel(warrior, [(5, 4)]))
        self.assertEqual(warrior.position, (3, 4))  # Two moves on grassland.
        warrior.moves = warrior.max_moves = 2
        env.map.grid[(4, 4)].set_terrain('PLAINS_HILLS')
        env.map.grid[(4, 4)].set_feature('FOREST')
        env.travel(warrior, [(4, 4)])
        self.assertEqual((warrior.position, warrior.moves), ((4, 4), 0))  # Full movement always moves one tile.
        warrior.moves = 1
        env.map.add_river((4, 4), (5, 4))
        for position in ((4, 3), (4, 5)):
            env.map.grid[position].set_terrain('GRASSLAND_MOUNTAIN')  # Leave only the river crossing.
        env.travel(warrior, [(5, 4)])
        self.assertEqual((warrior.position, warrior.moves), ((5, 4), 0))  # A river crossing ends the move.
        warrior.moves = 2
        env._add_unit(1, 'warrior', (6, 3))
        env.travel(warrior, [(8, 4)])
        self.assertEqual((warrior.position, warrior.moves), ((6, 4), 0))  # Next to an enemy: zone of control.

    def test_heavy_chariots_gain_a_move_on_flat_land(self):
        env = self.env
        chariot = env._add_unit(0, 'heavy_chariot', (1, 4))
        env._reset_moves(0)
        self.assertEqual(chariot.max_moves, 3)
        env.map.grid[(1, 4)].set_terrain('GRASSLAND_HILLS')
        env._reset_moves(0)
        self.assertEqual(chariot.max_moves, 2)

    def test_civilians_are_captured_and_stacking_is_limited(self):
        env = self.env
        settler = env._add_unit(1, 'settler', (4, 4))
        builder = env._add_unit(0, 'builder', (3, 4))
        self.assertTrue(env.blocked(builder, (4, 4)))
        warrior = env._add_unit(0, 'warrior', (2, 4))
        self.assertFalse(env.blocked(warrior, (3, 4)))  # A soldier may join its own civilian.
        env.travel(warrior, [(4, 4)])
        self.assertEqual(warrior.position, (4, 4))
        self.assertNotIn(settler, env.units)
        other = env._add_unit(0, 'warrior', (5, 4))
        self.assertTrue(env.blocked(other, (4, 4)))


class NavalTests(unittest.TestCase):
    def setUp(self):
        env = self.env = blank(make_game())
        for x in range(env.config.width):
            env.map.grid[(x, 4)].set_terrain('COAST')
        env.map.grid[(0, 4)].set_terrain('OCEAN')

    def test_land_units_embark_onto_coast_with_shipbuilding(self):
        env = self.env
        warrior = env._add_unit(0, 'warrior', (3, 3))
        self.assertIsNone(env.step_cost(warrior, (3, 3), (3, 4)))
        env.players[0].known.add('shipbuilding')
        self.assertEqual(env.step_cost(warrior, (3, 3), (3, 4)), ALL_MOVES)
        env.travel(warrior, [(3, 5)])
        self.assertTrue(env.embarked(warrior))
        self.assertEqual(warrior.moves, 0)
        self.assertEqual(env.unit_strength(warrior, False), R.EMBARKED_STRENGTH)
        self.assertEqual(env.attack_targets(warrior), [])
        self.assertIsNone(env.step_cost(warrior, (1, 4), (0, 4)))  # No ocean without Cartography.
        warrior.moves = 2
        env.travel(warrior, [(3, 5)])
        self.assertFalse(env.embarked(warrior))

    def test_ships_sail_the_coast_from_coastal_cities(self):
        env = self.env
        inland = found(env, 0, (2, 0))
        harbour = found(env, 0, (6, 3))
        env.players[0].known.add('sailing')
        self.assertFalse(env.can_build(inland, 'galley'))
        self.assertTrue(env.can_build(harbour, 'galley'))
        galley = env._add_unit(0, 'galley', harbour.position)
        self.assertEqual(env.step_cost(galley, harbour.position, (6, 4)), 1)
        self.assertIsNone(env.step_cost(galley, (6, 4), (6, 5)))
        self.assertIsNone(env.step_cost(galley, (1, 4), (0, 4)))
        env.travel(galley, [(3, 4)])
        self.assertEqual(galley.position, (4, 4))  # Three moves along the coast.
        env._add_unit(0, 'galley', harbour.position)
        self.assertIn('shipbuilding', env.players[0].boosted)

    def test_galleys_attack_the_shore_and_capture_coastal_cities(self):
        env = self.env
        found(env, 0, (1, 1))
        rival = found(env, 1, (5, 3))
        galley = env._add_unit(0, 'galley', (5, 4))
        archer = env._add_unit(1, 'archer', (4, 3))
        self.assertIn(archer, env.attack_targets(galley))
        self.assertIn(rival, env.attack_targets(galley))
        rival.hp = 1
        env.attack(galley, rival)
        self.assertEqual(rival.owner, 0)
        self.assertEqual(galley.position, rival.position)


class CombatTests(unittest.TestCase):
    def setUp(self):
        self.env = blank(make_game())

    def test_strength_modifiers(self):
        env = self.env
        spearman = env._add_unit(0, 'spearman', (2, 2))
        chariot = env._add_unit(1, 'heavy_chariot', (3, 2))
        self.assertEqual(env.unit_strength(spearman, True, chariot), 35)
        env.map.grid[(3, 2)].set_terrain('GRASSLAND_HILLS')
        self.assertEqual(env.unit_strength(chariot, False, spearman), 31)
        chariot.hp = 55
        self.assertEqual(env.unit_strength(chariot, False, spearman), 27)
        env.players[1].known.add('the_wheel')
        city = found(env, 1, (7, 7))
        self.assertEqual(env.city_strength(city), 28 + R.CITY_STRENGTH_BONUS)

    def test_attack_odds_match_simulated_fights(self):
        env = self.env
        attacker = env._add_unit(0, 'warrior', (2, 2))
        defender = env._add_unit(1, 'warrior', (3, 2))
        attacker.hp, defender.hp = 40, 30
        success, death = env.attack_odds(attacker, defender)
        rng = random.Random(3)
        trials, kills, deaths = 4000, 0, 0
        difference = env.unit_strength(attacker, True, defender) - env.unit_strength(defender, False, attacker)
        for _ in range(trials):
            dealt, taken = combat_damage(rng, difference), combat_damage(rng, -difference)
            died = taken >= attacker.hp
            deaths += died
            kills += not died and dealt >= defender.hp
        self.assertAlmostEqual(death, deaths / trials, delta=.03)
        self.assertAlmostEqual(success, kills / trials, delta=.03)
        self.assertAlmostEqual(death, damage_at_least(40, math.exp(-difference / 25)))

    def test_melee_kills_advance_and_ranged_attacks_draw_no_retaliation(self):
        env = self.env
        warrior = env._add_unit(0, 'warrior', (2, 2))
        victim = env._add_unit(1, 'warrior', (3, 2))
        victim.hp = 1
        env.attack(warrior, victim)
        self.assertNotIn(victim, env.units)
        self.assertEqual(warrior.position, (3, 2))
        self.assertEqual(warrior.moves, 0)
        archer = env._add_unit(0, 'archer', (5, 5))
        target = env._add_unit(1, 'spearman', (7, 5))
        env.attack(archer, target)
        self.assertEqual(archer.hp, 100)
        self.assertLess(target.hp, 100)
        with self.assertRaises(ValueError):
            env.attack(archer, target)  # No movement left.

    def test_walls_rams_and_ranged_attacks_on_cities(self):
        env = self.env
        city = found(env, 1, (5, 5))
        city.wall_hp = city.wall_hp_max = 100
        warrior = env._add_unit(0, 'warrior', (4, 5))
        self.assertEqual(env.wall_multiplier(warrior, city), .5)
        env._add_unit(0, 'battering_ram', (6, 5))
        self.assertEqual(env.wall_multiplier(warrior, city), 1.0)
        env._damage_city(city, 500, False, 1.0)
        self.assertEqual((city.wall_hp, city.hp), (0, 1))  # Ranged fire cannot take the last hit point.
        env._end_turn()
        self.assertEqual(city.hp, 1)  # A city damaged this turn does not heal.
        env._end_turn()
        self.assertEqual(city.hp, 1 + R.CITY_HEAL)

    def test_walled_cities_strike_the_weakest_unit_in_range(self):
        env = self.env
        city = found(env, 1, (5, 5))
        city.wall_hp = city.wall_hp_max = 100
        near = env._add_unit(0, 'warrior', (5, 3))
        weak = env._add_unit(0, 'warrior', (6, 6))
        far = env._add_unit(0, 'warrior', (1, 1))
        weak.hp = 60
        env._city_strikes(1)
        self.assertLess(weak.hp, 60)
        self.assertEqual((near.hp, far.hp), (100, 100))

    def test_capturing_the_only_rival_capital_wins_by_domination(self):
        env = self.env
        found(env, 0, (1, 1))
        capital = found(env, 1, (6, 6))
        garrison = env._add_unit(1, 'archer', capital.position)
        capital.population = 3
        warrior = env._add_unit(0, 'warrior', (5, 6))
        capital.hp = 1
        env.attack(warrior, capital)
        self.assertEqual((capital.owner, capital.population, capital.hp), (0, 2, 50))
        self.assertNotIn(garrison, env.units)
        self.assertEqual(warrior.position, capital.position)
        self.assertFalse(capital.has_palace)
        self.assertTrue(all(env.tile_owner[p].owner == 0 for p in capital.territory))
        self.assertEqual(env._conquest(), 'win')
        self.assertFalse(env.players[1].alive)


class GameTests(unittest.TestCase):
    def test_decisions_are_well_formed_through_random_and_heuristic_games(self):
        for seed, map_type, policy in ((1, 'continents', 'random'), (2, 'islands', 'heuristic'),
                                       (3, 'continents', 'heuristic')):
            env = EmpireGame(EmpireConfig(map_type=map_type, turn_limit=60), seed=seed)
            env.reset(seed=seed)
            rng = random.Random(seed)
            done = False
            with self.subTest(seed=seed, map_type=map_type, policy=policy):
                while not done:
                    for decision in env.observe():
                        spec = DECISION_KINDS[decision.kind]
                        self.assertIsInstance(decision, Decision)
                        self.assertEqual(len(decision.features), len(spec.features))
                        self.assertTrue(all(0 <= v <= limit for v, limit in zip(decision.features, spec.limits)))
                        self.assertTrue(decision.valid and set(decision.valid) <= set(range(len(spec.options))))
                    action = env.guided_action(rng) if policy == 'heuristic' else env.random_action(rng)
                    _, reward, done = env.step(action)
                    self.assertTrue(all(c.owner in (0, 1) for c in env.cities))
                    for unit in env.units:
                        layer = env.military_at if unit.spec.military else env.civilian_at
                        self.assertIs(layer[unit.position], unit)
                self.assertIn(env.last_info['outcome'], ('win', 'loss', 'timeout'))
                self.assertEqual(env.observe(), ())
                self.assertEqual(env.step(()), ((), 0.0, True))

    def test_land_units_know_whether_they_are_a_citys_only_defender(self):
        env = blank(make_game())
        city = found(env, 0, (4, 4))
        guard = env._add_unit(0, 'warrior', city.position)
        ram = env._add_unit(0, 'battering_ram', (5, 4))
        roamer = env._add_unit(0, 'warrior', (1, 1))
        self.assertEqual([env.post(u) for u in (guard, ram, roamer)], [2, 0, 0])
        second = env._add_unit(0, 'archer', (3, 4))
        self.assertEqual([env.post(guard), env.post(second)], [1, 1])
        decision = next(d for d in env._build_decisions(0) if d.entity == guard.id)
        self.assertEqual(dict(zip(DECISION_KINDS['land'].features, decision.features))['post'], 1)

    def test_invalid_actions_are_rejected(self):
        env = make_game()
        decisions = env.observe()
        for action in [(), 'abc', tuple(d.valid[0] for d in decisions)[:-1],
                       tuple(99 for _ in decisions), tuple(True for _ in decisions)]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)

    def test_same_seed_and_actions_reproduce_a_game(self):
        def play(seed):
            env = EmpireGame(EmpireConfig(turn_limit=40), seed=seed)
            env.reset(seed=seed)
            rng, trace, done = random.Random(5), [], False
            while not done:
                _, reward, done = env.step(env.guided_action(rng))
                trace.append((reward, env.score(0), env.score(1), len(env.units)))
            return trace
        self.assertEqual(play(11), play(11))
        self.assertNotEqual(play(11), play(12))

    def test_the_turn_limit_decides_by_score_and_ties_are_timeouts(self):
        env = make_game(turn_limit=10)
        done = False
        while not done:
            _, reward, done = act(env, holding)
        self.assertEqual(env.turn_number, 10)
        self.assertEqual(env.last_info['outcome'], 'timeout' if env.score(0) == env.score(1) else
                         'win' if env.score(0) > env.score(1) else 'loss')
        solo = make_game(rivals=0, turn_limit=10)
        done = False
        while not done:
            _, _, done = solo.step(solo.guided_action(random.Random(1)))
        self.assertEqual(solo.last_info['outcome'], 'timeout')
        self.assertGreater(solo.score(0), 0)

    def test_rewards_follow_score_and_combat(self):
        env = blank(make_game())
        found(env, 0, (1, 1))
        found(env, 1, (7, 7))
        warrior = env._add_unit(0, 'warrior', (4, 4))
        enemy = env._add_unit(1, 'warrior', (5, 4))
        enemy.hp = 1
        env._decisions = env._build_decisions(0)
        before = env.score(0)
        _, reward, _ = act(env, lambda d: 'hunt' if d.kind == 'land' else None)
        self.assertNotIn(enemy, env.units)
        self.assertAlmostEqual(reward, .5 + .1 * (env.score(0) - before))
        self.assertIn(warrior, env.units)

    def test_consulting_the_heuristic_changes_nothing(self):
        env = EmpireGame(EmpireConfig(turn_limit=60), seed=6)
        env.reset(seed=6)
        rng = random.Random(6)
        for _ in range(40):
            env.step(env.guided_action(rng))
        env.players[0].gold = 900  # Rich enough that the heuristic buys in several cities.
        env._decisions = env._build_decisions(0)

        def snapshot():
            return (env.players[0].gold, [(u.id, u.position, u.hp, u.moves) for u in env.units],
                    [(c.id, c.item, c.production, c.population) for c in env.cities], env.rng.getstate(),
                    env.observe())
        before = snapshot()
        action = env.guided_action(random.Random(1))
        self.assertEqual(snapshot(), before)
        buys = sum(DECISION_KINDS[d.kind].options[a] == 'buy' for d, a in zip(env.observe(), action))
        self.assertGreater(buys, 0)

    def test_heuristic_rival_expands_and_researches(self):
        env = EmpireGame(EmpireConfig(turn_limit=60), seed=4)
        env.reset(seed=4)
        while not env.step(env.guided_action(random.Random(4)))[2]:
            pass
        for owner in (0, 1):
            self.assertGreaterEqual(len(env.player_cities(owner)), 2)
            self.assertGreaterEqual(len(env.players[owner].known), 5)
        self.assertTrue(env.improvements)
        self.assertIsInstance(env.cities[0], City)
        self.assertLessEqual(max(hex_distance(p, c.position) for c in env.cities for p in c.territory), 3)


if __name__ == '__main__':
    unittest.main()
