"""Civ VI terrain types, features and yields in the simulator's hex map."""

import json
import random
import unittest
from pathlib import Path

from class_hex import FEATURES, TERRAINS, HexCell, HexMap, terrain_code

CATALOG = Path(__file__).resolve().parents[1] / 'civ6_bridge' / 'data' / 'terrain_types.json'


def catalog_yields(entry):
    return {name.removeprefix('YIELD_').lower(): amount for name, amount in entry['yields'].items()}


class TerrainTests(unittest.TestCase):
    def test_terrains_match_the_bundled_base_game_catalog(self):
        catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
        simulated = {spec.civ6_type: (name, spec) for name, spec in TERRAINS.items() if spec.civ6_type}
        self.assertEqual(set(simulated), set(catalog['terrains']))
        for civ6_type, entry in catalog['terrains'].items():
            name, spec = simulated[civ6_type]
            with self.subTest(terrain=name):
                cell = HexCell((0, 0), name)
                self.assertEqual(cell.yields, catalog_yields(entry))
                self.assertEqual(spec.impassable, entry['impassable'])
                self.assertEqual(spec.water, entry['is_water'])
                self.assertEqual(spec.elevation == 'HILLS', entry['is_hills'])
                self.assertEqual(spec.elevation == 'MOUNTAIN', entry['is_mountain'])
                self.assertEqual((spec.movement_cost, spec.defense),
                                 (entry['movement_cost'], entry['defense_modifier']))
        forest = catalog['features']['FEATURE_FOREST']
        spec = FEATURES['FOREST']
        self.assertEqual((spec.movement_change, spec.defense), (forest['movement_change'], forest['defense_modifier']))
        self.assertEqual(dict(spec.yields), catalog_yields(forest))

    def test_original_names_keep_their_rules(self):
        grass, forest, mountain = (HexCell((0, 0), name) for name in ('GRASSLAND', 'FOREST', 'MOUNTAIN'))
        self.assertEqual((forest.terrain_type, forest.feature), ('GRASSLAND', 'FOREST'))
        self.assertEqual(mountain.terrain_type, 'GRASSLAND_MOUNTAIN')
        self.assertEqual([(c.movement_cost, c.defense_bonus, c.block_path) for c in (grass, forest)],
                         [(1, 0, False), (2, 3, False)])
        self.assertTrue(mountain.block_path)
        self.assertEqual([terrain_code(c) for c in (grass, forest, mountain, HexCell((0, 0), 'EDGE'))], [0, 1, 2, 2])
        self.assertEqual(forest.yields, {'food': 2, 'production': 1})
        self.assertEqual(mountain.yields, {})
        forest.set_terrain('GRASSLAND')
        self.assertIsNone(forest.feature)

    def test_forests_grow_only_on_grassland_plains_and_tundra(self):
        for name in ('GRASSLAND', 'PLAINS_HILLS', 'TUNDRA'):
            HexCell((0, 0), name, 'FOREST')
        for name in ('DESERT', 'SNOW_HILLS', 'COAST', 'PLAINS_MOUNTAIN', 'EDGE'):
            with self.subTest(terrain=name), self.assertRaises(ValueError):
                HexCell((0, 0), name).set_feature('FOREST')
        with self.assertRaises(ValueError):
            HexCell((0, 0)).set_feature('JUNGLE')
        with self.assertRaises(ValueError):
            HexCell((0, 0), 'LAVA')

    def test_hills_and_forest_stack_and_reshape_keeps_valid_features(self):
        cell = HexCell((0, 0), 'FOREST')
        cell.reshape(elevation='HILLS')
        self.assertEqual((cell.terrain_type, cell.feature), ('GRASSLAND_HILLS', 'FOREST'))
        self.assertEqual((cell.movement_cost, cell.defense_bonus), (3, 6))
        self.assertEqual(cell.yields, {'food': 2, 'production': 2})
        cell.reshape(land_type='TUNDRA')
        self.assertEqual((cell.terrain_type, cell.feature), ('TUNDRA_HILLS', 'FOREST'))
        cell.reshape(land_type='DESERT')
        self.assertEqual((cell.terrain_type, cell.feature), ('DESERT_HILLS', None))
        water = HexCell((0, 0), 'OCEAN')
        water.reshape(land_type='PLAINS', elevation='HILLS')
        self.assertEqual(water.terrain_type, 'OCEAN')
        self.assertTrue(water.block_path)

    def test_water_next_to_land_is_coast_and_open_water_is_ocean(self):
        hex_map = HexMap(5, 5)
        for cell in hex_map.grid.values():
            cell.set_terrain('COAST')
        hex_map.grid[(0, 0)].set_terrain('PLAINS')
        hex_map.classify_water()
        self.assertEqual(hex_map.grid[(1, 0)].terrain_type, 'COAST')
        self.assertEqual(hex_map.grid[(4, 4)].terrain_type, 'OCEAN')
        self.assertEqual(hex_map.grid[(0, 0)].terrain_type, 'PLAINS')

    def test_land_types_form_regions_and_one_type_uses_no_randomness(self):
        rng = random.Random(3)
        state = rng.getstate()
        single = HexMap(6, 6)
        single.grid[(1, 1)].set_terrain('MOUNTAIN')
        single.grid[(2, 2)].set_terrain('FOREST')
        single.assign_land_types(('desert',), rng)
        self.assertEqual(rng.getstate(), state)
        self.assertEqual(single.grid[(1, 1)].terrain_type, 'DESERT_MOUNTAIN')
        self.assertEqual((single.grid[(2, 2)].terrain_type, single.grid[(2, 2)].feature), ('DESERT', None))
        seen = set()
        for seed in range(10):
            hex_map = HexMap(8, 8)
            hex_map.assign_land_types(('plains', 'snow'), random.Random(seed))
            types = {cell.terrain_type for cell in hex_map.grid.values()}
            self.assertLessEqual(types, {'PLAINS', 'SNOW'})
            seen |= types
        self.assertEqual(seen, {'PLAINS', 'SNOW'})

    def test_hills_skip_mountains_water_and_kept_tiles(self):
        hex_map = HexMap(4, 4)
        hex_map.grid[(0, 0)].set_terrain('MOUNTAIN')
        hex_map.grid[(1, 0)].set_terrain('COAST')
        rng = random.Random(1)
        state = rng.getstate()
        hex_map.raise_hills(0, rng)
        self.assertEqual(rng.getstate(), state)
        hex_map.raise_hills(1, rng, keep_flat={(2, 2)})
        self.assertEqual(hex_map.grid[(0, 0)].terrain_type, 'GRASSLAND_MOUNTAIN')
        self.assertEqual(hex_map.grid[(1, 0)].terrain_type, 'COAST')
        self.assertEqual(hex_map.grid[(2, 2)].terrain_type, 'GRASSLAND')
        self.assertEqual(sum(cell.terrain_type == 'GRASSLAND_HILLS' for cell in hex_map.grid.values()), 13)

    def test_rivers_never_border_water(self):
        hex_map = HexMap(8, 8)
        hex_map.grid[(3, 3)].set_terrain('COAST')
        with self.assertRaises(ValueError):
            hex_map.add_river((3, 3), (4, 3))
        for seed in range(10):
            hex_map = HexMap(8, 8)
            for position in [(3, 3), (4, 3), (3, 4), (5, 5), (6, 5)]:
                hex_map.grid[position].set_terrain('COAST')
            hex_map.generate_rivers(4, random.Random(seed))
            self.assertTrue(hex_map.rivers)
            for edge in hex_map.rivers:
                self.assertFalse(any(hex_map.grid[position].is_water for position in edge))


if __name__ == '__main__':
    unittest.main()
