"""Optional display tests run with SDL's in-memory video driver."""

import importlib.util
import os
import unittest
from unittest.mock import patch

from class_hex import direction_between
from game import C_Raider, Game, GameConfig


@unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional pygame not installed')
class RendererTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy',
            'PYGAME_HIDE_SUPPORT_PROMPT': '1',
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.env = Game(render=True, config=GameConfig(unit_count=2, unit_types=('warrior', 'archer')))
        self.env.reset(seed=7)
        self.addCleanup(self.env.close)

    def test_lazy_renderer_draw_and_close(self):
        self.assertIsNone(self.env._renderer)
        self.assertTrue(self.env.render_frame())
        renderer = self.env._renderer
        self.assertGreater(renderer.surface.get_width(), 0)
        self.assertTrue(renderer.pygame.display.get_init())
        self.env.close()
        self.assertFalse(renderer.pygame.display.get_init())

    def test_pause_single_step_speed_and_quit_events(self):
        self.env.render_frame()
        renderer = self.env._renderer
        p = renderer.pygame
        for key in [p.K_SPACE, p.K_MINUS, p.K_RIGHT]:
            p.event.post(p.event.Event(p.KEYDOWN, key=key))
        self.assertTrue(self.env.render_frame())
        self.assertTrue(renderer.paused)
        self.assertEqual(renderer.delay_ms, 50)
        p.event.post(p.event.Event(p.KEYDOWN, key=p.K_RIGHT))
        self.assertTrue(self.env.render_frame())
        self.assertTrue(renderer.paused)
        p.event.post(p.event.Event(p.KEYDOWN, key=p.K_ESCAPE))
        self.assertFalse(self.env.render_frame())
        self.assertTrue(self.env.quit)

    def test_draws_every_unit_type(self):
        for unit_types in [('warrior', 'spearman', 'slinger', 'archer'), ('heavy_chariot', 'battering_ram')]:
            env = Game(render=True, config=GameConfig(unit_count=len(unit_types), unit_types=unit_types))
            env.reset(seed=7)
            self.assertTrue(env.render_frame())
            env.close()

    def test_draws_rivers_along_tile_sides(self):
        from renderer import RIVER_COLOR, SIDE_CORNERS
        env = Game(render=True, config=GameConfig(unit_count=1, river_count=2))
        env.reset(seed=7)
        self.addCleanup(env.close)
        self.assertTrue(env.render_frame())
        renderer = env._renderer
        colors = []
        for edge in env.map.rivers:
            origin, neighbor = sorted(edge)
            polygon = renderer._polygon(origin)
            (x1, y1), (x2, y2) = (polygon[i] for i in SIDE_CORNERS[direction_between(origin, neighbor)])
            colors.append(tuple(renderer.surface.get_at((int((x1 + x2) / 2), int((y1 + y2) / 2))))[:3])
        self.assertGreater(colors.count(RIVER_COLOR), len(colors) // 2)

    def test_draws_intact_and_broken_ancient_walls(self):
        from renderer import RUINED_WALL_COLOR, WALL_COLOR
        env = Game(render=True, config=GameConfig(unit_count=1, walls='ancient'))
        env.reset(seed=7)
        self.addCleanup(env.close)
        for wall_hp, color in ((100, WALL_COLOR), (0, RUINED_WALL_COLOR)):
            env.city.wall_hp = wall_hp
            self.assertTrue(env.render_frame())
            renderer = env._renderer
            corners = [tuple(renderer.surface.get_at((int(x), int(y))))[:3] for x, y in renderer._wall_ring()]
            self.assertGreater(corners.count(color), 3)

    def test_draws_every_terrain_and_toggles_yields(self):
        from class_hex import TERRAINS
        from renderer import LAND_COLORS, MOUNTAIN_COLOR, WATER_COLORS, YIELD_COLORS
        env = Game(render=True, config=GameConfig(width=20, height=4, unit_count=1, city_position=(19, 3)))
        env.reset(seed=1)
        self.addCleanup(env.close)
        env.units[0].x, env.units[0].y = 18, 3
        names = list(TERRAINS)
        for x, name in enumerate(names):
            env.map.grid[(x, 0)].set_terrain(name)
        self.assertTrue(env.render_frame())
        renderer = env._renderer
        size = renderer.size
        for x, name in enumerate(names):
            spec = TERRAINS[name]
            expected = (WATER_COLORS[name] if spec.water else MOUNTAIN_COLOR
                        if spec.land_type is None or spec.elevation == 'MOUNTAIN' else LAND_COLORS[spec.land_type])
            left, top = renderer._position((x, 0))
            with self.subTest(terrain=name):
                self.assertEqual(tuple(renderer.surface.get_at((left + size // 2, top + int(size * .88))))[:3], expected)
        left, top = renderer._position((0, 0))  # Grassland: two food dots.
        dot = (int(left + size / 2 - 1.25 * max(2, size // 19)), int(top + size * .2))
        self.assertEqual(tuple(renderer.surface.get_at(dot))[:3], YIELD_COLORS['food'])
        p = renderer.pygame
        p.event.post(p.event.Event(p.KEYDOWN, key=p.K_y))
        self.assertTrue(env.render_frame())
        self.assertFalse(renderer.show_yields)
        self.assertEqual(tuple(renderer.surface.get_at(dot))[:3], LAND_COLORS['GRASSLAND'])

    def test_draws_buildings_and_lists_them_in_the_status(self):
        from renderer import GRANARY_COLOR, MONUMENT_COLOR
        self.env.render_frame()
        self.assertNotIn('Monument', self.env._renderer.status_text())
        env = Game(render=True, config=GameConfig(unit_count=1, walls='ancient', buildings=('monument', 'granary')))
        env.reset(seed=7)
        self.addCleanup(env.close)
        self.assertTrue(env.render_frame())
        renderer = env._renderer
        size = renderer.size
        self.assertIn('Monument, Granary (+1 food, +2 culture, +2 housing)', renderer.status_text())
        for name, height, color in (('monument', .1, MONUMENT_COLOR), ('granary', .05, GRANARY_COLOR)):
            x, base = renderer.building_spot(name)
            with self.subTest(building=name):
                self.assertEqual(tuple(renderer.surface.get_at((int(x), int(base - size * height))))[:3], color)

    def test_draws_territory_worked_tiles_and_economy_panel(self):
        from renderer import CITIZEN_COLOR, TERRITORY_COLOR
        env = Game(render=True, config=GameConfig(unit_count=1, economy=True, city_population=2))
        env.reset(seed=7)
        self.addCleanup(env.close)
        env.units[0].x, env.units[0].y = 0, 0
        self.assertTrue(env.render_frame())
        renderer = env._renderer
        worked = env.economy.worked_tiles()
        self.assertEqual(len(worked), 2)
        for position in worked:
            center = renderer.citizen_marker(position)
            self.assertEqual(tuple(renderer.surface.get_at((int(center[0]), int(center[1]))))[:3], CITIZEN_COLOR)
        # The ring's outer edge is outlined; the edge shared with the centre is not.
        x, y = env.city.position
        polygon = renderer._polygon((x + 1, y))  # East neighbour: its east side faces outside.
        cx = sum(px for px, _ in polygon) / 6
        cy = sum(py for _, py in polygon) / 6
        (x1, y1), (x2, y2) = ((cx + (px - cx) * .9, cy + (py - cy) * .9) for px, py in (polygon[1], polygon[2]))
        self.assertEqual(tuple(renderer.surface.get_at((int((x1 + x2) / 2), int((y1 + y2) / 2))))[:3], TERRITORY_COLOR)
        self.assertGreater(renderer.panel_height, 118)
        self.assertIn('Pop 2', env.economy.status_text())

    def test_draws_captured_cities_and_domination_victory(self):
        from renderer import WALL_COLOR
        env = Game(render=True, config=GameConfig(city_count=3, walls='ancient'))
        env.reset(seed=7)
        self.addCleanup(env.close)
        env.cities[1].death()
        self.assertTrue(env.render_frame())
        renderer = env._renderer
        self.assertIn('Cities captured 1/3', renderer.status_text())
        # Every city draws its own walls.
        for city in env.cities:
            x, y = renderer._wall_ring(city)[0]
            self.assertEqual(tuple(renderer.surface.get_at((int(x), int(y))))[:3], WALL_COLOR)
        for city in env.cities:
            city.death()
        env.step(env.encode_action(['SPACE'] * 3))
        self.assertEqual(env.last_info['outcome'], 'win')
        self.assertTrue(env.render_frame())
        self.assertIn('DOMINATION VICTORY', renderer.status_text())

    def test_draws_capital_raiders_and_capital_loss(self):
        from renderer import RAIDER_COLOR
        env = Game(render=True, config=GameConfig(player_capital=True))
        env.reset(seed=7)
        self.addCleanup(env.close)
        free = next(position for position in sorted(env.map.grid) if env.occupant(position) is None)
        env.raiders.append(C_Raider(*free, name_instance='Raider 1', strength=20, env=env))
        self.assertTrue(env.render_frame())
        renderer = env._renderer
        corner = renderer._polygon(free)[1]
        self.assertEqual(tuple(renderer.surface.get_at((int(corner[0]) - 1, int(corner[1]))))[:3], RAIDER_COLOR)
        self.assertIn('Capital 100/100', renderer.status_text())
        env.capital.death()
        env.step(env.encode_action(['SPACE'] * 3))
        self.assertTrue(env.last_info['capital_lost'])
        self.assertTrue(env.render_frame())
        self.assertIn('CAPITAL LOST', renderer.status_text())

    def test_draws_districts_civilians_farms_founded_cities_and_status_icons(self):
        from renderer import BLOCKED_COLOR, DISTRICT_STYLES, FOUNDED_COLOR
        env = Game(render=True, config=GameConfig(
            unit_count=1, economy=True, city_population=4, districts=('campus', 'encampment'),
            buildings=('monument', 'library', 'barracks'), city_builders=1, city_settlers=1,
            city_governor='pingala', city_pantheon='fertility_rites', city_war_weariness=500))
        env.reset(seed=7)
        self.addCleanup(env.close)
        economy = env.economy
        blocked = sorted(economy.workable_tiles())[0]
        env.units[0].x, env.units[0].y = blocked
        economy.farms.add(sorted(economy.workable_tiles())[0])
        economy.founded_cities.append((0, 0))
        self.assertTrue(env.render_frame())
        renderer = env._renderer
        surface = renderer.surface
        for district in ('campus', 'encampment'):
            corner = renderer._polygon(economy.districts[district])[1]
            with self.subTest(district=district):
                self.assertEqual(tuple(surface.get_at((int(corner[0]) - 1, int(corner[1]) + 1)))[:3],
                                 DISTRICT_STYLES[district][1])
        corner = renderer._polygon((0, 0))[1]
        self.assertEqual(tuple(surface.get_at((int(corner[0]) - 1, int(corner[1]) + 1)))[:3], FOUNDED_COLOR)
        center = renderer.citizen_marker(blocked)
        self.assertEqual(tuple(surface.get_at((int(center[0]), int(center[1]))))[:3], BLOCKED_COLOR)
        text = economy.status_text()
        for expected in ('Pop 4', 'Amenities', 'Loyalty', 'War weariness 500', 'Science', 'Pingala, establishing',
                         'toward Fertility rites'):
            self.assertIn(expected, text)
        self.assertIn('Civilians captured 0', renderer.status_text())
        # Every status field with an icon draws it; a turn later the panel still fits.
        icon_names = [icon for icon, _ in renderer.status_fields() + economy.status_fields() if icon]
        self.assertIn('mood_displeased', icon_names)
        for name in icon_names:
            renderer.icons.get(name, 18)
        env.step(env.encode_action(['SPACE']))
        self.assertTrue(env.render_frame())

    def test_human_slinger_shoots_adjacent_city(self):
        env = Game(render=True, human=True, config=GameConfig(unit_count=1, unit_types=('slinger',)))
        env.reset(seed=7)
        self.addCleanup(env.close)
        env.render_frame()
        renderer = env._renderer
        p = renderer.pygame
        env.units[0].x, env.units[0].y = 5, 4
        for key in [p.K_f, p.K_ESCAPE]:
            p.event.post(p.event.Event(p.KEYDOWN, key=key))
        renderer.human_loop()
        self.assertIn('ranged_attack', [e['type'] for e in env.events])

    def test_human_can_change_unit_and_shoot(self):
        self.env.human = True
        self.env.render_frame()
        renderer = self.env._renderer
        p = renderer.pygame
        self.env.units[1].x, self.env.units[1].y = 6, 4
        for key in [p.K_TAB, p.K_f, p.K_ESCAPE]:
            p.event.post(p.event.Event(p.KEYDOWN, key=key))
        renderer.human_loop()
        self.assertEqual(renderer.selected_unit, 1)
        self.assertIn('ranged_attack', [e['type'] for e in self.env.events])


@unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional pygame not installed')
class EmpireRendererTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy', 'PYGAME_HIDE_SUPPORT_PROMPT': '1',
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_draws_borders_roads_units_ships_and_both_players_status(self):
        import random
        from empire import EmpireConfig, EmpireGame
        from renderer import PLAYER_COLORS, ROAD_COLOR, EmpireRenderer
        env = EmpireGame(EmpireConfig(map_type='islands', turn_limit=40), seed=3, render=True)
        env.reset(seed=3)
        self.addCleanup(env.close)
        rng = random.Random(3)
        for _ in range(25):
            env.step(env.guided_action(rng))
        self.assertTrue(env.render_frame())
        renderer = env._renderer
        self.assertIsInstance(renderer, EmpireRenderer)
        surface = renderer.surface
        city = env.player_cities(0)[0]
        polygon = renderer._polygon(city.position)
        (x1, y1), (x2, y2) = polygon[3], polygon[4]  # The city's lower-left side carries its owner's outline.
        self.assertEqual(tuple(surface.get_at((round((x1 + x2) / 2), round((y1 + y2) / 2))))[:3], PLAYER_COLORS[0])
        # A road between two non-city tiles is drawn through the middle of their shared side.
        occupied = set(env.city_at) | {u.position for u in env.units}
        first, second = next((p, q) for p in sorted(env.map.grid) for q in env._neighbors(p)
                             if env._land(p) and env._land(q) and not {p, q} & occupied
                             and p not in env.resources and q not in env.resources)
        env.roads |= {first, second}
        renderer.draw()
        (x1, y1), (x2, y2) = renderer._center(first), renderer._center(second)
        self.assertEqual(tuple(surface.get_at((int((x1 + x2) / 2), int((y1 + y2) / 2))))[:3], ROAD_COLOR)
        galley = env._add_unit(1, 'galley', next(p for p in sorted(env.map.grid)
                                                if env.map.grid[p].terrain_type == 'COAST'
                                                and p not in env.military_at))
        renderer.draw()
        cx, cy = renderer._position(galley.position)
        self.assertEqual(tuple(surface.get_at((int(cx + renderer.size * .5), int(cy + renderer.size * .5))))[:3],
                         PLAYER_COLORS[1])
        p = renderer.pygame
        p.event.post(p.event.Event(p.KEYDOWN, key=p.K_ESCAPE))
        self.assertFalse(env.render_frame())
        self.assertTrue(env.quit)
        self.assertIn('Weasels', env.status_text(1))


if __name__ == '__main__':
    unittest.main()
