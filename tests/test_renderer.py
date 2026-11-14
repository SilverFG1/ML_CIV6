"""Optional display tests run with SDL's in-memory video driver."""

import importlib.util
import os
import unittest
from unittest.mock import patch

from game import Game, GameConfig


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


    def test_draws_city_growth_with_worked_and_blocked_tiles(self):
        env = Game(render=True, config=GameConfig(unit_count=2, city_population=3))
        env.reset(seed=7)
        self.addCleanup(env.close)
        env.units[0].x, env.units[0].y = 5, 4
        self.assertEqual(env.city_report()['blocked_tiles'], [[5, 4]])
        self.assertTrue(env.render_frame())
        env.city.death()
        self.assertTrue(env.render_frame())


    def test_draws_civilians_farms_and_founded_cities(self):
        env = Game(render=True, config=GameConfig(unit_count=1, city_population=2, city_builders=1, city_settlers=1))
        env.reset(seed=7)
        self.addCleanup(env.close)
        env.farms.add(env.territory()[0])
        env.founded_cities.append((0, 0))
        self.assertTrue(env.render_frame())
        env.civilians[0].leave('captured', 'civilian_captured')
        self.assertTrue(env.render_frame())


    def test_draws_districts_buildings_and_production(self):
        env = Game(render=True, config=GameConfig(
            unit_count=1, city_population=4, city_districts=('campus', 'encampment'),
            city_buildings=('monument', 'library', 'ancient_walls'), city_build_queue=('granary', 'barracks')))
        env.reset(seed=7)
        self.addCleanup(env.close)
        self.assertTrue(env.render_frame())
        self.assertGreater(env._renderer.panel, 118)  # room for the buildings and production line


if __name__ == '__main__':
    unittest.main()
