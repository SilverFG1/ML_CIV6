"""Optional display tests run with SDL's in-memory video driver."""

import importlib.util
import os
import unittest
from unittest.mock import patch

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

    def test_draws_captured_cities_and_domination_victory(self):
        env = Game(render=True, config=GameConfig(city_count=3, wall_hp=20))
        env.reset(seed=7)
        self.addCleanup(env.close)
        env.cities[1].death()
        self.assertTrue(env.render_frame())
        for city in env.cities:
            city.death()
        env.step(env.encode_action(['SPACE'] * 3))
        self.assertEqual(env.last_info['outcome'], 'win')
        self.assertTrue(env.render_frame())

    def test_draws_capital_raiders_and_capital_loss(self):
        env = Game(render=True, config=GameConfig(player_capital=True))
        env.reset(seed=7)
        self.addCleanup(env.close)
        free = next(position for position in sorted(env.map.grid) if env.occupant(position) is None)
        env.raiders.append(C_Raider(*free, name_instance='Raider 1', strength=20, env=env))
        self.assertTrue(env.render_frame())
        env.capital.death()
        env.step(env.encode_action(['SPACE'] * 3))
        self.assertTrue(env.last_info['capital_lost'])
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


if __name__ == '__main__':
    unittest.main()
