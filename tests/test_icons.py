"""The display's original icons are drawn in code and cover the game's content."""

import importlib.util
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import game
import icons


@unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional pygame not installed')
class IconTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'PYGAME_HIDE_SUPPORT_PROMPT': '1'})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        import pygame
        self.pygame = pygame
        self.icons = icons.IconSet(pygame)

    def test_every_icon_draws_visibly_at_small_and_large_sizes(self):
        for name in icons.ICONS:
            for size in (8, 18, 48):
                with self.subTest(name=name, size=size):
                    surface = self.icons.get(name, size)
                    self.assertEqual(surface.get_size(), (size, size))
                    covered = self.pygame.mask.from_surface(surface).count()
                    self.assertGreater(covered, size * size * .1)
                    self.assertLess(covered, size * size)  # icons have transparent corners

    def test_icons_cover_every_unit_district_building_civilian_and_mood(self):
        names = set(icons.ICONS)
        expected = ({f'unit_{name}' for name in game.UNIT_TYPES}
                    | {f'district_{name}' for name in game.DISTRICT_TYPES}
                    | {f'building_{name}' for name in game.BUILD_ITEMS if name not in game.DISTRICT_TYPES}
                    | {f'civilian_{name}' for name in game.CIVILIAN_KINDS}
                    | {f'mood_{mood}' for _, mood, _, _ in game.AMENITY_MOODS})
        self.assertEqual(expected - names, set())

    def test_icons_are_cached_and_unknown_names_are_rejected(self):
        self.assertIs(self.icons.get('food', 18), self.icons.get('food', 18))
        self.assertIsNot(self.icons.get('food', 18), self.icons.get('food', 24))
        with self.assertRaisesRegex(KeyError, 'gold'):
            self.icons.get('gold', 18)


class HeadlessIconTests(unittest.TestCase):
    def test_icon_module_does_not_import_pygame(self):
        script = "import icons, sys; assert 'pygame' not in sys.modules"
        result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
