"""Optional Pygame display and playback controls for the simulation."""

import time

import constants


class Renderer:
    def __init__(self, env):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError('Rendering requires: python -m pip install -r requirements-render.txt') from exc
        self.pygame = pygame
        self.env = env
        pygame.display.init()
        pygame.font.init()
        self.size = max(20, min(96, int(1180 / (env.config.width + .5)),
                               int(720 / (env.config.height * .75 + .25))))
        self.margin = 24
        width = max(640, int((env.config.width + .5) * self.size) + 2 * self.margin)
        height = int((env.config.height * .75 + .25) * self.size) + 2 * self.margin + 118
        self.surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption('ML_CIV6 — training and evaluation')
        self.font = pygame.font.SysFont('Segoe UI', 16)
        self.small_font = pygame.font.SysFont('Segoe UI', max(11, min(16, self.size // 5)))
        self.clock = pygame.time.Clock()
        self.paused = False
        self.delay_ms = None
        self._open = True
        self.selected_unit = 0
        self.images = {}
        for name, path in [('unit', 'data/Otter_Warrior.png'), ('city', 'data/Ottertopia.png')]:
            loaded = pygame.image.load(constants.resource_path(path)).convert_alpha()
            self.images[name] = pygame.transform.smoothscale(loaded, (self.size, self.size))

    def _position(self, position):
        x, y = position
        return (int(self.margin + (x + .5 * (y % 2)) * self.size),
                int(self.margin + y * self.size * .75))

    def _polygon(self, position):
        x, y = self._position(position)
        s = self.size
        return [(x + s / 2, y), (x + s, y + s / 4), (x + s, y + 3 * s / 4),
                (x + s / 2, y + s), (x, y + 3 * s / 4), (x, y + s / 4)]

    def _text(self, text, pos, color=(235, 235, 235), small=False):
        font = self.small_font if small else self.font
        self.surface.blit(font.render(text, True, color), pos)

    def _wrapped_text(self, text, pos):
        lines = ['']
        for word in text.split():
            proposed = (lines[-1] + ' ' + word).strip()
            if lines[-1] and self.font.size(proposed)[0] > self.surface.get_width() - 2 * self.margin:
                lines.append(word)
            else:
                lines[-1] = proposed
        for index, line in enumerate(lines):
            self._text(line, (pos[0], pos[1] + 22 * index))
        return len(lines)

    def draw(self):
        p = self.pygame
        self.surface.fill(constants.COLOR_DEFAULT_BG)
        colors = {'GRASSLAND': (107, 137, 78), 'FOREST': (42, 89, 56),
                  'MOUNTAIN': (99, 109, 121), 'EDGE': (99, 109, 121)}
        for cell in self.env.map.grid.values():
            polygon = self._polygon(cell.index)
            p.draw.polygon(self.surface, colors[cell.terrain_type], polygon)
            p.draw.polygon(self.surface, (44, 64, 48), polygon, 1)
            if cell.terrain_type == 'MOUNTAIN':
                x, y = self._position(cell.index)
                s = self.size
                p.draw.polygon(self.surface, (165, 175, 185),
                               [(x + s * .2, y + s * .7), (x + s * .5, y + s * .2),
                                (x + s * .8, y + s * .7)])
        for obj in self.env.objects:
            x, y = self._position(obj.position)
            if obj.alive:
                image = self.images['city' if obj is self.env.city else 'unit']
                self.surface.blit(image, (x, y - self.size // 6))
                if getattr(obj, 'unit_type', None) == 'archer':
                    self._text('ARCHER', (x + 2, y + 2), (255, 235, 140), small=True)
            else:
                s = self.size
                p.draw.line(self.surface, constants.COLOR_RED, (x + s / 3, y + s / 3),
                            (x + 2 * s / 3, y + 2 * s / 3), 3)
                p.draw.line(self.surface, constants.COLOR_RED, (x + 2 * s / 3, y + s / 3),
                            (x + s / 3, y + 2 * s / 3), 3)
            self._text(f'{obj.name_instance} {obj.hp:.0f}', (x + 2, y + self.size * .7), small=True)
        if self.env.human:
            unit = self.env.units[self.selected_unit]
            p.draw.polygon(self.surface, (255, 230, 100), self._polygon(unit.position), 3)
        y = self.surface.get_height() - 109
        city = self.env.city
        outcome = self.env.last_info['outcome']
        self._text(f'Turn {self.env.turn_number}   City {city.hp:.0f}/{city.hp_max:.0f}   '
                   f'Walls {city.wall_hp:.0f}   {outcome.upper()}', (self.margin, y))
        if self.env.human:
            controls = 'Tab: unit   Q/E/A/D/Z/X: move   Space: heal   F: shoot   R: restart   Esc: quit'
        else:
            controls = f'Space: pause   Right: one turn   +/-: speed   Esc: quit   Delay {self.delay_ms or 0}ms'
        control_lines = self._wrapped_text(controls, (self.margin, y + 27))
        if self.paused:
            self._text('PAUSED', (self.margin, y + 31 + 22 * control_lines), (255, 220, 100))
        p.display.flip()

    def _events(self):
        advance = False
        p = self.pygame
        for event in p.event.get():
            if event.type == p.QUIT or (event.type == p.KEYDOWN and event.key == p.K_ESCAPE):
                self._open = False
            elif event.type == p.KEYDOWN:
                if event.key == p.K_SPACE:
                    self.paused = not self.paused
                elif event.key == p.K_RIGHT:
                    advance = True
                elif event.key in (p.K_PLUS, p.K_EQUALS, p.K_KP_PLUS):
                    self.delay_ms = max(0, (self.delay_ms or 0) // 2 - 10)
                elif event.key in (p.K_MINUS, p.K_KP_MINUS):
                    self.delay_ms = min(3000, (self.delay_ms or 0) * 2 + 50)
        return advance

    def frame(self, delay_ms=0):
        if self.delay_ms is None:
            self.delay_ms = max(0, delay_ms)
        started = time.monotonic()
        while self._open:
            advance = self._events()
            self.draw()
            if advance or (not self.paused and (time.monotonic() - started) * 1000 >= self.delay_ms):
                break
            self.clock.tick(60)
        return self._open

    def human_loop(self):
        p = self.pygame
        bindings = {p.K_q: 'NW', p.K_e: 'NE', p.K_a: 'W', p.K_d: 'E',
                    p.K_z: 'SW', p.K_x: 'SE', p.K_SPACE: 'SPACE'}
        while self._open:
            for event in p.event.get():
                if event.type == p.QUIT or (event.type == p.KEYDOWN and event.key == p.K_ESCAPE):
                    self._open = False
                elif event.type == p.KEYDOWN:
                    if event.key == p.K_TAB:
                        self.selected_unit = (self.selected_unit + 1) % len(self.env.units)
                    elif event.key == p.K_r:
                        self.env.reset()
                    elif event.key in bindings or event.key == p.K_f:
                        directions = ['SPACE'] * len(self.env.units)
                        direction = 'SHOOT' if event.key == p.K_f else bindings[event.key]
                        if direction in self.env.unit_directions[self.selected_unit]:
                            directions[self.selected_unit] = direction
                            self.env.step(self.env.encode_action(directions))
            self.draw()
            self.clock.tick(60)

    def close(self):
        self._open = False
        self.pygame.display.quit()
        self.pygame.font.quit()
