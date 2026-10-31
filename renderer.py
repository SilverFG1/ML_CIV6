"""Optional Pygame display and playback controls for the simulation."""

import math
import time

import constants
from class_hex import direction_between

# Warriors share the plain unit sprite; other types get a short coloured label.
UNIT_LABELS = {
    'spearman': ('SPEARMAN', (170, 215, 255)),
    'heavy_chariot': ('CHARIOT', (235, 160, 235)),
    'slinger': ('SLINGER', (255, 190, 130)),
    'archer': ('ARCHER', (255, 235, 140)),
    'battering_ram': ('RAM', (200, 200, 200)),
}
RIVER_COLOR = (64, 140, 225)
# Polygon corner indices (from _polygon) for the side facing each neighbour.
SIDE_CORNERS = {'NE': (0, 1), 'E': (1, 2), 'SE': (2, 3), 'SW': (3, 4), 'W': (4, 5), 'NW': (5, 0)}
LAND_COLORS = {'GRASSLAND': (107, 137, 78), 'PLAINS': (158, 150, 84), 'DESERT': (214, 190, 126),
               'TUNDRA': (122, 128, 106), 'SNOW': (224, 229, 236)}
WATER_COLORS = {'COAST': (80, 140, 192), 'OCEAN': (40, 78, 130)}
MOUNTAIN_COLOR, PEAK_COLOR = (99, 109, 121), (165, 175, 185)
FOREST_COLOR = (42, 89, 56)
YIELD_COLORS = {'food': (104, 200, 88), 'production': (214, 130, 56), 'gold': (240, 206, 72),
                'science': (90, 160, 235), 'culture': (190, 110, 220), 'faith': (236, 236, 236)}
WALL_COLOR = (196, 184, 156)
RUINED_WALL_COLOR = (112, 102, 90)
WALL_OUTLINE = (70, 62, 52)


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
        self.show_yields = True
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
        for cell in self.env.map.grid.values():
            self._draw_tile(cell)
        self._draw_rivers()
        if self.show_yields:
            for cell in self.env.map.grid.values():
                self._draw_yields(cell)
        for obj in self.env.objects:
            x, y = self._position(obj.position)
            if obj.alive:
                image = self.images['city' if obj is self.env.city else 'unit']
                self.surface.blit(image, (x, y - self.size // 6))
                label = UNIT_LABELS.get(getattr(obj, 'unit_type', None))
                if label is not None:
                    self._text(label[0], (x + 2, y + 2), label[1], small=True)
            else:
                s = self.size
                p.draw.line(self.surface, constants.COLOR_RED, (x + s / 3, y + s / 3),
                            (x + 2 * s / 3, y + 2 * s / 3), 3)
                p.draw.line(self.surface, constants.COLOR_RED, (x + 2 * s / 3, y + s / 3),
                            (x + s / 3, y + 2 * s / 3), 3)
            if obj is self.env.city:
                self._draw_walls()
            self._text(f'{obj.name_instance} {obj.hp:.0f}', (x + 2, y + self.size * .7), small=True)
        if self.env.human:
            unit = self.env.units[self.selected_unit]
            p.draw.polygon(self.surface, (255, 230, 100), self._polygon(unit.position), 3)
        y = self.surface.get_height() - 109
        city = self.env.city
        outcome = self.env.last_info['outcome']
        walls = (f'Ancient walls {city.wall_hp:.0f}/{city.wall_hp_max:.0f}'
                 if self.env.config.walls == 'ancient' else f'Walls {city.wall_hp:.0f}')
        self._text(f'Turn {self.env.turn_number}   City {city.hp:.0f}/{city.hp_max:.0f}   '
                   f'{walls}   {outcome.upper()}', (self.margin, y))
        if self.env.human:
            controls = ('Tab: unit   Q/E/A/D/Z/X: move   Space: heal   F: shoot (archer/slinger)   '
                        'Y: yields   R: restart   Esc: quit')
        else:
            controls = (f'Space: pause   Right: one turn   +/-: speed   Y: yields   Esc: quit   '
                        f'Delay {self.delay_ms or 0}ms')
        control_lines = self._wrapped_text(controls, (self.margin, y + 27))
        if self.paused:
            self._text('PAUSED', (self.margin, y + 31 + 22 * control_lines), (255, 220, 100))
        p.display.flip()

    def _draw_tile(self, cell):
        p, s = self.pygame, self.size
        polygon = self._polygon(cell.index)
        x, y = self._position(cell.index)
        terrain = cell.spec
        if terrain.water:
            color = WATER_COLORS[cell.terrain_type]
        elif terrain.land_type is None or terrain.elevation == 'MOUNTAIN':
            color = MOUNTAIN_COLOR
        else:
            color = LAND_COLORS[terrain.land_type]
        if cell.feature == 'FOREST':
            color = tuple(round(a * .3 + b * .7) for a, b in zip(color, FOREST_COLOR))
        p.draw.polygon(self.surface, color, polygon)
        p.draw.polygon(self.surface, (44, 64, 48), polygon, 1)
        shade = tuple(round(c * .62) for c in color)
        if terrain.elevation == 'MOUNTAIN':
            p.draw.polygon(self.surface, PEAK_COLOR, [(x + s * .2, y + s * .7), (x + s * .5, y + s * .2),
                                                      (x + s * .8, y + s * .7)])
        elif terrain.elevation == 'HILLS':
            for left in (.16, .48):
                p.draw.arc(self.surface, shade, p.Rect(x + s * left, y + s * .5, s * .36, s * .3),
                           0, math.pi, max(2, s // 24))
        if cell.feature == 'FOREST':
            for tx, ty in ((.3, .42), (.5, .34), (.7, .42)):
                p.draw.polygon(self.surface, shade, [(x + s * (tx - .07), y + s * (ty + .12)),
                                                     (x + s * tx, y + s * (ty - .02)),
                                                     (x + s * (tx + .07), y + s * (ty + .12))])

    def _draw_yields(self, cell):
        """One small dot per yield point near the top of the tile."""
        dots = [YIELD_COLORS[name] for name, amount in cell.yields.items() for _ in range(amount)]
        if not dots:
            return
        x, y = self._position(cell.index)
        radius = max(2, self.size // 19)
        step = 2.5 * radius
        left = x + self.size / 2 - step * (len(dots) - 1) / 2
        for index, color in enumerate(dots):
            center = (left + index * step, y + self.size * .2)
            self.pygame.draw.circle(self.surface, color, center, radius)
            self.pygame.draw.circle(self.surface, (30, 30, 30), center, radius, 1)

    def _draw_rivers(self):
        width = max(3, self.size // 12)
        for edge in sorted(self.env.map.rivers, key=sorted):
            origin, neighbor = sorted(edge)
            polygon = self._polygon(origin)
            start, end = (polygon[i] for i in SIDE_CORNERS[direction_between(origin, neighbor)])
            self.pygame.draw.line(self.surface, RIVER_COLOR, start, end, width)
            for point in (start, end):  # Round the joints between river segments.
                self.pygame.draw.circle(self.surface, RIVER_COLOR, point, width / 2)

    def _wall_ring(self):
        """Corners of the city's walls, just inside its tile."""
        polygon = self._polygon(self.env.city.position)
        cx = sum(x for x, _ in polygon) / 6
        cy = sum(y for _, y in polygon) / 6
        return [(cx + (x - cx) * .9, cy + (y - cy) * .9) for x, y in polygon]

    def _draw_walls(self):
        """A stone ring with corner towers; broken walls leave only ruined towers."""
        city = self.env.city
        if city.wall_hp_max <= 0:
            return
        ring = self._wall_ring()
        intact = city.wall_hp > 0
        color = WALL_COLOR if intact else RUINED_WALL_COLOR
        if intact:
            self.pygame.draw.polygon(self.surface, WALL_OUTLINE, ring, max(5, self.size // 10))
            self.pygame.draw.polygon(self.surface, color, ring, max(3, self.size // 14))
        half = max(3, self.size // 16)
        for x, y in ring:
            tower = self.pygame.Rect(int(x - half), int(y - half), 2 * half, 2 * half)
            self.pygame.draw.rect(self.surface, color, tower)
            self.pygame.draw.rect(self.surface, WALL_OUTLINE, tower, 1)

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
                elif event.key == p.K_y:
                    self.show_yields = not self.show_yields
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
                    elif event.key == p.K_y:
                        self.show_yields = not self.show_yields
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
