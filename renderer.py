"""Optional Pygame display and playback controls for the simulation."""

import time

import constants

# Warriors share the plain unit sprite; other types get a short coloured label.
DISTRICT_STYLES = {
    'campus': ('CAMPUS', (95, 150, 230)),
    'holy_site': ('HOLY SITE', (240, 225, 150)),
    'encampment': ('ENCAMP', (225, 95, 80)),
}
UNIT_LABELS = {
    'spearman': ('SPEARMAN', (170, 215, 255)),
    'heavy_chariot': ('CHARIOT', (235, 160, 235)),
    'slinger': ('SLINGER', (255, 190, 130)),
    'archer': ('ARCHER', (255, 235, 140)),
    'battering_ram': ('RAM', (200, 200, 200)),
}


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
        # A longer status line (buildings and production) wraps onto extra rows.
        self.panel = 118 + (44 if env.config.builds else 0)
        height = int((env.config.height * .75 + .25) * self.size) + 2 * self.margin + self.panel
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
        s = self.size
        for pos in self.env.farms:
            # Furrows mark a farm.
            x, y = self._position(pos)
            for offset in (.62, .7, .78):
                p.draw.line(self.surface, (214, 186, 92), (x + s * .3, y + s * offset), (x + s * .7, y + s * offset), 2)
        for pos in self.env.founded_cities:
            x, y = self._position(pos)
            p.draw.polygon(self.surface, constants.COLOR_PURPLE, self._polygon(pos), 3)
            p.draw.rect(self.surface, constants.COLOR_PURPLE, (x + s * .35, y + s * .4, s * .3, s * .25))
            self._text('NEW CITY', (x + 2, y + s * .7), (235, 220, 255), small=True)
        for district, pos in self.env.districts.items():
            label, color = DISTRICT_STYLES[district]
            x, y = self._position(pos)
            p.draw.polygon(self.surface, color, self._polygon(pos), 4)
            p.draw.rect(self.surface, color, (x + s * .38, y + s * .3, s * .24, s * .24), 0 if district == 'encampment' else 2)
            self._text(label, (x + 4, y + s * .58), color, small=True)
        report = self.env.city_report()
        show_tiles = report is not None and self.env.city.alive
        if show_tiles:
            for pos in self.env.territory():
                p.draw.polygon(self.surface, constants.COLOR_PURPLE, self._polygon(pos), 2)
        for civilian in self.env.civilians:
            if not civilian.alive:
                continue
            x, y = self._position(civilian.position)
            center = (int(x + s / 2), int(y + s * .45))
            p.draw.circle(self.surface, constants.COLOR_PURPLE, center, max(6, int(s * .22)))
            p.draw.circle(self.surface, (40, 30, 50), center, max(6, int(s * .22)), 2)
            letter = 'B' if civilian.kind == 'builder' else 'S'
            glyph = self.font.render(letter, True, (40, 30, 50))
            self.surface.blit(glyph, glyph.get_rect(center=center))
            charges = f' ({civilian.charges})' if civilian.kind == 'builder' else ''
            self._text(civilian.name_instance + charges, (x + 2, y + s * .7), small=True)
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
            self._text(f'{obj.name_instance} {obj.hp:.0f}', (x + 2, y + self.size * .7), small=True)
        if show_tiles:
            # Drawn over units: gold marks a worked tile, red a tile a unit is blocking.
            for positions, color in ((report['worked_tiles'], (240, 200, 80)),
                                     (report['blocked_tiles'], constants.COLOR_RED)):
                for pos in positions:
                    x, y = self._position(pos)
                    center = (int(x + self.size * .82), int(y + self.size * .3))
                    p.draw.circle(self.surface, color, center, max(3, self.size // 14))
                    p.draw.circle(self.surface, (30, 30, 30), center, max(3, self.size // 14), 1)
        if self.env.human:
            unit = self.env.units[self.selected_unit]
            p.draw.polygon(self.surface, (255, 230, 100), self._polygon(unit.position), 3)
        y = self.surface.get_height() - self.panel + 9
        city = self.env.city
        outcome = self.env.last_info['outcome']
        status = [f'Turn {self.env.turn_number}', f'City {city.hp:.0f}/{city.hp_max:.0f}',
                  f'Walls {city.wall_hp:.0f}']
        if report is not None:
            status += [f'Pop {report["population"]}',
                       f'Food {report["food"]}/{report["growth_threshold"]} ({report["food_surplus"]:+d})']
        civilians = self.env.civilian_report()
        if civilians is not None:
            status += [f'Captured {civilians["captured"]}', f'Founded {len(civilians["founded_cities"])}']
        status.append(outcome.upper())
        if self.env.buildings:
            status.append('Built: ' + ', '.join(name.replace('_', ' ') for name in self.env.buildings))
        if report is not None and report.get('producing'):
            status.append(f'Producing {report["producing"].replace("_", " ")} '
                          f'{report["production_progress"]}/{report["production_cost"]} '
                          f'(+{report["production_yield"]})')
        lines = ['']
        for part in status:
            proposed = (lines[-1] + '   ' + part).strip()
            if lines[-1] and self.font.size(proposed)[0] > self.surface.get_width() - 2 * self.margin:
                lines.append(part)
            else:
                lines[-1] = proposed
        for index, line in enumerate(lines):
            self._text(line, (self.margin, y + 22 * index))
        y += 22 * (len(lines) - 1)
        if self.env.human:
            controls = 'Tab: unit   Q/E/A/D/Z/X: move   Space: heal   F: shoot (archer/slinger)   R: restart   Esc: quit'
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
