"""Optional Pygame display and playback controls for the simulation."""

import time

import constants
from icons import IconSet

# District tiles are outlined in their badge color and labeled under the icon.
DISTRICT_STYLES = {
    'campus': ('CAMPUS', (96, 158, 232)),
    'holy_site': ('HOLY SITE', (228, 168, 58)),
    'encampment': ('ENCAMP', (214, 66, 66)),
}
OUTCOME_COLORS = {'win': (140, 220, 120), 'loss': (240, 110, 100), 'free_city': (200, 170, 235),
                  'timeout': (235, 235, 235), 'ongoing': (190, 196, 204)}
STATUS_LINE = 24  # height of one status-bar row
STATUS_ICON = 18
STATUS_GAP = 16


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
        self.paused = False
        self.delay_ms = None
        self._open = True
        self.selected_unit = 0
        self.font = pygame.font.SysFont('Segoe UI', 16)
        self.small_font = pygame.font.SysFont('Segoe UI', max(11, min(16, self.size // 5)))
        self.icons = IconSet(pygame)
        width = max(640, int((env.config.width + .5) * self.size) + 2 * self.margin)
        self.panel = self._panel_height(width - 2 * self.margin)
        height = int((env.config.height * .75 + .25) * self.size) + 2 * self.margin + self.panel
        self.surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption('ML_CIV6 — training and evaluation')
        self.clock = pygame.time.Clock()
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

    def _icon(self, name, pos, size):
        self.surface.blit(self.icons.get(name, size), (int(pos[0]), int(pos[1])))

    def _controls(self):
        if self.env.human:
            return 'Tab: unit   Q/E/A/D/Z/X: move   Space: heal   F: shoot (archer/slinger)   R: restart   Esc: quit'
        return f'Space: pause   Right: one turn   +/-: speed   Esc: quit   Delay {self.delay_ms or 0}ms'

    def _wrap(self, text, width):
        """Split text into lines no wider than width."""
        lines = ['']
        for word in text.split():
            proposed = (lines[-1] + ' ' + word).strip()
            if lines[-1] and self.font.size(proposed)[0] > width:
                lines.append(word)
            else:
                lines[-1] = proposed
        return lines

    def _status_chips(self):
        """Status-bar entries as (icon name or None, text, color)."""
        env, city = self.env, self.env.city
        report = env.city_report()
        light = (235, 235, 235)
        outcome = env.last_info['outcome']
        chips = [('turn', f'Turn {env.turn_number}', light),
                 (None, outcome.upper(), OUTCOME_COLORS.get(outcome, light)),
                 ('city_hp', f'City {city.hp:.0f}/{city.hp_max:.0f}', light),
                 ('walls', f'Walls {city.wall_hp:.0f}', light)]
        if report is not None:
            chips += [('population', f'Pop {report["population"]}', light),
                      ('food', f'{report["food"]:g}/{report["growth_threshold"]} ({report["growth_surplus"]:+g})', light),
                      ('housing', f'Housing {report["housing"]:g}', light),
                      (f'mood_{report["mood"]}',
                       f'Amenities {report["amenities"]}/{report["amenities_needed"]} {report["mood"]}', light),
                      ('loyalty', f'Loyalty {report["loyalty"]:g} ({report["loyalty_per_turn"]:+g})', light)]
            if report['war_weariness']:
                chips.append(('war_weariness', f'War weariness {report["war_weariness"]}', light))
            if report['border_growth'] or report['tiles_claimed']:
                chips.append(('borders', f'Borders {report["border_culture"]:g}/{report["border_cost"]} '
                                         f'(+{report["border_growth"]:g})', light))
            for kind in ('culture', 'science'):
                if report[kind]:
                    chips.append((kind, f'{kind.capitalize()} +{report[kind]:g}', light))
            if 'pantheon' in report:
                belief = report['pantheon'].replace('_', ' ')
                chips.append(('pantheon', belief.capitalize(), light) if report['pantheon_founded'] else
                             ('faith', f'{report["faith_stored"]:g}/25 (+{report["faith"]:g}) toward {belief}', light))
            elif report['faith']:
                chips.append(('faith', f'Faith +{report["faith"]:g}', light))
            if 'governor' in report:
                text = report['governor'].capitalize()
                if report['governor_promotion']:
                    text += f' ({report["governor_promotion"].replace("_", " ")})'
                if not report['governor_established']:
                    turns = report['turns_to_establish']
                    text += f', establishing: {turns} turn{"" if turns == 1 else "s"}'
                chips.append(('governor', text, light))
            if report.get('producing'):
                chips.append(('production', f'Producing {report["producing"].replace("_", " ")} '
                                            f'{report["production_progress"]:g}/{report["production_cost"]} '
                                            f'(+{report["production_yield"]:g})', light))
        civilians = env.civilian_report()
        if civilians is not None:
            chips += [('captured', f'Captured {civilians["captured"]}', light),
                      ('settlement', f'Founded {len(civilians["founded_cities"])}', light)]
        chips += [(f'building_{name}', name.replace('_', ' ').capitalize(), light) for name in env.buildings]
        return chips

    def _layout(self, chips, width):
        """Place chips left to right, wrapping at width; returns [(row, x, chip)] and the row count."""
        placed, row, x = [], 0, 0
        for chip in chips:
            icon, text, _ = chip
            chip_width = self.font.size(text)[0] + (STATUS_ICON + 4 if icon else 0)
            if x and x + chip_width > width:
                row, x = row + 1, 0
            placed.append((row, x, chip))
            x += chip_width + STATUS_GAP
        return placed, row + 1

    def _panel_height(self, width):
        """Room for today's status bar plus a spare row, as values and buildings grow."""
        _, rows = self._layout(self._status_chips(), width)
        rows += 1 + bool(self.env.config.builds)
        return 10 + STATUS_LINE * rows + 6 + 22 * len(self._wrap(self._controls(), width)) + 22 + 8

    def _wrapped_text(self, text, pos):
        lines = self._wrap(text, self.surface.get_width() - 2 * self.margin)
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
            x, y = self._position(pos)
            self._icon('farm', (x + s * .31, y + s * .42), s * .38)
        for pos in self.env.founded_cities:
            x, y = self._position(pos)
            p.draw.polygon(self.surface, constants.COLOR_PURPLE, self._polygon(pos), 3)
            self._icon('settlement', (x + s * .27, y + s * .14), s * .46)
            self._text('NEW CITY', (x + 2, y + s * .66), (235, 220, 255), small=True)
        report = self.env.city_report()
        show_tiles = report is not None and self.env.city.alive
        if show_tiles:
            # The city's borders: every tile it owns, worked or not.
            for pos in self.env.owned_tiles:
                p.draw.polygon(self.surface, constants.COLOR_PURPLE, self._polygon(pos), 2)
        for district, pos in self.env.districts.items():
            label, color = DISTRICT_STYLES[district]
            x, y = self._position(pos)
            p.draw.polygon(self.surface, color, self._polygon(pos), 4)
            self._icon(f'district_{district}', (x + s * .29, y + s * .1), s * .42)
            self._text(label, (x + 4, y + s * .56), color, small=True)
        for civilian in self.env.civilians:
            if not civilian.alive:
                continue
            x, y = self._position(civilian.position)
            self._icon(f'civilian_{civilian.kind}', (x + s * .26, y + s * .16), s * .48)
            charges = f' ({civilian.charges})' if civilian.kind == 'builder' else ''
            self._text(civilian.name_instance + charges, (x + 2, y + s * .7), small=True)
        for obj in self.env.objects:
            x, y = self._position(obj.position)
            if obj.alive:
                image = self.images['city' if obj is self.env.city else 'unit']
                self.surface.blit(image, (x, y - self.size // 6))
                if obj is self.env.city:
                    if obj.wall_hp > 0:
                        self._icon('walls', (x + s * .64, y + s * .1), s * .3)
                else:
                    self._icon(f'unit_{obj.unit_type}', (x + s * .04, y + s * .18), s * .3)
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
        top = self.surface.get_height() - self.panel + 10
        placed, rows = self._layout(self._status_chips(), self.surface.get_width() - 2 * self.margin)
        for row, x, (icon, text, color) in placed:
            x, y = self.margin + x, top + row * STATUS_LINE
            if icon:
                self._icon(icon, (x, y + 1), STATUS_ICON)
                x += STATUS_ICON + 4
            self._text(text, (x, y), color)
        y = top + rows * STATUS_LINE + 6
        control_lines = self._wrapped_text(self._controls(), (self.margin, y))
        if self.paused:
            self._text('PAUSED', (self.margin, y + 22 * control_lines), (255, 220, 100))
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
