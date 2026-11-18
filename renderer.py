"""Optional Pygame display and playback controls for the simulation."""

import math
import time

import constants
from class_hex import direction_between, neighbor_positions

# Warriors share the plain unit sprite; other types get a short coloured label.
UNIT_LABELS = {
    'spearman': ('SPEARMAN', (170, 215, 255)),
    'heavy_chariot': ('CHARIOT', (235, 160, 235)),
    'slinger': ('SLINGER', (255, 190, 130)),
    'archer': ('ARCHER', (255, 235, 140)),
    'battering_ram': ('RAM', (200, 200, 200)),
}
RAIDER_COLOR = (255, 110, 100)
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
TERRITORY_COLOR, CITIZEN_COLOR = (232, 72, 72), (250, 250, 250)
MONUMENT_COLOR, GRANARY_COLOR, GRANARY_ROOF = (214, 208, 196), (206, 164, 92), (150, 104, 56)
# Horizontal position of each building's icon on the city tile.
BUILDING_SPOTS = {'monument': .24, 'granary': .76}
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
        # Text panel under the map; the economy adds a line or two.
        self.panel_height = self._panel_height()
        height = int((env.config.height * .75 + .25) * self.size) + 2 * self.margin + self.panel_height
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

    def _panel_height(self):
        return 118 + (44 if self.env.config.economy else 0)

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

    def _wrapped_text(self, text, pos, separator='   '):
        """Draw text over several lines, breaking only between separator-delimited fields."""
        lines = ['']
        for field in text.split(separator):
            proposed = lines[-1] + separator + field if lines[-1] else field
            if lines[-1] and self.font.size(proposed)[0] > self.surface.get_width() - 2 * self.margin:
                lines.append(field)
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
        if self.env.economy is not None:
            self._draw_territory()
        if self.show_yields:
            for cell in self.env.map.grid.values():
                self._draw_yields(cell)
        if self.env.economy is not None:
            for position in self.env.economy.worked_tiles():
                center = self.citizen_marker(position)
                p.draw.circle(self.surface, CITIZEN_COLOR, center, max(3, self.size // 14))
                p.draw.circle(self.surface, TERRITORY_COLOR, center, max(3, self.size // 14), 2)
        cities = self.env.cities
        capital = self.env.capital
        towns = cities + ([capital] if capital is not None else [])
        raiders = self.env.living_raiders()
        # Cities first, so a unit standing in a captured city is drawn on top.
        for obj in towns + raiders + self.env.units:
            x, y = self._position(obj.position)
            is_city = obj in towns
            if obj in raiders:
                p.draw.polygon(self.surface, RAIDER_COLOR, self._polygon(obj.position), 3)
            if obj.alive or is_city:
                image = self.images['city' if is_city else 'unit']
                self.surface.blit(image, (x, y - self.size // 6))
                label = UNIT_LABELS.get(getattr(obj, 'unit_type', None))
                if obj is capital:
                    label = ('CAPITAL', (150, 200, 255)) if obj.alive else ('LOST', RAIDER_COLOR)
                elif obj in raiders:
                    label = ('RAIDER', RAIDER_COLOR)
                elif is_city and not obj.alive:
                    label = ('CAPTURED', (140, 230, 140))
                if label is not None:
                    self._text(label[0], (x + 2, y + 2), label[1], small=True)
            else:
                s = self.size
                p.draw.line(self.surface, constants.COLOR_RED, (x + s / 3, y + s / 3),
                            (x + 2 * s / 3, y + 2 * s / 3), 3)
                p.draw.line(self.surface, constants.COLOR_RED, (x + 2 * s / 3, y + s / 3),
                            (x + s / 3, y + 2 * s / 3), 3)
            if obj in cities:
                self._draw_walls(obj)
                self._draw_buildings(obj)
            self._text(f'{obj.name_instance} {obj.hp:.0f}', (x + 2, y + self.size * .7), small=True)
        if self.env.human:
            unit = self.env.units[self.selected_unit]
            p.draw.polygon(self.surface, (255, 230, 100), self._polygon(unit.position), 3)
        y = self.surface.get_height() - self.panel_height + 9
        y += 22 * self._wrapped_text(self.status_text(), (self.margin, y))
        if self.env.economy is not None:
            y += 22 * self._wrapped_text(self.env.economy.status_text(), (self.margin, y))
        if self.env.human:
            controls = ('Tab: unit   Q/E/A/D/Z/X: move   Space: heal   F: shoot (archer/slinger)   '
                        'Y: yields   R: restart   Esc: quit')
        else:
            controls = (f'Space: pause   Right: one turn   +/-: speed   Y: yields   Esc: quit   '
                        f'Delay {self.delay_ms or 0}ms')
        control_lines = self._wrapped_text(controls, (self.margin, y + 5))
        if self.paused:
            self._text('PAUSED', (self.margin, y + 9 + 22 * control_lines), (255, 220, 100))
        p.display.flip()

    def status_text(self):
        env = self.env
        city, cities, capital = env.city, env.cities, env.capital
        outcome = env.last_info['outcome']
        if outcome == 'win' and len(cities) > 1:
            outcome = 'DOMINATION VICTORY'
        elif outcome == 'loss' and capital is not None and not capital.alive:
            outcome = 'CAPITAL LOST'
        else:
            outcome = outcome.upper()
        if len(cities) == 1:
            walls = (f'Ancient walls {city.wall_hp:.0f}/{city.wall_hp_max:.0f}'
                     if city.walls == 'ancient' else f'Walls {city.wall_hp:.0f}')
            status = f'City {city.hp:.0f}/{city.hp_max:.0f}   {walls}'
        else:
            status = f'Cities captured {env.cities_captured()}/{len(cities)}'
        if capital is not None:
            status += f'   Capital {capital.hp:.0f}/{capital.hp_max:.0f}'
        text = f'Turn {env.turn_number}   {status}   {outcome}'
        if city.buildings:
            effects = [f'+{amount} {kind}' for kind, amount in city.building_yields.items()]
            if city.building_housing:
                effects.append(f'+{city.building_housing} housing')
            names = ', '.join(name.title() for name in city.buildings)
            text += f'   {names} ({", ".join(effects)})'
        return text

    def citizen_marker(self, position):
        """Where a worked tile shows its citizen marker."""
        x, y = self._position(position)
        return x + self.size / 2, y + self.size * .84

    def _draw_territory(self):
        """Outline the city's territory just inside its edge tiles."""
        territory = self.env.economy.territory
        width = max(2, self.size // 22)
        for position in territory:
            polygon = self._polygon(position)
            cx = sum(x for x, _ in polygon) / 6
            cy = sum(y for _, y in polygon) / 6
            inset = [(cx + (x - cx) * .9, cy + (y - cy) * .9) for x, y in polygon]
            x, y = position
            for direction, (first, second) in SIDE_CORNERS.items():
                dx, dy = constants.movement_delta(direction, y)
                if (x + dx, y + dy) not in territory:
                    self.pygame.draw.line(self.surface, TERRITORY_COLOR, inset[first], inset[second], width)

    def building_spot(self, name, city=None):
        """Centre-bottom point of a building's icon on the city tile."""
        x, y = self._position((city or self.env.city).position)
        return x + self.size * BUILDING_SPOTS[name], y + self.size * .52

    def _draw_buildings(self, city=None):
        p, s = self.pygame, self.size
        outline = (40, 36, 32)
        city = city or self.env.city
        for name in city.buildings:
            cx, base = self.building_spot(name, city)
            if name == 'monument':
                obelisk = [(cx - s * .05, base), (cx - s * .03, base - s * .2), (cx, base - s * .25),
                           (cx + s * .03, base - s * .2), (cx + s * .05, base)]
                p.draw.polygon(self.surface, MONUMENT_COLOR, obelisk)
                p.draw.polygon(self.surface, outline, obelisk, 1)
            elif name == 'granary':
                body = p.Rect(int(cx - s * .07), int(base - s * .11), int(s * .14), int(s * .11))
                p.draw.rect(self.surface, GRANARY_COLOR, body)
                p.draw.rect(self.surface, outline, body, 1)
                roof = [(cx - s * .09, base - s * .11), (cx, base - s * .2), (cx + s * .09, base - s * .11)]
                p.draw.polygon(self.surface, GRANARY_ROOF, roof)
                p.draw.polygon(self.surface, outline, roof, 1)

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

    def _wall_ring(self, city=None):
        """Corners of the city's walls, just inside its tile."""
        polygon = self._polygon((city or self.env.city).position)
        cx = sum(x for x, _ in polygon) / 6
        cy = sum(y for _, y in polygon) / 6
        return [(cx + (x - cx) * .9, cy + (y - cy) * .9) for x, y in polygon]

    def _draw_walls(self, city=None):
        """A stone ring with corner towers; broken walls leave only ruined towers."""
        city = city or self.env.city
        if city.wall_hp_max <= 0:
            return
        ring = self._wall_ring(city)
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


PLAYER_COLORS = ((70, 150, 235), (225, 80, 70))
ROAD_COLOR = (150, 112, 70)
RESOURCE_COLORS = {'bonus': (240, 240, 240), 'strategic': (235, 120, 110), 'luxury': (205, 150, 245)}
IMPROVEMENT_LABELS = {'farm': 'F', 'mine': 'M', 'quarry': 'Q', 'pasture': 'P', 'plantation': 'Pl',
                      'fishing_boats': 'B'}
EMPIRE_UNIT_LABELS = {'warrior': 'WAR', 'slinger': 'SLG', 'archer': 'ARC', 'spearman': 'SPR',
                      'heavy_chariot': 'CHR', 'horseman': 'HRS', 'swordsman': 'SWD', 'battering_ram': 'RAM',
                      'galley': 'GAL', 'quadrireme': 'QDR', 'settler': 'SET', 'builder': 'BLD', 'trader': 'TRD'}


class EmpireRenderer(Renderer):
    """Draws an EmpireGame: territories, roads, resources, improvements, cities, units and ships."""

    def _panel_height(self):
        return 140

    def draw(self):
        p, env = self.pygame, self.env
        self.surface.fill(constants.COLOR_DEFAULT_BG)
        for cell in env.map.grid.values():
            self._draw_tile(cell)
        self._draw_rivers()
        self._draw_roads()
        self._draw_borders()
        for position in sorted(env.map.grid):
            self._draw_tile_details(position)
        for city in env.cities:
            self._draw_city(city)
        for unit in sorted(env.units, key=lambda u: not u.spec.military):
            self._draw_unit(unit)
        y = self.surface.get_height() - self.panel_height + 9
        outcome = env.last_info['outcome'].upper()
        y += 22 * self._wrapped_text(f'Turn {env.turn_number}/{env.config.turn_limit}   {outcome}', (self.margin, y))
        for player in env.players:
            y += 22 * self._wrapped_text(env.status_text(player.index), (self.margin, y))
        controls = (f'Space: pause   Right: one turn   +/-: speed   Y: yields   Esc: quit   '
                    f'Delay {self.delay_ms or 0}ms')
        lines = self._wrapped_text(controls, (self.margin, y + 5))
        if self.paused:
            self._text('PAUSED', (self.margin, y + 9 + 22 * lines), (255, 220, 100))
        p.display.flip()

    def _center(self, position):
        x, y = self._position(position)
        return x + self.size / 2, y + self.size / 2

    def _draw_roads(self):
        width = max(2, self.size // 16)
        roads = self.env.roads | set(self.env.city_at)
        for position in sorted(roads):
            for neighbor in neighbor_positions(position):
                if neighbor in roads and neighbor > position:
                    self.pygame.draw.line(self.surface, ROAD_COLOR, self._center(position), self._center(neighbor),
                                          width)

    def _draw_borders(self):
        """Outline each civilization's territory in its colour."""
        width = max(2, self.size // 20)
        owners = {position: city.owner for position, city in self.env.tile_owner.items()}
        for position, owner in owners.items():
            polygon = self._polygon(position)
            cx = sum(x for x, _ in polygon) / 6
            cy = sum(y for _, y in polygon) / 6
            inset = [(cx + (x - cx) * .9, cy + (y - cy) * .9) for x, y in polygon]
            x, y = position
            for direction, (first, second) in SIDE_CORNERS.items():
                dx, dy = constants.movement_delta(direction, y)
                if owners.get((x + dx, y + dy)) != owner:
                    self.pygame.draw.line(self.surface, PLAYER_COLORS[owner], inset[first], inset[second], width)

    def _draw_tile_details(self, position):
        env = self.env
        x, y = self._position(position)
        if self.show_yields:
            dots = [YIELD_COLORS[name] for name, amount in env.tile_yields(position).items()
                    for _ in range(int(amount))]
            radius = max(2, self.size // 19)
            step = 2.5 * radius
            left = x + self.size / 2 - step * (len(dots) - 1) / 2
            for index, color in enumerate(dots):
                center = (left + index * step, y + self.size * .2)
                self.pygame.draw.circle(self.surface, color, center, radius)
                self.pygame.draw.circle(self.surface, (30, 30, 30), center, radius, 1)
        resource = env.resources.get(position)
        if resource is not None:
            kind = __import__('empire_rules').RESOURCES[resource].kind
            self._text(resource[:3].upper(), (x + self.size * .12, y + self.size * .62), RESOURCE_COLORS[kind],
                       small=True)
        improvement = env.improvements.get(position)
        if improvement is not None:
            self._text(IMPROVEMENT_LABELS[improvement], (x + self.size * .62, y + self.size * .62),
                       (255, 230, 150), small=True)

    def _draw_city(self, city):
        x, y = self._position(city.position)
        self.surface.blit(self.images['city'], (x, y - self.size // 6))
        color = PLAYER_COLORS[city.owner]
        self.pygame.draw.polygon(self.surface, color, self._polygon(city.position), max(2, self.size // 18))
        if city.wall_hp_max > 0:
            ring = self._polygon(city.position)
            cx, cy = self._center(city.position)
            ring = [(cx + (px - cx) * .8, cy + (py - cy) * .8) for px, py in ring]
            wall = WALL_COLOR if city.wall_hp > 0 else RUINED_WALL_COLOR
            self.pygame.draw.polygon(self.surface, wall, ring, max(2, self.size // 16))
        label = self.small_font.render(f'{city.name} {city.population}', True, color)
        backing = label.get_rect(topleft=(x + 2, y + 2)).inflate(4, 2)
        self.pygame.draw.rect(self.surface, (20, 24, 28), backing)
        self.surface.blit(label, (x + 2, y + 2))
        self._text(f'{city.hp:.0f}', (x + 2, y + self.size * .7), small=True)

    def _draw_unit(self, unit):
        p, s = self.pygame, self.size
        x, y = self._position(unit.position)
        color = PLAYER_COLORS[unit.owner]
        military = unit.spec.military
        cx, cy = x + s * (.5 if military else .3), y + s * (.5 if military else .78)
        radius = max(5, s // (5 if military else 8))
        if unit.spec.domain == 'sea':
            hull = [(cx - radius, cy - radius / 3), (cx + radius, cy - radius / 3), (cx + radius / 2, cy + radius / 2),
                    (cx - radius / 2, cy + radius / 2)]
            p.draw.polygon(self.surface, color, hull)
            p.draw.polygon(self.surface, (20, 20, 20), hull, 1)
        else:
            p.draw.circle(self.surface, color, (cx, cy), radius)
            p.draw.circle(self.surface, (20, 20, 20), (cx, cy), radius, 1)
            if self.env.embarked(unit):
                p.draw.circle(self.surface, (240, 240, 240), (cx, cy), radius + 2, 1)
        label = EMPIRE_UNIT_LABELS[unit.kind]
        self._text(label, (cx - radius, cy - radius - 2), (250, 250, 250), small=True)
        if military and unit.hp < 100:
            self._text(f'{unit.hp:.0f}', (cx - radius, cy + radius - 4), (255, 210, 120), small=True)
