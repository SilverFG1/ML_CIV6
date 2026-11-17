"""A Civ VI-style empire game: found cities, grow, research, trade, and fight a rival.

Player 0 is the learner; any rival is played by the rule-based heuristic in
empire_policy. With one rival the map is mirrored top to bottom, so neither
start is favoured. Each player starts with a settler and a warrior. Settlers
found cities; cities grow within their housing and amenities, build units and
buildings, and claim tiles with culture; gold pays maintenance and buys items;
science researches a small tech tree; builders improve tiles; traders lay roads
between cities; and land units embark onto the coast once Shipbuilding is known,
alongside galleys and quadriremes.

A turn is one step. Every entity of player 0 decides each turn, so an
observation is a tuple of Decision records, one per city, unit and the research
queue, and an action is one option index per decision. A city that is still
building or research already under way has exactly one option: carry on. The
game ends on domination (holding every rival's original capital), elimination,
or at the turn limit, where the higher score wins.
"""

from collections import deque
from dataclasses import dataclass
import heapq
import math
import operator
import random
from typing import NamedTuple

import empire_rules as R
from class_hex import LAND_TYPES, HexMap, hex_distance, neighbor_positions
from economy import CITY_CENTER_MINIMUM, FOOD_PER_CITIZEN, WORK_RADIUS, border_cost, growth_multiplier
from economy import growth_threshold, tile_score
from game import MELEE_WALL_DAMAGE, RAM_SUPPORTED_CLASSES, RIVER_DEFENSE_BONUS, damage_at_least

MAP_TYPES = ('continents', 'islands')
# Civ VI game speeds scale production, research, growth and border costs.
SPEEDS = {'online': .5, 'quick': .67, 'standard': 1.0}
# The rival's yield multiplier (production, science and gold) for each difficulty preset.
DIFFICULTY_BONUS = {'easy': .75, 'normal': 1.0, 'hard': 1.25, 'curriculum': 1.25}
PLAYER_NAMES = ('Otters', 'Weasels')
CITY_NAMES = (
    ('Otterholt', 'Kelpford', 'Riverbend', 'Mossbank', 'Driftwood', 'Stonebrook', 'Reedwater', 'Pebblecove',
     'Willowmere', 'Fernhollow'),
    ('Ottertopia', 'Minkmoor', 'Stoatfield', 'Ferretford', 'Badgerburg', 'Martenmere', 'Sablestead',
     'Ermineholm', 'Polecat Point', 'Wolverwick'),
)
OBSERVATION_SCHEMA = 'ml_civ6.empire.v1'
# A movement cost that uses all of a unit's remaining movement: crossing a river
# without a bridge, embarking and disembarking.
ALL_MOVES = 'all'
SITE_DISTANCE_PENALTY = 3
CAPTURED_CITY_HP = 50
REWARD_SCALE = .1
VICTORY_REWARD = 10.0


class Decision(NamedTuple):
    """One entity's choice this turn: its kind, id, integer features and legal option indices."""
    kind: str
    entity: int
    features: tuple
    valid: tuple


@dataclass(frozen=True)
class DecisionKind:
    features: tuple
    limits: tuple                   # inclusive upper bound of each feature
    options: tuple


CITY_ITEMS = R.UNIT_NAMES + R.BUILDING_NAMES
DECISION_KINDS = {
    'research': DecisionKind(('stage', 'technologies', 'cities', 'coastal', 'threat', 'balance', 'separated'),
                             (3, 4, 4, 1, 2, 2, 1), R.TECHNOLOGY_NAMES),
    'city': DecisionKind(('stage', 'idle', 'capital', 'population', 'housing', 'amenities', 'food', 'production',
                          'sites', 'settlers', 'builders', 'defenders', 'threat', 'coastal', 'cities', 'walls', 'army'),
                         (3, 1, 1, 4, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 4, 1, 2), CITY_ITEMS + ('buy',)),
    'settler': DecisionKind(('stage', 'here', 'best', 'distance', 'threat', 'escort', 'cities'),
                            (3, 3, 3, 3, 2, 1, 4), ('found', 'site_1', 'site_2', 'site_3', 'hold')),
    'builder': DecisionKind(('stage', 'here', 'targets', 'distance', 'charges', 'threat'),
                            (3, 3, 3, 3, 2, 2), ('improve', 'goto_resource', 'goto_tile', 'hold')),
    'land': DecisionKind(('stage', 'unit_type', 'health', 'threat', 'contact', 'odds', 'enemy_city', 'balance',
                          'territory', 'post', 'escort'),
                         (3, len(R.LAND_MILITARY) - 1, 3, 2, 2, 3, 3, 2, 2, 2, 1),
                         ('fortify', 'garrison', 'defend', 'hunt', 'siege', 'escort')),
    'naval': DecisionKind(('stage', 'unit_type', 'health', 'threat', 'contact', 'odds', 'enemy_city', 'balance'),
                          (3, len(R.NAVAL_UNITS) - 1, 3, 2, 2, 3, 3, 2), ('fortify', 'hunt', 'raid', 'patrol')),
    'trader': DecisionKind(('stage', 'routes'), (3, 1), ('route', 'hold')),
}
UNIT_KIND_ORDER = ('land', 'naval', 'settler', 'builder', 'trader')
OPTION_INDEX = {kind: {name: index for index, name in enumerate(spec.options)}
                for kind, spec in DECISION_KINDS.items()}


def combat_damage(rng, difference):
    """The siege game's damage roll: round(roll * exp(difference / 25) * scale)."""
    return round(rng.randint(24, 36) * math.exp(difference / 25) * rng.uniform(.75, 1))


def _bucket(value, bounds):
    """Index of the first bound value is below; len(bounds) when above all of them."""
    for index, bound in enumerate(bounds):
        if value < bound:
            return index
    return len(bounds)


@dataclass(frozen=True)
class EmpireConfig:
    width: int = 14
    height: int = 13
    rivals: int = 1
    map_type: str = 'continents'
    turn_limit: int = 100
    land_types: tuple = ('grassland', 'plains')
    hills_density: float = .15
    forest_density: float = .15
    mountain_density: float = .05
    water_density: float = .05
    river_count: int = 2
    resource_density: float = .15
    rival_bonus: float = 1.0
    speed: str = 'online'
    difficulty: str = 'normal'      # recorded for the runner; rival_bonus sets the rival's strength

    def __post_init__(self):
        for name, low, high in (('width', 8, 32), ('height', 7, 31), ('turn_limit', 10, 500)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f'{name} must be an integer between {low} and {high}')
        if isinstance(self.rivals, bool) or self.rivals not in (0, 1):
            raise ValueError('rivals must be 0 or 1')
        if self.rivals and self.height % 2 == 0:
            raise ValueError('height must be odd with a rival, so the map can be mirrored')
        if self.difficulty not in DIFFICULTY_BONUS:
            raise ValueError(f'difficulty must be one of {", ".join(DIFFICULTY_BONUS)}')
        if self.speed not in SPEEDS:
            raise ValueError(f'speed must be one of {", ".join(SPEEDS)}')
        if self.map_type not in MAP_TYPES:
            raise ValueError(f'map_type must be one of {", ".join(MAP_TYPES)}')
        try:
            object.__setattr__(self, 'land_types', tuple(self.land_types))
        except TypeError as exc:
            raise ValueError('land_types must be a sequence of land type names') from exc
        if (not self.land_types or any(name not in LAND_TYPES for name in self.land_types)
                or len(set(self.land_types)) != len(self.land_types)):
            raise ValueError(f'land_types must name distinct types from {", ".join(LAND_TYPES)}')
        for name in ('hills_density', 'forest_density', 'mountain_density', 'water_density', 'resource_density'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be between 0 and 1')
        if self.mountain_density + self.water_density + self.forest_density > .6:
            raise ValueError('combined mountain, water and forest density must be at most 0.6')
        if self.resource_density > .5:
            raise ValueError('resource_density must be at most 0.5')
        if isinstance(self.river_count, bool) or not isinstance(self.river_count, int) or not 0 <= self.river_count <= 10:
            raise ValueError('river_count must be an integer between 0 and 10')
        if (isinstance(self.rival_bonus, bool) or not isinstance(self.rival_bonus, (int, float))
                or not math.isfinite(self.rival_bonus) or not .25 <= self.rival_bonus <= 3):
            raise ValueError('rival_bonus must be between 0.25 and 3')


class Player:
    def __init__(self, index, bonus):
        self.index = index
        self.name = PLAYER_NAMES[index]
        self.bonus = float(bonus)
        self.gold = 0.0
        self.known = set()
        self.research = None
        self.progress = {}
        self.boosted = set()
        self.settlers_trained = 0
        self.cities_founded = 0
        self.met = set()
        self.alive = True
        self.routes = []            # (origin city id, destination city id)


class Unit:
    def __init__(self, uid, owner, kind, position):
        if kind not in R.UNITS:
            raise ValueError(f'Unknown unit type: {kind}')
        self.id, self.owner, self.kind = uid, owner, kind
        self.x, self.y = position
        self.hp = 100.0
        self.max_moves = float(R.UNITS[kind].moves)
        self.moves = self.max_moves
        self.charges = R.UNITS[kind].charges
        self.acted = False
        self.target = None

    @property
    def spec(self):
        return R.UNITS[self.kind]

    @property
    def position(self):
        return self.x, self.y

    @property
    def decision_kind(self):
        if self.kind in ('settler', 'builder', 'trader'):
            return self.kind
        return 'naval' if self.spec.domain == 'sea' else 'land'

    def __repr__(self):
        return f'Unit({self.id}, {self.kind}, owner={self.owner}, at={self.position}, hp={self.hp:.0f})'


class City:
    def __init__(self, cid, owner, name, position, capital):
        self.id, self.owner, self.original_owner, self.name = cid, owner, owner, name
        self.x, self.y = position
        self.capital = capital
        self.hp = self.hp_max = float(R.CITY_HP)
        self.wall_hp = self.wall_hp_max = 0.0
        self.population = 1
        self.food = self.production = self.culture = 0.0
        self.tiles_acquired = 0
        self.item = None
        self.buildings = []
        self.territory = set()
        self.damaged = False

    @property
    def position(self):
        return self.x, self.y

    @property
    def has_palace(self):
        return self.capital and self.owner == self.original_owner

    def __repr__(self):
        return f'City({self.name}, owner={self.owner}, pop={self.population}, hp={self.hp:.0f})'


class EmpireGame:
    """The empire environment. reset() and step() return a tuple of Decision records for player 0."""

    observation_mode = 'empire'

    def __init__(self, config=None, seed=None, render=False, ml_ai=True, observation='empire'):
        if observation != 'empire':
            raise ValueError("EmpireGame only provides the 'empire' observation")
        self.config = config or EmpireConfig()
        self.render = render
        self.ml_ai = ml_ai
        self.rng = random.Random(seed)
        self.observation_schema = OBSERVATION_SCHEMA
        self.decision_kinds = DECISION_KINDS
        self.human = False
        self.quit = False
        self._renderer = None
        self.map = None
        self.players = []
        self.units = []
        self.cities = []
        self.events = []
        self.turn_number = 0
        self.last_info = {'outcome': 'ongoing', 'events': []}
        self._done = False
        self._decisions = ()
        # Geometry depends only on the map's size: on-map neighbours and tiles within a radius.
        cells = [(x, y) for y in range(self.config.height) for x in range(self.config.width)]
        on_map = set(cells)
        self._adjacent = {p: [q for q in neighbor_positions(p) if q in on_map] for p in cells}
        self._within_cache = {}
        self._distance_cache = {}

    # ------------------------------------------------------------------ setup

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        c = self.config
        self.players = [Player(index, c.rival_bonus if index else 1.0) for index in range(1 + c.rivals)]
        self.units, self.cities = [], []
        self.military_at, self.civilian_at, self.city_at, self.tile_owner = {}, {}, {}, {}
        self.improvements, self.roads, self.resources = {}, set(), {}
        self._next_id = 1
        self._site_cache = None
        self._distance_cache = {}
        self._city_version = 0
        self.turn_number = 0
        self.events = []
        self.last_info = {'outcome': 'ongoing', 'events': []}
        self._done = self.quit = False
        self.starts = self._generate_map()
        for player, start in zip(self.players, self.starts):
            self._add_unit(player.index, 'settler', start)
            self._add_unit(player.index, 'warrior', start)
        self._reset_moves()
        self._decisions = self._build_decisions(0)
        return self._decisions

    def game_initialize(self, ep_number=0, seed=None):
        self.episode_number = ep_number
        return self.reset(seed)

    def _generate_map(self):
        for attempt in range(60):
            relax = 1.0 if attempt < 40 else 0.0
            self.map = HexMap(self.config.height, self.config.width)
            self.resources = {}
            self._shape_map(relax)
            starts = self._choose_starts()
            if starts is not None:
                return starts
        raise RuntimeError('could not generate a playable empire map')

    def _shape_map(self, relax):
        c, rng, grid = self.config, self.rng, self.map.grid
        water, mountains = c.water_density * relax, c.mountain_density * relax
        for (x, y), cell in grid.items():
            edge = x in (0, c.width - 1) or y in (0, c.height - 1)
            draw = rng.random()
            if (edge and draw < .6) or draw < water:
                cell.set_terrain('COAST')
            elif not edge and draw < water + mountains:
                cell.set_terrain('GRASSLAND_MOUNTAIN')
            else:
                cell.set_terrain('GRASSLAND')
        if c.map_type == 'islands':
            # One row of water between the two halves; it is coast wherever land touches it.
            for x in range(c.width):
                grid[(x, c.height // 2)].set_terrain('COAST')
        self.map.assign_land_types(c.land_types, rng)
        self.map.raise_hills(c.hills_density, rng)
        for position in sorted(grid):
            cell = grid[position]
            if (cell.spec.land_type in ('GRASSLAND', 'PLAINS', 'TUNDRA') and not cell.spec.impassable
                    and rng.random() < c.forest_density * relax):
                cell.set_feature('FOREST')
        self.map.classify_water()
        self.map.generate_rivers(c.river_count, rng)
        self._place_resources()
        if c.rivals:
            self._mirror()
            self.map.classify_water()
            self.resources = {p: name for p, name in self.resources.items() if self._resource_fits(name, p)}

    def _resource_fits(self, name, position):
        spec, cell = R.RESOURCES[name], self.map.grid[position]
        if cell.terrain_type not in spec.terrains:
            return False
        return cell.feature == spec.feature

    def _place_resources(self):
        density = self.config.resource_density
        if density <= 0:
            return
        for position in sorted(self.map.grid):
            if self.rng.random() >= density:
                continue
            options = [name for name in R.RESOURCE_NAMES if self._resource_fits(name, position)]
            if options:
                self.resources[position] = self.rng.choice(options)

    def mirror(self, position):
        x, y = position
        return x, self.config.height - 1 - y

    def _mirror(self):
        """Copy the top half onto the bottom half; the middle row maps onto itself."""
        middle = self.config.height // 2
        grid = self.map.grid
        for (x, y), cell in grid.items():
            if y > middle:
                source = grid[self.mirror((x, y))]
                cell.set_terrain(source.terrain_type)
                if source.feature is not None:
                    cell.set_feature(source.feature)
        top = {p: name for p, name in self.resources.items() if p[1] <= middle}
        self.resources = {**top, **{self.mirror(p): name for p, name in top.items() if p[1] < middle}}
        rivers = {edge for edge in self.map.rivers if all(p[1] <= middle for p in edge)}
        self.map.rivers = rivers | {frozenset(self.mirror(p) for p in edge) for edge in rivers}

    def _land(self, position):
        cell = self.map.grid.get(position)
        return cell is not None and not cell.is_water and not cell.spec.impassable

    def _choose_starts(self):
        c, grid = self.config, self.map.grid
        last_row = c.height // 2 - 2 if c.rivals else c.height - 2
        candidates = []
        for (x, y) in sorted(grid):
            if not (1 <= x <= c.width - 2 and 1 <= y <= last_row) or not self._land((x, y)):
                continue
            ring = [p for p in self._within((x, y), 2) if p != (x, y)]
            if sum(self._land(p) for p in ring) < 8:
                continue
            candidates.append((self._start_value((x, y)), (x, y)))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        start = self.rng.choice(candidates[:3])[1]
        starts = [start] + ([self.mirror(start)] if c.rivals else [])
        if c.rivals and c.map_type == 'continents':
            reach = self._bfs([start], self._land)
            if starts[1] not in reach:
                return None
        for position in starts:
            self.resources.pop(position, None)
            grid[position].set_feature(None)
        return starts

    def _start_value(self, position):
        value = sum(tile_score(self.tile_yields(p)) for p in self._within(position, 2)
                    if p != position and not self.map.grid[p].spec.impassable)
        return value + self._center_bonus(position)

    def _within(self, position, radius):
        """Tiles on the map within radius of position (shared lists: do not modify them)."""
        key = position, radius
        tiles = self._within_cache.get(key)
        if tiles is None:
            x0, y0 = position
            tiles = self._within_cache[key] = [
                (x, y) for y in range(max(0, y0 - radius), min(self.config.height, y0 + radius + 1))
                for x in range(max(0, x0 - radius - 1), min(self.config.width, x0 + radius + 2))
                if hex_distance((x, y), position) <= radius]
        return tiles

    def _neighbors(self, position):
        """On-map neighbours of position (a shared list: do not modify it)."""
        return self._adjacent[position]

    def _bfs(self, starts, passable):
        """Fewest steps from any start through passable tiles; starts need not be passable."""
        adjacent = self._adjacent
        distances = {tuple(p): 0 for p in starts}
        pending = deque(distances)
        while pending:
            position = pending.popleft()
            for neighbor in adjacent[position]:
                if neighbor not in distances and passable(neighbor):
                    distances[neighbor] = distances[position] + 1
                    pending.append(neighbor)
        return distances

    # ------------------------------------------------------------- entities

    def _add_unit(self, owner, kind, position):
        unit = Unit(self._next_id, owner, kind, position)
        self._next_id += 1
        self.units.append(unit)
        self._occupancy(unit)[position] = unit
        if kind == 'galley' and sum(u.kind == 'galley' and u.owner == owner for u in self.units) >= 2:
            self._boost(owner, 'two_galleys')
        return unit

    def _occupancy(self, unit):
        return self.military_at if unit.spec.military else self.civilian_at

    def _relocate(self, unit, position):
        layer = self._occupancy(unit)
        if layer.get(unit.position) is unit:
            del layer[unit.position]
        unit.x, unit.y = position
        layer[position] = unit

    def _remove_unit(self, unit):
        layer = self._occupancy(unit)
        if layer.get(unit.position) is unit:
            del layer[unit.position]
        if unit in self.units:
            self.units.remove(unit)

    def _event(self, kind, **details):
        self.events.append({'type': kind, **details})

    def player_units(self, owner, kind=None):
        return [u for u in self.units if u.owner == owner and (kind is None or u.kind == kind)]

    def player_cities(self, owner):
        return [city for city in self.cities if city.owner == owner]

    def enemies(self, owner):
        return [p.index for p in self.players if p.index != owner and p.alive]

    def embarked(self, unit):
        return unit.spec.domain == 'land' and self.map.grid[unit.position].is_water

    def can_embark(self, owner):
        return R.EMBARK_TECHNOLOGY in self.players[owner].known

    def coastal(self, position):
        return any(self.map.grid[p].terrain_type == 'COAST' for p in self._neighbors(position))

    def has_road(self, position):
        return position in self.roads or position in self.city_at

    # --------------------------------------------------------------- cities

    def legal_site(self, owner, position):
        """Whether owner may found a city here: open land, far enough from every city, not in foreign territory."""
        if not self._land(position):
            return False
        holder = self.tile_owner.get(position)
        if holder is not None and holder.owner != owner:
            return False
        return all(hex_distance(position, city.position) >= R.MIN_CITY_DISTANCE for city in self.cities)

    def _center_bonus(self, position):
        neighbors = self._neighbors(position)
        if any(self.map.has_river(position, p) for p in neighbors):
            bonus = 4
        elif any(self.map.grid[p].is_water for p in neighbors):
            bonus = 2
        else:
            bonus = 0
        return bonus + (1 if self.map.grid[position].spec.elevation == 'HILLS' else 0)

    def site_values(self):
        """Value of every tile where some player could found a city: the yields of unowned tiles within two."""
        # Values depend only on the cities, their territory and the improvements, which only accumulate.
        key = (self._city_version, len(self.improvements))
        if self._site_cache is not None and self._site_cache[0] == key:
            return self._site_cache[1]
        values = {}
        for position in self.map.grid:
            if not self._land(position) or any(hex_distance(position, city.position) < R.MIN_CITY_DISTANCE
                                               for city in self.cities):
                continue
            value = self._center_bonus(position)
            for p in self._within(position, 2):
                if p == position or p in self.tile_owner or self.map.grid[p].spec.impassable:
                    continue
                value += tile_score(self.tile_yields(p))
                resource = self.resources.get(p)
                if resource is not None and R.RESOURCES[resource].kind == 'luxury':
                    value += 2
            values[position] = value
        self._site_cache = key, values
        return values

    def found_city(self, unit):
        if unit.kind != 'settler' or not self.legal_site(unit.owner, unit.position):
            raise ValueError('settler cannot found a city here')
        player = self.players[unit.owner]
        names = CITY_NAMES[unit.owner]
        name = names[player.cities_founded % len(names)]
        if player.cities_founded >= len(names):
            name += f' {player.cities_founded // len(names) + 1}'
        city = City(self._next_id, unit.owner, name, unit.position, capital=player.cities_founded == 0)
        self._next_id += 1
        player.cities_founded += 1
        self._remove_unit(unit)
        self.cities.append(city)
        self.city_at[city.position] = city
        for position in [city.position] + self._neighbors(city.position):
            if position not in self.tile_owner:
                self._claim(city, position)
        self._city_version += 1
        self._event('city_founded', player=unit.owner, city=city.name, position=list(city.position))
        if self.coastal(city.position):
            self._boost(unit.owner, 'coastal_city')
        return city

    def _claim(self, city, position):
        city.territory.add(position)
        self.tile_owner[position] = city

    def tile_yields(self, position):
        totals = dict(self.map.grid[position].yields)
        resource = self.resources.get(position)
        if resource is not None:
            R.add_yields(totals, R.RESOURCES[resource].yields)
        improvement = self.improvements.get(position)
        if improvement is not None:
            R.add_yields(totals, R.IMPROVEMENTS[improvement].yields)
        return totals

    def _center_yields(self, city):
        totals = self.tile_yields(city.position)
        for kind, minimum in CITY_CENTER_MINIMUM:
            totals[kind] = max(totals.get(kind, 0), minimum)
        return totals

    def worked_tiles(self, city):
        """Citizens work the best owned tiles not blocked by enemy units, trading for food rather than starve."""
        grid = self.map.grid
        blocked = {u.position for u in self.units if u.owner != city.owner}
        scored = {p: self.tile_yields(p) for p in city.territory
                  if p != city.position and not grid[p].spec.impassable and p not in blocked}
        ranked = sorted(scored, key=lambda p: (-tile_score(scored[p]), -scored[p].get('food', 0),
                                               hex_distance(p, city.position), p))
        worked, spare = ranked[:city.population], ranked[city.population:]

        def food(position):
            return scored[position].get('food', 0)

        base_food = self._center_yields(city).get('food', 0) + self._building_yields(city).get('food', 0)
        base_food += sum(amount for kind, amount in R.TRADE_ROUTE_YIELDS if kind == 'food') * self._routes_from(city)
        while spare and base_food + sum(map(food, worked)) < FOOD_PER_CITIZEN * city.population:
            poorest = min(worked, key=lambda p: (food(p), -worked.index(p)))
            richest = max(spare, key=lambda p: (food(p), -spare.index(p)))
            if food(richest) <= food(poorest):
                break
            worked[worked.index(poorest)] = richest
            spare.remove(richest)
        return worked

    def _building_yields(self, city):
        totals = {}
        for name in city.buildings:
            R.add_yields(totals, R.BUILDINGS[name].yields)
        if city.has_palace:
            R.add_yields(totals, R.PALACE_YIELDS)
        return totals

    def _routes_from(self, city):
        return sum(origin == city.id for origin, _ in self.players[city.owner].routes)

    def housing(self, city):
        """2, or 3 next to water, or 5 next to a river; plus buildings and half a house per improved farm, pasture,
        plantation or fishing boats in the city's territory."""
        neighbors = self._neighbors(city.position)
        if any(self.map.has_river(city.position, p) for p in neighbors):
            base = 5
        elif any(self.map.grid[p].is_water for p in neighbors):
            base = 3
        else:
            base = 2
        base += sum(R.BUILDINGS[name].housing for name in city.buildings)
        return base + sum(R.IMPROVEMENTS[self.improvements[p]].housing for p in city.territory if p in self.improvements)

    def improved_resources(self, owner):
        """Resource types on improved tiles in owner's territory."""
        found = set()
        for city in self.player_cities(owner):
            for position in city.territory:
                name = self.resources.get(position)
                if name is not None and self.improvements.get(position) == R.RESOURCES[name].improvement:
                    found.add(name)
        return found

    def strategic_resources(self, owner):
        return {name for name in self.improved_resources(owner) if R.RESOURCES[name].kind == 'strategic'}

    def luxuries(self, owner):
        """Distinct luxury types on improved tiles in owner's territory."""
        return sorted(name for name in self.improved_resources(owner) if R.RESOURCES[name].kind == 'luxury')

    def amenities(self, owner):
        """Amenities available in each of owner's cities: each luxury type supplies up to four, neediest first."""
        cities = sorted(self.player_cities(owner), key=lambda c: c.id)
        available = {city.id: 0 for city in cities}
        for _ in self.luxuries(owner):
            needy = sorted(cities, key=lambda c: (available[c.id] - R.amenities_required(c.population), c.id))
            for city in needy[:R.LUXURY_CITIES]:
                available[city.id] += 1
        return available

    def amenity_surplus(self, city, available=None):
        available = self.amenities(city.owner) if available is None else available
        return available.get(city.id, 0) - R.amenities_required(city.population)

    def city_yields(self, city, available=None):
        """This turn's yields: centre, worked tiles, buildings, Palace and trade routes, then citizens' science,
        the amenity mood and the owner's bonus."""
        totals = self._center_yields(city)
        for position in self.worked_tiles(city):
            R.add_yields(totals, self.tile_yields(position).items())
        R.add_yields(totals, self._building_yields(city).items())
        for _ in range(self._routes_from(city)):
            R.add_yields(totals, R.TRADE_ROUTE_YIELDS)
        totals['science'] = totals.get('science', 0) + R.SCIENCE_PER_CITIZEN * city.population
        _, _, multiplier = R.amenity_mood(self.amenity_surplus(city, available))
        bonus = self.players[city.owner].bonus
        for kind in list(totals):
            if kind != 'food':
                totals[kind] *= multiplier * (bonus if kind in ('production', 'science', 'gold') else 1)
        return totals

    def food_surplus(self, city, yields=None):
        yields = self.city_yields(city) if yields is None else yields
        return yields.get('food', 0) - FOOD_PER_CITIZEN * city.population

    def scaled(self, cost):
        """A base cost at this game's speed."""
        return math.ceil(cost * SPEEDS[self.config.speed])

    def item_cost(self, owner, item):
        return self.scaled(R.item_cost(item, self.players[owner].settlers_trained))

    def tech_cost(self, tech):
        return self.scaled(R.TECHNOLOGIES[tech].cost)

    def growth_threshold(self, population):
        return self.scaled(growth_threshold(population))

    def border_cost(self, tiles_acquired):
        return border_cost(tiles_acquired) * SPEEDS[self.config.speed]

    def item_price(self, owner, item):
        return R.GOLD_PER_PRODUCTION * self.item_cost(owner, item)

    def has_building(self, city, name):
        return name in city.buildings

    def improvable(self, owner, position):
        """The improvement a builder of owner could place here, or None."""
        holder = self.tile_owner.get(position)
        if (holder is None or holder.owner != owner or position in self.city_at or position in self.improvements):
            return None
        known = self.players[owner].known
        cell = self.map.grid[position]
        resource = self.resources.get(position)
        if resource is not None:
            improvement = R.RESOURCES[resource].improvement
            tech = R.IMPROVEMENTS[improvement].technology
            return improvement if tech is None or tech in known else None
        if cell.is_water or cell.spec.impassable or cell.feature is not None:
            return None
        if cell.spec.elevation == 'HILLS' and 'mining' in known:
            return 'mine'
        return 'farm' if cell.spec.land_type in R.FARM_LAND else None

    def can_build(self, city, item):
        player = self.players[city.owner]
        if item in R.UNITS:
            spec = R.UNITS[item]
            if spec.technology is not None and spec.technology not in player.known:
                return False
            if spec.domain == 'sea' and not self.coastal(city.position):
                return False
            if item == 'settler' and city.population < R.SETTLER_MIN_POPULATION:
                return False
            if spec.resource is not None and spec.resource not in self.strategic_resources(city.owner):
                return False
            if item == 'trader':
                return (len(self.player_cities(city.owner)) >= 2
                        and len(player.routes) + len(self.player_units(city.owner, 'trader'))
                        + sum(c.item == 'trader' for c in self.player_cities(city.owner)) < self.route_capacity(city.owner))
            return True
        spec = R.BUILDINGS[item]
        if item in city.buildings or (spec.technology is not None and spec.technology not in player.known):
            return False
        if spec.requires == 'river':
            return any(self.map.has_river(city.position, p) for p in self._neighbors(city.position))
        return True

    def route_capacity(self, owner):
        return R.BASE_TRADE_ROUTES + sum('market' in c.buildings for c in self.player_cities(owner))

    # ------------------------------------------------------------- research

    def _boost(self, owner, event):
        player = self.players[owner]
        for name, tech in R.TECHNOLOGIES.items():
            if tech.boost == event and name not in player.known and name not in player.boosted:
                player.boosted.add(name)
                player.progress[name] = player.progress.get(name, 0.0) + R.BOOST_SHARE * self.tech_cost(name)
                self._event('technology_boosted', player=owner, technology=name)

    def available_technologies(self, owner):
        return R.available_technologies(self.players[owner].known)

    def _set_research(self, owner, tech):
        player = self.players[owner]
        if player.research is None:
            if tech not in self.available_technologies(owner):
                raise ValueError(f'{tech} cannot be researched now')
            player.research = tech

    def _research(self, owner, science):
        player = self.players[owner]
        if player.research is None:
            return
        tech = player.research
        player.progress[tech] = player.progress.get(tech, 0.0) + science
        cost = self.tech_cost(tech)
        if player.progress[tech] >= cost:
            overflow = player.progress.pop(tech) - cost
            player.known.add(tech)
            player.research = None
            player.progress['_overflow'] = player.progress.get('_overflow', 0.0) + overflow
            self._event('technology_researched', player=owner, technology=tech)
            if tech == R.BRIDGE_TECHNOLOGY or tech == R.EMBARK_TECHNOLOGY:
                self._site_cache = None

    # ------------------------------------------------------------- movement

    def step_cost(self, unit, origin, destination):
        """Movement points to step between adjacent tiles; None if unit cannot enter; ALL_MOVES for all it has left."""
        grid = self.map.grid
        cell = grid.get(destination)
        if cell is None:
            return None
        city = self.city_at.get(destination)
        if unit.spec.domain == 'sea':
            if city is not None:
                return 1 if city.owner == unit.owner and self.coastal(destination) else None
            return 1 if cell.terrain_type == 'COAST' else None
        if cell.spec.impassable or (city is not None and city.owner != unit.owner):
            return None
        if cell.is_water:
            if cell.terrain_type != 'COAST' or not self.can_embark(unit.owner):
                return None
            return 1 if grid[origin].is_water else ALL_MOVES
        if grid[origin].is_water:
            return ALL_MOVES
        river = self.map.has_river(origin, destination)
        if (self.has_road(origin) and self.has_road(destination)
                and (not river or R.BRIDGE_TECHNOLOGY in self.players[unit.owner].known)):
            return 1
        return ALL_MOVES if river else cell.movement_cost

    def blocked(self, unit, position):
        """Whether another unit stops unit from entering a tile (an enemy civilian is captured instead)."""
        military, civilian = self.military_at.get(position), self.civilian_at.get(position)
        if unit.spec.military:
            return military is not None and military is not unit
        return ((civilian is not None and civilian is not unit)
                or (military is not None and military.owner != unit.owner))

    def search(self, unit, goals=(), limit=40):
        """Cheapest movement cost from unit to the tiles it can reach, ignoring zones of control.

        The search stops once it settles the cheapest of goals; without one it covers everything in reach.
        """
        start = unit.position
        costs, previous = {start: 0.0}, {}
        heap, counter = [(0.0, 0, start)], 1
        adjacent = self._adjacent
        while heap:
            cost, _, position = heapq.heappop(heap)
            if cost > costs[position] or cost > limit:
                continue
            if position in goals:
                break
            for neighbor in adjacent[position]:
                step = self.step_cost(unit, position, neighbor)
                if step is None or self.blocked(unit, neighbor):
                    continue
                total = cost + (unit.max_moves if step == ALL_MOVES else step)
                if total < costs.get(neighbor, math.inf):
                    costs[neighbor], previous[neighbor] = total, position
                    heapq.heappush(heap, (total, counter, neighbor))
                    counter += 1
        return costs, previous

    def zone_of_control(self, unit, position):
        return any(other.owner != unit.owner and not self.embarked(other)
                   for p in self._neighbors(position)
                   for other in (self.military_at.get(p),) if other is not None)

    def walk(self, unit, path):
        """Move unit along path while it has movement; returns whether it moved."""
        moved = False
        for step in path[1:]:
            if unit.moves <= 0:
                break
            cost = self.step_cost(unit, unit.position, step)
            if cost is None or self.blocked(unit, step):
                break
            if cost == ALL_MOVES:
                cost = unit.moves
            elif cost > unit.moves and unit.moves < unit.max_moves:
                break
            civilian = self.civilian_at.get(step)
            if civilian is not None and civilian.owner != unit.owner:
                if not unit.spec.military or self.embarked(unit) or unit.spec.unit_class == 'support':
                    break
                self._kill(civilian, unit.owner)
            self._relocate(unit, step)
            unit.moves = max(0.0, unit.moves - cost)
            unit.acted = moved = True
            if self.zone_of_control(unit, step):
                unit.moves = 0.0
        return moved

    def travel(self, unit, destinations):
        """Head for the cheapest reachable destination, or the reachable tile nearest any of them."""
        destinations = set(destinations)
        if not destinations or unit.position in destinations or unit.moves <= 0:
            return False
        costs, previous = self.search(unit, destinations)
        reachable = [p for p in destinations if p in costs]
        if reachable:
            goal = min(reachable, key=lambda p: (costs[p], p))
        else:
            def gap(p):
                return min(hex_distance(p, d) for d in destinations)
            goal = min(costs, key=lambda p: (gap(p), costs[p], p))
            if gap(goal) >= gap(unit.position):
                return False
        path = [goal]
        while path[-1] != unit.position:
            path.append(previous[path[-1]])
        return self.walk(unit, path[::-1])

    def _reset_moves(self, owner=None):
        for unit in self.units:
            if owner is not None and unit.owner != owner:
                continue
            cell = self.map.grid[unit.position]
            flat = cell.spec.elevation == 'FLAT' and cell.feature is None and not cell.is_water
            unit.max_moves = float(unit.spec.moves + (1 if unit.kind == 'heavy_chariot' and flat else 0))
            unit.moves = unit.max_moves
            unit.acted = False

    # --------------------------------------------------------------- combat

    def best_strength(self, owner, ranged=False):
        known = self.players[owner].known
        names = [n for n in R.LAND_MILITARY if R.UNITS[n].technology is None or R.UNITS[n].technology in known]
        if ranged:
            return max([R.WALL_MIN_RANGED_STRENGTH] + [R.UNITS[n].ranged_strength for n in names])
        return max(R.UNITS[n].strength for n in names if R.UNITS[n].unit_class in ('melee', 'anti_cavalry',
                                                                                     'heavy_cavalry'))

    def city_strength(self, city):
        return self.best_strength(city.owner) + R.CITY_STRENGTH_BONUS

    def unit_strength(self, unit, attacking, opponent=None, ranged=False):
        spec = unit.spec
        embarked = self.embarked(unit)
        if embarked and not attacking:
            strength = R.EMBARKED_STRENGTH
        else:
            strength = spec.ranged_strength if ranged else spec.strength
        strength -= math.floor((100 - unit.hp) / R.WOUND_STEP)
        if (isinstance(opponent, Unit) and spec.unit_class == 'anti_cavalry'
                and opponent.spec.unit_class in R.CAVALRY_CLASSES):
            strength += R.ANTI_CAVALRY_BONUS
        if not attacking and not embarked and spec.domain == 'land':
            strength += self.map.grid[unit.position].defense_bonus
        return strength

    def _river_defense(self, attacker, target_position):
        if attacker.spec.domain == 'land' and self.map.has_river(attacker.position, target_position):
            return RIVER_DEFENSE_BONUS
        return 0

    def wall_multiplier(self, unit, city):
        if unit.spec.unit_class in RAM_SUPPORTED_CLASSES and any(
                u.kind == 'battering_ram' and u.owner == unit.owner and hex_distance(u.position, city.position) == 1
                for u in self.units):
            return 1.0
        return MELEE_WALL_DAMAGE

    def attack_targets(self, unit):
        """Enemy units and cities unit can attack from where it stands now."""
        spec = unit.spec
        if not spec.can_attack or unit.moves <= 0 or self.embarked(unit):
            return []
        enemies = set(self.enemies(unit.owner))
        targets = []
        if spec.ranged:
            for position in self._within(unit.position, spec.attack_range):
                if position == unit.position:
                    continue
                military, city = self.military_at.get(position), self.city_at.get(position)
                if city is not None and city.owner in enemies:
                    targets.append(city)
                elif military is not None and military.owner in enemies:
                    targets.append(military)
            return targets
        for position in self._neighbors(unit.position):
            military, city = self.military_at.get(position), self.city_at.get(position)
            water = self.map.grid[position].is_water
            if city is not None and city.owner in enemies:
                targets.append(city)
            elif military is not None and military.owner in enemies and (spec.domain == 'sea' or not water):
                targets.append(military)
        return targets

    def attack_odds(self, unit, target):
        """(chance the attack destroys or captures target, chance unit dies to the retaliation)."""
        ranged = unit.spec.ranged
        attack = self.unit_strength(unit, True, target, ranged)
        if isinstance(target, City):
            defense = self.city_strength(target) + (0 if ranged else self._river_defense(unit, target.position))
            if ranged:
                return 0.0, 0.0
            needed = target.hp + (target.wall_hp / self.wall_multiplier(unit, target) if target.wall_hp > 0 else 0)
        else:
            defense = (self.unit_strength(target, False, unit)
                       + (0 if ranged else self._river_defense(unit, target.position)))
            needed = target.hp
        kill = damage_at_least(needed, math.exp((attack - defense) / 25))
        if ranged:
            return kill, 0.0
        death = damage_at_least(unit.hp, math.exp((defense - attack) / 25))
        return kill * (1 - death), death

    def attack(self, unit, target):
        if target not in self.attack_targets(unit):
            raise ValueError('target is not attackable')
        unit.moves, unit.acted = 0.0, True
        if isinstance(target, City):
            self._attack_city(unit, target)
        else:
            self._attack_unit(unit, target)

    def _attack_unit(self, unit, defender):
        ranged = unit.spec.ranged
        river = 0 if ranged else self._river_defense(unit, defender.position)
        difference = self.unit_strength(unit, True, defender, ranged) - self.unit_strength(defender, False, unit) - river
        dealt = combat_damage(self.rng, difference)
        self._event('unit_attacked', player=unit.owner, unit=unit.kind, target=defender.kind)
        if not ranged:
            taken = combat_damage(self.rng, -difference)
            self._damage_unit(unit, taken, defender.owner, defender.kind)
            if unit.hp <= 0:
                dealt = min(dealt, defender.hp - 1)
        target_position = defender.position
        self._damage_unit(defender, dealt, unit.owner, unit.kind)
        if (not ranged and unit.hp > 0 and defender.hp <= 0 and unit.spec.domain == 'land'
                and not self.blocked(unit, target_position) and target_position not in self.city_at):
            civilian = self.civilian_at.get(target_position)
            if civilian is not None and civilian.owner != unit.owner:
                self._kill(civilian, unit.owner)
            self._relocate(unit, target_position)

    def _damage_unit(self, unit, amount, by, by_kind=None):
        amount = min(unit.hp, max(0.0, float(amount)))
        unit.hp -= amount
        if unit.hp <= 0:
            self._kill(unit, by, by_kind)

    def _kill(self, unit, by, by_kind=None):
        unit.hp = 0.0
        self._remove_unit(unit)
        self._event('unit_killed', player=unit.owner, unit=unit.kind, military=unit.spec.military, by=by)
        if by_kind == 'slinger' and by is not None:
            self._boost(by, 'slinger_kill')

    def _attack_city(self, unit, city):
        ranged = unit.spec.ranged
        river = 0 if ranged else self._river_defense(unit, city.position)
        difference = self.unit_strength(unit, True, city, ranged) - self.city_strength(city) - river
        dealt = combat_damage(self.rng, difference)
        self._event('city_attacked', player=unit.owner, city=city.name)
        if ranged:
            self._damage_city(city, dealt, False, 1.0)
            return
        taken = combat_damage(self.rng, -difference)
        self._damage_unit(unit, taken, city.owner)
        self._damage_city(city, dealt, unit.hp > 0, self.wall_multiplier(unit, city))
        if city.hp <= 0 and unit.hp > 0:
            self._capture(city, unit)

    def _damage_city(self, city, damage, can_capture, wall_multiplier):
        damage = max(0.0, float(damage))
        if city.wall_hp > 0:
            if damage * wall_multiplier < city.wall_hp:
                wall_damage, damage = damage * wall_multiplier, 0.0
            else:
                wall_damage = city.wall_hp
                damage -= wall_damage / wall_multiplier
            city.wall_hp -= wall_damage
            city.damaged = city.damaged or wall_damage > 0
        actual = min(city.hp if can_capture else max(0.0, city.hp - 1), damage)
        if actual > 0:
            city.hp -= actual
            city.damaged = True

    def _capture(self, city, unit):
        loser = city.owner
        for layer in (self.military_at, self.civilian_at):
            occupant = layer.get(city.position)
            if occupant is not None and occupant.owner == loser:
                self._kill(occupant, unit.owner)
        self.players[loser].routes = [r for r in self.players[loser].routes if city.id not in r]
        city.owner = unit.owner
        city.hp, city.wall_hp = float(CAPTURED_CITY_HP), 0.0
        city.population = max(1, city.population - 1)
        city.item, city.production, city.food = None, 0.0, 0.0
        city.damaged = True
        self._city_version += 1
        self._event('city_captured', city=city.name, by=unit.owner, player=loser)
        if not self.blocked(unit, city.position):
            self._relocate(unit, city.position)
        self._check_elimination(loser)

    def _check_elimination(self, owner):
        player = self.players[owner]
        if player.alive and not self.player_cities(owner) and not self.player_units(owner, 'settler'):
            player.alive = False
            for unit in list(self.player_units(owner)):
                self._remove_unit(unit)
            self._event('player_eliminated', player=owner)

    def _city_strikes(self, owner):
        """Walled cities shoot the weakest enemy unit within range."""
        for city in sorted(self.player_cities(owner), key=lambda c: c.id):
            if city.wall_hp <= 0:
                continue
            enemies = [u for u in self.units if u.owner != owner and u.spec.military
                       and hex_distance(u.position, city.position) <= R.CITY_RANGED_RANGE]
            if not enemies:
                continue
            target = min(enemies, key=lambda u: (u.hp, u.id))
            strength = self.best_strength(owner, ranged=True)
            damage = combat_damage(self.rng, strength - self.unit_strength(target, False))
            self._event('city_struck', player=owner, city=city.name, target=target.kind)
            self._damage_unit(target, damage, owner)

    # --------------------------------------------------------- builders etc

    def builder_work_tile(self, unit):
        """Where unit's 'improve' would build: its own tile, or an adjacent coast resource for fishing boats."""
        if self.improvable(unit.owner, unit.position) is not None:
            return unit.position
        if self.embarked(unit):
            return None
        for position in sorted(self._neighbors(unit.position)):
            if self.map.grid[position].is_water and self.improvable(unit.owner, position) is not None:
                return position
        return None

    def improve(self, unit):
        position = self.builder_work_tile(unit)
        if position is None:
            raise ValueError('nothing to improve here')
        improvement = self.improvable(unit.owner, position)
        self.improvements[position] = improvement
        unit.charges -= 1
        unit.acted = True
        unit.moves = 0.0
        self._event('tile_improved', player=unit.owner, improvement=improvement, position=list(position))
        resource = self.resources.get(position)
        if improvement == 'quarry':
            self._boost(unit.owner, 'quarry')
        elif improvement == 'mine' and resource is not None:
            self._boost(unit.owner, 'mine_resource')
            if resource == 'iron':
                self._boost(unit.owner, 'mine_iron')
        elif improvement == 'pasture':
            self._boost(unit.owner, 'pasture')
        elif improvement == 'farm' and resource is not None:
            self._boost(unit.owner, 'farm_resource')
        if unit.charges <= 0:
            self._remove_unit(unit)

    def land_path(self, owner, origin, destination):
        """Fewest-step path over open land between two of owner's cities, avoiding foreign cities."""
        def passable(p):
            city = self.city_at.get(p)
            return self._land(p) and (city is None or city.owner == owner)
        distances = self._bfs([origin], passable)
        if destination not in distances:
            return None
        path = [destination]
        while path[-1] != origin:
            path.append(min((p for p in self._neighbors(path[-1]) if distances.get(p) == distances[path[-1]] - 1)))
        return path[::-1]

    def route_destinations(self, owner, origin):
        """Own cities a trader from origin can reach over land, nearest first."""
        used = {destination for start, destination in self.players[owner].routes if start == origin.id}
        options = []
        for city in self.player_cities(owner):
            if city is origin or city.id in used:
                continue
            path = self.land_path(owner, origin.position, city.position)
            if path is not None:
                options.append((len(path), city.id, city))
        return [city for _, _, city in sorted(options, key=lambda option: option[:2])]

    def trader_origin(self, unit):
        own = self.player_cities(unit.owner)
        if not own:
            return None
        return min(own, key=lambda c: (hex_distance(c.position, unit.position), c.id))

    def establish_route(self, unit):
        origin = self.trader_origin(unit)
        player = self.players[unit.owner]
        if origin is None or len(player.routes) >= self.route_capacity(unit.owner):
            raise ValueError('no trade route is available')
        destinations = self.route_destinations(unit.owner, origin)
        if not destinations:
            raise ValueError('no trade route is available')
        destination = destinations[0]
        for position in self.land_path(unit.owner, origin.position, destination.position):
            self.roads.add(position)
        player.routes.append((origin.id, destination.id))
        self._remove_unit(unit)
        self._event('trade_route', player=unit.owner, origin=origin.name, destination=destination.name)
        self._boost(unit.owner, 'trade_route')

    # ------------------------------------------------------------- turn end

    def _city_turn(self, city, available):
        player = self.players[city.owner]
        yields = self.city_yields(city, available)
        surplus = self.food_surplus(city, yields)
        if surplus > 0:
            _, growth, _ = R.amenity_mood(self.amenity_surplus(city, available))
            surplus *= growth_multiplier(city.population, self.housing(city)) * growth
        city.food += surplus
        if city.food < 0:
            city.food = 0.0
            if city.population > 1:
                city.population -= 1
                self._event('city_starved', player=city.owner, city=city.name, population=city.population)
        elif city.food >= self.growth_threshold(city.population):
            city.food -= self.growth_threshold(city.population)
            city.population += 1
            self._event('city_grew', player=city.owner, city=city.name, population=city.population)
        if city.item is not None:
            city.production += yields.get('production', 0)
            cost = self.item_cost(city.owner, city.item)
            if city.production >= cost:
                if self._complete(city, city.item):
                    city.production -= cost
                    city.item = None
                else:
                    city.production = min(city.production, cost)
        candidates = self._border_candidates(city)
        if candidates:
            city.culture += yields.get('culture', 0)
            cost = self.border_cost(city.tiles_acquired)
            if city.culture >= cost:
                city.culture -= cost
                city.tiles_acquired += 1
                self._claim(city, candidates[0])
                self._city_version += 1
                self._event('border_expanded', player=city.owner, city=city.name, tile=list(candidates[0]))
        return yields

    def _border_candidates(self, city):
        candidates = {q for p in city.territory for q in self._neighbors(p)
                      if q not in self.tile_owner and hex_distance(q, city.position) <= WORK_RADIUS}
        return sorted(candidates, key=lambda p: (-tile_score(self.tile_yields(p)),
                                                 -self.tile_yields(p).get('food', 0),
                                                 hex_distance(p, city.position), p))

    def _spawn_tile(self, city, kind):
        """The city tile, or a free adjacent tile of the right domain, for a new unit."""
        probe = Unit(0, city.owner, kind, city.position)
        layer = self._occupancy(probe)
        if city.position not in layer:
            return city.position
        sea = probe.spec.domain == 'sea'
        for position in sorted(self._neighbors(city.position)):
            cell = self.map.grid[position]
            if position in layer or position in self.city_at or cell.spec.impassable:
                continue
            if (sea and cell.terrain_type == 'COAST') or (not sea and not cell.is_water):
                if not self.blocked(probe, position) and not any(
                        u.owner != city.owner for u in (self.military_at.get(position), self.civilian_at.get(position))
                        if u is not None):
                    return position
        return None

    def _complete(self, city, item):
        player = self.players[city.owner]
        if item in R.UNITS:
            if item == 'settler' and city.population < R.SETTLER_MIN_POPULATION:
                return False
            position = self._spawn_tile(city, item)
            if position is None:
                return False
            if item == 'settler':
                city.population -= 1
                player.settlers_trained += 1
            unit = self._add_unit(city.owner, item, position)
            unit.moves = 0.0
        else:
            city.buildings.append(item)
            if item == 'ancient_walls':
                city.wall_hp = city.wall_hp_max = float(R.WALL_HP)
                self._boost(city.owner, 'ancient_walls')
        self._event('item_completed', player=city.owner, city=city.name, item=item)
        return True

    def net_income(self, owner):
        """Gold per turn from owner's cities after maintenance."""
        available = self.amenities(owner)
        gold = sum(self.city_yields(c, available).get('gold', 0) for c in self.player_cities(owner))
        return gold - self.maintenance(owner)

    def maintenance(self, owner):
        units = sum(u.spec.maintenance for u in self.player_units(owner))
        return units + sum(R.BUILDINGS[name].maintenance for c in self.player_cities(owner) for name in c.buildings)

    def _heal(self, unit):
        if unit.acted or unit.hp >= 100:
            return
        holder = self.tile_owner.get(unit.position)
        if self.embarked(unit):
            return
        if unit.spec.domain == 'sea' and (holder is None or holder.owner != unit.owner):
            return
        city = self.city_at.get(unit.position)
        if city is not None and city.owner == unit.owner:
            amount = R.HEAL_IN_CITY
        elif holder is None:
            amount = R.HEAL_NEUTRAL
        else:
            amount = R.HEAL_FRIENDLY if holder.owner == unit.owner else R.HEAL_ENEMY
        unit.hp = min(100.0, unit.hp + amount)

    def _end_turn(self):
        for player in self.players:
            if not player.alive:
                continue
            available = self.amenities(player.index)
            gold = science = 0.0
            for city in sorted(self.player_cities(player.index), key=lambda c: c.id):
                yields = self._city_turn(city, available)
                gold += yields.get('gold', 0)
                science += yields.get('science', 0)
            player.gold += gold - self.maintenance(player.index)
            if player.research is not None:
                science += player.progress.pop('_overflow', 0.0)
            self._research(player.index, science)
            if player.gold < 0:
                # Bankrupt: one unit that costs maintenance is disbanded each turn.
                paid = [u for u in self.player_units(player.index) if u.spec.maintenance]
                player.gold = 0.0
                if paid:
                    disbanded = max(paid, key=lambda u: (u.spec.maintenance, u.id))
                    self._remove_unit(disbanded)
                    self._event('unit_disbanded', player=player.index, unit=disbanded.kind)
        for unit in list(self.units):
            self._heal(unit)
        for city in self.cities:
            if not city.damaged:
                city.hp = min(city.hp_max, city.hp + R.CITY_HEAL)
            city.damaged = False
        self._meet()
        self._reset_moves()

    def _meet(self):
        for player in self.players:
            for other in self.players:
                if other.index <= player.index or other.index in player.met:
                    continue
                mine = [u.position for u in self.player_units(player.index)] + [
                    c.position for c in self.player_cities(player.index)]
                theirs = [u.position for u in self.player_units(other.index)] + [
                    c.position for c in self.player_cities(other.index)]
                if any(hex_distance(a, b) <= 3 for a in mine for b in theirs):
                    player.met.add(other.index)
                    other.met.add(player.index)
                    self._boost(player.index, 'met_civilization')
                    self._boost(other.index, 'met_civilization')

    # ---------------------------------------------------------------- score

    def score(self, owner):
        w = R.SCORE_WEIGHTS
        cities = self.player_cities(owner)
        return (w['city'] * len(cities) + w['citizen'] * sum(c.population for c in cities)
                + w['technology'] * len(self.players[owner].known)
                + w['building'] * sum(len(c.buildings) for c in cities))

    def _dominates(self, owner):
        """Whether owner holds every other player's original capital (a never-founded one counts if eliminated)."""
        for other in self.players:
            if other.index == owner:
                continue
            capital = next((c for c in self.cities if c.capital and c.original_owner == other.index), None)
            if capital is None:
                if other.alive:
                    return False
            elif capital.owner != owner:
                return False
        return len(self.players) > 1

    def _conquest(self):
        """'win' or 'loss' by domination or elimination, else 'ongoing'."""
        if not self.players[0].alive:
            return 'loss'
        if self._dominates(0):
            return 'win'
        if any(self._dominates(p.index) for p in self.players[1:] if p.alive):
            return 'loss'
        return 'ongoing'

    def _outcome(self):
        outcome = self._conquest()
        if outcome != 'ongoing':
            return outcome
        if self.turn_number >= self.config.turn_limit:
            rival = max((self.score(p.index) for p in self.players[1:]), default=None)
            if rival is None or rival == self.score(0):
                return 'timeout'  # No rival to outscore, or an exact tie.
            return 'win' if self.score(0) > rival else 'loss'
        return 'ongoing'

    def summary(self):
        """Numbers worth averaging over evaluation games."""
        cities = self.player_cities(0)
        rival = max((self.score(p.index) for p in self.players[1:]), default=0)
        return {'score': self.score(0), 'rival_score': rival, 'cities': len(cities),
                'population': sum(c.population for c in cities), 'technologies': len(self.players[0].known),
                'gold': round(self.players[0].gold, 1)}

    # ------------------------------------------------------------ decisions

    def observe(self):
        return self._decisions

    def valid_actions(self):
        return tuple(decision.valid for decision in self._decisions)

    def decisions(self, owner):
        return self._decisions if owner == 0 else self._build_decisions(owner)

    def stage(self):
        left = self.config.turn_limit - self.turn_number
        return 0 if left <= 10 else 1 if left <= 25 else 2 if left <= 50 else 3

    def threats(self, owner):
        """Enemy military units inside owner's territory or within three tiles of one of its cities."""
        cities = self.player_cities(owner)
        found = []
        for unit in self.units:
            if unit.owner == owner or not unit.spec.military:
                continue
            holder = self.tile_owner.get(unit.position)
            if (holder is not None and holder.owner == owner) or any(
                    hex_distance(unit.position, c.position) <= 3 for c in cities):
                found.append(unit)
        return found

    def military_power(self, owner):
        return sum(max(u.spec.strength, u.spec.ranged_strength) * u.hp / 100 for u in self.player_units(owner)
                   if u.spec.military)

    def balance(self, owner):
        rivals = [self.military_power(index) for index in self.enemies(owner)]
        if not rivals:
            return 2
        ratio = (self.military_power(owner) + 1) / (max(rivals) + 1)
        return 0 if ratio < .75 else 2 if ratio > 4 / 3 else 1

    def separated(self, owner):
        """Whether no rival city or land unit is reachable over land from owner's cities and land units."""
        starts = [c.position for c in self.player_cities(owner)] + [u.position for u in self.player_units(owner)
                                                                    if u.spec.domain == 'land']
        targets = {c.position for c in self.cities if c.owner != owner} | {
            u.position for u in self.units if u.owner != owner and u.spec.domain == 'land'}
        if not starts or not targets:
            return 0
        reach = self._bfs(starts, lambda p: self._land(p) or p in targets)
        return int(not targets & set(reach))

    def enemy_city_distances(self, owner, sea=False):
        """Steps from each tile to the nearest enemy city, over coast for ships and over land (and coast once
        embarking is known) for land units."""
        embark = self.can_embark(owner)
        key = owner, sea, embark, self._city_version
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached
        targets = [c.position for c in self.cities if c.owner != owner]
        grid = self.map.grid
        if sea:
            def passable(p):
                return grid[p].terrain_type == 'COAST'
        else:
            def passable(p):
                return self._land(p) or (embark and grid[p].terrain_type == 'COAST')
        distances = self._bfs(targets, lambda p: p not in self.city_at and passable(p))
        self._distance_cache[key] = distances
        return distances

    def ranked_sites(self, unit, values=None):
        """Sites for a settler, best first: (score, position, value), with score = value minus travel."""
        values = self.site_values() if values is None else values
        embark = self.can_embark(unit.owner)

        def passable(p):
            cell = self.map.grid.get(p)
            return cell is not None and (self._land(p) or (embark and cell.terrain_type == 'COAST'))
        reach = self._bfs([unit.position], passable)
        reserved = {u.target for u in self.player_units(unit.owner, 'settler') if u is not unit and u.target}
        ranked = [(value - SITE_DISTANCE_PENALTY * reach[p], p, value) for p, value in values.items()
                  if p in reach and p not in reserved and self.legal_site(unit.owner, p)
                  and not any(u.owner != unit.owner for u in (self.military_at.get(p), self.civilian_at.get(p))
                              if u is not None)]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[:3]

    def builder_targets(self, unit, resources):
        """Improvable tiles for a builder, nearest first, skipping other builders' targets."""
        reserved = {u.target for u in self.player_units(unit.owner, 'builder') if u is not unit and u.target}
        targets = []
        for city in self.player_cities(unit.owner):
            for position in city.territory:
                improvement = self.improvable(unit.owner, position)
                if improvement is None or position in reserved:
                    continue
                if (position in self.resources) != resources:
                    continue
                targets.append((hex_distance(unit.position, position), -tile_score(self.tile_yields(position)),
                                position))
        return [position for *_, position in sorted(targets)]

    def _enemy_units_near(self, owner, position, radius):
        return [u for u in self.units if u.owner != owner and u.spec.military
                and hex_distance(u.position, position) <= radius]

    def _build_decisions(self, owner):
        player = self.players[owner]
        if not player.alive or self._done:
            return ()
        stage = self.stage()
        cities = sorted(self.player_cities(owner), key=lambda c: c.id)
        city_count = min(4, len(cities))
        threats = self.threats(owner)
        threat = min(2, len(threats))
        balance = self.balance(owner)
        decisions = []
        available = self.available_technologies(owner)
        if player.research is not None or available:
            index = OPTION_INDEX['research']
            valid = ((index[player.research],) if player.research is not None
                     else tuple(index[name] for name in available))
            features = (stage, min(4, len(player.known) // 3), city_count,
                        int(any(self.coastal(c.position) for c in cities)), threat, balance, self.separated(owner))
            decisions.append(Decision('research', -1, features, valid))
        values = self.site_values()
        settlers = len(self.player_units(owner, 'settler')) + sum(c.item == 'settler' for c in cities)
        builders = self.player_units(owner, 'builder')
        military = [u for u in self.player_units(owner) if u.spec.military]
        amenities = self.amenities(owner)
        for city in cities:
            decisions.append(self._city_decision(city, stage, city_count, values, settlers, builders, military,
                                                 amenities))
        land_distances = self.enemy_city_distances(owner)
        sea_distances = self.enemy_city_distances(owner, sea=True)
        unescorted = int(any(not any(self.military_at.get(p) is not None and self.military_at[p].owner == owner
                                     for p in [s.position] + self._neighbors(s.position))
                             for s in self.player_units(owner, 'settler')))
        units = sorted(self.player_units(owner), key=lambda u: (UNIT_KIND_ORDER.index(u.decision_kind), u.id))
        for unit in units:
            kind = unit.decision_kind
            if kind == 'settler':
                decisions.append(self._settler_decision(unit, stage, city_count, values))
            elif kind == 'builder':
                decisions.append(self._builder_decision(unit, stage))
            elif kind == 'trader':
                origin = self.trader_origin(unit)
                routes = int(origin is not None and len(player.routes) < self.route_capacity(owner)
                             and bool(self.route_destinations(owner, origin)))
                index = OPTION_INDEX['trader']
                valid = ((index['route'],) if routes else ()) + (index['hold'],)
                decisions.append(Decision('trader', unit.id, (stage, routes), valid))
            else:
                distances = sea_distances if kind == 'naval' else land_distances
                decisions.append(self._military_decision(unit, kind, stage, threats, balance, distances, unescorted))
        return tuple(decisions)

    def _city_decision(self, city, stage, city_count, values, settlers, builders, military, amenities):
        owner = city.owner
        index = OPTION_INDEX['city']
        if city.item is None:
            valid = tuple(index[item] for item in CITY_ITEMS if self.can_build(city, item))
        else:
            valid = (index[city.item],)
            if self.players[owner].gold >= self.item_price(owner, city.item):
                valid += (index['buy'],)
        yields = self.city_yields(city, amenities)
        surplus = self.amenity_surplus(city, amenities)
        improvable = any(self.improvable(owner, p) is not None for p in city.territory)
        builder_level = 0 if not improvable else 2 if builders else 1
        defenders = sum(u.spec.domain == 'land' and hex_distance(u.position, city.position) <= 1 for u in military)
        threat = len(self._enemy_units_near(owner, city.position, 3))
        sites = sum(hex_distance(p, city.position) <= 6 and self.legal_site(owner, p) for p in values)
        army = 0 if len(military) < len(self.player_cities(owner)) else 1 if len(military) < 2 * len(
            self.player_cities(owner)) + 2 else 2
        features = (stage, int(city.item is None), int(city.has_palace),
                    _bucket(city.population, (2, 3, 5, 8)),
                    _bucket(self.housing(city) - city.population, (1, 2)),
                    0 if surplus < 0 else 1 if surplus == 0 else 2,
                    _bucket(self.food_surplus(city, yields), (.01, 3)),
                    _bucket(yields.get('production', 0), (3, 5, 8)),
                    min(2, sites), min(2, settlers), builder_level, min(2, defenders), min(2, threat),
                    int(self.coastal(city.position)), city_count, int(city.wall_hp_max > 0), army)
        return Decision('city', city.id, features, valid)

    def _settler_decision(self, unit, stage, city_count, values):
        ranked = self.ranked_sites(unit, values)
        index = OPTION_INDEX['settler']
        legal = self.legal_site(unit.owner, unit.position)
        if legal:
            here_value = values.get(unit.position, 0)
            best_value = max([here_value] + [value for *_, value in ranked])
            ratio = here_value / max(1, best_value)
            here = 1 if ratio < .75 else 2 if ratio < .95 else 3
        else:
            here = 0
        if ranked:
            best = _bucket(ranked[0][2], (40, 60)) + 1
            distance = hex_distance(unit.position, ranked[0][1])
            distance = 0 if distance == 0 else 1 if distance <= 2 else 2 if distance <= 5 else 3
        else:
            best, distance = 0, 3
        threat = min(2, len(self._enemy_units_near(unit.owner, unit.position, 2)))
        escort = int(any(self.military_at.get(p) is not None and self.military_at[p].owner == unit.owner
                         for p in [unit.position] + self._neighbors(unit.position)))
        valid = ((index['found'],) if legal else ()) + tuple(index[f'site_{k}'] for k in range(1, len(ranked) + 1))
        valid += (index['hold'],)
        return Decision('settler', unit.id, (stage, here, best, distance, threat, escort, city_count), valid)

    def _builder_decision(self, unit, stage):
        index = OPTION_INDEX['builder']
        work = self.builder_work_tile(unit)
        if work is None:
            here = 0
        else:
            resource = self.resources.get(work)
            here = 1 if resource is None else 3 if R.RESOURCES[resource].kind == 'luxury' else 2
        resources = self.builder_targets(unit, True)
        tiles = self.builder_targets(unit, False)
        targets = len(resources) + len(tiles)
        nearest = min([hex_distance(unit.position, p) for p in resources[:1] + tiles[:1]], default=None)
        distance = 3 if nearest is None else 0 if nearest == 0 else 1 if nearest <= 2 else 2 if nearest <= 5 else 3
        threat = min(2, len(self._enemy_units_near(unit.owner, unit.position, 2)))
        valid = (((index['improve'],) if work is not None else ())
                 + ((index['goto_resource'],) if resources else ()) + ((index['goto_tile'],) if tiles else ())
                 + (index['hold'],))
        features = (stage, here, _bucket(targets, (1, 2, 4)), distance, min(2, unit.charges - 1), threat)
        return Decision('builder', unit.id, features, valid)

    def best_attack(self, unit, targets=None):
        """The attackable target with the best odds: (target, success chance, death chance) or None."""
        targets = self.attack_targets(unit) if targets is None else targets
        best = None
        for target in targets:
            success, death = self.attack_odds(unit, target)
            if isinstance(target, City) and unit.spec.ranged:
                success = .5  # Bombarding wears the city down but cannot take it.
            key = (success - death, -target.hp, -target.id)
            if best is None or key > best[0]:
                best = key, target, success, death
        return None if best is None else best[1:]

    def _military_decision(self, unit, kind, stage, threats, balance, distances, unescorted):
        owner = unit.owner
        threat = min(2, len(threats))
        index = OPTION_INDEX[kind]
        targets = self.attack_targets(unit)
        contact = 2 if any(isinstance(t, City) for t in targets) else 1 if targets else 0
        best = self.best_attack(unit, targets)
        if best is None:
            odds = 0
        else:
            _, success, death = best
            odds = 3 if unit.spec.ranged or death < .1 or success >= .5 else 1 if death >= .5 else 2
        distance = distances.get(unit.position)
        if contact == 2:
            enemy_city = 0
        elif distance is None:
            enemy_city = 3
        else:
            enemy_city = 1 if distance <= 3 else 2 if distance <= 8 else 3
        health = 3 if unit.hp >= 100 else min(2, int(3 * unit.hp / 100))
        enemy_units = any(u.owner != owner for u in self.units)
        enemy_cities = any(c.owner != owner for c in self.cities)
        if kind == 'naval':
            unit_type = R.NAVAL_UNITS.index(unit.kind)
            coastal_enemy = any(c.owner != owner and self.coastal(c.position) for c in self.cities)
            own_coastal = any(self.coastal(c.position) for c in self.player_cities(owner))
            valid = ((index['fortify'],) + ((index['hunt'],) if enemy_units else ())
                     + ((index['raid'],) if coastal_enemy else ()) + ((index['patrol'],) if own_coastal else ()))
            features = (stage, unit_type, health, threat, contact, odds, enemy_city, balance)
            return Decision('naval', unit.id, features, valid)
        holder = self.tile_owner.get(unit.position)
        territory = 1 if holder is None else 0 if holder.owner == owner else 2
        fighter = unit.spec.can_attack
        post = self.post(unit)
        valid = ((index['fortify'],) + ((index['garrison'],) if self.player_cities(owner) else ())
                 + ((index['defend'],) if fighter and threats else ())
                 + ((index['hunt'],) if fighter and enemy_units else ())
                 + ((index['siege'],) if enemy_cities else ())
                 + ((index['escort'],) if self.player_units(owner, 'settler') else ()))
        features = (stage, R.LAND_MILITARY.index(unit.kind), health, threat, contact, odds, enemy_city, balance,
                    territory, post, unescorted)
        return Decision('land', unit.id, features, valid)

    def post(self, unit):
        """0 away from own cities, 1 one of several defenders beside a city, 2 a city's only defender."""
        cities = [c for c in self.player_cities(unit.owner) if hex_distance(unit.position, c.position) <= 1]
        if not cities or not unit.spec.can_attack:
            return 0
        city = min(cities, key=lambda c: (hex_distance(unit.position, c.position), c.id))
        others = any(u is not unit and u.owner == unit.owner and u.spec.can_attack and u.spec.domain == 'land'
                     and hex_distance(u.position, city.position) <= 1 for u in self.units)
        return 1 if others else 2

    # --------------------------------------------------------------- orders

    def _entity(self, decision):
        if decision.kind == 'research':
            return None
        if decision.kind == 'city':
            return next((c for c in self.cities if c.id == decision.entity), None)
        return next((u for u in self.units if u.id == decision.entity), None)

    def _play(self, owner, decisions, choices):
        player = self.players[owner]
        self._reset_moves(owner)
        for decision, choice in zip(decisions, choices):
            option = DECISION_KINDS[decision.kind].options[choice]
            entity = self._entity(decision)
            if decision.kind == 'research':
                self._set_research(owner, option)
            elif decision.kind == 'city':
                if entity is None or entity.owner != owner:
                    continue
                if option == 'buy':
                    price = self.item_price(owner, entity.item)
                    if player.gold >= price and self._complete(entity, entity.item):
                        player.gold -= price
                        entity.item, entity.production = None, 0.0
                        self._event('item_purchased', player=owner, city=entity.name, price=price)
                elif entity.item is None:
                    entity.item = option
            elif entity is not None and entity.owner == owner:
                self._order(entity, decision.kind, option)
        self._city_strikes(owner)

    def _order(self, unit, kind, option):
        if kind == 'settler':
            if option == 'hold':
                unit.target = None
            elif option == 'found':
                if self.legal_site(unit.owner, unit.position):
                    self.found_city(unit)
            elif option.startswith('site_'):
                ranked = self.ranked_sites(unit)
                rank = int(option[-1]) - 1
                if rank < len(ranked):
                    site = ranked[rank][1]
                    unit.target = site
                    self.travel(unit, [site])
        elif kind == 'builder':
            if option == 'improve':
                if self.builder_work_tile(unit) is not None:
                    self.improve(unit)
            elif option in ('goto_resource', 'goto_tile'):
                targets = self.builder_targets(unit, option == 'goto_resource')
                if targets:
                    goal = targets[0]
                    if self.map.grid[goal].is_water and not self.can_embark(unit.owner):
                        goals = [p for p in self._neighbors(goal) if self._land(p)]
                    else:
                        goals = [goal]
                    self.travel(unit, goals)
                    unit.target = goal
        elif kind == 'trader':
            if option == 'route':
                origin = self.trader_origin(unit)
                if (origin is not None and len(self.players[unit.owner].routes) < self.route_capacity(unit.owner)
                        and self.route_destinations(unit.owner, origin)):
                    self.establish_route(unit)
        elif kind == 'land':
            self._land_order(unit, option)
        elif kind == 'naval':
            self._naval_order(unit, option)

    def _strike(self, unit, candidates):
        """Attack the best of candidates that is attackable now; returns whether it attacked."""
        targets = [t for t in self.attack_targets(unit) if t in candidates]
        best = self.best_attack(unit, targets)
        if best is None:
            return False
        self.attack(unit, best[0])
        return True

    def _engage(self, unit, targets):
        """Attack one of targets if possible, otherwise close in and attack if movement remains."""
        if not targets or self._strike(unit, targets):
            return
        reach = unit.spec.attack_range if unit.spec.ranged else 1
        goals = set()
        for target in targets:
            for p in self._within(target.position, reach):
                if p != target.position and p not in self.city_at:
                    goals.add(p)
        if unit.position not in goals:
            self.travel(unit, goals)
        self._strike(unit, targets)

    def _nearest(self, unit, things):
        return sorted(things, key=lambda t: (hex_distance(unit.position, t.position), t.id))

    def garrison_target(self, unit):
        """The own city most in need of a defender, nearest first."""
        cities = self.player_cities(unit.owner)
        if not cities:
            return None

        def need(city):
            threat = len(self._enemy_units_near(unit.owner, city.position, 3))
            defenders = sum(u.spec.military and u.spec.domain == 'land' and u is not unit
                            and hex_distance(u.position, city.position) <= 1 for u in self.units if u.owner == unit.owner)
            return threat - defenders
        return min(cities, key=lambda c: (-need(c), hex_distance(unit.position, c.position), c.id))

    def _land_order(self, unit, option):
        owner = unit.owner
        enemy_units = [u for u in self.units if u.owner != owner]
        enemy_cities = [c for c in self.cities if c.owner != owner]
        if option == 'garrison' or (option == 'defend' and not self.threats(owner)):
            city = self.garrison_target(unit)
            if city is None:
                return
            if self.military_at.get(city.position) in (None, unit):
                goals = [city.position]
            else:
                goals = [p for p in self._neighbors(city.position) if self._land(p)]
            if unit.position not in goals:
                self.travel(unit, goals)
        elif option == 'defend':
            threats = self._nearest(unit, self.threats(owner))
            self._engage(unit, threats[:1] + [t for t in threats[1:] if t in self.attack_targets(unit)])
        elif option == 'hunt':
            near = self._nearest(unit, enemy_units)
            self._engage(unit, near[:1] + [t for t in near[1:] if t in self.attack_targets(unit)])
        elif option == 'siege':
            if not enemy_cities:
                return
            distances = self.enemy_city_distances(owner)
            target = min(enemy_cities, key=lambda c: (min([distances.get(p, math.inf) for p in self._neighbors(
                c.position)] + [math.inf]), hex_distance(unit.position, c.position), c.id))
            if unit.spec.can_attack:
                self._engage(unit, [target])
            elif hex_distance(unit.position, target.position) > 1:
                self.travel(unit, [p for p in self._neighbors(target.position) if self._land(p)])
        elif option == 'escort':
            settlers = self._nearest(unit, self.player_units(owner, 'settler'))
            if settlers and hex_distance(unit.position, settlers[0].position) > 0:
                self.travel(unit, [settlers[0].position] + self._neighbors(settlers[0].position))

    def _naval_order(self, unit, option):
        owner = unit.owner
        if option == 'hunt':
            near = self._nearest(unit, [u for u in self.units if u.owner != owner])
            self._engage(unit, near[:1] + [t for t in near[1:] if t in self.attack_targets(unit)])
        elif option == 'raid':
            cities = self._nearest(unit, [c for c in self.cities if c.owner != owner and self.coastal(c.position)])
            if cities:
                self._engage(unit, cities[:1])
        elif option == 'patrol':
            coastal = [c for c in self.player_cities(owner) if self.coastal(c.position)]
            if coastal:
                city = min(coastal, key=lambda c: (-len(self._enemy_units_near(owner, c.position, 3)),
                                                   hex_distance(unit.position, c.position), c.id))
                goals = [p for p in [city.position] + self._neighbors(city.position)
                         if p == city.position or self.map.grid[p].terrain_type == 'COAST']
                if unit.position not in goals:
                    self.travel(unit, goals)

    # ------------------------------------------------------------------ API

    def guided_action(self, rng=None):
        """The rule-based heuristic's choices for player 0 this turn."""
        import empire_policy
        return empire_policy.choose(self, 0, self._decisions, rng or self.rng)

    def random_action(self, rng):
        return tuple(rng.choice(decision.valid) for decision in self._decisions)

    def _check_action(self, action):
        if isinstance(action, (str, bytes)) or not hasattr(action, '__len__') or len(action) != len(self._decisions):
            raise ValueError(f'action must hold one option index for each of the {len(self._decisions)} decisions')
        choices = []
        for decision, choice in zip(self._decisions, action):
            try:
                index = None if isinstance(choice, bool) else operator.index(choice)
            except TypeError:
                index = None
            if index not in decision.valid:
                raise ValueError(f'option {choice!r} is not valid for the {decision.kind} decision')
            choices.append(index)
        return tuple(choices)

    def step(self, action):
        if not self.players:
            raise RuntimeError('Call reset before step')
        if self._done:
            return self._decisions, 0.0, True
        choices = self._check_action(action)
        import empire_policy
        self.events = []
        self.turn_number += 1
        score_before = self.score(0)
        # Players alternate moving first, so neither gains from always striking or settling first.
        order = [p.index for p in self.players]
        if self.turn_number % 2 == 0:
            order = order[1:] + order[:1]
        for index in order:
            if self._conquest() != 'ongoing':
                break
            if index == 0:
                self._play(0, self._decisions, choices)
            elif self.players[index].alive:
                decisions = self._build_decisions(index)
                self._play(index, decisions, empire_policy.choose(self, index, decisions, self.rng))
        if self._conquest() == 'ongoing':
            self._end_turn()
        outcome = self._outcome()
        self._done = outcome != 'ongoing'
        self.last_info = {'outcome': outcome, 'events': [dict(e) for e in self.events], 'turn': self.turn_number}
        reward = self._reward(score_before, outcome)
        self._decisions = self._build_decisions(0)
        return self._decisions, reward, self._done

    def _reward(self, score_before, outcome):
        reward = REWARD_SCALE * (self.score(0) - score_before)
        for event in self.events:
            kind = event['type']
            if kind == 'unit_killed':
                value = .5 if event['military'] else .25
                if event['player'] == 0:
                    reward -= value
                elif event['by'] == 0:
                    reward += value
            elif kind == 'city_captured':
                reward += 3.0 if event['by'] == 0 else -3.0 if event['player'] == 0 else 0.0
        if outcome == 'win':
            reward += VICTORY_REWARD
        elif outcome == 'loss':
            reward -= VICTORY_REWARD
        return float(reward)

    def status_text(self, owner=0):
        player = self.players[owner]
        if player.research:
            research = (f'{player.research.replace("_", " ").title()} '
                        f'{player.progress.get(player.research, 0):.0f}/{self.tech_cost(player.research)}')
        else:
            research = 'complete' if not self.available_technologies(owner) else 'none'
        return (f'{player.name}: score {self.score(owner)}   cities {len(self.player_cities(owner))}   '
                f'gold {player.gold:.0f}   techs {len(player.known)}   research {research}')

    def render_frame(self, delay_ms=0):
        if not self.render:
            return True
        if self._renderer is None:
            from renderer import EmpireRenderer
            self._renderer = EmpireRenderer(self)
        self.quit = not self._renderer.frame(delay_ms)
        return not self.quit

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
