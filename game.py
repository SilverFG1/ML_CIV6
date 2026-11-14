"""A small Civ-like combat environment with isolated state and seeded randomness."""

from dataclasses import dataclass
import heapq
import itertools
import math
import random

import numpy as np

import constants
from class_hex import HexMap, TERRAIN_CODES


def hex_coords(position):
    x, y = position
    q = x - (y - y % 2) // 2
    return q, -q - y, y


def hex_distance(first, second):
    return max(abs(a - b) for a, b in zip(hex_coords(first), hex_coords(second)))


@dataclass(frozen=True)
class UnitSpec:
    """Default combat profile for a unit type."""
    strength: float
    unit_class: str = 'melee'
    ranged_strength: float = 0
    attack_range: int = 1

    @property
    def ranged(self):
        return self.ranged_strength > 0

    @property
    def support(self):
        """Support units cannot attack or capture the city."""
        return self.unit_class == 'support'


# New unit types use their Civ VI base strengths. Archers keep this project's
# original 12/20 values.
UNIT_SPECS = {
    'warrior': UnitSpec(strength=20),
    'spearman': UnitSpec(strength=25, unit_class='anti_cavalry'),
    'heavy_chariot': UnitSpec(strength=28, unit_class='heavy_cavalry'),
    'slinger': UnitSpec(strength=5, unit_class='ranged', ranged_strength=15, attack_range=1),
    'archer': UnitSpec(strength=12, unit_class='ranged', ranged_strength=20, attack_range=2),
    'battering_ram': UnitSpec(strength=0, unit_class='support'),
}
UNIT_TYPES = tuple(UNIT_SPECS)

# Melee attacks deal this share of their damage to walls. A battering ram next
# to the city restores full wall damage for the classes it supports.
MELEE_WALL_DAMAGE = 0.5
RAM_SUPPORTED_CLASSES = ('melee', 'anti_cavalry')

# City economy, enabled when the city starts with population. Yields follow
# Civ VI: each citizen eats 2 food, the city center makes at least 2 food and
# 1 production, grassland makes 2 food, and woods add 1 production. Mountains
# cannot be worked. A newly founded city owns the six tiles around its center,
# and a tile with an enemy unit on it cannot be worked.
FOOD_PER_CITIZEN = 2
CITY_CENTER_YIELD = (2, 1)  # (food, production)
TILE_YIELDS = {'GRASSLAND': (2, 0), 'FOREST': (2, 1)}
MAX_CITY_POPULATION = 20
# The attacker loses this much reward when the city grows and gains it when
# the city starves.
POPULATION_REWARD = 1.0


# Civilians belong to a populated city. A builder has Civ VI's 3 build charges
# and farms flat grassland in the city's territory (+1 food); Civ VI does not
# allow farms on woods, so forests are left alone. A settler walks to the
# nearest tile at least 4 hexes from every city (Civ VI's minimum spacing) and
# founds a city there. An attacking unit that moves onto a civilian captures it.
CIVILIAN_KINDS = ('builder', 'settler')
BUILDER_CHARGES = 3
FARM_FOOD = 1
MIN_CITY_DISTANCE = 4
MAX_CIVILIANS = 2  # of each kind
CAPTURE_REWARDS = {'builder': 2.0, 'settler': 4.0}
CITY_FOUNDED_PENALTY = 5.0
# The heuristic hunts a civilian within this many hexes instead of besieging.
HUNT_RANGE = 3


@dataclass(frozen=True)
class DistrictSpec:
    tech: str
    cost: int = 54  # base cost; Civ VI raises it with tech and civic progress
    yield_type: str | None = None  # what its adjacency bonus produces
    mountain: int = 0  # adjacency per adjacent mountain
    woods_pair: int = 0  # adjacency per two adjacent woods (forests)
    district_pair: int = 0  # adjacency per two adjacent districts (the city center counts)
    distance: int = 1  # hexes from the city center


@dataclass(frozen=True)
class BuildingSpec:
    tech: str
    cost: int
    district: str = 'city_center'
    yields: tuple = ()  # (yield type, amount) pairs
    walls: float = 0  # outer defense added to the city's walls


# Ancient-era districts and buildings (Gathering Storm). The Water Mill needs a
# river, which these maps lack. The Government Plaza is omitted because its
# buildings need Classical governments. The Encampment cannot be next to the
# city center; this sim has no border growth, so it goes two hexes out.
DISTRICTS = {
    'campus': DistrictSpec('Writing', yield_type='science', mountain=1, district_pair=1),
    'holy_site': DistrictSpec('Astrology', yield_type='faith', mountain=1, woods_pair=1, district_pair=1),
    'encampment': DistrictSpec('Bronze Working', distance=2),
}
BUILDINGS = {
    'monument': BuildingSpec('(none)', 60, yields=(('culture', 2),)),
    'granary': BuildingSpec('Pottery', 65, yields=(('food', 1),)),
    'ancient_walls': BuildingSpec('Masonry', 80, walls=100),
    'library': BuildingSpec('Writing', 80, 'campus', yields=(('science', 2),)),
    'shrine': BuildingSpec('Astrology', 65, 'holy_site', yields=(('faith', 2),)),
    'barracks': BuildingSpec('Bronze Working', 80, 'encampment', yields=(('production', 1),)),
}
# Walls and encampments give the city a ranged strike; Civ VI uses the
# strongest ranged unit's strength, or 15 without one. The city has no units.
CITY_STRIKE_STRENGTH = 15
STRIKE_RANGE = 2
# The attacker loses this much reward whenever the city completes a build.
BUILD_REWARD = 1.0


def district_limit(population):
    """Specialty districts a city of this size may have (Civ VI: one, plus one per 3 citizens)."""
    return 1 + (population - 1) // 3


def growth_threshold(population):
    """Food a city of this size needs to gain a citizen (Civ VI: 15 + 8n + n^1.5)."""
    n = population - 1
    return math.floor(15 + 8 * n + n ** 1.5)


@dataclass(frozen=True)
class GameConfig:
    width: int = 8
    height: int = 8
    unit_count: int = 3
    city_position: tuple | None = None
    city_strength: float = 28
    wall_hp: float = 0
    ranged_strength: float = 0
    city_heals: bool = True
    mountain_density: float = 0
    forest_density: float = 0
    unit_strengths: tuple = ()
    unit_types: tuple = ()
    difficulty: str = 'normal'
    city_population: int = 0
    city_builders: int = 0
    city_settlers: int = 0
    city_districts: tuple = ()
    city_buildings: tuple = ()
    city_build_queue: tuple = ()

    @property
    def city_growth(self):
        """Population, worked tiles and growth are simulated only for a populated city."""
        return self.city_population > 0

    def __post_init__(self):
        for name in ('width', 'height'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 3 <= value <= 32:
                raise ValueError(f'{name} must be an integer between 3 and 32')
        if isinstance(self.unit_count, bool) or not isinstance(self.unit_count, int) or not 1 <= self.unit_count <= 4:
            raise ValueError('unit_count must be between 1 and 4')
        for name in ('city_strength', 'ranged_strength'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f'{name} must be finite and between 0 and 100')
        if not math.isfinite(self.wall_hp) or not 0 <= self.wall_hp <= 1000:
            raise ValueError('wall_hp must be finite and between 0 and 1000')
        for name in ('mountain_density', 'forest_density'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be between 0 and 1')
        if self.mountain_density + self.forest_density > 0.8:
            raise ValueError('combined terrain density must be at most 0.8')
        if self.unit_strengths and len(self.unit_strengths) != self.unit_count:
            raise ValueError('unit_strengths must contain one strength for every unit')
        if any(not math.isfinite(s) or not 0 < s <= 100 for s in self.unit_strengths):
            raise ValueError('unit strengths must be finite and in (0, 100]')
        if self.unit_types and (len(self.unit_types) != self.unit_count
                               or any(t not in UNIT_SPECS for t in self.unit_types)):
            raise ValueError(f'unit_types must name one of {", ".join(UNIT_TYPES)} for every unit')
        if self.unit_types and all(UNIT_SPECS[t].support for t in self.unit_types):
            raise ValueError('unit_types must include at least one unit that can attack')
        if self.difficulty not in ('easy', 'normal', 'hard', 'curriculum'):
            raise ValueError('Unknown difficulty')
        if (isinstance(self.city_population, bool) or not isinstance(self.city_population, int)
                or not 0 <= self.city_population <= MAX_CITY_POPULATION):
            raise ValueError(f'city_population must be an integer between 0 and {MAX_CITY_POPULATION}')
        for name in ('city_builders', 'city_settlers'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_CIVILIANS:
                raise ValueError(f'{name} must be an integer between 0 and {MAX_CIVILIANS}')
        if (self.city_builders or self.city_settlers) and not self.city_growth:
            raise ValueError('builders and settlers need a city with population (city_population >= 1)')
        self._validate_builds()
        if self.city_position is not None:
            if len(self.city_position) != 2 or any(isinstance(v, bool) or not isinstance(v, int)
                                                 for v in self.city_position):
                raise ValueError('city_position must contain two integers')
            x, y = self.city_position
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError('city_position must lie inside the map')


    def _validate_builds(self):
        """Districts and buildings exist once each, and a building needs its district first."""
        groups = (('city_districts', DISTRICTS), ('city_buildings', BUILDINGS),
                  ('city_build_queue', {**DISTRICTS, **BUILDINGS}))
        for name, table in groups:
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(item not in table for item in value):
                raise ValueError(f'{name} must name items from: {", ".join(table)}')
        items = self.city_districts + self.city_buildings + self.city_build_queue
        if not items:
            return
        if not self.city_growth:
            raise ValueError('districts and buildings need a city with population (city_population >= 1)')
        if len(set(items)) != len(items):
            raise ValueError('each district and building can be built only once')
        if len(self.city_districts) > district_limit(self.city_population):
            limit = district_limit(self.city_population)
            raise ValueError(f'a city of {self.city_population} can have at most {limit} '
                             f'district{"s" if limit != 1 else ""}')
        available = {'city_center', *self.city_districts}
        for item in self.city_buildings + self.city_build_queue:
            if item in BUILDINGS and BUILDINGS[item].district not in available:
                raise ValueError(f'{item} needs a {BUILDINGS[item].district} built or queued before it')
            available.add(item)

    @property
    def builds(self):
        """Every district and building in the scenario, built or queued."""
        return self.city_districts + self.city_buildings + self.city_build_queue

class C_Sprite:
    """A simulation object; sprite is retained for older construction callers."""
    def __init__(self, x, y, sprite=None, name_instance='Object', hp=100,
                 hp_max=100, strength=20, env=None):
        self.x, self.y = int(x), int(y)
        self.sprite = sprite
        self.name_instance = name_instance
        self.hp_max = float(hp_max)
        self.hp = float(hp)
        self.strength = float(strength)
        self.alive = self.hp > 0
        self.status = 'alive' if self.alive else 'dead'
        self.status_default = 'alive'
        self.env = env

    @property
    def position(self):
        return self.x, self.y

    def _event(self, kind, **details):
        if self.env is not None:
            self.env.events.append({'type': kind, 'object': self.name_instance, **details})


class C_Unit(C_Sprite):
    def __init__(self, *args, strength_ranged=0, unit_type='warrior', **kwargs):
        if unit_type not in UNIT_SPECS:
            raise ValueError(f'Unknown unit type: {unit_type}')
        super().__init__(*args, **kwargs)
        self.strength_ranged = float(strength_ranged)
        self.unit_type = unit_type
        self.attack_range = UNIT_SPECS[unit_type].attack_range

    @property
    def ranged(self):
        return UNIT_SPECS[self.unit_type].ranged

    @property
    def support(self):
        return UNIT_SPECS[self.unit_type].support

    def move(self, dx, dy):
        if self.env is None:
            raise RuntimeError('Unit must belong to a Game before moving')
        self.env.move_unit(self, (int(dx), int(dy)))

    def take_damage(self, damage, aggressor_alive=True):
        if not self.alive:
            return
        actual = min(self.hp, max(0, float(damage)))
        self.hp -= actual
        self._event('unit_damaged', amount=actual)
        if self.hp <= 0:
            self.death_unit()

    def death_unit(self):
        if self.alive:
            self.hp = 0
            self.alive = False
            self.status = 'dead'
            self._event('unit_died')


class C_City(C_Sprite):
    def __init__(self, x, y, sprite=None, name_instance='City', hp=100,
                 hp_max=100, wall_hp=0, strength=28, strength_ranged=0,
                 ranged_combat=False, heal=True, env=None, population=0):
        super().__init__(x, y, sprite, name_instance, hp, hp_max, strength, env)
        self.wall_hp = float(wall_hp)
        self.wall_hp_max = float(wall_hp)
        self.strength_ranged = float(strength_ranged)
        self.ranged_combat = ranged_combat
        self.heal = heal
        self.population = int(population)
        self.food = 0

    def take_turn(self):
        if self.env is not None and self.alive:
            self.env.city_turn()

    def take_damage(self, damage, aggressor_alive=True, wall_multiplier=1.0):
        """Walls absorb damage first, scaled by wall_multiplier; any excess hits the city."""
        if not self.alive:
            return
        if not 0 < wall_multiplier <= 1:
            raise ValueError('wall_multiplier must lie in (0, 1]')
        damage = max(0, float(damage))
        if self.wall_hp > 0:
            if damage * wall_multiplier < self.wall_hp:
                wall_damage, damage = damage * wall_multiplier, 0
            else:
                wall_damage = self.wall_hp
                damage -= wall_damage / wall_multiplier
            if wall_damage:
                self.wall_hp -= wall_damage
                self._event('wall_damaged', amount=wall_damage)
        actual = min(self.hp if aggressor_alive else max(0, self.hp - 1), damage)
        if actual:
            self.hp -= actual
            self.status = 'took damage'
            self._event('city_damaged', amount=actual)
        if self.hp <= 0:
            self.death()

    def death(self):
        if self.alive:
            self.hp = 0
            self.alive = False
            self.status = 'dead'
            self._event('city_captured')


class C_Civilian(C_Sprite):
    """The city's builder or settler. Civilians cannot fight; attackers capture them."""
    def __init__(self, x, y, kind, name_instance, env=None):
        if kind not in CIVILIAN_KINDS:
            raise ValueError(f'Unknown civilian: {kind}')
        super().__init__(x, y, name_instance=name_instance, strength=0, env=env)
        self.kind = kind
        self.charges = BUILDER_CHARGES if kind == 'builder' else 0
        self.status = 'active'

    def leave(self, status, event, **details):
        """Remove the civilian from play as captured, settled or used up."""
        if self.alive:
            self.alive = False
            self.status = status
            self._event(event, civilian=self.kind, **details)


def attack(aggressor, target, ranged=False, rng=None):
    """Compute combat damage using the environment's private RNG."""
    rng = rng or (aggressor.env.rng if aggressor.env is not None else random.Random())
    offense = aggressor.strength_ranged if ranged else aggressor.strength
    defense = target.strength
    if isinstance(target, C_Unit) and target.env is not None:
        defense += target.env.map.grid[target.position].defense_bonus
    strength_diff = offense - defense
    damage_out = round(rng.randint(24, 36) * math.exp(strength_diff / 25) * rng.uniform(.75, 1))
    if ranged:
        return damage_out, 0
    retaliation_diff = target.strength - aggressor.strength
    if aggressor.env is not None:
        retaliation_diff -= aggressor.env.map.grid[aggressor.position].defense_bonus
    damage_taken = round(rng.randint(24, 36) * math.exp(retaliation_diff / 25) * rng.uniform(.75, 1))
    return damage_out, damage_taken


class Game:
    def __init__(self, human=False, ml_ai=False, render=False, config=None, seed=None):
        self.config = config or GameConfig()
        self.human, self.ml_ai, self.render = human, ml_ai, render
        self.rng = random.Random(seed)
        self.unit_types = tuple(self.config.unit_types or ('warrior',) * self.config.unit_count)
        self.unit_directions = [constants.DIRECTIONS + (('SHOOT',) if UNIT_SPECS[kind].ranged else ())
                                for kind in self.unit_types]
        self.actions = tuple(itertools.product(*self.unit_directions))
        self.action_count = len(self.actions)
        self._action_index = {a: i for i, a in enumerate(self.actions)}
        self.civilian_kinds = (('builder',) * self.config.city_builders
                               + ('settler',) * self.config.city_settlers)
        self.observation_schema = (f'ml_civ6.v2:{self.config.width}x{self.config.height}:'
                                   + ','.join(self.unit_types)
                                   + (':city_growth' if self.config.city_growth else '')
                                   + (':civilians=' + ','.join(self.civilian_kinds) if self.civilian_kinds else '')
                                   + (':built=' + ','.join(self.config.city_districts + self.config.city_buildings)
                                      if self.config.city_districts or self.config.city_buildings else '')
                                   + (':queue=' + ','.join(self.config.city_build_queue)
                                      if self.config.city_build_queue else ''))
        self._renderer = None
        self.quit = False
        self.units = []
        self.civilians = []
        self.farms = set()
        self.founded_cities = []
        self.districts = {}
        self.buildings = []
        self.build_progress = {}
        self.completed = []
        self.city = None
        self.events = []
        self.last_info = {'outcome': 'ongoing', 'events': []}
        self.turn_number = 0
        self._done = False
        self._valid_cache_key = None

    @property
    def objects(self):
        return self.units + [self.city] if self.city is not None else self.units

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        c = self.config
        city_pos = tuple(c.city_position or (c.width // 2, c.height // 2))
        self.map = HexMap(c.height, c.width)
        # Keep enough connected grassland for starting units and a possible siege.
        for _ in range(100):
            for cell in self.map.grid.values():
                draw = self.rng.random()
                terrain = ('MOUNTAIN' if draw < c.mountain_density else
                           'FOREST' if draw < c.mountain_density + c.forest_density else 'GRASSLAND')
                cell.set_terrain(terrain)
            self.map.grid[city_pos].set_terrain('GRASSLAND')
            component = self.map.reachable(city_pos)
            adjacent = self.map.grid[city_pos].get_neighbors(self.map.grid)
            needed = c.unit_count + len(self.civilian_kinds) + 1 + ('encampment' in c.city_districts)
            if len(component) >= needed and sum(not cell.block_path for cell in adjacent) >= min(c.unit_count, 3):
                break
        else:
            for cell in self.map.grid.values():
                cell.set_terrain('GRASSLAND')
            component = self.map.reachable(city_pos)
        self.city = C_City(*city_pos, name_instance='Ottertopia', strength=c.city_strength,
                           wall_hp=c.wall_hp, strength_ranged=c.ranged_strength,
                           ranged_combat=c.ranged_strength > 0, heal=c.city_heals, env=self,
                           population=c.city_population)
        self.units, self.civilians = [], []
        self.farms = set()
        self.founded_cities = []
        self.districts, self.buildings, self.build_progress, self.completed = {}, [], {}, []
        self._valid_cache_key = self._path_costs = None
        # Starting districts and buildings exist before the squad arrives.
        for item in c.city_districts + c.city_buildings:
            self.complete_build(item, announce=False)
        if 'encampment' in self.districts:
            component = self.map.reachable(city_pos)
        start_positions = self.rng.sample(sorted(component - {city_pos}), c.unit_count)
        specs = [UNIT_SPECS[kind] for kind in self.unit_types]
        strengths = c.unit_strengths or tuple(spec.strength for spec in specs)
        names = ('Otto', 'Fynn', 'Victor', 'Ada')
        self.units = [C_Unit(*pos, name_instance=names[i], strength=strengths[i], env=self,
                             unit_type=self.unit_types[i], strength_ranged=specs[i].ranged_strength)
                      for i, pos in enumerate(start_positions)]
        self.civilians = self._place_civilians(component) if self.civilian_kinds else []
        self.turn_number = 0
        self.events = []
        self.last_info = {'outcome': 'ongoing', 'events': []}
        self._done = self.quit = False
        self._valid_cache_key = None
        self._path_costs = None
        return self.get_observation()

    def _place_civilians(self, component):
        """Start civilians on free territory tiles, then on the free tiles nearest the city."""
        taken = {u.position for u in self.units} | {self.city.position}
        territory = [pos for pos in self.territory() if pos in component and pos not in taken]
        self.rng.shuffle(territory)
        nearby = sorted((pos for pos in component if pos not in taken and pos not in territory),
                        key=lambda pos: (hex_distance(pos, self.city.position), pos[1], pos[0]))
        spots = territory + nearby
        counts = {}
        civilians = []
        for kind, pos in zip(self.civilian_kinds, spots):
            counts[kind] = counts.get(kind, 0) + 1
            name = kind.capitalize() + ('' if counts[kind] == 1 else f' {counts[kind]}')
            civilians.append(C_Civilian(*pos, kind, name, env=self))
        return civilians

    def civilian_at(self, position):
        return next((c for c in self.civilians if c.alive and c.position == tuple(position)), None)

    def game_initialize(self, ep_number=0, seed=None):
        self.episode_number = ep_number
        return self.reset(seed)

    def occupant(self, position, exclude=None):
        return next((obj for obj in self.objects if obj is not exclude
                     and obj.alive and obj.position == tuple(position)), None)

    def valid_actions(self):
        """Legal joint actions account for moves executed earlier in the turn."""
        if self.city is None:
            raise RuntimeError('Call reset before using the environment')
        terrain = tuple(cell.terrain_type for cell in self.map.grid.values())
        key = (tuple((u.position, u.alive) for u in self.units), self.city.position, terrain,
               tuple((c.position, c.alive) for c in self.civilians))
        if key == self._valid_cache_key:
            return list(self._valid_cache)
        result = []
        occupied = {u.position for u in self.units if u.alive}

        def visit(index, action, locations):
            if index == len(self.units):
                result.append(self._action_index[tuple(action)])
                return
            unit = self.units[index]
            if not unit.alive:
                visit(index + 1, action + ['SPACE'], locations)
                return
            remaining = locations - {unit.position}
            for direction in self.unit_directions[index]:
                if direction == 'SHOOT':
                    if self.city.alive and hex_distance(unit.position, self.city.position) <= unit.attack_range:
                        visit(index + 1, action + [direction], locations)
                    continue
                dx, dy = constants.movement_delta(direction, unit.y)
                destination = (unit.x + dx, unit.y + dy)
                cell = self.map.grid.get(destination)
                if cell is None or cell.block_path or destination in remaining:
                    continue
                if unit.support and (destination == self.city.position or self.civilian_at(destination)):
                    continue
                final = unit.position if destination == self.city.position else destination
                visit(index + 1, action + [direction], remaining | {final})

        visit(0, [], occupied)
        self._valid_cache_key, self._valid_cache = key, result
        return list(result)

    def action_mask(self):
        mask = np.zeros(self.action_count, dtype=bool)
        mask[self.valid_actions()] = True
        return mask

    def encode_action(self, directions):
        try:
            return self._action_index[tuple(directions)]
        except KeyError as exc:
            raise ValueError('Expected one valid direction per unit') from exc

    def move_unit(self, unit, delta):
        if not unit.alive or not self.city.alive:
            return
        dx, dy = delta
        destination = unit.x + dx, unit.y + dy
        cell = self.map.grid.get(destination)
        if cell is None or cell.block_path:
            unit.status = 'hit wall'
            unit._event('invalid_move')
            return
        civilian = self.civilian_at(destination)
        if civilian is not None:
            if unit.support:
                unit._event('blocked_move')
                return
            civilian.leave('captured', 'civilian_captured', by=unit.name_instance)
            unit.x, unit.y = destination
            unit.status = 'captured'
            return
        target = self.occupant(destination, exclude=unit)
        if target is self.city and unit.support:
            unit._event('invalid_move')
        elif target is self.city:
            damage_out, damage_taken = attack(unit, self.city)
            unit._event('attacked')
            unit.take_damage(damage_taken)
            self.city.take_damage(damage_out, aggressor_alive=unit.alive,
                                  wall_multiplier=self.melee_wall_multiplier(unit))
            if unit.alive:
                unit.status = 'attacked'
        elif target is not None:
            unit._event('blocked_move')
        elif dx == 0 and dy == 0:
            healed = min(10, unit.hp_max - unit.hp)
            if healed > 0:
                unit.hp += healed
                unit.status = 'healed'
                unit._event('unit_healed', amount=healed)
        else:
            distance_before = hex_distance(unit.position, self.city.position)
            unit.x, unit.y = destination
            unit.status = 'alive'
            unit._event('moved', progress=distance_before - hex_distance(destination, self.city.position),
                        cost=cell.movement_cost)

    def ram_in_position(self):
        return any(u.alive and u.support and hex_distance(u.position, self.city.position) == 1
                   for u in self.units)

    def melee_wall_multiplier(self, unit):
        if UNIT_SPECS[unit.unit_type].unit_class in RAM_SUPPORTED_CLASSES and self.ram_in_position():
            return 1.0
        return MELEE_WALL_DAMAGE

    def city_turn(self):
        if not self.city.alive:
            return
        if self.city.ranged_combat:
            targets = [u for u in self.units if u.alive and hex_distance(u.position, self.city.position) <= 2]
            if targets:
                target = self.rng.choice(targets)
                damage, _ = attack(self.city, target, ranged=True)
                target.take_damage(damage)
        encampment = self.districts.get('encampment')
        if encampment is not None:
            targets = [u for u in self.units if u.alive and hex_distance(u.position, encampment) <= STRIKE_RANGE]
            if targets:
                target = self.rng.choice(targets)
                damage, _ = attack(self.city, target, ranged=True)
                self.city._event('encampment_strike', target=target.name_instance)
                target.take_damage(damage)
        if self.city.heal:
            besiegers = sum(u.alive and hex_distance(u.position, self.city.position) == 1 for u in self.units)
            if besiegers < min(3, self.config.unit_count):
                healed = min(10, self.city.hp_max - self.city.hp)
                if healed > 0:
                    self.city.hp += healed
                    self.city.status = 'healed'
                    self.city._event('city_healed', amount=healed)
        for civilian in self.civilians:
            if civilian.alive:
                (self.builder_turn if civilian.kind == 'builder' else self.settler_turn)(civilian)
        if self.config.city_growth:
            self.grow_city()
        if self.config.city_build_queue:
            self.produce()

    def is_built(self, item):
        return item in self.districts or item in self.buildings

    def adjacency(self, district, pos):
        """A district's Civ VI adjacency bonus at pos; the city center counts as a district."""
        spec = DISTRICTS[district]
        neighbors = self.map.grid[pos].get_neighbors(self.map.grid)
        mountains = sum(cell.terrain_type == 'MOUNTAIN' for cell in neighbors)
        woods = sum(cell.terrain_type == 'FOREST' and cell.district is None for cell in neighbors)
        districts = sum(cell.district is not None or cell.index == self.city.position for cell in neighbors)
        return spec.mountain * mountains + spec.woods_pair * (woods // 2) + spec.district_pair * (districts // 2)

    def district_site(self, district):
        """The free tile with the best adjacency, then the lowest yields, or None.

        Campuses and holy sites go next to the city center; encampments two hexes out.
        """
        occupied = ({u.position for u in self.units if u.alive}
                    | {c.position for c in self.civilians if c.alive})
        sites = [pos for pos, cell in self.map.grid.items()
                 if hex_distance(pos, self.city.position) == DISTRICTS[district].distance
                 and cell.terrain_type in TILE_YIELDS and cell.district is None and pos not in occupied]
        if not sites:
            return None
        return max(sites, key=lambda pos: (self.adjacency(district, pos), -sum(self.tile_yield(pos)), -pos[1], -pos[0]))

    def can_build(self, item):
        if self.is_built(item):
            return False
        if item in DISTRICTS:
            return (len(self.districts) < district_limit(self.city.population)
                    and self.district_site(item) is not None)
        district = BUILDINGS[item].district
        return district == 'city_center' or district in self.districts

    def complete_build(self, item, announce=True):
        """Add a district or building to the city; return False if it cannot be built now."""
        city = self.city
        if item in DISTRICTS:
            site = self.district_site(item)
            if site is None:
                return False
            cell = self.map.grid[site]
            cell.district, cell.fortified = item, item == 'encampment'
            self.farms.discard(site)
            self.districts[item] = site
            self._valid_cache_key = self._path_costs = None
        else:
            district = BUILDINGS[item].district
            if district != 'city_center' and district not in self.districts:
                return False
            self.buildings.append(item)
            city.wall_hp += BUILDINGS[item].walls
            city.wall_hp_max += BUILDINGS[item].walls
        if item in ('ancient_walls', 'encampment') and city.strength_ranged == 0:
            city.strength_ranged = CITY_STRIKE_STRENGTH
        if item == 'ancient_walls':
            city.ranged_combat = True
        if announce:
            self.completed.append(item)
            city._event('city_built', item=item)
        return True

    def current_build(self):
        """The first item in the queue that the city can work on now."""
        return next((item for item in self.config.city_build_queue if self.can_build(item)), None)

    @staticmethod
    def build_cost(item):
        return (DISTRICTS.get(item) or BUILDINGS[item]).cost

    def produce(self):
        """Put this turn's production into the queue; overflow goes to the next item."""
        production = self.city_yields()[1]
        while production > 0:
            item = self.current_build()
            if item is None:
                return  # nothing can be built now, so the production is lost
            needed = self.build_cost(item) - self.build_progress.get(item, 0)
            if production < needed:
                self.build_progress[item] = self.build_progress.get(item, 0) + production
                return
            self.build_progress[item] = self.build_cost(item)
            production -= needed
            self.complete_build(item)

    def _steps_to(self, goals, blocked):
        """Hex steps from every reachable tile to the nearest goal, avoiding blocked tiles."""
        steps = {goal: 0 for goal in goals}
        pending = list(steps)
        for pos in pending:
            for cell in self.map.grid[pos].get_neighbors(self.map.grid):
                if not cell.block_path and cell.index not in blocked and cell.index not in steps:
                    steps[cell.index] = steps[pos] + 1
                    pending.append(cell.index)
        return steps

    def _civilian_step(self, civilian, goals):
        """Move one hex toward the nearest reachable goal, around units and the city."""
        blocked = ({u.position for u in self.units if u.alive} | {self.city.position}
                   | {c.position for c in self.civilians if c.alive and c is not civilian})
        steps = self._steps_to([goal for goal in goals if goal not in blocked], blocked)
        here = steps.get(civilian.position)
        options = [(steps[cell.index], cell.index[1], cell.index[0])
                   for cell in self.map.grid[civilian.position].get_neighbors(self.map.grid)
                   if cell.index in steps and cell.index not in blocked]
        if here is not None and options and min(options)[0] < here:
            _, y, x = min(options)
            civilian.x, civilian.y = x, y

    def farm_sites(self):
        """Unfarmed grassland in the city's territory."""
        return [pos for pos in self.territory()
                if self.map.grid[pos].terrain_type == 'GRASSLAND' and pos not in self.farms]

    def builder_turn(self, builder):
        sites = self.farm_sites()
        if builder.position not in sites:
            self._civilian_step(builder, sites)
            return
        self.farms.add(builder.position)
        builder.charges -= 1
        builder._event('farm_built', position=list(builder.position), charges=builder.charges)
        if builder.charges == 0:
            builder.leave('used up', 'builder_used_up')

    def settlement_sites(self):
        """Tiles far enough from every city to found a new one."""
        cities = [self.city.position] + self.founded_cities
        return [pos for pos, cell in self.map.grid.items() if not cell.block_path
                and all(hex_distance(pos, city) >= MIN_CITY_DISTANCE for city in cities)]

    def settler_turn(self, settler):
        sites = self.settlement_sites()
        if settler.position not in sites:
            self._civilian_step(settler, sites)
            return
        self.founded_cities.append(settler.position)
        settler.leave('settled', 'city_founded', position=list(settler.position))

    def territory(self):
        """Workable tiles the city owns: the six around its center, minus mountains."""
        return [cell.index for cell in self.map.grid[self.city.position].get_neighbors(self.map.grid)
                if cell.terrain_type in TILE_YIELDS and cell.district is None]

    def blocked_tiles(self):
        """Territory tiles that cannot be worked because a unit stands on them."""
        occupied = {u.position for u in self.units if u.alive}
        return [pos for pos in self.territory() if pos in occupied]

    def worked_tiles(self):
        """Citizens work the free tiles with the most food, then production.

        Citizens are reassigned whenever units move, so this is derived from the
        current positions. Citizens with no free tile still eat.
        """
        if not self.config.city_growth:
            return []
        blocked = set(self.blocked_tiles())
        free = [pos for pos in self.territory() if pos not in blocked]
        free.sort(key=lambda pos: (tuple(-v for v in self.tile_yield(pos)), pos[1], pos[0]))
        return free[:self.city.population]

    def tile_yield(self, pos):
        """Return a territory tile's (food, production), including any farm."""
        food, production = TILE_YIELDS[self.map.grid[pos].terrain_type]
        return food + (FARM_FOOD if pos in self.farms else 0), production

    def city_yields(self):
        """Return (food, production) from the city center and worked tiles."""
        food, production = CITY_CENTER_YIELD
        for pos in self.worked_tiles():
            tile_food, tile_production = self.tile_yield(pos)
            food, production = food + tile_food, production + tile_production
        for building in self.buildings:
            yields = dict(BUILDINGS[building].yields)
            food, production = food + yields.get('food', 0), production + yields.get('production', 0)
        return food, production

    def city_output(self):
        """All the city's yields per turn: food, production, culture, science and faith."""
        food, production = self.city_yields()
        output = {'food': food, 'production': production, 'culture': 0, 'science': 0, 'faith': 0}
        for district, pos in self.districts.items():
            if DISTRICTS[district].yield_type:
                output[DISTRICTS[district].yield_type] += self.adjacency(district, pos)
        for building in self.buildings:
            for kind, amount in BUILDINGS[building].yields:
                if kind not in ('food', 'production'):
                    output[kind] += amount
        return output

    def food_surplus(self):
        return self.city_yields()[0] - FOOD_PER_CITIZEN * self.city.population

    def city_report(self):
        """Population, stored food and tile use, or None when growth is off."""
        if not self.config.city_growth:
            return None
        city = self.city
        food, production = self.city_yields()
        surplus = food - FOOD_PER_CITIZEN * city.population
        threshold = growth_threshold(city.population)
        return {'population': city.population, 'food': city.food, 'growth_threshold': threshold,
                'food_yield': food, 'production_yield': production, 'food_surplus': surplus,
                'turns_to_grow': math.ceil((threshold - city.food) / surplus) if surplus > 0 else None,
                'worked_tiles': [list(pos) for pos in self.worked_tiles()],
                'blocked_tiles': [list(pos) for pos in self.blocked_tiles()],
                'farms': [list(pos) for pos in sorted(self.farms, key=lambda pos: (pos[1], pos[0]))],
                **{kind: amount for kind, amount in self.city_output().items() if kind in ('culture', 'science', 'faith')},
                'districts': {name: list(pos) for name, pos in self.districts.items()},
                'buildings': list(self.buildings), **self.production_report()}

    def production_report(self):
        """What the city is producing, when it has a build queue."""
        if not self.config.city_build_queue:
            return {}
        item = self.current_build()
        return {'producing': item, 'production_progress': self.build_progress.get(item, 0) if item else 0,
                'production_cost': self.build_cost(item) if item else None, 'completed': list(self.completed)}

    def civilian_report(self):
        """Each civilian's state and the cities settlers have founded, or None without civilians."""
        if not self.civilian_kinds:
            return None
        return {'civilians': [{'name': c.name_instance, 'kind': c.kind, 'position': list(c.position),
                               'status': c.status, 'charges': c.charges} for c in self.civilians],
                'captured': sum(c.status == 'captured' for c in self.civilians),
                'founded_cities': [list(pos) for pos in self.founded_cities]}

    def grow_city(self):
        """Store this turn's surplus; grow at the threshold, or starve once food runs out."""
        city = self.city
        city.food += self.food_surplus()
        threshold = growth_threshold(city.population)
        if city.food >= threshold:
            city.food -= threshold
            city.population += 1
            city._event('city_grew', population=city.population)
        elif city.food < 0:
            city.food = 0
            if city.population > 1:
                city.population -= 1
                city._event('city_starved', population=city.population)

    def shoot(self, unit):
        if not unit.alive or not self.city.alive:
            return
        if not unit.ranged or hex_distance(unit.position, self.city.position) > unit.attack_range:
            unit._event('invalid_move')
            return
        damage, _ = attack(unit, self.city, ranged=True)
        unit._event('ranged_attack')
        # Capturing requires a melee action; a ranged strike leaves at least 1 HP.
        self.city.take_damage(damage, aggressor_alive=False)

    def _outcome(self):
        if not self.city.alive or self.city.hp <= 0:
            return 'win'
        if not any(u.alive and u.hp > 0 and not u.support for u in self.units):
            return 'loss'
        return 'ongoing'

    def get_rewards(self):
        """Reward only the events of the most recent turn; reading is harmless."""
        reward = -.05
        for event in self.events:
            kind = event['type']
            amount = event.get('amount', 0)
            if kind == 'city_damaged':
                reward += 3 * amount / self.city.hp_max
            elif kind == 'wall_damaged':
                reward += 2 * amount / max(1, self.city.wall_hp_max)
            elif kind == 'city_healed':
                reward -= 3 * amount / self.city.hp_max
            elif kind == 'unit_damaged':
                reward -= .5 * amount / 100
            elif kind == 'unit_died':
                reward -= 5
            elif kind == 'unit_healed':
                reward += .02 * amount / 100
            elif kind in ('invalid_move', 'blocked_move'):
                reward -= .2
            elif kind == 'moved':
                reward += .05 * event['progress'] - .01 * (event['cost'] - 1)
            elif kind == 'city_grew':
                reward -= POPULATION_REWARD
            elif kind == 'city_starved':
                reward += POPULATION_REWARD
            elif kind == 'civilian_captured':
                reward += CAPTURE_REWARDS[event['civilian']]
            elif kind == 'city_founded':
                reward -= CITY_FOUNDED_PENALTY
            elif kind == 'city_built':
                reward -= BUILD_REWARD
        if self.last_info['outcome'] == 'win':
            reward += 10
        elif self.last_info['outcome'] == 'loss':
            reward -= 10
        return float(reward)

    def step(self, action=0):
        if self.city is None:
            raise RuntimeError('Call reset before step')
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < self.action_count:
            raise ValueError(f'action must be an integer in [0, {self.action_count})')
        if self._done:
            return self.get_observation(), 0.0, True
        self.events = []
        self.turn_number += 1
        if self._outcome() == 'ongoing':
            for unit, direction in zip(self.units, self.actions[action]):
                if direction == 'SHOOT':
                    self.shoot(unit)
                else:
                    self.move_unit(unit, constants.movement_delta(direction, unit.y))
                if not self.city.alive:
                    break
            if self._outcome() == 'ongoing':
                self.city_turn()
        outcome = self._outcome()
        self._done = outcome != 'ongoing'
        self.last_info = {'outcome': outcome, 'events': [event.copy() for event in self.events],
                          'turn': self.turn_number}
        if self.config.city_growth:
            self.last_info['city'] = self.city_report()
        if self.civilian_kinds:
            self.last_info['civilians'] = self.civilian_report()
        reward = self.get_rewards()
        return self.get_observation(), reward, self._done

    def get_observation(self):
        if self.city is None:
            raise RuntimeError('Call reset before get_observation')
        c = self.config
        city = self.city
        values = [city.x / (c.width - 1), city.y / (c.height - 1),
                  city.hp / city.hp_max, city.wall_hp / 1000,
                  city.strength / 100, city.strength_ranged / 100, float(city.heal)]
        for unit in self.units:
            values.extend([unit.x / (c.width - 1), unit.y / (c.height - 1),
                           unit.hp / unit.hp_max, float(unit.alive), unit.strength / 100,
                           float(unit.y % 2), unit.strength_ranged / 100, unit.attack_range / 2])
        # An encampment blocks movement, so it reads like a mountain.
        values.extend(1.0 if self.map.grid[(x, y)].fortified else TERRAIN_CODES[self.map.grid[(x, y)].terrain_type] / 2
                      for y in range(c.height) for x in range(c.width))
        if c.city_growth:
            # Growth progress is coarse (quarters) to keep the tabular state space small.
            progress = min(1.0, city.food / growth_threshold(city.population))
            values.extend([city.population / MAX_CITY_POPULATION, math.floor(4 * progress) / 4,
                           self.food_surplus() / 10])
        for civilian in self.civilians:
            values.extend([civilian.x / (c.width - 1), civilian.y / (c.height - 1),
                           float(civilian.alive), civilian.charges / BUILDER_CHARGES])
        values.extend(float(self.is_built(item)) for item in c.builds)
        if c.city_build_queue:
            item = self.current_build()
            progress = self.build_progress.get(item, 0) / self.build_cost(item) if item else 0
            values.append(math.floor(4 * progress) / 4)
        return np.asarray(values, dtype=float)

    def _distances(self):
        key = (self.city.position, tuple(cell.terrain_type for cell in self.map.grid.values()))
        if self._path_costs is not None and self._path_costs[0] == key:
            return self._path_costs[1]
        costs = {self.city.position: 0}
        pending = [(0, self.city.position)]
        while pending:
            cost, pos = heapq.heappop(pending)
            if cost != costs[pos]:
                continue
            for neighbor in self.map.grid[pos].get_neighbors(self.map.grid):
                if neighbor.block_path:
                    continue
                new_cost = cost + self.map.grid[pos].movement_cost
                if new_cost < costs.get(neighbor.index, float('inf')):
                    costs[neighbor.index] = new_cost
                    heapq.heappush(pending, (new_cost, neighbor.index))
        self._path_costs = key, costs
        return costs

    def _hunt_steps(self, unit):
        """Steps to the nearest civilian within HUNT_RANGE, for a healthy unit that can capture."""
        if not unit.alive or unit.support or unit.hp < .35 * unit.hp_max:
            return None
        nearby = [c for c in self.civilians if c.alive and hex_distance(c.position, unit.position) <= HUNT_RANGE]
        if not nearby:
            return None
        target = min(nearby, key=lambda c: hex_distance(c.position, unit.position))
        return self._steps_to([target.position], {self.city.position})

    def guided_action(self, rng=None):
        rng = rng or self.rng
        costs = self._distances()
        hunts = [self._hunt_steps(unit) for unit in self.units]
        scores = []
        for action in self.valid_actions():
            score = 0
            for unit, direction, hunt in zip(self.units, self.actions[action], hunts):
                if not unit.alive:
                    continue
                if hunt is not None:
                    # Chase the civilian; moving onto it captures it.
                    if direction == 'SHOOT':
                        score += hunt.get(unit.position, 10000) + 1
                    else:
                        dx, dy = constants.movement_delta(direction, unit.y)
                        score += hunt.get((unit.x + dx, unit.y + dy), 10000)
                    continue
                if direction == 'SHOOT':
                    score += costs.get(unit.position, 10000) - 2
                    if self.city.hp <= 1 and self.city.wall_hp == 0:
                        score += 5
                    continue
                dx, dy = constants.movement_delta(direction, unit.y)
                destination = unit.x + dx, unit.y + dy
                score += costs.get(destination, 10000)
                if direction == 'SPACE':
                    if unit.hp < .35 * unit.hp_max:
                        score -= 2
                    elif unit.support and hex_distance(unit.position, self.city.position) == 1:
                        score -= 1
                    else:
                        score += .3
            scores.append((score, action))
        best = min(score for score, _ in scores)
        return rng.choice([action for score, action in scores if score == best])

    def get_current_state(self):
        state = {obj.name_instance: {'health': obj.hp, 'position': [obj.x, obj.y],
                                    'alive': obj.alive} for obj in self.objects}
        if self.config.city_growth:
            state[self.city.name_instance].update(self.city_report())
        for civilian in self.civilians:
            state[civilian.name_instance] = {'position': [civilian.x, civilian.y], 'alive': civilian.alive,
                                             'kind': civilian.kind, 'status': civilian.status,
                                             'charges': civilian.charges}
        return state

    def render_frame(self, delay_ms=0):
        if not self.render:
            return True
        if self._renderer is None:
            from renderer import Renderer
            self._renderer = Renderer(self)
        self.quit = not self._renderer.frame(delay_ms)
        return not self.quit

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def game_main_loop(self, action=0):
        """Interactive human mode: select units with Tab and use Q/E/A/D/Z/X."""
        self.render = True
        if self.city is None:
            self.reset()
        from renderer import Renderer
        self._renderer = Renderer(self)
        self._renderer.human_loop()
        self.close()


if __name__ == '__main__':
    env = Game(human=True, render=True)
    env.reset()
    env.game_main_loop()
