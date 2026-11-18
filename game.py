"""A small Civ-like combat environment with isolated state and seeded randomness."""

import collections
from dataclasses import dataclass
import itertools
import math
import random

import numpy as np

import constants
from economy import CityEconomy
from class_hex import LAND_TYPES, YIELD_TYPES, HexMap, hex_distance, terrain_code
from class_hex import hex_coords  # noqa: F401  Kept importable from game, where it used to live.


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

# Civ VI river defense: a defender gains this much strength against melee
# attacks from across a river. Ranged attacks are unaffected.
RIVER_DEFENSE_BONUS = 5
MAX_RIVERS = 10


@dataclass(frozen=True)
class WallSpec:
    """City walls: their outer defense (wall HP) and the ranged strike they give the city."""
    hp: float = 0
    ranged_strength: float = 0
    cost: int = 0


# Civ VI Ancient Walls give 100 outer defense and let the city shoot units
# within two tiles. The strike matches the owner's strongest ranged unit, or 15
# without one; the city here has no units, so it uses 15. They cost 80 production.
WALL_SPECS = {
    'none': WallSpec(),
    'ancient': WallSpec(hp=100, ranged_strength=15, cost=80),
}
WALL_TYPES = tuple(WALL_SPECS)


@dataclass(frozen=True)
class BuildingSpec:
    """A Civ VI city-centre building: production cost, yields and housing."""
    civ6_type: str
    cost: int
    yields: tuple = ()  # (yield type, amount) pairs
    housing: int = 0
    technology: str | None = None


# Base-game values from CivFanatics. Some wikis list a later cost of 50 for both.
# Neither building affects combat; with the city economy on, their yields and
# housing feed the city's growth, production and borders.
BUILDING_SPECS = {
    'monument': BuildingSpec('BUILDING_MONUMENT', cost=60, yields=(('culture', 2),)),
    'granary': BuildingSpec('BUILDING_GRANARY', cost=65, yields=(('food', 1),), housing=2,
                            technology='TECH_POTTERY'),
}
BUILDING_TYPES = tuple(BUILDING_SPECS)

# What the city can build with the economy on, and its default order.
BUILD_ITEMS = BUILDING_TYPES + ('ancient_walls',)
BUILD_COSTS = {**{name: spec.cost for name, spec in BUILDING_SPECS.items()},
               'ancient_walls': WALL_SPECS['ancient'].cost}
DEFAULT_BUILD_QUEUE = ('monument', 'granary', 'ancient_walls')
MAX_CITY_POPULATION = 20

# Each enemy city is a rival civilization's capital. Capturing all of them is a
# domination victory.
CITY_NAMES = ('Ottertopia', 'Kelpford', 'Clamhaven', 'Driftwood')
MAX_CITIES = len(CITY_NAMES)
MIN_CITY_DISTANCE = 3

# With player_capital set, the squad has its own capital to defend. Enemy cities
# send raiders to capture it; losing it loses the game.
PLAYER_CAPITAL_NAME = 'Otterholt'
MAX_RAIDERS = 4
RAID_THREAT_RADIUS = 3


def open_neighbors(position, passable):
    """Count the neighboring tiles of position for which passable(tile) is true."""
    x, y = position
    return sum(passable((x + dx, y + dy)) for dx, dy in
               (constants.movement_delta(direction, y) for direction in constants.DIRECTIONS[:-1]))


def place_cities(capital, candidates, count, rng=None, fits=None):
    """Pick count city sites, starting with capital, at least MIN_CITY_DISTANCE apart.

    fits(site) can reject sites, e.g. ones without room for a siege. Sites are
    drawn at random when rng is given, otherwise in a fixed scan order.
    Returns None when the remaining cities do not fit.
    """
    positions = [tuple(capital)]
    for _ in range(count - 1):
        options = sorted(p for p in candidates
                         if all(hex_distance(p, q) >= MIN_CITY_DISTANCE for q in positions)
                         and (fits is None or fits(p)))
        if not options:
            return None
        positions.append(rng.choice(options) if rng is not None else options[0])
    return positions


def place_capital(enemy_positions, candidates, rng=None, fits=None):
    """Pick the player's capital: the site farthest from the nearest enemy city.

    It must be at least MIN_CITY_DISTANCE from every enemy city and pass fits.
    Ties are drawn at random when rng is given, otherwise the first in scan order.
    Returns None when no site qualifies.
    """
    gaps = {}
    for site in candidates:
        gap = min(hex_distance(site, enemy) for enemy in enemy_positions)
        if gap >= MIN_CITY_DISTANCE and (fits is None or fits(site)):
            gaps[site] = gap
    if not gaps:
        return None
    best = max(gaps.values())
    farthest = sorted(site for site, gap in gaps.items() if gap == best)
    return rng.choice(farthest) if rng is not None else farthest[0]

# attack() draws damage as round(roll * exp(diff / 25) * scale) with these ranges.
DAMAGE_ROLLS = range(24, 37)
DAMAGE_SCALE = (.75, 1.0)

# Tactical observations describe each unit relative to the city, so what is
# learned on one map or seed transfers to others. Every per-unit command maps
# to one of TACTICS in the current position; -1 marks an unavailable command.
# Holding position is HEAL for a wounded unit and WAIT at full health, since
# only the first restores health.
COMMANDS = constants.DIRECTIONS + ('SHOOT',)
TACTICS = ('ATTACK', 'SHOOT', 'HEAL', 'WAIT', 'ADVANCE', 'SIDESTEP', 'RETREAT')
TACTICAL_FEATURES = ('unit_type', 'distance', 'health', 'risk', 'city_health', 'walls',
                     'allies_adjacent', 'allies_away', 'ram_ready', 'city_ranged', 'city_heals')
# Inclusive upper bound of each feature, in TACTICAL_FEATURES order.
TACTICAL_FEATURE_LIMITS = (len(UNIT_SPECS) - 1, 6, 3, 3, 4, 2, 3, 3, 1, 1, 1)
# Per-unit block: exact state fractions, the integer features, then one tactic
# per command. The fractions are not learned from directly; they make two
# observations equal only when nothing changed.
TACTICAL_HEADER = ('alive', 'health_fraction', 'city_health_fraction', 'wall_fraction')
TACTICAL_FIRST_TACTIC = len(TACTICAL_HEADER) + len(TACTICAL_FEATURES)
TACTICAL_UNIT_WIDTH = TACTICAL_FIRST_TACTIC + len(COMMANDS)
OBSERVATION_MODES = ('raw', 'tactical')
TACTIC_INDEX = {name: index for index, name in enumerate(TACTICS)}


def damage_at_least(threshold, multiplier):
    """Exact probability that attack() deals at least threshold damage.

    Damage is round(roll * multiplier * scale) for an integer roll and a
    uniform scale, so the event is roll * multiplier * scale >= ceil(threshold) - 0.5.
    """
    needed = math.ceil(threshold) - .5
    if needed <= 0:
        return 1.0
    low, high = DAMAGE_SCALE
    total = 0.0
    for roll in DAMAGE_ROLLS:
        cutoff = needed / (roll * multiplier)
        total += min(1.0, max(0.0, (high - cutoff) / (high - low)))
    return total / len(DAMAGE_ROLLS)


def risk_level(probability):
    """Bucket a death probability: impossible, unlikely, plausible, likely."""
    if probability <= 0:
        return 0
    return 1 if probability < .15 else 2 if probability < .5 else 3


@dataclass(frozen=True)
class GameConfig:
    width: int = 8
    height: int = 8
    unit_count: int = 3
    city_position: tuple | None = None
    city_strength: float = 28
    wall_hp: float = 0
    walls: str = 'none'
    buildings: tuple = ()
    ranged_strength: float = 0
    city_heals: bool = True
    mountain_density: float = 0
    forest_density: float = 0
    land_types: tuple = ()
    hills_density: float = 0
    water_density: float = 0
    river_count: int = 0
    unit_strengths: tuple = ()
    unit_types: tuple = ()
    difficulty: str = 'normal'
    economy: bool = False
    city_population: int = 1
    build_queue: tuple = ()
    city_count: int = 1
    player_capital: bool = False
    capital_strength: float = 28
    raid_interval: int = 6
    raider_strength: float = 20

    def __post_init__(self):
        try:
            object.__setattr__(self, 'buildings', tuple(self.buildings))
        except TypeError as exc:
            raise ValueError('buildings must be a sequence of building names') from exc
        if (any(not isinstance(name, str) or name not in BUILDING_SPECS for name in self.buildings)
                or len(set(self.buildings)) != len(self.buildings)):
            raise ValueError(f'buildings must name distinct buildings from {", ".join(BUILDING_TYPES)}')
        if not isinstance(self.walls, str) or self.walls not in WALL_SPECS:
            raise ValueError(f'walls must be one of {", ".join(WALL_TYPES)}')
        # Walls supply their defaults for wall HP and ranged strength left at zero.
        spec = WALL_SPECS[self.walls]
        if spec.hp and not self.wall_hp:
            object.__setattr__(self, 'wall_hp', spec.hp)
        if spec.ranged_strength and not self.ranged_strength:
            object.__setattr__(self, 'ranged_strength', spec.ranged_strength)
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
        for name in ('mountain_density', 'forest_density', 'hills_density', 'water_density'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be between 0 and 1')
        if self.mountain_density + self.water_density + self.forest_density > 0.8:
            raise ValueError('combined mountain, water and forest density must be at most 0.8')
        try:
            object.__setattr__(self, 'land_types', tuple(self.land_types))
        except TypeError as exc:
            raise ValueError('land_types must be a sequence of land type names') from exc
        if (any(not isinstance(name, str) or name not in LAND_TYPES for name in self.land_types)
                or len(set(self.land_types)) != len(self.land_types)):
            raise ValueError(f'land_types must name distinct types from {", ".join(LAND_TYPES)}')
        if (isinstance(self.river_count, bool) or not isinstance(self.river_count, int)
                or not 0 <= self.river_count <= MAX_RIVERS):
            raise ValueError(f'river_count must be an integer between 0 and {MAX_RIVERS}')
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
        if not isinstance(self.economy, bool):
            raise ValueError('economy must be true or false')
        if (isinstance(self.city_population, bool) or not isinstance(self.city_population, int)
                or not 1 <= self.city_population <= MAX_CITY_POPULATION):
            raise ValueError(f'city_population must be an integer between 1 and {MAX_CITY_POPULATION}')
        try:
            object.__setattr__(self, 'build_queue', tuple(self.build_queue))
        except TypeError as exc:
            raise ValueError('build_queue must be a sequence of build items') from exc
        if (any(not isinstance(name, str) or name not in BUILD_ITEMS for name in self.build_queue)
                or len(set(self.build_queue)) != len(self.build_queue)):
            raise ValueError(f'build_queue must name distinct items from {", ".join(BUILD_ITEMS)}')
        if not self.economy and (self.city_population != 1 or self.build_queue):
            raise ValueError('city_population and build_queue need the city economy')
        if self.city_position is not None:
            if len(self.city_position) != 2 or any(isinstance(v, bool) or not isinstance(v, int)
                                                 for v in self.city_position):
                raise ValueError('city_position must contain two integers')
            x, y = self.city_position
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError('city_position must lie inside the map')
        self._validate_cities()

    def _validate_cities(self):
        """Check the domination and player-capital settings, and that the map can hold the cities."""
        if (isinstance(self.city_count, bool) or not isinstance(self.city_count, int)
                or not 1 <= self.city_count <= MAX_CITIES):
            raise ValueError(f'city_count must be between 1 and {MAX_CITIES}')
        if not isinstance(self.player_capital, bool):
            raise ValueError('player_capital must be True or False')
        for name in ('capital_strength', 'raider_strength'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 100:
                raise ValueError(f'{name} must be finite and in (0, 100]')
        if (isinstance(self.raid_interval, bool) or not isinstance(self.raid_interval, int)
                or not 0 <= self.raid_interval <= 1000):
            raise ValueError('raid_interval must be an integer between 0 and 1000')
        capital = self.city_position or (self.width // 2, self.height // 2)
        every_tile = [(x, y) for y in range(self.height) for x in range(self.width)]
        siege_room = min(self.unit_count, 3)

        def fits(site):
            return open_neighbors(site, lambda p: 0 <= p[0] < self.width and 0 <= p[1] < self.height) >= siege_room

        enemy_sites = place_cities(capital, every_tile, self.city_count, fits=fits)
        if enemy_sites is None:
            raise ValueError(f'the map cannot fit {self.city_count} cities at least {MIN_CITY_DISTANCE} '
                             f'hexes apart, each with {siege_room} open neighboring tiles')
        if self.player_capital and place_capital(enemy_sites, every_tile, fits=fits) is None:
            raise ValueError(f"the map has no room for the player's capital at least {MIN_CITY_DISTANCE} "
                             f'hexes from every enemy city')

    @property
    def extended_terrain(self):
        """Whether maps can hold terrain beyond the original grassland, forest and mountains."""
        return bool(self.hills_density or self.water_density or set(self.land_types) - {'grassland'})


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
    damage_event, death_event = 'unit_damaged', 'unit_died'

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
        self._event(self.damage_event, amount=actual)
        if self.hp <= 0:
            self.death_unit()

    def death_unit(self):
        if self.alive:
            self.hp = 0
            self.alive = False
            self.status = 'dead'
            self._event(self.death_event)


class C_Raider(C_Unit):
    """An enemy unit sent to capture the player's capital."""
    damage_event, death_event = 'raider_damaged', 'raider_killed'


class C_City(C_Sprite):
    damage_event, wall_event, capture_event, heal_event = (
        'city_damaged', 'wall_damaged', 'city_captured', 'city_healed')

    def __init__(self, x, y, sprite=None, name_instance='City', hp=100,
                 hp_max=100, wall_hp=0, strength=28, strength_ranged=0,
                 ranged_combat=False, heal=True, env=None, buildings=(), walls='none'):
        super().__init__(x, y, sprite, name_instance, hp, hp_max, strength, env)
        if any(name not in BUILDING_SPECS for name in buildings):
            raise ValueError(f'buildings must be among {", ".join(BUILDING_TYPES)}')
        self.buildings = tuple(buildings)
        self.walls = walls
        self.wall_hp = float(wall_hp)
        self.wall_hp_max = float(wall_hp)
        self.strength_ranged = float(strength_ranged)
        self.ranged_combat = ranged_combat
        self.heal = heal

    @property
    def building_yields(self):
        """Yields added by the city's buildings, e.g. {'food': 1, 'culture': 2}."""
        totals = {}
        for name in self.buildings:
            for kind, amount in BUILDING_SPECS[name].yields:
                totals[kind] = totals.get(kind, 0) + amount
        return {kind: totals[kind] for kind in YIELD_TYPES if totals.get(kind)}

    @property
    def building_housing(self):
        return sum(BUILDING_SPECS[name].housing for name in self.buildings)

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
                self._event(self.wall_event, amount=wall_damage)
        actual = min(self.hp if aggressor_alive else max(0, self.hp - 1), damage)
        if actual:
            self.hp -= actual
            self.status = 'took damage'
            self._event(self.damage_event, amount=actual)
        if self.hp <= 0:
            self.death()

    def death(self):
        if self.alive:
            self.hp = 0
            self.alive = False
            self.status = 'dead'
            self._event(self.capture_event)


class C_Capital(C_City):
    """The player's own capital; losing it loses the game."""
    damage_event, wall_event, capture_event, heal_event = (
        'capital_damaged', 'capital_wall_damaged', 'capital_captured', 'capital_healed')


def attack(aggressor, target, ranged=False, rng=None):
    """Compute combat damage using the environment's private RNG."""
    rng = rng or (aggressor.env.rng if aggressor.env is not None else random.Random())
    offense = aggressor.strength_ranged if ranged else aggressor.strength
    defense = target.strength
    if isinstance(target, C_Unit) and target.env is not None:
        defense += target.env.map.grid[target.position].defense_bonus
    river_defense = 0
    if not ranged and aggressor.env is not None and aggressor.env.map.has_river(aggressor.position, target.position):
        river_defense = RIVER_DEFENSE_BONUS
    strength_diff = offense - defense - river_defense
    damage_out = round(rng.randint(24, 36) * math.exp(strength_diff / 25) * rng.uniform(.75, 1))
    if ranged:
        return damage_out, 0
    retaliation_diff = target.strength + river_defense - aggressor.strength
    if aggressor.env is not None:
        retaliation_diff -= aggressor.env.map.grid[aggressor.position].defense_bonus
    damage_taken = round(rng.randint(24, 36) * math.exp(retaliation_diff / 25) * rng.uniform(.75, 1))
    return damage_out, damage_taken


class Game:
    """The siege environment.

    reset() and step() return the observation for the configured mode: 'raw'
    is get_observation()'s absolute state vector, and 'tactical' is
    tactical_observation()'s egocentric per-unit encoding.
    """

    def __init__(self, human=False, ml_ai=False, render=False, config=None, seed=None, observation='raw'):
        if observation not in OBSERVATION_MODES:
            raise ValueError(f'observation must be one of {", ".join(OBSERVATION_MODES)}')
        self.config = config or GameConfig()
        self.human, self.ml_ai, self.render = human, ml_ai, render
        self.observation_mode = observation
        self.rng = random.Random(seed)
        self.unit_types = tuple(self.config.unit_types or ('warrior',) * self.config.unit_count)
        self.unit_directions = [constants.DIRECTIONS + (('SHOOT',) if UNIT_SPECS[kind].ranged else ())
                                for kind in self.unit_types]
        self.actions = tuple(itertools.product(*self.unit_directions))
        self.action_count = len(self.actions)
        self._action_index = {a: i for i, a in enumerate(self.actions)}
        if observation == 'tactical':
            # Egocentric features do not depend on the map size, and terrain, rivers,
            # the economy and the number of enemy cities reach them only through the
            # odds and features they change. Raiders change what a unit's objective
            # can be, so models with and without a capital are kept apart.
            self.observation_schema = ('ml_civ6.tactical.v1:' + ','.join(self.unit_types)
                                       + (':capital' if self.config.player_capital else ''))
        else:
            # Raw features for terrain, rivers, the economy, further cities and the
            # capital are appended only when they are on, so scenarios without them
            # keep their observation format.
            self.observation_schema = (f'ml_civ6.v2:{self.config.width}x{self.config.height}:'
                                       + ','.join(self.unit_types)
                                       + (f':cities={self.config.city_count}' if self.config.city_count > 1 else '')
                                       + (':capital' if self.config.player_capital else '')
                                       + (':terrain' if self.config.extended_terrain else '')
                                       + (':rivers' if self.config.river_count else '')
                                       + (':economy' if self.config.economy else ''))
        self._renderer = None
        self.quit = False
        self.units = []
        self.cities = []
        self.capital = None
        self.raiders = []
        self.economy = None
        self.events = []
        self.turn_number = 0
        self.last_info = self._info('ongoing')
        self._done = False
        self._valid_cache_key = None
        self._route_cache = {}

    @property
    def city(self):
        """The first enemy city (the enemy capital); single-city callers use this."""
        return self.cities[0] if self.cities else None

    @property
    def objects(self):
        return self.units + self.raiders + self.cities + ([self.capital] if self.capital is not None else [])

    def living_raiders(self):
        return [raider for raider in self.raiders if raider.alive]

    def raider_at(self, position):
        return next((r for r in self.raiders if r.alive and r.position == tuple(position)), None)

    def living_cities(self):
        return [city for city in self.cities if city.alive]

    def cities_captured(self):
        return sum(not city.alive for city in self.cities)

    def city_at(self, position):
        return next((city for city in self.cities if city.alive and city.position == tuple(position)), None)

    def enemy_at(self, position):
        """The uncaptured enemy city or living raider on a tile, which a move there attacks."""
        return self.city_at(position) or self.raider_at(position)

    def nearest_city_distance(self, position):
        return min((hex_distance(position, city.position) for city in self.living_cities()), default=0)

    def shoot_target(self, unit):
        """The closest enemy city or raider in range.

        Ties go to the target with the least HP plus walls, then to raiders.
        """
        in_range = [(hex_distance(unit.position, city.position), city.hp + city.wall_hp, 1, index, city)
                    for index, city in enumerate(self.cities)
                    if city.alive and hex_distance(unit.position, city.position) <= unit.attack_range]
        in_range += [(hex_distance(unit.position, raider.position), raider.hp, 0, index, raider)
                     for index, raider in enumerate(self.raiders)
                     if raider.alive and hex_distance(unit.position, raider.position) <= unit.attack_range]
        return min(in_range)[-1] if in_range else None

    def _info(self, outcome):
        return {'outcome': outcome, 'events': [event.copy() for event in self.events], 'turn': self.turn_number,
                'cities_captured': self.cities_captured(),
                'capital_lost': self.capital is not None and not self.capital.alive}

    def summary(self):
        """End-of-game figures that evaluation averages: cities captured and whether the capital fell."""
        result = {}
        if self.config.city_count > 1:
            result['cities_captured'] = self.cities_captured()
        if self.config.player_capital:
            result['capital_lost'] = int(self.capital is not None and not self.capital.alive)
        return result

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        c = self.config
        enemy_capital = tuple(c.city_position or (c.width // 2, c.height // 2))
        self.map = HexMap(c.height, c.width)

        def passable(position):
            cell = self.map.grid.get(position)
            return cell is not None and not cell.block_path

        def fits(site):
            # Room for a full siege, so the city can be stopped from healing.
            return open_neighbors(site, passable) >= min(c.unit_count, 3)

        # Keep enough connected open land for starting units, every city, and a possible siege.
        blocked_density = c.mountain_density + c.water_density
        for _ in range(100):
            for cell in self.map.grid.values():
                draw = self.rng.random()
                terrain = ('MOUNTAIN' if draw < c.mountain_density else
                           'COAST' if draw < blocked_density else
                           'FOREST' if draw < blocked_density + c.forest_density else 'GRASSLAND')
                cell.set_terrain(terrain)
            self.map.grid[enemy_capital].set_terrain('GRASSLAND')
            component = self.map.reachable(enemy_capital)
            adjacent = self.map.grid[enemy_capital].get_neighbors(self.map.grid)
            if (len(component) >= c.unit_count + c.city_count + c.player_capital
                    and sum(not cell.block_path for cell in adjacent) >= min(c.unit_count, 3)):
                city_positions = place_cities(enemy_capital, component, c.city_count, self.rng, fits)
                home = None
                if city_positions is not None and c.player_capital:
                    home = place_capital(city_positions, component, self.rng, fits)
                if city_positions is not None and (home is not None or not c.player_capital):
                    break
        else:
            for cell in self.map.grid.values():
                cell.set_terrain('GRASSLAND')
            component = self.map.reachable(enemy_capital)
            city_positions = place_cities(enemy_capital, component, c.city_count, fits=fits)
            home = place_capital(city_positions, component, fits=fits) if c.player_capital else None
        sites = city_positions + ([home] if home is not None else [])
        for position in sites:
            self.map.grid[position].set_terrain('GRASSLAND')
        if c.extended_terrain:
            # None of these steps block movement, so the start area stays connected.
            self.map.assign_land_types(c.land_types or LAND_TYPES[:1], self.rng)
            self.map.raise_hills(c.hills_density, self.rng, keep_flat=set(sites))
            self.map.classify_water()
        # Rivers never block movement, so they cannot disconnect the start area.
        self.map.generate_rivers(c.river_count, self.rng)
        free = sorted(component - set(sites))
        if home is None:
            start_positions = self.rng.sample(free, c.unit_count)
        else:
            # The squad starts around its own capital.
            nearby = sorted(free, key=lambda position: (hex_distance(position, home), position))
            radius = max(2, hex_distance(nearby[c.unit_count - 1], home))
            start_positions = self.rng.sample([p for p in nearby if hex_distance(p, home) <= radius],
                                              c.unit_count)
        specs = [UNIT_SPECS[kind] for kind in self.unit_types]
        strengths = c.unit_strengths or tuple(spec.strength for spec in specs)
        names = ('Otto', 'Fynn', 'Victor', 'Ada')
        self.units = [C_Unit(*pos, name_instance=names[i], strength=strengths[i], env=self,
                             unit_type=self.unit_types[i], strength_ranged=specs[i].ranged_strength)
                      for i, pos in enumerate(start_positions)]
        # Every enemy city shares the scenario's strength, walls and buildings.
        self.cities = [C_City(*position, name_instance=CITY_NAMES[i], strength=c.city_strength,
                              wall_hp=c.wall_hp, strength_ranged=c.ranged_strength,
                              ranged_combat=c.ranged_strength > 0, heal=c.city_heals, env=self,
                              buildings=c.buildings, walls=c.walls)
                       for i, position in enumerate(city_positions)]
        self.capital = (C_Capital(*home, name_instance=PLAYER_CAPITAL_NAME, strength=c.capital_strength, env=self)
                        if home is not None else None)
        self.raiders = []
        # The economy runs in the enemy capital, the first city.
        self.economy = (CityEconomy(self, c.city_population, c.build_queue or DEFAULT_BUILD_QUEUE, BUILD_COSTS)
                        if c.economy else None)
        self.turn_number = 0
        self.events = []
        self.last_info = self._info('ongoing')
        self._done = self.quit = False
        self._valid_cache_key = None
        self._route_cache = {}
        return self.observe()

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
        key = (tuple((u.position, u.alive) for u in self.units),
               tuple((city.position, city.alive) for city in self.cities),
               tuple((raider.position, raider.alive) for raider in self.raiders), terrain)
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
                    if self.shoot_target(unit) is not None:
                        visit(index + 1, action + [direction], locations)
                    continue
                dx, dy = constants.movement_delta(direction, unit.y)
                destination = (unit.x + dx, unit.y + dy)
                cell = self.map.grid.get(destination)
                if cell is None or cell.block_path or destination in remaining:
                    continue
                if self.capital is not None and destination == self.capital.position:
                    continue
                attacks = self.enemy_at(destination) is not None
                if unit.support and attacks:
                    continue
                final = unit.position if attacks else destination
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
        if not unit.alive or not self.living_cities():
            return
        dx, dy = delta
        destination = unit.x + dx, unit.y + dy
        cell = self.map.grid.get(destination)
        if cell is None or cell.block_path:
            unit.status = 'hit wall'
            unit._event('invalid_move')
            return
        target = self.occupant(destination, exclude=unit)
        if target is not None and target is self.capital:
            unit._event('blocked_move')
        elif isinstance(target, (C_City, C_Raider)) and unit.support:
            unit._event('invalid_move')
        elif isinstance(target, C_Raider):
            damage_out, damage_taken = attack(unit, target)
            unit._event('attacked')
            unit.take_damage(damage_taken)
            target.take_damage(damage_out)
            if unit.alive:
                unit.status = 'attacked'
        elif isinstance(target, C_City):
            damage_out, damage_taken = attack(unit, target)
            unit._event('attacked')
            unit.take_damage(damage_taken)
            target.take_damage(damage_out, aggressor_alive=unit.alive,
                               wall_multiplier=self.melee_wall_multiplier(unit, target))
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
            distance_before = self.nearest_city_distance(unit.position)
            origin = unit.position
            unit.x, unit.y = destination
            unit.status = 'alive'
            unit._event('moved', progress=distance_before - self.nearest_city_distance(destination),
                        cost=self.map.movement_cost(origin, destination))

    def ram_in_position(self, city=None):
        """Whether a battering ram stands next to city (default: the enemy capital)."""
        city = city or self.city
        return any(u.alive and u.support and hex_distance(u.position, city.position) == 1
                   for u in self.units)

    def melee_wall_multiplier(self, unit, city=None):
        if UNIT_SPECS[unit.unit_type].unit_class in RAM_SUPPORTED_CLASSES and self.ram_in_position(city):
            return 1.0
        return MELEE_WALL_DAMAGE

    def city_turn(self):
        """Each city that has not been captured shoots and heals; then the enemy capital's economy runs."""
        for city in self.living_cities():
            if city.ranged_combat:
                targets = [u for u in self.units if u.alive and hex_distance(u.position, city.position) <= 2]
                if targets:
                    target = self.rng.choice(targets)
                    damage, _ = attack(city, target, ranged=True)
                    target.take_damage(damage)
            if city.heal:
                besiegers = sum(u.alive and hex_distance(u.position, city.position) == 1 for u in self.units)
                if besiegers < min(3, self.config.unit_count):
                    healed = min(10, city.hp_max - city.hp)
                    if healed > 0:
                        city.hp += healed
                        city.status = 'healed'
                        city._event(city.heal_event, amount=healed)
        if self.economy is not None and self.city.alive:
            self.economy.turn()

    def raider_turn(self):
        """Raiders next to the capital attack it. The rest march toward it, stepping around
        the squad when they can and attacking a unit that blocks every way forward."""
        steps = self._raider_steps()
        for raider in self.living_raiders():
            if not self.capital.alive:
                break
            if hex_distance(raider.position, self.capital.position) == 1:
                damage_out, damage_taken = attack(raider, self.capital)
                raider._event('raider_attacked')
                raider.take_damage(damage_taken)
                self.capital.take_damage(damage_out, aggressor_alive=raider.alive,
                                         wall_multiplier=MELEE_WALL_DAMAGE)
                continue
            here = steps.get(raider.position, math.inf)
            moves, blockers = [], []
            for order, direction in enumerate(constants.DIRECTIONS[:-1]):
                dx, dy = constants.movement_delta(direction, raider.y)
                destination = raider.x + dx, raider.y + dy
                after = steps.get(destination)
                if after is None or after >= here:
                    continue
                occupant = self.occupant(destination)
                if occupant is None:
                    moves.append((after, order, destination))
                elif occupant in self.units:
                    blockers.append((occupant.hp, self.units.index(occupant), occupant))
            if moves:
                raider.x, raider.y = min(moves)[2]
            elif blockers:
                defender = min(blockers)[-1]
                damage_out, damage_taken = attack(raider, defender)
                raider._event('raider_attacked')
                defender.take_damage(damage_out)
                raider.take_damage(damage_taken)

    def raiding_city(self):
        """The enemy city whose turn it is to raid this turn, or None.

        Every raid_interval turns one city sends a raider, the cities taking turns in order,
        so capturing cities slows the raids. A captured city's turn passes without a raid.
        """
        interval = self.config.raid_interval
        if self.capital is None or not interval or self.turn_number % interval:
            return None
        city = self.cities[(self.turn_number // interval - 1) % len(self.cities)]
        return city if city.alive else None

    def spawn_raiders(self):
        """Send a raider from the raiding city, on the free tile beside it nearest the capital."""
        city = self.raiding_city()
        if city is None or len(self.living_raiders()) >= MAX_RAIDERS:
            return
        steps = self._raider_steps()
        sites = []
        for order, direction in enumerate(constants.DIRECTIONS[:-1]):
            dx, dy = constants.movement_delta(direction, city.y)
            site = city.x + dx, city.y + dy
            cell = self.map.grid.get(site)
            if cell is not None and not cell.block_path and self.occupant(site) is None:
                sites.append((steps.get(site, math.inf), order, site))
        if not sites:
            return
        slot = next((i for i, r in enumerate(self.raiders) if not r.alive), len(self.raiders))
        raider = C_Raider(*min(sites)[2], name_instance=f'Raider {slot + 1}',
                          strength=self.config.raider_strength, env=self)
        if slot == len(self.raiders):
            self.raiders.append(raider)
        else:
            self.raiders[slot] = raider
        raider._event('raider_spawned')

    def capital_turn(self):
        """The capital heals 10 HP at the end of any turn with no raider next to it."""
        capital = self.capital
        if any(hex_distance(r.position, capital.position) == 1 for r in self.living_raiders()):
            return
        healed = min(10, capital.hp_max - capital.hp)
        if healed > 0:
            capital.hp += healed
            capital._event(capital.heal_event, amount=healed)

    def is_built(self, item):
        """Whether the city already has a build item; any walls count as walls."""
        if item == 'ancient_walls':
            return self.city.wall_hp_max > 0
        return item in self.city.buildings

    def complete_build(self, item):
        """Add a finished building to the city; walls arrive at full health with their ranged strike."""
        city = self.city
        if item == 'ancient_walls':
            spec = WALL_SPECS['ancient']
            city.walls = 'ancient'
            city.wall_hp = city.wall_hp_max = float(spec.hp)
            if not city.ranged_combat:
                city.strength_ranged, city.ranged_combat = float(spec.ranged_strength), True
        else:
            city.buildings += (item,)

    def shoot(self, unit):
        if not unit.alive or not self.living_cities():
            return
        target = self.shoot_target(unit) if unit.ranged else None
        if target is None:
            unit._event('invalid_move')
            return
        damage, _ = attack(unit, target, ranged=True)
        unit._event('ranged_attack')
        if isinstance(target, C_Raider):
            target.take_damage(damage)
        else:
            # Capturing requires a melee action; a ranged strike leaves at least 1 HP.
            target.take_damage(damage, aggressor_alive=False)

    def _outcome(self):
        if all(not city.alive or city.hp <= 0 for city in self.cities):
            return 'win'
        if self.capital is not None and not self.capital.alive:
            return 'loss'
        if not any(u.alive and u.hp > 0 and not u.support for u in self.units):
            return 'loss'
        return 'ongoing'

    def _city_named(self, name):
        return next((city for city in self.cities if city.name_instance == name), self.city)

    def get_rewards(self):
        """Reward only the events of the most recent turn; reading is harmless."""
        reward = -.05
        for event in self.events:
            kind = event['type']
            amount = event.get('amount', 0)
            if kind == 'city_damaged':
                reward += 3 * amount / self._city_named(event['object']).hp_max
            elif kind == 'wall_damaged':
                reward += 2 * amount / max(1, self._city_named(event['object']).wall_hp_max)
            elif kind == 'city_healed':
                reward -= 3 * amount / self._city_named(event['object']).hp_max
            elif kind == 'capital_damaged':
                reward -= 3 * amount / self.capital.hp_max
            elif kind == 'raider_damaged':
                reward += .5 * amount / 100
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
        # Each city captured before the last earns 5; the last one brings the 10-point victory.
        captures = sum(event['type'] == 'city_captured' for event in self.events)
        if self.last_info['outcome'] == 'win':
            reward += 10 + 5 * max(0, captures - 1)
        else:
            reward += 5 * captures
        if self.last_info['outcome'] == 'loss':
            reward -= 10
        return float(reward)

    def step(self, action=0):
        if self.city is None:
            raise RuntimeError('Call reset before step')
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < self.action_count:
            raise ValueError(f'action must be an integer in [0, {self.action_count})')
        if self._done:
            return self.observe(), 0.0, True
        self.events = []
        self.turn_number += 1
        if self._outcome() == 'ongoing':
            for unit, direction in zip(self.units, self.actions[action]):
                if direction == 'SHOOT':
                    self.shoot(unit)
                else:
                    self.move_unit(unit, constants.movement_delta(direction, unit.y))
                if not self.living_cities():
                    break
            if self._outcome() == 'ongoing':
                self.city_turn()
            if self.capital is not None and self._outcome() == 'ongoing':
                self.raider_turn()
                if self._outcome() == 'ongoing':
                    self.spawn_raiders()
                    self.capital_turn()
        outcome = self._outcome()
        self._done = outcome != 'ongoing'
        self.last_info = self._info(outcome)
        reward = self.get_rewards()
        return self.observe(), reward, self._done

    def observe(self):
        """Return the observation for this environment's configured mode."""
        return self.tactical_observation() if self.observation_mode == 'tactical' else self.get_observation()

    def get_observation(self):
        if self.city is None:
            raise RuntimeError('Call reset before get_observation')
        c = self.config
        values = []
        for city in self.cities:
            values.extend([city.x / (c.width - 1), city.y / (c.height - 1),
                           city.hp / city.hp_max, city.wall_hp / 1000,
                           city.strength / 100, city.strength_ranged / 100, float(city.heal)])
        for unit in self.units:
            values.extend([unit.x / (c.width - 1), unit.y / (c.height - 1),
                           unit.hp / unit.hp_max, float(unit.alive), unit.strength / 100,
                           float(unit.y % 2), unit.strength_ranged / 100, unit.attack_range / 2])
        if self.capital is not None:
            capital = self.capital
            values.extend([capital.x / (c.width - 1), capital.y / (c.height - 1),
                           capital.hp / capital.hp_max, capital.strength / 100])
            for slot in range(MAX_RAIDERS):
                raider = self.raiders[slot] if slot < len(self.raiders) else None
                if raider is not None and raider.alive:
                    values.extend([raider.x / (c.width - 1), raider.y / (c.height - 1),
                                   raider.hp / raider.hp_max, 1.0])
                else:
                    values.extend([0.0, 0.0, 0.0, 0.0])
        cells = [self.map.grid[(x, y)] for y in range(c.height) for x in range(c.width)]
        if c.extended_terrain:
            # Combat-relevant properties: impassable, extra movement cost, defense bonus.
            for cell in cells:
                blocked = cell.block_path
                values.extend([float(blocked), 0.0 if blocked else (cell.movement_cost - 1) / 2,
                               0.0 if blocked else cell.defense_bonus / 6])
        else:
            values.extend(terrain_code(cell) / 2 for cell in cells)
        if c.river_count:
            values.extend(float(flag) for y in range(c.height) for x in range(c.width)
                          for flag in self.map.river_flags((x, y)))
        if self.economy is not None:
            values.extend(self.economy.observation(BUILD_ITEMS))
        return np.asarray(values, dtype=float)

    def target_city(self):
        """The city the squad focuses on: the weakest, then the closest to the squad."""
        attackers = [u for u in self.units if u.alive and not u.support]
        options = [(city.hp + city.wall_hp, sum(hex_distance(u.position, city.position) for u in attackers),
                    index, city) for index, city in enumerate(self.cities) if city.alive]
        return min(options)[-1] if options else None

    def objective(self, unit, threat=None):
        """What unit is working toward: a raider threatening the capital, else the target city.

        Battering rams cannot fight raiders, so they always head for the target city.
        After the last capture this is the enemy capital, so terminal states stay encodable.
        """
        if threat is not None and not unit.support:
            return threat
        return self.target_city() or self.city

    def _adjacency(self):
        """Passable neighbors of every passable tile, cached for the current terrain."""
        terrain = tuple(cell.terrain_type for cell in self.map.grid.values())
        cached = self._route_cache.get('adjacency')
        if cached is None or cached[0] != terrain:
            grid = self.map.grid
            adjacency = {position: [neighbor.index for neighbor in cell.get_neighbors(grid)
                                    if not neighbor.block_path]
                         for position, cell in grid.items() if not cell.block_path}
            self._route_cache.clear()
            cached = self._route_cache['adjacency'] = terrain, adjacency
        return cached[1]

    def _steps_to(self, goal, blocked):
        """Fewest turns from every reachable tile to goal, never passing through blocked tiles."""
        adjacency = self._adjacency()
        key = (goal, blocked)
        steps = self._route_cache.get(key)
        if steps is None:
            if len(self._route_cache) >= 64:
                # Raiders move every turn; keep the cache small.
                self._route_cache = {'adjacency': self._route_cache['adjacency']}
            steps = {goal: 0}
            pending = collections.deque([goal])
            while pending:
                position = pending.popleft()
                for neighbor in adjacency.get(position, ()):
                    if neighbor not in steps and neighbor not in blocked:
                        steps[neighbor] = steps[position] + 1
                        pending.append(neighbor)
            self._route_cache[key] = steps
        return steps

    def _route_blockers(self, target):
        """Uncaptured cities other than target, and the player's capital, which routes go around."""
        blocked = frozenset(city.position for city in self.living_cities() if city is not target)
        return blocked | {self.capital.position} if self.capital is not None else blocked

    def step_distances(self, target=None):
        """Fewest turns from each reachable tile to target (default: the target city).

        Every unit moves one hex per turn, so forests add no turns; they carry
        only a small reward penalty and a defense bonus. Routes go around other
        uncaptured cities and the player's capital.
        """
        target = target or self.target_city() or self.city
        return self._steps_to(target.position, self._route_blockers(target))

    def route_distances(self, unit, target=None):
        """Fewest turns to target for unit, routing around the squad's other living units.

        Covers unit's tile and its neighbors; tiles that other units occupy are
        absent. When allies fill the target's ring in front of a unit, the move
        toward a free approach is the one that shortens its route.
        """
        target = target or self.target_city() or self.city
        steps = self.step_distances(target)
        blocked = {other.position for other in self.units if other.alive and other is not unit}
        if not blocked:
            return steps
        adjacency = self._adjacency()
        blocked |= self._route_blockers(target)
        targets = {unit.position, *adjacency.get(unit.position, ())} - blocked
        routes = {target.position: 0}
        targets.discard(target.position)
        pending = collections.deque([target.position])
        while pending and targets:
            position = pending.popleft()
            for neighbor in adjacency.get(position, ()):
                if neighbor not in routes and neighbor not in blocked:
                    routes[neighbor] = routes[position] + 1
                    targets.discard(neighbor)
                    pending.append(neighbor)
        return routes

    def _raider_steps(self):
        """Fewest turns for raiders from every tile to the player's capital, around enemy cities."""
        return self._steps_to(self.capital.position, frozenset(city.position for city in self.living_cities()))

    def river_defense(self, unit, target=None):
        """The target's extra strength against a melee attack by unit from across a river."""
        target = target or self.city
        return RIVER_DEFENSE_BONUS if self.map.has_river(unit.position, target.position) else 0

    def melee_death_probability(self, unit, target=None):
        """Chance that the retaliation of target (default: the enemy capital) kills unit if it attacks now."""
        target = target or self.city
        bonus = self.map.grid[unit.position].defense_bonus
        strength = target.strength + self.river_defense(unit, target)
        return damage_at_least(unit.hp, math.exp((strength - unit.strength - bonus) / 25))

    def capture_probability(self, unit, target=None):
        """Chance that a melee attack by unit now captures target (default: the enemy capital).

        The blow must break any walls and then deal the city's remaining
        health, and the attacker must survive the retaliation, which resolves
        first. For a raider it is the chance the blow kills it, which does not
        depend on the attacker surviving.
        """
        target = target or self.city
        if not unit.alive or unit.support or not target.alive:
            return 0.0
        river = self.river_defense(unit, target)
        if isinstance(target, C_Raider):
            bonus = self.map.grid[target.position].defense_bonus
            return damage_at_least(target.hp, math.exp((unit.strength - target.strength - bonus - river) / 25))
        needed = target.hp
        if target.wall_hp > 0:
            needed += target.wall_hp / self.melee_wall_multiplier(unit, target)
        hit = damage_at_least(needed, math.exp((unit.strength - target.strength - river) / 25))
        return hit * (1 - self.melee_death_probability(unit, target))

    def tactical_observation(self):
        """Encode each unit's situation relative to its objective (see objective()).

        Every unit contributes TACTICAL_UNIT_WIDTH values: the TACTICAL_HEADER
        flag and fractions, the integer TACTICAL_FEATURES, and the TACTICS index
        that each of COMMANDS would perform from the current position (-1 when
        the command is unavailable). The objective is the target city, or a
        raider that threatens the player's capital; a raider has no walls, does
        not shoot or heal, and no ram helps against it. Moves are classified by
        the fewest turns to the objective along a route around the other living
        units, so the encoding does not depend on absolute coordinates, the map
        size or the seed. Whether a move is legal this turn is left to
        valid_actions(). A dead unit's block is zero with every command unavailable.
        """
        if self.city is None:
            raise RuntimeError('Call reset before tactical_observation')
        threat = self.threat()
        first_feature, first_tactic = len(TACTICAL_HEADER), TACTICAL_FIRST_TACTIC
        values = np.zeros(len(self.units) * TACTICAL_UNIT_WIDTH)
        for index, unit in enumerate(self.units):
            block = values[index * TACTICAL_UNIT_WIDTH:(index + 1) * TACTICAL_UNIT_WIDTH]
            block[first_tactic:] = -1
            if not unit.alive:
                continue
            target = self.objective(unit, threat)
            city = isinstance(target, C_City)
            others = [other for other in self.units if other is not unit]
            adjacent = [other.alive and hex_distance(other.position, target.position) == 1 for other in others]
            steps = self.route_distances(unit, target)
            distance = steps.get(unit.position, math.inf)
            health = 3 if unit.hp >= unit.hp_max else min(2, int(3 * unit.hp / unit.hp_max))
            risk = 0 if unit.support else risk_level(self.melee_death_probability(unit, target))
            target_health = min(4, max(0, math.ceil(5 * target.hp / target.hp_max) - 1))
            wall_max = target.wall_hp_max if city else 0
            walls = 0 if not wall_max or target.wall_hp <= 0 else 1 if target.wall_hp <= wall_max / 2 else 2
            shared = ((int(self.ram_in_position(target)), int(target.ranged_combat), int(target.heal))
                      if city else (0, 0, 0))
            block[:first_feature] = (1, unit.hp / unit.hp_max, target.hp / target.hp_max,
                                     target.wall_hp / wall_max if wall_max else 0.0)
            block[first_feature:first_tactic] = (
                UNIT_TYPES.index(unit.unit_type), min(distance, TACTICAL_FEATURE_LIMITS[1]), health, risk,
                target_health, walls, sum(adjacent),
                sum(other.alive and not other.support and not near for other, near in zip(others, adjacent)),
                *shared)
            for command_index, command in enumerate(COMMANDS):
                block[first_tactic + command_index] = self._command_tactic(unit, command, distance, steps)
        return values

    def _command_tactic(self, unit, command, distance, steps):
        if command == 'SHOOT':
            usable = unit.ranged and self.shoot_target(unit) is not None
            return TACTIC_INDEX['SHOOT'] if usable else -1
        if command == 'SPACE':
            return TACTIC_INDEX['HEAL' if unit.hp < unit.hp_max else 'WAIT']
        dx, dy = constants.movement_delta(command, unit.y)
        destination = unit.x + dx, unit.y + dy
        cell = self.map.grid.get(destination)
        if cell is None or cell.block_path or (self.capital is not None and destination == self.capital.position):
            return -1
        if self.enemy_at(destination) is not None:
            return -1 if unit.support else TACTIC_INDEX['ATTACK']
        after = steps.get(destination, math.inf)
        return TACTIC_INDEX['ADVANCE' if after < distance else 'SIDESTEP' if after == distance else 'RETREAT']

    def raid_damage(self, raider):
        """Expected damage a raider deals the capital before the capital's counterattacks kill it."""
        capital, average_roll = self.capital, 30 * .875
        per_attack = average_roll * math.exp((raider.strength - capital.strength) / 25)
        per_counter = average_roll * math.exp((capital.strength - raider.strength) / 25)
        return per_attack * math.ceil(raider.hp / per_counter)

    def threat(self):
        """The raider the squad should stop first, or None while the capital can hold.

        Only raiders within RAID_THREAT_RADIUS of the capital count, and the squad turns back
        only when they could together destroy it (with a 25% safety margin). A nearly taken
        enemy city is finished first.
        """
        if self.capital is None or not self.capital.alive:
            return None
        target = self.target_city()
        if target is not None and target.hp + target.wall_hp <= 40:
            return None
        near = [(hex_distance(r.position, self.capital.position), r.hp, index, r)
                for index, r in enumerate(self.raiders)
                if r.alive and hex_distance(r.position, self.capital.position) <= RAID_THREAT_RADIUS]
        if not near or 1.25 * sum(self.raid_damage(entry[-1]) for entry in near) < self.capital.hp:
            return None
        return min(near)[-1]

    def guided_action(self, rng=None):
        """Choose a joint action with a risk-aware siege heuristic.

        The squad besieges one city at a time (target_city()). Units close in by
        the fewest turns, and archers and slingers shoot when in range. Melee
        attacks are made only when the city's retaliation cannot kill the
        attacker while the city is encircled (a small risk is accepted while it
        can still heal), unless a capture is nearly certain. Otherwise units heal
        in place, hold next to the city until enough of them surround it to stop
        its healing, and wait just outside a ranged city's reach until the squad
        has closed up. When raiders could take the player's capital, the
        fighting units turn back to stop the nearest one first. Ties use rng.
        """
        rng = rng or self.rng
        city = self.target_city() or self.city
        threat = self.threat()
        steps = self.step_distances(city)
        alive = [u for u in self.units if u.alive]
        needed = min(3, self.config.unit_count)
        encircled = sum(hex_distance(u.position, city.position) == 1 for u in alive) >= needed
        # Attack freely once the city cannot heal, or when too few units survive to stop it.
        ready = not city.heal or encircled or len(alive) < needed
        # While the city cannot heal, waiting is nearly free: accept no chance of death.
        tolerance = 0.0 if not city.heal or encircled else .1
        staging = city.ranged_combat and any(steps.get(u.position, math.inf) > 3 for u in alive if not u.support)
        scores = []
        for index, unit in enumerate(self.units):
            if not unit.alive:
                scores.append({})
            elif threat is not None and not unit.support:
                # A raider does not heal or shoot, so only the retaliation's risk holds a unit back.
                scores.append(self._command_scores(index, threat, self.step_distances(threat), True, .1, False))
            else:
                scores.append(self._command_scores(index, city, steps, ready, tolerance, staging))
        best, choices = math.inf, []
        for action in self.valid_actions():
            score = sum(scores[index][command] for index, command in enumerate(self.actions[action])
                        if self.units[index].alive)
            if score < best - 1e-9:
                best, choices = score, [action]
            elif score <= best + 1e-9:
                choices.append(action)
        return rng.choice(choices)

    def _command_scores(self, index, target, steps, ready, tolerance, staging):
        """Cost of each usable command for one unit working toward target; lower is better."""
        unit = self.units[index]
        distance = steps.get(unit.position, math.inf)
        death = 0.0 if unit.support else self.melee_death_probability(unit, target)
        capture = self.capture_probability(unit, target)
        wounded = unit.hp < unit.hp_max
        shot = self.shoot_target(unit) if unit.ranged else None
        # Across a river from the target, a free tile on its side may be one step away.
        dry_bank = not unit.support and self.river_defense(unit, target) > 0 and self._dry_bank_nearby(unit, target)
        scores = {}
        for command in self.unit_directions[index]:
            if command == 'SHOOT':
                if shot is not None:
                    # Ranged fire cannot take a city's last hit point once its walls are down.
                    useless = isinstance(shot, C_City) and shot.hp <= 1 and shot.wall_hp <= 0
                    scores[command] = 5.0 if useless else -1.0
                continue
            if command == 'SPACE':
                if distance == 1:
                    holding = wounded or unit.support or not ready or death > tolerance
                    scores[command] = 0.0 if holding else .5
                else:
                    # A wounded unit near the target recovers before engaging.
                    recovering = wounded and death > tolerance and distance <= 2
                    scores[command] = distance + (0.0 if recovering else .4)
                continue
            dx, dy = constants.movement_delta(command, unit.y)
            destination = unit.x + dx, unit.y + dy
            cell = self.map.grid.get(destination)
            if cell is None or cell.block_path or (unit.support and destination == target.position):
                continue
            if destination == target.position:
                if capture >= .9 or (capture >= .5 and death <= tolerance):
                    scores[command] = -3.0
                elif death > tolerance:
                    scores[command] = 2.0 + 10 * death
                elif shot is not None or not ready:
                    scores[command] = 1.0
                else:
                    scores[command] = -.5
                if dry_bank:
                    # Prefer to attack from a bank that the river does not separate from the target.
                    scores[command] += 2.0
                continue
            if self.enemy_at(destination) is not None:
                # Another city or raider is not the objective; attack it only if nothing else is possible.
                scores[command] = 50.0
                continue
            after = steps.get(destination, math.inf)
            if staging and distance == 3 and after == 2:
                scores[command] = 3.5
            elif unit.ranged and unit.attack_range >= 2 and distance == 2 and after == 1:
                scores[command] = 1.5
            else:
                scores[command] = after + .01 * (self.map.movement_cost(unit.position, destination) - 1)
                if dry_bank and after == 1 and not self.map.has_river(destination, target.position):
                    # Cross to the target's side of the river rather than wait or attack across it.
                    scores[command] -= .75
        return scores

    def _dry_bank_nearby(self, unit, target=None):
        """Whether unit can step to a free tile beside target that no river separates from it."""
        target = target or self.city
        occupied = {other.position for other in self.units if other.alive and other is not unit}
        for neighbor in self.map.grid[unit.position].get_neighbors(self.map.grid):
            position = neighbor.index
            if (not neighbor.block_path and position != target.position and position not in occupied
                    and self.occupant(position) is None
                    and hex_distance(position, target.position) == 1
                    and not self.map.has_river(position, target.position)):
                return True
        return False

    def get_current_state(self):
        return {obj.name_instance: {'health': obj.hp, 'position': [obj.x, obj.y],
                                   'alive': obj.alive} for obj in self.objects}

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
