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

# Each enemy city is a rival civilization's capital. Capturing all of them is a
# domination victory.
CITY_NAMES = ('Ottertopia', 'Kelpford', 'Clamhaven', 'Driftwood')
MAX_CITIES = len(CITY_NAMES)
MIN_CITY_DISTANCE = 3


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


# With player_capital set, the squad has its own capital to defend. Enemy cities
# send raiders to capture it; losing it loses the game.
PLAYER_CAPITAL_NAME = 'Otterholt'
MAX_RAIDERS = 4
RAID_THREAT_RADIUS = 3


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
    city_count: int = 1
    player_capital: bool = False
    capital_strength: float = 28
    raid_interval: int = 6
    raider_strength: float = 20

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
        if self.city_position is not None:
            if len(self.city_position) != 2 or any(isinstance(v, bool) or not isinstance(v, int)
                                                 for v in self.city_position):
                raise ValueError('city_position must contain two integers')
            x, y = self.city_position
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError('city_position must lie inside the map')
        if (isinstance(self.city_count, bool) or not isinstance(self.city_count, int)
                or not 1 <= self.city_count <= MAX_CITIES):
            raise ValueError(f'city_count must be between 1 and {MAX_CITIES}')
        capital = self.city_position or (self.width // 2, self.height // 2)
        every_tile = [(x, y) for y in range(self.height) for x in range(self.width)]
        siege_room = min(self.unit_count, 3)

        def fits(site):
            return open_neighbors(site, lambda p: 0 <= p[0] < self.width and 0 <= p[1] < self.height) >= siege_room

        if not isinstance(self.player_capital, bool):
            raise ValueError('player_capital must be True or False')
        for name in ('capital_strength', 'raider_strength'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 100:
                raise ValueError(f'{name} must be finite and in (0, 100]')
        if (isinstance(self.raid_interval, bool) or not isinstance(self.raid_interval, int)
                or not 0 <= self.raid_interval <= 1000):
            raise ValueError('raid_interval must be an integer between 0 and 1000')
        enemy_sites = place_cities(capital, every_tile, self.city_count, fits=fits)
        if enemy_sites is None:
            raise ValueError(f'the map cannot fit {self.city_count} cities at least {MIN_CITY_DISTANCE} '
                             f'hexes apart, each with {siege_room} open neighboring tiles')
        if self.player_capital and place_capital(enemy_sites, every_tile, fits=fits) is None:
            raise ValueError(f"the map has no room for the player's capital at least {MIN_CITY_DISTANCE} "
                             f'hexes from every enemy city')


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
                 ranged_combat=False, heal=True, env=None):
        super().__init__(x, y, sprite, name_instance, hp, hp_max, strength, env)
        self.wall_hp = float(wall_hp)
        self.wall_hp_max = float(wall_hp)
        self.strength_ranged = float(strength_ranged)
        self.ranged_combat = ranged_combat
        self.heal = heal

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
        self.observation_schema = (f'ml_civ6.v2:{self.config.width}x{self.config.height}:'
                                   + ','.join(self.unit_types))
        if self.config.city_count > 1:
            self.observation_schema += f':cities={self.config.city_count}'
        if self.config.player_capital:
            self.observation_schema += ':capital'
        self._renderer = None
        self.quit = False
        self.units = []
        self.cities = []
        self.capital = None
        self.raiders = []
        self.events = []
        self.last_info = {'outcome': 'ongoing', 'events': [], 'cities_captured': 0, 'capital_lost': False}
        self.turn_number = 0
        self._done = False
        self._valid_cache_key = None

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

        # Keep enough connected grassland for starting units and a possible siege.
        for _ in range(100):
            for cell in self.map.grid.values():
                draw = self.rng.random()
                terrain = ('MOUNTAIN' if draw < c.mountain_density else
                           'FOREST' if draw < c.mountain_density + c.forest_density else 'GRASSLAND')
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
        self.cities = [C_City(*position, name_instance=CITY_NAMES[i], strength=c.city_strength,
                              wall_hp=c.wall_hp, strength_ranged=c.ranged_strength,
                              ranged_combat=c.ranged_strength > 0, heal=c.city_heals, env=self)
                       for i, position in enumerate(city_positions)]
        self.capital = (C_Capital(*home, name_instance=PLAYER_CAPITAL_NAME, strength=c.capital_strength, env=self)
                        if home is not None else None)
        self.raiders = []
        self.turn_number = 0
        self.events = []
        self.last_info = {'outcome': 'ongoing', 'events': [], 'cities_captured': 0, 'capital_lost': False}
        self._done = self.quit = False
        self._valid_cache_key = None
        self._path_cache = {}
        return self.get_observation()

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
                attacks = self.city_at(destination) is not None or self.raider_at(destination) is not None
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
            unit.x, unit.y = destination
            unit.status = 'alive'
            unit._event('moved', progress=distance_before - self.nearest_city_distance(destination),
                        cost=cell.movement_cost)

    def ram_in_position(self, city=None):
        city = city or self.city
        return any(u.alive and u.support and hex_distance(u.position, city.position) == 1
                   for u in self.units)

    def melee_wall_multiplier(self, unit, city=None):
        if UNIT_SPECS[unit.unit_type].unit_class in RAM_SUPPORTED_CLASSES and self.ram_in_position(city):
            return 1.0
        return MELEE_WALL_DAMAGE

    def city_turn(self):
        """Each city that has not been captured shoots and heals."""
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
                        city._event('city_healed', amount=healed)

    def raider_turn(self):
        """Raiders next to the capital attack it. The rest march toward it, stepping around
        the squad when they can and attacking a unit that blocks every way forward."""
        costs = self._raider_costs()
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
            here = costs.get(raider.position, float('inf'))
            steps, blockers = [], []
            for order, direction in enumerate(constants.DIRECTIONS[:-1]):
                dx, dy = constants.movement_delta(direction, raider.y)
                destination = raider.x + dx, raider.y + dy
                cost = costs.get(destination)
                if cost is None or cost >= here:
                    continue
                occupant = self.occupant(destination)
                if occupant is None:
                    steps.append((cost, order, destination))
                elif occupant in self.units:
                    blockers.append((occupant.hp, self.units.index(occupant), occupant))
            if steps:
                raider.x, raider.y = min(steps)[2]
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
        costs = self._raider_costs()
        sites = []
        for order, direction in enumerate(constants.DIRECTIONS[:-1]):
            dx, dy = constants.movement_delta(direction, city.y)
            site = city.x + dx, city.y + dy
            cell = self.map.grid.get(site)
            if cell is not None and not cell.block_path and self.occupant(site) is None:
                sites.append((costs.get(site, float('inf')), order, site))
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
            elif kind == 'capital_damaged':
                reward -= 3 * amount / self.capital.hp_max
            elif kind == 'raider_damaged':
                reward += .5 * amount / 100
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
            return self.get_observation(), 0.0, True
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
        self.last_info = {'outcome': outcome, 'events': [event.copy() for event in self.events],
                          'turn': self.turn_number, 'cities_captured': self.cities_captured(),
                          'capital_lost': self.capital is not None and not self.capital.alive}
        reward = self.get_rewards()
        return self.get_observation(), reward, self._done

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
        values.extend(TERRAIN_CODES[self.map.grid[(x, y)].terrain_type] / 2
                      for y in range(c.height) for x in range(c.width))
        return np.asarray(values, dtype=float)

    def target_city(self):
        """The city the heuristic squad focuses on: the weakest, then the closest to the squad."""
        attackers = [u for u in self.units if u.alive and not u.support]
        options = [(city.hp + city.wall_hp, sum(hex_distance(u.position, city.position) for u in attackers),
                    index, city) for index, city in enumerate(self.cities) if city.alive]
        return min(options)[-1] if options else None

    def _path_costs_to(self, goal, blocked):
        """Path cost from every tile to goal, never passing through blocked tiles."""
        key = (goal, blocked, tuple(cell.terrain_type for cell in self.map.grid.values()))
        if key in self._path_cache:
            return self._path_cache[key]
        if len(self._path_cache) >= 64:
            # Raider targets move every turn; keep the cache small.
            self._path_cache.clear()
        costs = {goal: 0}
        pending = [(0, goal)]
        while pending:
            cost, pos = heapq.heappop(pending)
            if cost != costs[pos]:
                continue
            for neighbor in self.map.grid[pos].get_neighbors(self.map.grid):
                if neighbor.block_path or neighbor.index in blocked:
                    continue
                new_cost = cost + self.map.grid[pos].movement_cost
                if new_cost < costs.get(neighbor.index, float('inf')):
                    costs[neighbor.index] = new_cost
                    heapq.heappush(pending, (new_cost, neighbor.index))
        self._path_cache[key] = costs
        return costs

    def _distances(self, target=None):
        """Path cost for the squad from every tile to the target (a city or a raider).

        Uncaptured cities other than the target, and the player's capital, block paths.
        """
        target = target or self.target_city()
        if target is None:
            return {}
        blocked = frozenset(city.position for city in self.living_cities() if city is not target)
        if self.capital is not None:
            blocked |= {self.capital.position}
        return self._path_costs_to(target.position, blocked)

    def _raider_costs(self):
        """Path cost for raiders from every tile to the player's capital."""
        return self._path_costs_to(self.capital.position,
                                   frozenset(city.position for city in self.living_cities()))

    def raid_damage(self, raider):
        """Expected damage a raider deals the capital before the capital's counterattacks kill it."""
        capital, average_roll = self.capital, 30 * .875
        per_attack = average_roll * math.exp((raider.strength - capital.strength) / 25)
        per_counter = average_roll * math.exp((capital.strength - raider.strength) / 25)
        return per_attack * math.ceil(raider.hp / per_counter)

    def threat(self):
        """The raider the heuristic squad should stop first, or None while the capital can hold.

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
        rng = rng or self.rng
        target = self.target_city()
        city_costs = self._distances(target)
        # A raider closing on the capital comes before the siege. Rams cannot fight it.
        threat = self.threat()
        defense_costs = self._distances(threat) if threat is not None else None
        # After taking a city, regroup: hurt units rest, and the squad attacks the
        # next city together instead of one unit at a time.
        regroup = self.cities_captured() > 0 and target is not None
        if regroup:
            attackers = [u for u in self.units if u.alive and not u.support]
            adjacent = sum(hex_distance(u.position, target.position) == 1 for u in attackers)
            slots = sum(not cell.block_path for cell in self.map.grid[target.position].get_neighbors(self.map.grid))
            assembled = adjacent >= min(3, len(attackers), slots)
        scores = []
        for action in self.valid_actions():
            score = 0
            for unit, direction in zip(self.units, self.actions[action]):
                if not unit.alive:
                    continue
                defending = defense_costs is not None and not unit.support
                costs = defense_costs if defending else city_costs
                sieging = regroup and not defending
                if direction == 'SHOOT':
                    shot = self.shoot_target(unit)
                    score += costs.get(unit.position, 10000) - 2
                    if isinstance(shot, C_City) and shot.hp <= 1 and shot.wall_hp == 0:
                        score += 5
                    continue
                dx, dy = constants.movement_delta(direction, unit.y)
                destination = unit.x + dx, unit.y + dy
                score += costs.get(destination, 10000)
                if sieging and destination == target.position and not assembled:
                    score += 3
                if direction == 'SPACE':
                    if unit.hp < .35 * unit.hp_max:
                        score -= 2
                    elif sieging and unit.hp < .7 * unit.hp_max:
                        score -= 2
                    elif sieging and not assembled and hex_distance(unit.position, target.position) == 1:
                        score -= 1
                    elif unit.support and target is not None and hex_distance(unit.position, target.position) == 1:
                        score -= 1
                    else:
                        score += .3
            scores.append((score, action))
        best = min(score for score, _ in scores)
        return rng.choice([action for score, action in scores if score == best])

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
