"""A small Civ-like combat environment with isolated state and seeded randomness."""

import collections
from dataclasses import dataclass
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
    ranged_strength: float = 0
    city_heals: bool = True
    mountain_density: float = 0
    forest_density: float = 0
    unit_strengths: tuple = ()
    unit_types: tuple = ()
    difficulty: str = 'normal'

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
            # Egocentric features do not depend on the map size.
            self.observation_schema = 'ml_civ6.tactical.v1:' + ','.join(self.unit_types)
        else:
            self.observation_schema = (f'ml_civ6.v2:{self.config.width}x{self.config.height}:'
                                       + ','.join(self.unit_types))
        self._renderer = None
        self.quit = False
        self.units = []
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
            if len(component) >= c.unit_count + 1 and sum(not cell.block_path for cell in adjacent) >= min(c.unit_count, 3):
                break
        else:
            for cell in self.map.grid.values():
                cell.set_terrain('GRASSLAND')
            component = self.map.reachable(city_pos)
        start_positions = self.rng.sample(sorted(component - {city_pos}), c.unit_count)
        specs = [UNIT_SPECS[kind] for kind in self.unit_types]
        strengths = c.unit_strengths or tuple(spec.strength for spec in specs)
        names = ('Otto', 'Fynn', 'Victor', 'Ada')
        self.units = [C_Unit(*pos, name_instance=names[i], strength=strengths[i], env=self,
                             unit_type=self.unit_types[i], strength_ranged=specs[i].ranged_strength)
                      for i, pos in enumerate(start_positions)]
        self.city = C_City(*city_pos, name_instance='Ottertopia', strength=c.city_strength,
                           wall_hp=c.wall_hp, strength_ranged=c.ranged_strength,
                           ranged_combat=c.ranged_strength > 0, heal=c.city_heals, env=self)
        self.turn_number = 0
        self.events = []
        self.last_info = {'outcome': 'ongoing', 'events': []}
        self._done = self.quit = False
        self._valid_cache_key = None
        self._step_cache = None
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
        key = (tuple((u.position, u.alive) for u in self.units), self.city.position, terrain)
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
                if unit.support and destination == self.city.position:
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
        if self.city.heal:
            besiegers = sum(u.alive and hex_distance(u.position, self.city.position) == 1 for u in self.units)
            if besiegers < min(3, self.config.unit_count):
                healed = min(10, self.city.hp_max - self.city.hp)
                if healed > 0:
                    self.city.hp += healed
                    self.city.status = 'healed'
                    self.city._event('city_healed', amount=healed)

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
            return self.observe(), 0.0, True
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
        reward = self.get_rewards()
        return self.observe(), reward, self._done

    def observe(self):
        """Return the observation for this environment's configured mode."""
        return self.tactical_observation() if self.observation_mode == 'tactical' else self.get_observation()

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
        values.extend(TERRAIN_CODES[self.map.grid[(x, y)].terrain_type] / 2
                      for y in range(c.height) for x in range(c.width))
        return np.asarray(values, dtype=float)

    def _terrain_routes(self):
        """Cache terrain-only distances and passable neighbors for this map."""
        key = (self.city.position, tuple(cell.terrain_type for cell in self.map.grid.values()))
        if self._step_cache is None or self._step_cache[0] != key:
            grid = self.map.grid
            adjacency = {position: [neighbor.index for neighbor in cell.get_neighbors(grid)
                                    if not neighbor.block_path]
                         for position, cell in grid.items() if not cell.block_path}
            steps = {self.city.position: 0}
            pending = collections.deque([self.city.position])
            while pending:
                position = pending.popleft()
                for neighbor in adjacency[position]:
                    if neighbor not in steps:
                        steps[neighbor] = steps[position] + 1
                        pending.append(neighbor)
            self._step_cache = key, steps, adjacency
        return self._step_cache[1], self._step_cache[2]

    def step_distances(self):
        """Fewest turns from each reachable tile to the city.

        Every unit moves one hex per turn, so forests add no turns; they carry
        only a small reward penalty and a defense bonus.
        """
        return self._terrain_routes()[0]

    def route_distances(self, unit):
        """Fewest turns to the city for unit, routing around the squad's other living units.

        Covers unit's tile and its neighbors; tiles that other units occupy are
        absent. When allies fill the city's ring in front of a unit, the move
        toward a free approach is the one that shortens its route.
        """
        steps, adjacency = self._terrain_routes()
        blocked = {other.position for other in self.units if other.alive and other is not unit}
        if not blocked:
            return steps
        targets = {unit.position, *adjacency.get(unit.position, ())} - blocked
        routes = {self.city.position: 0}
        targets.discard(self.city.position)
        pending = collections.deque([self.city.position])
        while pending and targets:
            position = pending.popleft()
            for neighbor in adjacency[position]:
                if neighbor not in routes and neighbor not in blocked:
                    routes[neighbor] = routes[position] + 1
                    targets.discard(neighbor)
                    pending.append(neighbor)
        return routes

    def melee_death_probability(self, unit):
        """Chance that the city's retaliation kills unit if it attacks in melee now."""
        bonus = self.map.grid[unit.position].defense_bonus
        return damage_at_least(unit.hp, math.exp((self.city.strength - unit.strength - bonus) / 25))

    def capture_probability(self, unit):
        """Chance that a melee attack by unit now captures the city.

        The blow must break any walls and then deal the city's remaining
        health, and the attacker must survive the retaliation, which resolves
        first.
        """
        city = self.city
        if not unit.alive or unit.support or not city.alive:
            return 0.0
        needed = city.hp
        if city.wall_hp > 0:
            needed += city.wall_hp / self.melee_wall_multiplier(unit)
        hit = damage_at_least(needed, math.exp((unit.strength - city.strength) / 25))
        return hit * (1 - self.melee_death_probability(unit))

    def tactical_observation(self):
        """Encode each unit's situation relative to the city.

        Every unit contributes TACTICAL_UNIT_WIDTH values: the TACTICAL_HEADER
        flag and fractions, the integer TACTICAL_FEATURES, and the TACTICS index
        that each of COMMANDS would perform from the current position (-1 when
        the command is unavailable). Moves are classified by the fewest turns to
        the city along a route around the other living units, so the encoding
        does not depend on absolute coordinates, the map size or the seed.
        Whether a move is legal this turn is left to valid_actions(). A dead
        unit's block is zero with every command unavailable.
        """
        if self.city is None:
            raise RuntimeError('Call reset before tactical_observation')
        city = self.city
        adjacent = [u.alive and hex_distance(u.position, city.position) == 1 for u in self.units]
        attackers = [u.alive and not u.support for u in self.units]
        city_health = min(4, max(0, math.ceil(5 * city.hp / city.hp_max) - 1))
        walls = 0 if city.wall_hp <= 0 else 1 if city.wall_hp <= city.wall_hp_max / 2 else 2
        shared = (int(self.ram_in_position()), int(city.ranged_combat), int(city.heal))
        first_feature, first_tactic = len(TACTICAL_HEADER), TACTICAL_FIRST_TACTIC
        header = (city.hp / city.hp_max, city.wall_hp / city.wall_hp_max if city.wall_hp_max else 0.0)
        values = np.zeros(len(self.units) * TACTICAL_UNIT_WIDTH)
        for index, unit in enumerate(self.units):
            block = values[index * TACTICAL_UNIT_WIDTH:(index + 1) * TACTICAL_UNIT_WIDTH]
            block[first_tactic:] = -1
            if not unit.alive:
                continue
            steps = self.route_distances(unit)
            distance = steps.get(unit.position, math.inf)
            health = 3 if unit.hp >= unit.hp_max else min(2, int(3 * unit.hp / unit.hp_max))
            risk = 0 if unit.support else risk_level(self.melee_death_probability(unit))
            others = [other for other in range(len(self.units)) if other != index]
            block[:first_feature] = (1, unit.hp / unit.hp_max, *header)
            block[first_feature:first_tactic] = (
                UNIT_TYPES.index(unit.unit_type), min(distance, TACTICAL_FEATURE_LIMITS[1]), health, risk,
                city_health, walls, sum(adjacent[other] for other in others),
                sum(attackers[other] and not adjacent[other] for other in others), *shared)
            for command_index, command in enumerate(COMMANDS):
                block[first_tactic + command_index] = self._command_tactic(unit, command, distance, steps)
        return values

    def _command_tactic(self, unit, command, distance, steps):
        city = self.city
        if command == 'SHOOT':
            usable = unit.ranged and city.alive and hex_distance(unit.position, city.position) <= unit.attack_range
            return TACTIC_INDEX['SHOOT'] if usable else -1
        if command == 'SPACE':
            return TACTIC_INDEX['HEAL' if unit.hp < unit.hp_max else 'WAIT']
        dx, dy = constants.movement_delta(command, unit.y)
        destination = unit.x + dx, unit.y + dy
        cell = self.map.grid.get(destination)
        if cell is None or cell.block_path:
            return -1
        if destination == city.position:
            return -1 if unit.support else TACTIC_INDEX['ATTACK']
        after = steps.get(destination, math.inf)
        return TACTIC_INDEX['ADVANCE' if after < distance else 'SIDESTEP' if after == distance else 'RETREAT']

    def guided_action(self, rng=None):
        """Choose a joint action with a risk-aware siege heuristic.

        Units close in by the fewest turns, and archers and slingers shoot when
        in range. Melee attacks are made only when the city's retaliation
        cannot kill the attacker while the city is encircled (a small risk is
        accepted while it can still heal), unless a capture is nearly certain.
        Otherwise units heal in place, hold next to the city until enough of
        them surround it to stop its healing, and wait just outside a ranged
        city's reach until the squad has closed up. Ties use rng.
        """
        rng = rng or self.rng
        city = self.city
        steps = self.step_distances()
        alive = [u for u in self.units if u.alive]
        needed = min(3, self.config.unit_count)
        encircled = sum(hex_distance(u.position, city.position) == 1 for u in alive) >= needed
        # Attack freely once the city cannot heal, or when too few units survive to stop it.
        ready = not city.heal or encircled or len(alive) < needed
        # While the city cannot heal, waiting is nearly free: accept no chance of death.
        tolerance = 0.0 if not city.heal or encircled else .1
        staging = city.ranged_combat and any(steps.get(u.position, math.inf) > 3 for u in alive if not u.support)
        scores = [self._command_scores(index, steps, ready, tolerance, staging) if unit.alive else {}
                  for index, unit in enumerate(self.units)]
        best, choices = math.inf, []
        for action in self.valid_actions():
            score = sum(scores[index][command] for index, command in enumerate(self.actions[action])
                        if self.units[index].alive)
            if score < best - 1e-9:
                best, choices = score, [action]
            elif score <= best + 1e-9:
                choices.append(action)
        return rng.choice(choices)

    def _command_scores(self, index, steps, ready, tolerance, staging):
        """Cost of each usable command for one unit; lower is better."""
        unit, city = self.units[index], self.city
        distance = steps.get(unit.position, math.inf)
        death = 0.0 if unit.support else self.melee_death_probability(unit)
        capture = self.capture_probability(unit)
        wounded = unit.hp < unit.hp_max
        in_range = unit.ranged and hex_distance(unit.position, city.position) <= unit.attack_range
        scores = {}
        for command in self.unit_directions[index]:
            if command == 'SHOOT':
                if in_range:
                    # Ranged fire cannot take the last hit point once the walls are down.
                    scores[command] = 5.0 if city.hp <= 1 and city.wall_hp <= 0 else -1.0
                continue
            if command == 'SPACE':
                if distance == 1:
                    holding = wounded or unit.support or not ready or death > tolerance
                    scores[command] = 0.0 if holding else .5
                else:
                    # A wounded unit near the city recovers before engaging.
                    recovering = wounded and death > tolerance and distance <= 2
                    scores[command] = distance + (0.0 if recovering else .4)
                continue
            dx, dy = constants.movement_delta(command, unit.y)
            destination = unit.x + dx, unit.y + dy
            cell = self.map.grid.get(destination)
            if cell is None or cell.block_path or (unit.support and destination == city.position):
                continue
            if destination == city.position:
                if capture >= .9 or (capture >= .5 and death <= tolerance):
                    scores[command] = -3.0
                elif death > tolerance:
                    scores[command] = 2.0 + 10 * death
                elif in_range or not ready:
                    scores[command] = 1.0
                else:
                    scores[command] = -.5
                continue
            after = steps.get(destination, math.inf)
            if staging and distance == 3 and after == 2:
                scores[command] = 3.5
            elif unit.ranged and unit.attack_range >= 2 and distance == 2 and after == 1:
                scores[command] = 1.5
            else:
                scores[command] = after + .01 * (cell.movement_cost - 1)
        return scores

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
