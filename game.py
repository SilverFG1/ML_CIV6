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
                               or any(t not in ('warrior', 'archer') for t in self.unit_types)):
            raise ValueError('unit_types must contain warrior or archer for every unit')
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
        super().__init__(*args, **kwargs)
        self.strength_ranged = float(strength_ranged)
        self.unit_type = unit_type
        self.attack_range = 2 if unit_type == 'archer' else 1

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

    def take_damage(self, damage, aggressor_alive=True):
        if not self.alive:
            return
        damage = max(0, float(damage))
        wall_damage = min(self.wall_hp, damage)
        if wall_damage:
            self.wall_hp -= wall_damage
            damage -= wall_damage
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
    def __init__(self, human=False, ml_ai=False, render=False, config=None, seed=None):
        self.config = config or GameConfig()
        self.human, self.ml_ai, self.render = human, ml_ai, render
        self.rng = random.Random(seed)
        self.unit_types = tuple(self.config.unit_types or ('warrior',) * self.config.unit_count)
        self.unit_directions = [constants.DIRECTIONS + (('SHOOT',) if kind == 'archer' else ())
                                for kind in self.unit_types]
        self.actions = tuple(itertools.product(*self.unit_directions))
        self.action_count = len(self.actions)
        self._action_index = {a: i for i, a in enumerate(self.actions)}
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
        strengths = c.unit_strengths or tuple(12 if t == 'archer' else 20 for t in self.unit_types)
        names = ('Otto', 'Fynn', 'Victor', 'Ada')
        self.units = [C_Unit(*pos, name_instance=names[i], strength=strengths[i], env=self,
                             unit_type=self.unit_types[i],
                             strength_ranged=20 if self.unit_types[i] == 'archer' else 0)
                      for i, pos in enumerate(start_positions)]
        self.city = C_City(*city_pos, name_instance='Ottertopia', strength=c.city_strength,
                           wall_hp=c.wall_hp, strength_ranged=c.ranged_strength,
                           ranged_combat=c.ranged_strength > 0, heal=c.city_heals, env=self)
        self.turn_number = 0
        self.events = []
        self.last_info = {'outcome': 'ongoing', 'events': []}
        self._done = self.quit = False
        self._valid_cache_key = None
        self._path_costs = None
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
        if target is self.city:
            damage_out, damage_taken = attack(unit, self.city)
            unit._event('attacked')
            unit.take_damage(damage_taken)
            self.city.take_damage(damage_out, aggressor_alive=unit.alive)
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
        if unit.unit_type != 'archer' or hex_distance(unit.position, self.city.position) > unit.attack_range:
            unit._event('invalid_move')
            return
        damage, _ = attack(unit, self.city, ranged=True)
        unit._event('ranged_attack')
        # Capturing requires a melee action; a ranged strike leaves at least 1 HP.
        self.city.take_damage(damage, aggressor_alive=False)

    def _outcome(self):
        if not self.city.alive or self.city.hp <= 0:
            return 'win'
        if not any(u.alive and u.hp > 0 for u in self.units):
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
        values.extend(TERRAIN_CODES[self.map.grid[(x, y)].terrain_type] / 2
                      for y in range(c.height) for x in range(c.width))
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

    def guided_action(self, rng=None):
        rng = rng or self.rng
        costs = self._distances()
        scores = []
        for action in self.valid_actions():
            score = 0
            for unit, direction in zip(self.units, self.actions[action]):
                if not unit.alive:
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
                    score += -2 if unit.hp < .35 * unit.hp_max else .3
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
