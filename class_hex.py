"""Hex geometry and terrain, independent of the rendering library."""

from dataclasses import dataclass
import functools
import itertools
import math

from constants import DIRECTIONS, movement_delta

TERRAIN_CODES = {'GRASSLAND': 0, 'FOREST': 1, 'MOUNTAIN': 2, 'EDGE': 2}

# Crossing a river uses all of a two-move unit's movement, like entering a forest.
RIVER_CROSSING_COST = 2
# Each river edge is reported once, by the tile on its west/north side.
RIVER_OBSERVATION_DIRECTIONS = ('E', 'SE', 'SW')
# Chance that a growing river takes the branch that best follows its flow.
RIVER_FLOW_BIAS = 0.7


def neighbor_positions(position):
    """All six neighbouring coordinates, including those off the map."""
    x, y = position
    return [(x + dx, y + dy) for dx, dy in (movement_delta(d, y) for d in DIRECTIONS[:-1])]


def direction_between(origin, destination):
    """Return the direction from origin to an adjacent destination, else None."""
    x, y = origin
    for direction in DIRECTIONS[:-1]:
        dx, dy = movement_delta(direction, y)
        if (x + dx, y + dy) == tuple(destination):
            return direction
    return None


def _center(position):
    x, y = position
    return x + .5 * (y % 2), y * math.sqrt(3) / 2


def _edge_corners(edge):
    """A hex side's two corners, each named by the three tiles that meet there."""
    first, second = sorted(edge)
    shared = set(neighbor_positions(first)) & set(neighbor_positions(second))
    return [frozenset((first, second, third)) for third in sorted(shared)]


def _corner_edges(corner):
    return [frozenset(pair) for pair in itertools.combinations(sorted(corner), 2)]


@functools.lru_cache(maxsize=32)
def _map_edges(num_rows, num_columns):
    """Every side shared by two tiles of a map this size, in a fixed order."""
    tiles = {(x, y) for y in range(num_rows) for x in range(num_columns)}
    edges = {frozenset((p, q)) for p in tiles for q in neighbor_positions(p) if q in tiles}
    return tuple(sorted(edges, key=sorted))


@dataclass
class HexCell:
    index: tuple
    terrain_type: str = 'GRASSLAND'

    def __post_init__(self):
        self.set_terrain(self.terrain_type)

    def set_terrain(self, terrain_type):
        terrain_type = terrain_type.upper()
        if terrain_type not in TERRAIN_CODES:
            raise ValueError(f'Unknown terrain: {terrain_type}')
        self.terrain_type = terrain_type

    @property
    def movement_cost(self):
        return 2 if self.terrain_type == 'FOREST' else 1

    @property
    def block_path(self):
        return self.terrain_type in ('MOUNTAIN', 'EDGE')

    @property
    def defense_bonus(self):
        return 3 if self.terrain_type == 'FOREST' else 0

    def get_neighbors(self, grid):
        return [grid[position] for position in neighbor_positions(self.index) if position in grid]


class HexMap:
    def __init__(self, num_rows, num_columns, cell_size=None, cell_offset=None):
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.grid = {(x, y): HexCell((x, y)) for y in range(num_rows)
                     for x in range(num_columns)}
        # Rivers run along hex sides; each is stored as the pair of tiles it separates.
        self.rivers = set()

    def reachable(self, origin):
        """Return the traversable component containing origin."""
        visited = {origin}
        pending = [origin]
        while pending:
            for cell in self.grid[pending.pop()].get_neighbors(self.grid):
                if not cell.block_path and cell.index not in visited:
                    visited.add(cell.index)
                    pending.append(cell.index)
        return visited

    def _is_map_edge(self, edge):
        return all(position in self.grid for position in edge)

    def add_river(self, first, second):
        """Place a river on the side shared by two adjacent tiles on the map."""
        first, second = tuple(first), tuple(second)
        edge = frozenset((first, second))
        if len(edge) != 2 or not self._is_map_edge(edge) or direction_between(first, second) is None:
            raise ValueError('A river must lie between two adjacent tiles on the map')
        self.rivers.add(edge)

    def has_river(self, first, second):
        return frozenset((tuple(first), tuple(second))) in self.rivers

    def river_flags(self, position):
        """River presence on a tile's east, southeast and southwest sides."""
        x, y = position
        return tuple(self.has_river(position, (x + dx, y + dy)) for dx, dy in
                     (movement_delta(d, y) for d in RIVER_OBSERVATION_DIRECTIONS))

    def movement_cost(self, origin, destination):
        """Cost of stepping between adjacent tiles, including any river crossing."""
        cost = self.grid[tuple(destination)].movement_cost
        if self.has_river(origin, destination):
            cost = max(cost, RIVER_CROSSING_COST)
        return cost

    def generate_rivers(self, count, rng):
        """Add up to count rivers that meander across the map along hex sides.

        Each river starts on a random side and grows in both directions,
        usually following a random flow direction, until it leaves the map or
        joins an earlier river. No randomness is used when count is zero.
        """
        if count <= 0:
            return
        map_edges = _map_edges(self.num_rows, self.num_columns)
        existing = {corner for edge in self.rivers for corner in _edge_corners(edge)}
        for _ in range(count):
            candidates = [edge for edge in map_edges if edge not in self.rivers]
            if not candidates:
                return
            start = rng.choice(candidates)
            angle = rng.uniform(0, 2 * math.pi)
            flow = (math.cos(angle), math.sin(angle))
            self.rivers.add(start)
            corners = sorted(_edge_corners(start), key=lambda c: self._progress(c, flow))
            visited = set(corners)
            limit = self.num_rows + self.num_columns
            # Grow downstream from the leading corner and upstream from the trailing one.
            for corner, direction in ((corners[1], flow), (corners[0], (-flow[0], -flow[1]))):
                self._grow_river(corner, start, direction, rng, visited, existing, limit)
            existing |= visited  # The corners this river passes through.

    def _progress(self, corner, direction):
        points = [_center(position) for position in corner]
        return sum(direction[0] * px + direction[1] * py for px, py in points) / len(points)

    def _grow_river(self, corner, previous, direction, rng, visited, existing, limit):
        for _ in range(limit):
            if corner in existing:
                return  # Joined an earlier river.
            options = []
            for edge in _corner_edges(corner):
                if edge == previous or edge in self.rivers or not self._is_map_edge(edge):
                    continue
                following = next(c for c in _edge_corners(edge) if c != corner)
                if following not in visited:
                    options.append((edge, following))
            if not options:
                return  # Reached the map border.
            if len(options) > 1 and rng.random() < RIVER_FLOW_BIAS:
                edge, following = max(options, key=lambda option: self._progress(option[1], direction))
            else:
                edge, following = rng.choice(options)
            self.rivers.add(edge)
            visited.add(following)
            corner, previous = following, edge
