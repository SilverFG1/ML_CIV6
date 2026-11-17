"""Hex geometry and terrain, independent of the rendering library."""

from collections import deque
from dataclasses import dataclass
import functools
import itertools
import math

from constants import DIRECTIONS, movement_delta

YIELD_TYPES = ('food', 'production', 'gold', 'science', 'culture', 'faith')
LAND_TYPES = ('grassland', 'plains', 'desert', 'tundra', 'snow')


@dataclass(frozen=True)
class TerrainSpec:
    """Base-game Civ VI terrain rules (see civ6_bridge/data/terrain_types.json)."""
    civ6_type: str | None
    land_type: str | None = None  # None for water and the map edge
    elevation: str = 'FLAT'       # FLAT, HILLS or MOUNTAIN
    movement_cost: int = 1
    defense: int = 0
    impassable: bool = False
    water: bool = False
    yields: tuple = ()            # (yield type, amount) pairs


@dataclass(frozen=True)
class FeatureSpec:
    """A Civ VI feature that sits on top of a terrain, such as a forest."""
    civ6_type: str
    movement_change: int = 0
    defense: int = 0
    yields: tuple = ()
    land_types: tuple = ()        # land types it can grow on, flat or hills


def _land(civ6_type, land_type, elevation='FLAT', **yields):
    hills = elevation == 'HILLS'
    return TerrainSpec(civ6_type, land_type, elevation, movement_cost=2 if hills else 1,
                       defense=3 if hills else 0, impassable=elevation == 'MOUNTAIN',
                       yields=tuple(yields.items()))


# Hills add one production and +3 defense and cost two moves. Mountains are
# impassable and yield nothing. Land units cannot embark, so water blocks them.
TERRAINS = {
    'GRASSLAND': _land('TERRAIN_GRASS', 'GRASSLAND', food=2),
    'GRASSLAND_HILLS': _land('TERRAIN_GRASS_HILLS', 'GRASSLAND', 'HILLS', food=2, production=1),
    'GRASSLAND_MOUNTAIN': _land('TERRAIN_GRASS_MOUNTAIN', 'GRASSLAND', 'MOUNTAIN'),
    'PLAINS': _land('TERRAIN_PLAINS', 'PLAINS', food=1, production=1),
    'PLAINS_HILLS': _land('TERRAIN_PLAINS_HILLS', 'PLAINS', 'HILLS', food=1, production=2),
    'PLAINS_MOUNTAIN': _land('TERRAIN_PLAINS_MOUNTAIN', 'PLAINS', 'MOUNTAIN'),
    'DESERT': _land('TERRAIN_DESERT', 'DESERT'),
    'DESERT_HILLS': _land('TERRAIN_DESERT_HILLS', 'DESERT', 'HILLS', production=1),
    'DESERT_MOUNTAIN': _land('TERRAIN_DESERT_MOUNTAIN', 'DESERT', 'MOUNTAIN'),
    'TUNDRA': _land('TERRAIN_TUNDRA', 'TUNDRA', food=1),
    'TUNDRA_HILLS': _land('TERRAIN_TUNDRA_HILLS', 'TUNDRA', 'HILLS', food=1, production=1),
    'TUNDRA_MOUNTAIN': _land('TERRAIN_TUNDRA_MOUNTAIN', 'TUNDRA', 'MOUNTAIN'),
    'SNOW': _land('TERRAIN_SNOW', 'SNOW'),
    'SNOW_HILLS': _land('TERRAIN_SNOW_HILLS', 'SNOW', 'HILLS', production=1),
    'SNOW_MOUNTAIN': _land('TERRAIN_SNOW_MOUNTAIN', 'SNOW', 'MOUNTAIN'),
    'COAST': TerrainSpec('TERRAIN_COAST', water=True, yields=(('food', 1), ('gold', 1))),
    'OCEAN': TerrainSpec('TERRAIN_OCEAN', water=True, yields=(('food', 1),)),
    'EDGE': TerrainSpec(None, impassable=True),  # Simulator-only blocker; never generated.
}
FEATURES = {
    'FOREST': FeatureSpec('FEATURE_FOREST', movement_change=1, defense=3, yields=(('production', 1),),
                          land_types=('GRASSLAND', 'PLAINS', 'TUNDRA')),
}
# Earlier names still work: FOREST is forested grassland, MOUNTAIN a grassland mountain.
TERRAIN_ALIASES = {'FOREST': ('GRASSLAND', 'FOREST'), 'MOUNTAIN': ('GRASSLAND_MOUNTAIN', None)}

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


def hex_coords(position):
    x, y = position
    q = x - (y - y % 2) // 2
    return q, -q - y, y


def hex_distance(first, second):
    return max(abs(a - b) for a, b in zip(hex_coords(first), hex_coords(second)))


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


def terrain_code(cell):
    """The original one-value tile code: 0 open, 1 rough (forest or hills), 2 impassable."""
    if cell.block_path:
        return 2
    return 1 if cell.movement_cost > 1 or cell.defense_bonus else 0


@dataclass
class HexCell:
    index: tuple
    terrain_type: str = 'GRASSLAND'
    feature: str | None = None

    def __post_init__(self):
        feature = self.feature
        self.set_terrain(self.terrain_type)
        if feature is not None:
            self.set_feature(feature)

    def set_terrain(self, terrain_type):
        """Replace the tile's terrain and clear any feature."""
        name, feature = TERRAIN_ALIASES.get(terrain_type.upper(), (terrain_type.upper(), None))
        if name not in TERRAINS:
            raise ValueError(f'Unknown terrain: {terrain_type}')
        self.terrain_type, self.feature = name, None
        if feature is not None:
            self.set_feature(feature)

    def set_feature(self, feature):
        """Add a feature such as FOREST, or remove it with None."""
        if feature is None:
            self.feature = None
            return
        name = feature.upper()
        if name not in FEATURES:
            raise ValueError(f'Unknown feature: {feature}')
        terrain = self.spec
        if terrain.impassable or terrain.water or terrain.land_type not in FEATURES[name].land_types:
            raise ValueError(f'{name} cannot grow on {self.terrain_type}')
        self.feature = name

    def reshape(self, land_type=None, elevation=None):
        """Change the land type or elevation of a land tile, dropping a feature that no longer fits."""
        terrain = self.spec
        if terrain.land_type is None:
            return  # Water and the map edge have no land type.
        land_type = (land_type or terrain.land_type).upper()
        elevation = (elevation or terrain.elevation).upper()
        feature = self.feature
        self.set_terrain(land_type if elevation == 'FLAT' else f'{land_type}_{elevation}')
        if feature is not None and not self.spec.impassable and land_type in FEATURES[feature].land_types:
            self.set_feature(feature)

    @property
    def spec(self):
        return TERRAINS[self.terrain_type]

    @property
    def feature_spec(self):
        return FEATURES[self.feature] if self.feature is not None else None

    @property
    def is_water(self):
        return self.spec.water

    @property
    def movement_cost(self):
        feature = self.feature_spec
        return self.spec.movement_cost + (feature.movement_change if feature else 0)

    @property
    def block_path(self):
        return self.spec.impassable or self.spec.water

    @property
    def defense_bonus(self):
        feature = self.feature_spec
        return self.spec.defense + (feature.defense if feature else 0)

    @property
    def yields(self):
        """Base-game yields of the terrain and feature, e.g. {'food': 2, 'production': 1}."""
        totals = dict(self.spec.yields)
        for name, amount in self.feature_spec.yields if self.feature_spec else ():
            totals[name] = totals.get(name, 0) + amount
        return {name: totals[name] for name in YIELD_TYPES if totals.get(name)}

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

    def assign_land_types(self, land_types, rng):
        """Split the map into regions of the given land types, keeping each tile's elevation.

        Each tile takes the type of its nearest randomly placed region centre,
        found by flooding outwards from all centres at once. No randomness is
        used for a single land type.
        """
        land_types = tuple(dict.fromkeys(name.upper() for name in land_types))
        if len(land_types) == 1:
            for cell in self.grid.values():
                cell.reshape(land_type=land_types[0])
            return
        positions = sorted(self.grid)
        count = min(len(positions), max(2, round(len(positions) / 12)))
        region = {position: rng.choice(land_types) for position in rng.sample(positions, count)}
        pending = deque(region)
        while pending:
            position = pending.popleft()
            for neighbor in neighbor_positions(position):
                if neighbor in self.grid and neighbor not in region:
                    region[neighbor] = region[position]
                    pending.append(neighbor)
        for position, cell in self.grid.items():
            cell.reshape(land_type=region[position])

    def raise_hills(self, density, rng, keep_flat=()):
        """Turn about `density` of the flat land (not mountains or water) into hills."""
        if density <= 0:
            return
        for position in sorted(self.grid):
            cell = self.grid[position]
            if cell.spec.land_type is None or cell.spec.elevation != 'FLAT' or position in keep_flat:
                continue
            if rng.random() < density:
                cell.reshape(elevation='HILLS')

    def classify_water(self):
        """Water next to land is coast; water further out is ocean."""
        for cell in self.grid.values():
            if cell.is_water:
                near_land = any(n.spec.land_type is not None for n in cell.get_neighbors(self.grid))
                cell.set_terrain('COAST' if near_land else 'OCEAN')

    def _can_hold_river(self, edge):
        return all(position in self.grid and not self.grid[position].is_water for position in edge)

    def add_river(self, first, second):
        """Place a river on the side shared by two adjacent land tiles on the map."""
        first, second = tuple(first), tuple(second)
        edge = frozenset((first, second))
        if len(edge) != 2 or not self._can_hold_river(edge) or direction_between(first, second) is None:
            raise ValueError('A river must lie between two adjacent land tiles on the map')
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
            candidates = [edge for edge in map_edges if edge not in self.rivers and self._can_hold_river(edge)]
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
                if edge == previous or edge in self.rivers or not self._can_hold_river(edge):
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
