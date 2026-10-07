"""Hex geometry and terrain, independent of the rendering library."""

from dataclasses import dataclass

from constants import DIRECTIONS, movement_delta

TERRAIN_CODES = {'GRASSLAND': 0, 'FOREST': 1, 'MOUNTAIN': 2, 'EDGE': 2}


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
        x, y = self.index
        return [grid[(x + dx, y + dy)] for dx, dy in
                (movement_delta(direction, y) for direction in DIRECTIONS[:-1])
                if (x + dx, y + dy) in grid]


class HexMap:
    def __init__(self, num_rows, num_columns, cell_size=None, cell_offset=None):
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.grid = {(x, y): HexCell((x, y)) for y in range(num_rows)
                     for x in range(num_columns)}

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
