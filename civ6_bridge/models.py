"""Validated, JSON-friendly contracts for the optional live-game bridge."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math


SCHEMA = "civ6.live.v1"


class BridgeError(RuntimeError):
    """The live game could not be read or an order could not be verified."""


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _coordinate(value, name="center"):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two coordinates")
    return tuple(_integer(item, name) for item in value)


def _json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json(item)
        return
    raise ValueError("snapshot metadata must contain finite JSON values")


@dataclass(frozen=True)
class Order:
    kind: str
    unit_index: int
    x: int | None = None
    y: int | None = None

    def __post_init__(self):
        if self.kind not in ("move", "attack", "fortify", "skip"):
            raise ValueError("order kind must be move, attack, fortify, or skip")
        _integer(self.unit_index, "unit_index")
        if self.kind in ("move", "attack"):
            _integer(self.x, "x")
            _integer(self.y, "y")
        elif self.x is not None or self.y is not None:
            raise ValueError("only move and attack orders accept target coordinates")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ScoutMove:
    """An adjacent, engine-validated move with observed exploration information."""

    x: int
    y: int
    unrevealed_tiles: int
    visible_threats: int = 0

    def __post_init__(self):
        for name in ("x", "y", "unrevealed_tiles", "visible_threats"):
            _integer(getattr(self, name), name)


@dataclass(frozen=True)
class UnitState:
    index: int
    unit_id: int
    name: str
    x: int
    y: int
    hp: int
    moves_remaining: float
    unit_type: str | None = None
    max_moves: float | None = None
    base_combat_strength: int | None = None
    base_sight_range: int | None = None
    scout_moves: tuple[ScoutMove, ...] = ()

    def __post_init__(self):
        for name in ("index", "unit_id", "x", "y", "hp"):
            _integer(getattr(self, name), name)
        if self.hp > 100:
            raise ValueError("hp must be a percentage between 0 and 100")
        if not isinstance(self.name, str):
            raise ValueError("unit name must be a string")
        for name in ("moves_remaining", "max_moves"):
            moves = getattr(self, name)
            if moves is None and name == "max_moves":
                continue
            if isinstance(moves, bool) or not isinstance(moves, (int, float)) or not math.isfinite(moves) or moves < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.unit_type is not None and not isinstance(self.unit_type, str):
            raise ValueError("unit_type must be a string or None")
        for name in ("base_combat_strength", "base_sight_range"):
            if getattr(self, name) is not None:
                _integer(getattr(self, name), name)
        if not isinstance(self.scout_moves, tuple) or not all(isinstance(move, ScoutMove) for move in self.scout_moves):
            raise ValueError("scout_moves must be a tuple of ScoutMove objects")
        if len(self.scout_moves) > 6 or len({(move.x, move.y) for move in self.scout_moves}) != len(self.scout_moves):
            raise ValueError("scout_moves must contain at most six unique destinations")

    @property
    def is_scout(self):
        return self.unit_type == "UNIT_SCOUT"

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or not isinstance(value.get("scout_moves", []), list):
            raise ValueError("serialized unit must be a dictionary with a scout_moves list")
        fields = dict(value)
        fields["scout_moves"] = tuple(ScoutMove(**move) for move in value.get("scout_moves", []))
        return cls(**fields)


@dataclass(frozen=True)
class Snapshot:
    turn: int
    player_id: int
    turn_active: bool
    units: tuple[UnitState, ...]
    overview: dict
    tiles: tuple[dict, ...] = ()
    center: tuple | None = None
    radius: int = 2

    def __post_init__(self):
        _integer(self.turn, "turn")
        _integer(self.player_id, "player_id")
        if not isinstance(self.turn_active, bool):
            raise ValueError("turn_active must be a boolean")
        if not isinstance(self.units, tuple) or not all(isinstance(unit, UnitState) for unit in self.units):
            raise ValueError("units must be a tuple of UnitState objects")
        if len({unit.index for unit in self.units}) != len(self.units) or len({unit.unit_id for unit in self.units}) != len(self.units):
            raise ValueError("unit identities must be unique")
        if not isinstance(self.overview, dict):
            raise ValueError("overview must be a dictionary")
        _json(self.overview)
        if not isinstance(self.tiles, tuple) or not all(isinstance(tile, dict) for tile in self.tiles):
            raise ValueError("tiles must be a tuple of dictionaries")
        for tile in self.tiles:
            _json(tile)
        if self.center is not None:
            object.__setattr__(self, "center", _coordinate(self.center))
        if _integer(self.radius, "radius") > 4:
            raise ValueError("radius must be between 0 and 4")

    def to_dict(self):
        result = asdict(self)
        result["schema"] = SCHEMA
        result["units"] = list(result["units"])
        for unit in result["units"]:
            unit["scout_moves"] = list(unit["scout_moves"])
        result["tiles"] = list(result["tiles"])
        result["center"] = list(self.center) if self.center is not None else None
        return result

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("schema") != SCHEMA:
            raise ValueError(f"snapshot schema must be {SCHEMA}")
        try:
            if not isinstance(value["units"], list) or not isinstance(value.get("tiles", []), list):
                raise ValueError("serialized units and tiles must be lists")
            return cls(value["turn"], value["player_id"], value["turn_active"],
                       tuple(UnitState.from_dict(item) for item in value["units"]), deepcopy(value["overview"]),
                       tuple(deepcopy(value.get("tiles", []))), value.get("center"), value.get("radius", 2))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid snapshot: {exc}") from exc
