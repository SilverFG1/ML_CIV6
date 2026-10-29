"""Base-game terrain reference metadata, separate from active game rules.

An annotation describes static XML definitions. Actual movement, visibility,
combat, and passability remain the game engine's responsibility.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


SCHEMA = "civ6.terrain_catalog.v1"
BASE_TYPES = ("GRASS", "PLAINS", "DESERT", "TUNDRA", "SNOW", "COAST", "OCEAN")
SOURCE_FILES = ("Base/Assets/Gameplay/Data/Terrains.xml", "Base/Assets/Gameplay/Data/Features.xml")
TERRAIN_FLAGS = ("is_water", "is_shallow_water", "is_hills", "is_mountain", "impassable")


def _int(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def _rows(root, table):
    records = []
    for section in root.findall(table):
        for row in section:
            if row.tag != "Row":
                raise ValueError(f"{table}: only complete Row definitions are supported; {row.tag} is unsupported")
            values = dict(row.attrib)
            for child in row:
                if child.tag in values or len(child):
                    raise ValueError(f"{table}: duplicate or nested field {child.tag}")
                values[child.tag] = (child.text or "").strip()
            records.append(values)
    return records


def _number(row, field, default=0):
    try:
        return int(row[field]) if field in row else default
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid XML integer {field}={row[field]!r}") from exc


def _flag(row, field):
    value = row.get(field, "false").lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError(f"invalid XML boolean {field}={value!r}")
    return value in ("true", "1")


def _base_type(identifier):
    for base in BASE_TYPES:
        if identifier in (f"TERRAIN_{base}", f"TERRAIN_{base}_HILLS", f"TERRAIN_{base}_MOUNTAIN"):
            return base
    return identifier


def _definitions(root, table, id_field, classes=None):
    result = {}
    for row in _rows(root, table):
        identifier = row.get(id_field)
        if not identifier or identifier in result or not row.get("Name"):
            raise ValueError(f"{table}: every Row needs a unique {id_field} and Name")
        entry = {"name_key": row["Name"], "impassable": _flag(row, "Impassable"),
                 "defense_modifier": _number(row, "DefenseModifier"), "yields": {}}
        if table == "Terrains":
            if "MovementCost" not in row:
                raise ValueError(f"{identifier}: MovementCost is required")
            entry.update(terrain_type=identifier, terrain_class=classes.get(identifier), base_type=_base_type(identifier),
                         is_water=_flag(row, "Water"), is_shallow_water=_flag(row, "ShallowWater"),
                         is_hills=_flag(row, "Hills"), is_mountain=_flag(row, "Mountain"),
                         movement_cost=_number(row, "MovementCost"))
        else:
            entry.update(feature_type=identifier, movement_change=_number(row, "MovementChange"))
        result[identifier] = entry
    if not result and table == "Terrains":
        raise ValueError(f"XML has no {table} Row definitions")
    for row in _rows(root, "Terrain_YieldChanges" if table == "Terrains" else "Feature_YieldChanges"):
        identifier, yield_type = row.get(id_field), row.get("YieldType")
        if identifier not in result or not yield_type or yield_type in result[identifier]["yields"]:
            raise ValueError(f"invalid or duplicate yield reference in {table}")
        result[identifier]["yields"][yield_type] = _number(row, "YieldChange")
    return result


class TerrainCatalog:
    """A validated, serializable reference catalog; unknown identifiers stay unknown."""

    def __init__(self, payload):
        self._validate(payload)
        self._data = deepcopy(payload)

    @staticmethod
    def _validate(payload):
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise ValueError(f"terrain catalog schema must be {SCHEMA}")
        if set(payload) != {"schema", "source", "terrains", "features"}:
            raise ValueError("terrain catalog must contain schema, source, terrains, and features only")
        source = payload["source"]
        if not isinstance(source, dict) or source.get("kind") != "base_game_xml" or source.get("rules_scope") != "base_game_reference":
            raise ValueError("terrain catalog source must identify base_game_xml / base_game_reference")
        if set(source) != {"kind", "rules_scope", "files", "sha256"} or source["files"] not in (list(SOURCE_FILES), [SOURCE_FILES[0]]):
            raise ValueError("terrain catalog source must list base Terrains.xml and optional Features.xml")
        if not isinstance(source["sha256"], dict) or set(source["sha256"]) != set(source["files"]):
            raise ValueError("terrain catalog needs a hash for each source file")
        if any(not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
               for digest in source["sha256"].values()):
            raise ValueError("terrain catalog source hashes must be SHA256 hex strings")
        for table, prefix in (("terrains", "TERRAIN_"), ("features", "FEATURE_")):
            entries = payload[table]
            if not isinstance(entries, dict) or (table == "terrains" and not entries):
                raise ValueError(f"catalog {table} must be a dictionary (terrains cannot be empty)")
            terrain = table == "terrains"
            id_field = "terrain_type" if terrain else "feature_type"
            required = {id_field, "name_key", "impassable", "defense_modifier", "yields"}
            required |= {"terrain_class", "base_type", "movement_cost", *TERRAIN_FLAGS} if terrain else {"movement_change"}
            for identifier, entry in entries.items():
                if not isinstance(identifier, str) or not identifier.startswith(prefix) or not isinstance(entry, dict):
                    raise ValueError(f"invalid {table} identifier or definition")
                if set(entry) != required or entry[id_field] != identifier or not isinstance(entry["name_key"], str) or not entry["name_key"]:
                    raise ValueError(f"invalid fields for {identifier}")
                for flag in TERRAIN_FLAGS if terrain else ("impassable",):
                    if not isinstance(entry[flag], bool):
                        raise ValueError(f"{identifier}.{flag} must be boolean")
                _int(entry["defense_modifier"], f"{identifier}.defense_modifier")
                _int(entry["movement_cost" if terrain else "movement_change"], f"{identifier}.movement", 0 if terrain else None)
                if terrain and ((entry["terrain_class"] is not None and
                                 (not isinstance(entry["terrain_class"], str) or not entry["terrain_class"].startswith("TERRAIN_CLASS_")))
                                or not isinstance(entry["base_type"], str) or entry["base_type"] != _base_type(identifier)):
                    raise ValueError(f"invalid class or base_type for {identifier}")
                yields = entry["yields"]
                if not isinstance(yields, dict) or any(not isinstance(key, str) or not key.startswith("YIELD_") for key in yields):
                    raise ValueError(f"invalid yield names for {identifier}")
                for key, amount in yields.items():
                    _int(amount, f"{identifier}.{key}")

    @classmethod
    def default(cls):
        return cls.load(Path(__file__).with_name("data") / "terrain_types.json")

    @classmethod
    def load(cls, path):
        try:
            with Path(path).open(encoding="utf-8") as handle:
                return cls(json.load(handle, object_pairs_hook=_pairs, parse_constant=_reject_constant))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid terrain catalog: {exc}") from exc

    @classmethod
    def from_game_path(cls, path):
        location = Path(path)
        if location.is_file():
            if location.name.lower() != "terrains.xml":
                raise ValueError("direct XML input must be Terrains.xml; sibling Features.xml is optional")
            folder = location.parent
        else:
            candidates = (location / "Base/Assets/Gameplay/Data", location / "Assets/Gameplay/Data", location)
            folder = next((item for item in candidates if (item / "Terrains.xml").is_file()), candidates[0])
        roots, hashes = {}, {}
        for filename in SOURCE_FILES:
            input_path = folder / Path(filename).name
            if filename == SOURCE_FILES[1] and not input_path.exists():
                continue
            try:
                content = input_path.read_bytes()
                roots[filename] = ET.fromstring(content)
            except (ET.ParseError, OSError) as exc:
                raise ValueError(f"cannot import base-game {input_path.name}: {exc}") from exc
            hashes[filename] = hashlib.sha256(content).hexdigest()
            if roots[filename].tag != "GameInfo":
                raise ValueError(f"{input_path.name}: expected GameInfo XML root")
        classes = {}
        for row in _rows(roots[SOURCE_FILES[0]], "TerrainClass_Terrains"):
            terrain, terrain_class = row.get("TerrainType"), row.get("TerrainClassType")
            if not terrain or not terrain_class or terrain in classes:
                raise ValueError("invalid or duplicate TerrainClass_Terrains reference")
            classes[terrain] = terrain_class
        return cls({"schema": SCHEMA, "source": {"kind": "base_game_xml", "rules_scope": "base_game_reference",
                     "files": list(hashes), "sha256": hashes},
                    "terrains": _definitions(roots[SOURCE_FILES[0]], "Terrains", "TerrainType", classes),
                    "features": _definitions(roots[SOURCE_FILES[1]], "Features", "FeatureType") if SOURCE_FILES[1] in roots else {}})

    @property
    def terrains(self):
        return deepcopy(self._data["terrains"])

    @property
    def features(self):
        return deepcopy(self._data["features"])

    def to_dict(self):
        return deepcopy(self._data)

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                             prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(self._data, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def annotate_tile(self, tile):
        if not isinstance(tile, dict):
            raise ValueError("tile must be a dictionary")
        result = deepcopy(tile)
        terrain_type, feature_type = tile.get("terrain"), tile.get("feature")
        terrain = self._data["terrains"].get(terrain_type) if isinstance(terrain_type, str) else None
        no_feature = feature_type is None or (isinstance(feature_type, str) and feature_type.lower() == "none")
        feature = ({"impassable": False, "movement_change": 0, "defense_modifier": 0} if no_feature else
                   self._data["features"].get(feature_type) if isinstance(feature_type, str) else None)
        terrain, feature = terrain or {}, feature or {}
        terrain_impassable, feature_impassable = terrain.get("impassable"), feature.get("impassable")
        static_impassable = (True if terrain_impassable is True or feature_impassable is True else
                             None if terrain_impassable is None or feature_impassable is None else False)
        result["terrain_info"] = {
            "rules_scope": self._data["source"]["rules_scope"],
            "known_terrain": bool(terrain), "known_feature": bool(feature),
            "terrain_type": deepcopy(terrain_type), "feature_type": deepcopy(feature_type),
            "base_type": terrain.get("base_type", deepcopy(terrain_type)),
            **{flag: terrain.get(flag) for flag in TERRAIN_FLAGS if flag != "impassable"},
            "terrain_impassable": terrain_impassable, "feature_impassable": feature_impassable,
            "static_impassable": static_impassable, "base_movement_cost": terrain.get("movement_cost"),
            "feature_movement_change": feature.get("movement_change"),
            "base_defense_modifier": terrain.get("defense_modifier"),
            "feature_defense_modifier": feature.get("defense_modifier"),
        }
        return result
