"""Fake transports exercise live-order boundaries without Civ VI or civ6-mcp."""

import asyncio
from dataclasses import dataclass, replace
import re
import unittest

from civ6_bridge.backend import COMPLETE, TURN_QUERY, RealCiv6Client
from civ6_bridge.models import BridgeError, Order


@dataclass
class Overview:
    turn: int = 12
    player_id: int = 0
    enabled_victories: set = None
    gold: float = 50.0
    total_land: int = 1000
    rankings: list = None


@dataclass
class Unit:
    unit_index: int = 7
    unit_id: int = 65543
    name: str = "Warrior"
    x: int = 4
    y: int = 5
    health: int = 80
    max_health: int = 100
    moves_remaining: float = 2.0
    unit_type: str = "UNIT_WARRIOR"
    max_moves: float = 2.0
    combat_strength: int = 20


@dataclass
class Tile:
    x: int = 5
    y: int = 5
    visibility: str = "visible"
    yields: tuple = (2, 1, 0, 0, 0, 0)
    units: str = ""
    terrain: str = "TERRAIN_GRASS"
    feature: str | None = None
    is_hills: bool = False


class FakeLua:
    def __init__(self):
        self.overview = Overview(enabled_victories={"VICTORY_CONQUEST"})
        self.units = [Unit()]
        self.tiles = [Tile(), Tile(x=6, visibility="revealed", units="hidden enemy")]

    def build_overview_query(self):
        return 'OVERVIEW_QUERY\nprint("---END---")'

    def build_units_query(self):
        return 'UNITS_QUERY\nprint("---END---")'

    def build_map_area_query(self, x, y, radius):
        return f'MAP_QUERY|{x}|{y}|{radius}\nprint("---END---")'

    def parse_overview_response(self, lines):
        return self.overview

    def parse_units_response(self, lines):
        return self.units

    def parse_map_response(self, lines):
        return self.tiles

    def build_move_unit(self, index, x, y):
        return f'ACTION_MOVE|{index}|{x}|{y}\nprint("---END---")'

    def build_attack_unit(self, index, x, y):
        return f'ACTION_ATTACK|{index}|{x}|{y}\nprint("---END---")'

    def build_skip_unit(self, index):
        return f'ACTION_SKIP|{index}\nprint("---END---")'

    def build_fortify_unit(self, index):
        return f'ACTION_FORTIFY|{index}\nprint("---END---")'


class FakeConnection:
    gamecore_index = 1
    ingame_index = 2

    def __init__(self, lua):
        self.lua = lua
        self.active = True
        self.commands = []
        self.actions = []
        self.closed = False
        self.truncated = False
        self.action_response = None
        self.action_error = None
        self.move_executes = True
        self.scout_response = None
        self.scout_available = True
        self.scout_danger = 0
        self.scout_score = 4

    async def connect(self):
        pass

    async def disconnect(self):
        self.closed = True

    async def execute_read(self, code, timeout):
        return await self.execute(code, True)

    async def execute_write(self, code, timeout):
        return await self.execute(code, False)

    async def execute(self, code, gamecore):
        self.commands.append((gamecore, code))
        if "ACTION_" in code or "OK:SCOUT_MOVE_REQUESTED|" in code:
            self.actions.append((gamecore, code))
            if self.action_error is not None:
                raise self.action_error
            if self.action_response is not None:
                return self.action_response
            if "ACTION_MOVE" in code and self.move_executes:
                x, y = map(int, re.search(r"ACTION_MOVE\|\d+\|(\d+)\|(\d+)", code).groups())
                self.lua.units[0] = replace(self.lua.units[0], x=x, y=y, moves_remaining=1.0)
            if "OK:SCOUT_MOVE_REQUESTED|" in code and self.move_executes:
                x, y = map(int, re.search(r"OK:SCOUT_MOVE_REQUESTED\|(\d+)\|(\d+)", code).groups())
                self.lua.units[0] = replace(self.lua.units[0], x=x, y=y, moves_remaining=1.0)
            if "ACTION_SKIP" in code:
                self.lua.units[0] = replace(self.lua.units[0], moves_remaining=0.0)
            return ["OK:REQUESTED"]
        if self.truncated:
            return []
        if "MLCIV6_STATE|" in code:
            overview = self.lua.overview
            return [f"MLCIV6_STATE|{overview.turn}|{overview.player_id}|{str(self.active).lower()}", COMPLETE]
        if 'print("MLCIV6_SCOUT|' in code:
            if self.scout_response is not None:
                return self.scout_response
            index = int(re.search(r"UnitManager.GetUnit\(me, (\d+)\)", code).group(1))
            unit = next(unit for unit in self.lua.units if unit.unit_index == index)
            rows = [f"MLCIV6_SCOUT|{unit.unit_id}|{index}|2",
                    f"MLCIV6_SCOUT_STATUS|{unit.unit_id}|{'ready' if self.scout_available else 'unavailable'}"]
            if self.scout_available:
                rows.append(f"MLCIV6_SCOUT_MOVE|{unit.unit_id}|{unit.x + 1}|{unit.y}|{self.scout_score}|{self.scout_danger}")
            return rows + [COMPLETE]
        return [COMPLETE]


class BackendTests(unittest.IsolatedAsyncioTestCase):
    def client(self, **kwargs):
        lua = FakeLua()
        connection = FakeConnection(lua)
        return RealCiv6Client(connection=connection, lua=lua, **kwargs), connection, lua

    async def test_snapshot_filters_fog_and_normalizes_upstream_metadata(self):
        client, connection, _ = self.client()
        async with client:
            snapshot = await client.snapshot()
            self.assertEqual(snapshot.center, (4, 5))
            self.assertEqual(snapshot.units[0].unit_id, 65543)
            self.assertEqual(snapshot.units[0].unit_type, "UNIT_WARRIOR")
            self.assertEqual(snapshot.units[0].max_moves, 2.0)
            self.assertEqual(snapshot.units[0].base_combat_strength, 20)
            self.assertEqual(snapshot.overview["gold"], 50.0)
            self.assertNotIn("total_land", snapshot.overview)
            self.assertNotIn("rankings", snapshot.overview)
            self.assertEqual(snapshot.tiles[0]["yields"], [2, 1, 0, 0, 0, 0])
            self.assertEqual(len(snapshot.tiles), 1)
            self.assertEqual(snapshot.tiles[0]["terrain_info"]["base_type"], "GRASS")
            self.assertFalse(any("ACTION_" in code for _, code in connection.commands))
        self.assertTrue(connection.closed)

    async def test_visible_tiles_preserve_terrain_variants_features_and_unknown_mods(self):
        client, _, lua = self.client()
        lua.tiles = [Tile(terrain="TERRAIN_PLAINS_HILLS", feature="FEATURE_FOREST", is_hills=True),
                     Tile(x=6, terrain="TERRAIN_SNOW_MOUNTAIN"),
                     Tile(x=7, terrain="TERRAIN_OCEAN", feature="FEATURE_ICE"),
                     Tile(x=8, terrain="TERRAIN_MODDED", feature="FEATURE_MODDED"),
                     Tile(x=9, visibility="revealed", terrain="TERRAIN_DESERT")]
        async with client:
            snapshot = await client.snapshot()
        self.assertEqual(len(snapshot.tiles), 4)
        hills, mountain, ocean, modded = snapshot.tiles
        self.assertEqual(hills["terrain"], "TERRAIN_PLAINS_HILLS")
        self.assertEqual(hills["feature"], "FEATURE_FOREST")
        self.assertTrue(hills["terrain_info"]["is_hills"])
        self.assertEqual(hills["terrain_info"]["base_movement_cost"], 2)
        self.assertEqual(hills["terrain_info"]["feature_movement_change"], 1)
        self.assertTrue(mountain["terrain_info"]["static_impassable"])
        self.assertTrue(ocean["terrain_info"]["is_water"])
        self.assertTrue(ocean["terrain_info"]["static_impassable"])
        self.assertFalse(modded["terrain_info"]["known_terrain"])
        self.assertEqual(modded["terrain"], "TERRAIN_MODDED")

    async def test_missing_game_state_closes_connection(self):
        client, connection, _ = self.client()
        connection.gamecore_index = None
        with self.assertRaisesRegex(BridgeError, "Load a Civilization VI save"):
            async with client:
                pass
        self.assertTrue(connection.closed)

    async def test_partial_read_is_not_an_empty_world(self):
        client, connection, _ = self.client()
        connection.truncated = True
        async with client:
            with self.assertRaisesRegex(BridgeError, "Incomplete game read"):
                await client.snapshot()

    async def test_stale_or_inactive_unit_is_rejected_before_dispatch(self):
        for changed in ("identity", "position", "turn", "player", "inactive"):
            client, connection, lua = self.client()
            async with client:
                expected = await client.snapshot()
                if changed == "identity":
                    lua.units[0] = replace(lua.units[0], unit_id=131079)
                elif changed == "position":
                    lua.units[0] = replace(lua.units[0], x=3)
                elif changed == "turn":
                    lua.overview.turn += 1
                elif changed == "player":
                    lua.overview.player_id += 1
                else:
                    connection.active = False
                with self.subTest(changed=changed), self.assertRaises(BridgeError):
                    await client.perform(Order("move", 7, 5, 5), expected)
                self.assertEqual(connection.actions, [])

    async def test_move_guard_and_readback_protect_the_execution_boundary(self):
        client, connection, _ = self.client()
        async with client:
            expected = await client.snapshot()
            result = await client.perform(Order("move", 7, 5, 5), expected)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["observed"]["units"][0]["x"], 5)
        self.assertEqual(len(connection.actions), 1)
        gamecore, code = connection.actions[0]
        self.assertFalse(gamecore)
        for guard in ("unit:GetID() ~= 65543", "IsTurnActive()", "IsTurnActiveComplete()",
                      "Game.GetCurrentGameTurn() ~= 12", "me ~= 0", "ML_CIV6_LAST_ORDER_TOKEN",
                      "IsVisible(5, 5)", "Map.GetAdjacentPlot", "FOREIGN_UNIT", "CITY_TARGET",
                      "CanStartOperation(unit, UnitOperationTypes.MOVE_TO, nil, params)"):
            self.assertIn(guard, code)
        self.assertLess(code.index("ML_CIV6_LAST_ORDER_TOKEN ="), code.index("ACTION_MOVE"))
        # Civ VI's ActionPanel waits while IsTurnActiveComplete is false:
        # "complete" refers to activation, not the player having ended a turn.
        self.assertNotIn("not player:IsTurnActiveComplete()", TURN_QUERY)
        self.assertNotIn("not player:IsTurnActiveComplete()", code)

    async def test_acknowledgement_alone_never_verifies_move_or_combat(self):
        for kind in ("move", "attack", "fortify"):
            client, connection, _ = self.client()
            connection.move_executes = False
            async with client:
                expected = await client.snapshot()
                order = Order(kind, 7, 5, 5) if kind != "fortify" else Order(kind, 7)
                result = await client.perform(order, expected)
            with self.subTest(kind=kind):
                self.assertEqual(result["status"], "requested")
                self.assertIn("unconfirmed", result["message"])
                if kind == "attack":
                    code = connection.actions[0][1]
                    self.assertIn("REMOTE_MELEE_TARGET", code)
                    self.assertIn("UnitOperationTypes.RANGE_ATTACK, nil, params", code)
                    self.assertIn("UnitOperationTypes.AIR_ATTACK, nil, params", code)
                    self.assertLess(code.index("REMOTE_MELEE_TARGET"), code.index("ACTION_ATTACK"))

    async def test_skip_uses_gamecore_and_verifies_remaining_moves(self):
        client, connection, _ = self.client()
        async with client:
            expected = await client.snapshot()
            result = await client.perform(Order("skip", 7), expected)
        self.assertEqual(result["status"], "verified")
        self.assertTrue(connection.actions[0][0])
        self.assertIn("FindID(7)", connection.actions[0][1])

    async def test_mutation_timeout_is_unknown_and_never_retried(self):
        client, connection, _ = self.client()
        connection.action_error = TimeoutError("socket timed out")
        async with client:
            expected = await client.snapshot()
            result = await client.perform(Order("skip", 7), expected)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("Do not retry", result["message"])
        self.assertEqual(len(connection.actions), 1)

    async def test_missing_or_duplicate_ack_is_unknown_and_engine_errors_reject(self):
        for response, status in [([], "unknown"), (["ERR:BRIDGE_DUPLICATE"], "unknown"),
                                 (["ERR:NO_MOVES"], "rejected"), (["ERR:STOPPED_SHORT"], "unknown")]:
            client, connection, _ = self.client()
            connection.action_response = response
            async with client:
                expected = await client.snapshot()
                result = await client.perform(Order("skip", 7), expected)
            with self.subTest(response=response):
                self.assertEqual(result["status"], status)
                self.assertEqual(len(connection.actions), 1)

    async def test_connect_handshake_is_bounded(self):
        client, connection, _ = self.client(timeout=0.01)

        async def hang():
            await asyncio.Event().wait()

        connection.connect = hang
        with self.assertRaises(BridgeError):
            async with client:
                pass
        self.assertTrue(connection.closed)

    async def test_scout_snapshot_preserves_native_stats_and_queries_candidates(self):
        client, connection, lua = self.client()
        lua.units[0] = replace(lua.units[0], name="Renamed explorer", unit_type="UNIT_SCOUT",
                               max_moves=4.0, moves_remaining=3.0, combat_strength=13)
        async with client:
            snapshot = await client.snapshot()
        unit = snapshot.units[0]
        self.assertTrue(unit.is_scout)
        self.assertEqual(unit.unit_type, "UNIT_SCOUT")
        self.assertEqual(unit.max_moves, 4.0)
        self.assertEqual(unit.base_combat_strength, 13)
        self.assertEqual(unit.base_sight_range, 2)
        self.assertEqual((unit.scout_moves[0].x, unit.scout_moves[0].y), (5, 5))
        self.assertEqual(unit.scout_moves[0].unrevealed_tiles, 4)
        scout_queries = [(core, code) for core, code in connection.commands if 'print("MLCIV6_SCOUT|' in code]
        self.assertEqual(len(scout_queries), 1)
        self.assertFalse(scout_queries[0][0])
        self.assertIn("unit:GetID() ~= 65543", scout_queries[0][1])
        self.assertIn("unit:GetX() ~= 4", scout_queries[0][1])
        self.assertIn("unit:GetMovesRemaining() ~= 3.0", scout_queries[0][1])

    async def test_localized_name_does_not_create_scout_candidates(self):
        client, connection, lua = self.client()
        lua.units[0] = replace(lua.units[0], name="Scout")
        async with client:
            snapshot = await client.snapshot()
        self.assertFalse(snapshot.units[0].is_scout)
        self.assertFalse(any('print("MLCIV6_SCOUT|' in code for _, code in connection.commands))

    async def test_unavailable_scout_api_has_no_fallback_moves(self):
        client, connection, lua = self.client()
        lua.units[0] = replace(lua.units[0], unit_type="UNIT_SCOUT")
        connection.scout_available = False
        async with client:
            with self.assertRaisesRegex(BridgeError, "APIs are unavailable"):
                await client.snapshot()
        self.assertEqual(connection.actions, [])

    async def test_scout_safe_move_is_revalidated_in_same_lua_dispatch(self):
        client, connection, lua = self.client()
        lua.units[0] = replace(lua.units[0], unit_type="UNIT_SCOUT")
        async with client:
            expected = await client.snapshot()
            result = await client.perform(Order("move", 7, 5, 5), expected)
        self.assertEqual(result["status"], "verified")
        code = connection.actions[0][1]
        for check in ("SCOUT_TARGET_CHANGED", "GetReachableMovement", "HasMovedIntoZOC",
                      "IsUnitVisible", "row[4] == 0", "plot:IsWater()", "plot:IsImpassable()"):
            self.assertIn(check, code)
            self.assertLess(code.index(check), code.index("OK:SCOUT_MOVE_REQUESTED|"))
        self.assertNotIn("UnitOperationMoveModifiers.ATTACK", code)

    async def test_scout_threat_or_changed_candidate_prevents_dispatch(self):
        for changed in ("danger", "score"):
            client, connection, lua = self.client()
            lua.units[0] = replace(lua.units[0], unit_type="UNIT_SCOUT")
            async with client:
                expected = await client.snapshot()
                if changed == "danger":
                    connection.scout_danger = 1
                else:
                    connection.scout_score += 1
                with self.assertRaisesRegex(BridgeError, "selected unit changed"):
                    await client.perform(Order("move", 7, 5, 5), expected)
            self.assertEqual(connection.actions, [])

    async def test_malformed_or_truncated_scout_query_is_not_empty_observation(self):
        for rows in ([COMPLETE], ["MLCIV6_SCOUT|65543|7|2", COMPLETE],
                     ["MLCIV6_SCOUT|65543|7|2", "MLCIV6_SCOUT_STATUS|65543|ready"],
                     ["MLCIV6_SCOUT|999|7|2", "MLCIV6_SCOUT_STATUS|999|ready", COMPLETE]):
            client, connection, lua = self.client()
            lua.units[0] = replace(lua.units[0], unit_type="UNIT_SCOUT")
            connection.scout_response = rows
            async with client:
                with self.subTest(rows=rows), self.assertRaises(BridgeError):
                    await client.snapshot()


if __name__ == "__main__":
    unittest.main()
