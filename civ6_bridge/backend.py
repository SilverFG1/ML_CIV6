"""Optional direct FireTuner client using civ6-mcp's connection and Lua builders.

No MCP server or language model is started. Reads and orders are bounded, and
commands are never replayed automatically after a transport failure.
"""

import asyncio
from dataclasses import asdict, is_dataclass, replace
import math
from uuid import uuid4

from .models import BridgeError, Order, Snapshot, UnitState, _coordinate, _integer
from .scouts import build_scout_move, build_scout_move_guard, build_scout_query, parse_scout_response
from .terrain import TerrainCatalog


SENTINEL = "---END---"  # civ6-mcp v1.1.11's pinned protocol terminator
COMPLETE = "MLCIV6_COMPLETE"
LOCAL_OVERVIEW_FIELDS = frozenset((
    "turn", "player_id", "civ_name", "leader_name", "gold", "gold_per_turn",
    "science_yield", "culture_yield", "faith", "current_research", "current_civic",
    "num_cities", "num_units", "score", "diplomatic_favor", "favor_per_turn",
    "explored_land", "our_religion", "total_population", "era_name", "era_score",
    "era_dark_threshold", "era_golden_threshold", "gold_income", "total_maintenance",
    "unit_maintenance", "unit_breakdown",
))
TURN_QUERY = '''
local me = Game.GetLocalPlayer()
local player = Players[me]
if not player then print("ERR:NO_LOCAL_PLAYER"); print("---END---"); return end
local ok, active = pcall(function()
    return player:IsHuman() and player:IsAlive() and player:IsTurnActive()
        and player:IsTurnActiveComplete()
end)
if not ok then print("ERR:TURN_GUARD_UNAVAILABLE"); print("---END---"); return end
print("MLCIV6_STATE|" .. Game.GetCurrentGameTurn() .. "|" .. me .. "|" .. tostring(active))
print("---END---")
'''


def _plain(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_plain(item) for item in (sorted(value) if isinstance(value, set) else value)]
    return value


def _load_api():
    try:
        from civ_mcp.connection import GameConnection
        from civ_mcp import lua
    except ImportError as exc:
        raise BridgeError("Live mode requires the optional civ6-mcp dependency; see requirements-civ6.txt.") from exc

    class SingleSendConnection(GameConnection):
        async def _execute_and_collect(self, state_index, lua_code, timeout):
            # Upstream retries after socket errors. Replaying a game order is
            # unsafe, so call the pinned transport exactly once for every query.
            if not self.is_connected:
                raise ConnectionError("FireTuner disconnected before dispatch")
            async with self._lock:
                return await self._locked_execute(state_index, lua_code, timeout)

    return SingleSendConnection(), lua


class RealCiv6Client:
    def __init__(self, timeout=10, *, connection=None, lua=None, terrain_catalog=None):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number")
        if (connection is None) != (lua is None):
            raise ValueError("connection and lua must be provided together")
        if terrain_catalog is not None and not isinstance(terrain_catalog, TerrainCatalog):
            raise ValueError("terrain_catalog must be a TerrainCatalog")
        self.timeout, self.connection, self.lua = timeout, connection, lua
        self.terrain_catalog = terrain_catalog if terrain_catalog is not None else TerrainCatalog.default()
        self._orders = asyncio.Lock()
        self._opened = False

    async def __aenter__(self):
        if self.connection is None:
            self.connection, self.lua = _load_api()
        try:
            async with asyncio.timeout(self.timeout):
                await self.connection.connect()
                if self.connection.gamecore_index is None or self.connection.ingame_index is None:
                    raise BridgeError("Load a Civilization VI save first; FireTuner needs GameCore_Tuner and InGame states.")
            self._opened = True
            return self
        except Exception as exc:
            await self._close()
            if isinstance(exc, BridgeError):
                raise
            raise BridgeError(f"Cannot connect to local Civilization VI FireTuner: {exc}") from exc

    async def _close(self):
        self._opened = False
        try:
            async with asyncio.timeout(self.timeout):
                await self.connection.disconnect()
        except Exception:
            pass  # A failed socket close must not hide the original result.

    async def __aexit__(self, *_):
        await self._close()

    async def _execute(self, code, *, gamecore=False):
        if not self._opened:
            raise BridgeError("Use RealCiv6Client inside an async context manager")
        method = self.connection.execute_read if gamecore else self.connection.execute_write
        return await method(code, timeout=self.timeout)

    async def _query(self, code, *, gamecore=False):
        terminator = f'print("{SENTINEL}")'
        if terminator not in code:
            raise BridgeError("Unsupported civ6-mcp query protocol; install the pinned dependency")
        lines = await self._execute(code.replace(terminator, f'print("{COMPLETE}"); {terminator}'), gamecore=gamecore)
        errors = [line for line in lines if line.startswith("ERR:")]
        if errors or COMPLETE not in lines:
            raise BridgeError("Incomplete game read: " + ("; ".join(errors) or "missing completion marker"))
        return [line for line in lines if line != COMPLETE]

    async def _turn(self):
        lines = await self._query(TURN_QUERY)
        try:
            rows = [line.split("|") for line in lines if line.startswith("MLCIV6_STATE|")]
            if len(rows) != 1 or len(rows[0]) != 4 or rows[0][3] not in ("true", "false"):
                raise ValueError("missing turn state")
            return int(float(rows[0][1])), int(float(rows[0][2])), rows[0][3] == "true"
        except (ValueError, IndexError) as exc:
            raise BridgeError("Cannot verify local player and turn activity") from exc

    async def snapshot(self, center=None, radius=2):
        if center is not None:
            center = _coordinate(center)
        if _integer(radius, "radius") > 4:
            raise ValueError("radius must be between 0 and 4")
        try:
            async with asyncio.timeout(self.timeout):
                before = await self._turn()
                raw_overview = _plain(self.lua.parse_overview_response(await self._query(self.lua.build_overview_query())))
                # The upstream overview also includes world statistics and other
                # players' rankings. Keep only information about our own player.
                overview = {key: value for key, value in raw_overview.items() if key in LOCAL_OVERVIEW_FIELDS}
                raw_units = self.lua.parse_units_response(await self._query(self.lua.build_units_query()))
                units = tuple(UnitState(u.unit_index, u.unit_id, u.name, u.x, u.y,
                                        round(100 * u.health / u.max_health), u.moves_remaining,
                                        unit_type=getattr(u, "unit_type", None),
                                        max_moves=getattr(u, "max_moves", None),
                                        base_combat_strength=getattr(u, "combat_strength", None))
                              for u in raw_units)
                enriched = []
                for unit in units:
                    if unit.unit_type == "UNIT_SCOUT":
                        sight, moves = parse_scout_response(await self._query(build_scout_query(unit)), unit)
                        unit = replace(unit, base_sight_range=sight, scout_moves=moves)
                    enriched.append(unit)
                units = tuple(enriched)
                selected_center = center or ((units[0].x, units[0].y) if units else None)
                tiles = ()
                if selected_center is not None:
                    raw_tiles = self.lua.parse_map_response(await self._query(
                        self.lua.build_map_area_query(*selected_center, radius), gamecore=True))
                    # Upstream revealed tiles contain live ownership metadata.
                    # Export currently visible tiles only, including their units.
                    tiles = tuple(self.terrain_catalog.annotate_tile(_plain(tile))
                                  for tile in raw_tiles if tile.visibility == "visible")
                after = await self._turn()
                if before != after or (overview["turn"], overview["player_id"]) != before[:2]:
                    raise BridgeError("Player or turn changed while reading; take another snapshot")
                return Snapshot(*after, units, overview, tiles, selected_center, radius)
        except BridgeError:
            raise
        except Exception as exc:
            raise BridgeError(f"Cannot read Civilization VI state: {exc}") from exc

    def _guarded_order(self, order, expected, unit, token):
        # All interpolated fields are validated integers or a generated hex token.
        lookup = (f"Players[me]:GetUnits():FindID({unit.index})" if order.kind == "skip"
                  else f"UnitManager.GetUnit(me, {unit.index})")
        guard = f'''
local function reject(reason) print("ERR:BRIDGE_" .. reason); print("{SENTINEL}") end
if _G.ML_CIV6_LAST_ORDER_TOKEN == "{token}" then reject("DUPLICATE"); return end
_G.ML_CIV6_LAST_ORDER_TOKEN = "{token}"
local me = Game.GetLocalPlayer()
if me ~= {expected.player_id} or Game.GetCurrentGameTurn() ~= {expected.turn} then reject("STALE_TURN"); return end
local player = Players[me]
local ok, active = pcall(function()
    return player:IsHuman() and player:IsAlive() and player:IsTurnActive() and player:IsTurnActiveComplete()
end)
if not ok or not active then reject("INACTIVE_TURN"); return end
local unit = {lookup}
if not unit or unit:GetID() ~= {unit.unit_id} then reject("STALE_UNIT"); return end
if unit:GetX() ~= {unit.x} or unit:GetY() ~= {unit.y} or unit:GetMovesRemaining() ~= {unit.moves_remaining} then reject("STALE_STATE"); return end
'''
        if order.kind in ("move", "attack"):
            guard += f'''
local visibility = PlayersVisibility[me]
if not visibility or not visibility:IsVisible({order.x}, {order.y}) then reject("TARGET_NOT_VISIBLE"); return end
local adjacent = false
for direction = 0, 5 do
    local plot = Map.GetAdjacentPlot(unit:GetX(), unit:GetY(), direction)
    if plot and plot:GetX() == {order.x} and plot:GetY() == {order.y} then adjacent = true end
end
'''
        if order.kind == "move":
            foreign = ("other:GetOwner() ~= me and visibility:IsUnitVisible(other)"
                       if unit.unit_type == "UNIT_SCOUT" else "other:GetOwner() ~= me")
            guard += f'''
if not adjacent then reject("TARGET_NOT_ADJACENT"); return end
local target = Map.GetPlot({order.x}, {order.y})
if not target or target:IsCity() then reject("CITY_TARGET"); return end
local occupants = Map.GetUnitsAt({order.x}, {order.y})
if occupants then for other in occupants:Units() do
    if {foreign} then reject("FOREIGN_UNIT"); return end
end end
local params = {{}}
params[UnitOperationTypes.PARAM_X] = {order.x}
params[UnitOperationTypes.PARAM_Y] = {order.y}
if not UnitManager.CanStartOperation(unit, UnitOperationTypes.MOVE_TO, nil, params) then reject("ILLEGAL_MOVE"); return end
'''
            if unit.unit_type == "UNIT_SCOUT":
                guard += build_scout_move_guard(order)
        elif order.kind == "attack":
            guard += f'''
-- A distant melee MOVE_TO could queue travel for future turns. Only allow a
-- nonadjacent target when the engine accepts a direct ranged or air strike.
if not adjacent then
    local params = {{}}
    params[UnitOperationTypes.PARAM_X] = {order.x}
    params[UnitOperationTypes.PARAM_Y] = {order.y}
    local direct = UnitManager.CanStartOperation(unit, UnitOperationTypes.RANGE_ATTACK, nil, params)
    if not direct and UnitOperationTypes.AIR_ATTACK then
        direct = UnitManager.CanStartOperation(unit, UnitOperationTypes.AIR_ATTACK, nil, params)
    end
    if not direct then reject("REMOTE_MELEE_TARGET"); return end
end
'''
        elif order.kind == "skip":
            # FinishMoves is the engine's GameCore operation; upstream provides
            # no separate CanStartOperation check for this unit-local command.
            guard += '''
if unit:GetMovesRemaining() <= 0 or not UnitManager.FinishMoves then reject("CANNOT_SKIP"); return end
'''
        builder = getattr(self.lua, "build_" + order.kind + "_unit")
        args = (unit.index, order.x, order.y) if order.kind in ("move", "attack") else (unit.index,)
        action = build_scout_move(order) if unit.unit_type == "UNIT_SCOUT" and order.kind == "move" else builder(*args)
        return "do\n" + guard + "\ndo\n" + action + "\nend\nend"

    async def perform(self, order, expected_snapshot):
        if not isinstance(order, Order) or not isinstance(expected_snapshot, Snapshot):
            raise ValueError("perform requires an Order and its expected Snapshot")
        sent = False
        result = {"status": "unknown", "message": "Order outcome is unknown", "order": order.to_dict()}
        try:
            async with asyncio.timeout(self.timeout), self._orders:
                current = await self.snapshot(expected_snapshot.center, expected_snapshot.radius)
                if not current.turn_active or (current.turn, current.player_id) != (expected_snapshot.turn, expected_snapshot.player_id):
                    raise BridgeError("Order rejected: player/turn changed or local turn is inactive")
                unit = next((u for u in current.units if u.index == order.unit_index), None)
                reviewed = next((u for u in expected_snapshot.units if u.index == order.unit_index), None)
                if unit is None or reviewed is None or unit != reviewed:
                    raise BridgeError("Order rejected: selected unit changed; take another snapshot")
                if unit.unit_type == "UNIT_SCOUT" and order.kind == "move" and not any(
                        (move.x, move.y) == (order.x, order.y) and move.visible_threats == 0
                        for move in unit.scout_moves):
                    raise BridgeError("Order rejected: scout destination is not a safe current-turn candidate")
                code = self._guarded_order(order, current, unit, uuid4().hex)
                sent = True  # From this point, timeout or socket errors are indeterminate.
                lines = await self._execute(code, gamecore=order.kind == "skip")
                result["acknowledgement"] = lines
                if any(line.startswith("ERR:BRIDGE_DUPLICATE") for line in lines):
                    result["message"] = "Duplicate dispatch suppressed; previous outcome is unknown. Do not retry automatically."
                    return result
                if not any(line.startswith("OK:") for line in lines):
                    errors = [line for line in lines if line.startswith("ERR:")]
                    # STOPPED_SHORT may follow a partial melee move, so it is not a clean rejection.
                    result["status"] = "rejected" if errors and not any("STOPPED_SHORT" in line for line in errors) else "unknown"
                    result["message"] = "; ".join(errors) or "No acknowledgement; do not retry automatically."
                    return result
                observed = await self.snapshot(current.center, current.radius)
                result["observed"] = observed.to_dict()
                after = next((u for u in observed.units if u.unit_id == unit.unit_id), None)
                same_turn = (observed.turn, observed.player_id) == (current.turn, current.player_id)
                verified = same_turn and after is not None and (
                    (order.kind == "move" and (after.x, after.y) == (order.x, order.y)) or
                    (order.kind == "skip" and after.moves_remaining == 0))
                result.update(status="verified" if verified else "requested", message=(
                    "Order effect confirmed by a fresh game read." if verified else
                    "Game acknowledged the request; its intended effect is unconfirmed (including any combat damage)."))
                return result
        except Exception as exc:
            if sent:
                result["message"] = f"Order may have executed, but verification failed: {exc}. Do not retry automatically."
                return result
            if isinstance(exc, BridgeError):
                raise
            raise BridgeError(f"Order was not dispatched: {exc}") from exc
