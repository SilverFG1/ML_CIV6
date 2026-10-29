"""Bounded, player-visible observations for one-step native scout movement."""

import re

from .models import BridgeError, ScoutMove, UnitState


SCOUT_HEADER = "MLCIV6_SCOUT"
SCOUT_STATUS = "MLCIV6_SCOUT_STATUS"
SCOUT_MOVE = "MLCIV6_SCOUT_MOVE"

# This is shared by observation and dispatch so a candidate is checked again
# immediately before its ordinary MOVE_TO request. No native automation starts.
_CANDIDATE_HELPERS = '''
local function scoutBaseSight(unit)
    local info = GameInfo.Units[unit:GetType()]
    local sight = info and tonumber(info.BaseSightRange)
    if not sight or sight < 0 or sight ~= math.floor(sight) then return nil end
    return sight
end
local function scoutCandidateRows(unit, me)
    local info = GameInfo.Units[unit:GetType()]
    if not info or info.UnitType ~= "UNIT_SCOUT" then error("NOT_SCOUT") end
    local vis = PlayersVisibility[me]
    if not vis or not vis.IsUnitVisible or not UnitManager.GetReachableMovement
        or not UnitManager.CanStartOperation or not Map.GetAdjacentPlot then
        error("CANDIDATE_API_UNAVAILABLE")
    end
    local sight = scoutBaseSight(unit)
    if sight == nil then error("SIGHT_UNAVAILABLE") end
    local rows = {}
    if unit:GetMovesRemaining() <= 0 or unit:HasMovedIntoZOC() then return rows end
    local movement = UnitManager.GetReachableMovement(unit)
    if type(movement) ~= "table" then error("REACHABILITY_UNAVAILABLE") end
    local reachable = {}
    for _, index in ipairs(movement) do reachable[index] = true end
    local function area(origin, radius)
        local visited = {[origin:GetIndex()] = true}
        local all = {origin}
        local frontier = {origin}
        for depth = 1, radius do
            local nextFrontier = {}
            for _, plot in ipairs(frontier) do
                for direction = 0, 5 do
                    local adjacent = Map.GetAdjacentPlot(plot:GetX(), plot:GetY(), direction)
                    if adjacent and not visited[adjacent:GetIndex()] then
                        visited[adjacent:GetIndex()] = true
                        table.insert(all, adjacent)
                        table.insert(nextFrontier, adjacent)
                    end
                end
            end
            frontier = nextFrontier
        end
        return all
    end
    local origin = Map.GetPlot(unit:GetX(), unit:GetY())
    if not origin then error("SCOUT_POSITION_UNAVAILABLE") end
    local diplomacy = Players[me]:GetDiplomacy()
    local threats = {}
    -- Radius two contains every plot within one tile of an adjacent candidate.
    for _, plot in ipairs(area(origin, 2)) do
        if vis:IsVisible(plot:GetIndex()) then
            local occupants = Map.GetUnitsAt(plot:GetX(), plot:GetY())
            if occupants then for other in occupants:Units() do
                if vis:IsUnitVisible(other) then
                local owner = other:GetOwner()
                if owner ~= me
                    and (Players[owner]:IsBarbarian() or diplomacy:IsAtWarWith(owner)) then
                    local otherInfo = GameInfo.Units[other:GetType()]
                    if otherInfo and ((otherInfo.Combat or 0) > 0 or (otherInfo.RangedCombat or 0) > 0) then
                        table.insert(threats, plot)
                    end
                end
                end
            end end
        end
    end
    local seen = {}
    for direction = 0, 5 do
        local plot = Map.GetAdjacentPlot(unit:GetX(), unit:GetY(), direction)
        if plot and not seen[plot:GetIndex()] and vis:IsVisible(plot:GetIndex()) then
            seen[plot:GetIndex()] = true
            if reachable[plot:GetIndex()] and not plot:IsWater()
                and not plot:IsImpassable() and not plot:IsCity() then
                local blocked = false
                local occupants = Map.GetUnitsAt(plot:GetX(), plot:GetY())
                if occupants then for other in occupants:Units() do
                    if other:GetOwner() == me or vis:IsUnitVisible(other) then
                    local otherInfo = GameInfo.Units[other:GetType()]
                    if other:GetOwner() ~= me or not otherInfo
                        or otherInfo.FormationClass == info.FormationClass then blocked = true end
                    end
                end end
                local params = {}
                params[UnitOperationTypes.PARAM_X] = plot:GetX()
                params[UnitOperationTypes.PARAM_Y] = plot:GetY()
                if not blocked and UnitManager.CanStartOperation(unit, UnitOperationTypes.MOVE_TO, nil, params) then
                    local unexplored = 0
                    -- Only visibility masks of these geometric neighbors are read.
                    -- Terrain, resources and occupants in fog are never inspected.
                    for _, nearby in ipairs(area(plot, math.min(sight, 4))) do
                        if not vis:IsRevealed(nearby:GetIndex()) then unexplored = unexplored + 1 end
                    end
                    local danger = 0
                    local nearbyThreatPlots = {}
                    for _, nearby in ipairs(area(plot, 1)) do
                        nearbyThreatPlots[nearby:GetIndex()] = true
                    end
                    for _, threat in ipairs(threats) do
                        if nearbyThreatPlots[threat:GetIndex()] then
                            danger = danger + 1
                        end
                    end
                    table.insert(rows, {plot:GetX(), plot:GetY(), unexplored, danger})
                end
            end
        end
    end
    return rows
end
'''


def build_scout_query(unit):
    """Read six current-turn candidates, or explicitly report unavailable APIs."""
    if not isinstance(unit, UnitState) or unit.unit_type != "UNIT_SCOUT":
        raise ValueError("scout query requires an observed native UNIT_SCOUT")
    return _CANDIDATE_HELPERS + f'''
local me = Game.GetLocalPlayer()
local unit = UnitManager.GetUnit(me, {unit.index})
if not unit or unit:GetID() ~= {unit.unit_id} or unit:GetX() ~= {unit.x}
    or unit:GetY() ~= {unit.y} or unit:GetMovesRemaining() ~= {unit.moves_remaining} then
    print("ERR:STALE_SCOUT"); print("---END---"); return
end
local info = GameInfo.Units[unit:GetType()]
if not info or info.UnitType ~= "UNIT_SCOUT" then
    print("ERR:NOT_SCOUT"); print("---END---"); return
end
local sight = scoutBaseSight(unit)
print("{SCOUT_HEADER}|{unit.unit_id}|{unit.index}|" .. (sight and tostring(sight) or "none"))
local player = Players[me]
if not player or not player:IsTurnActive() or not player:IsTurnActiveComplete() then
    print("{SCOUT_STATUS}|{unit.unit_id}|ready")
    print("---END---"); return
end
local ok, rows = pcall(function() return scoutCandidateRows(unit, me) end)
if not ok then
    print("{SCOUT_STATUS}|{unit.unit_id}|unavailable")
else
    print("{SCOUT_STATUS}|{unit.unit_id}|ready")
    for _, row in ipairs(rows) do
        print("{SCOUT_MOVE}|{unit.unit_id}|" .. row[1] .. "|" .. row[2] .. "|" .. row[3] .. "|" .. row[4])
    end
end
print("---END---")
'''


def _natural(text):
    if not isinstance(text, str) or not re.fullmatch(r"[0-9]+(?:\.0+)?", text):
        raise ValueError("expected a nonnegative integer")
    return int(text.split(".", 1)[0])


def parse_scout_response(lines, unit):
    """Validate identity, availability and candidate bounds before exposing them."""
    try:
        headers = [line.split("|") for line in lines if line.startswith(SCOUT_HEADER + "|")]
        statuses = [line.split("|") for line in lines if line.startswith(SCOUT_STATUS + "|")]
        if len(headers) != 1 or len(headers[0]) != 4 or len(statuses) != 1 or len(statuses[0]) != 3:
            raise ValueError("missing or duplicate scout header/status")
        header, status = headers[0], statuses[0]
        if (_natural(header[1]), _natural(header[2])) != (unit.unit_id, unit.index):
            raise ValueError("scout identity changed")
        if _natural(status[1]) != unit.unit_id or status[2] not in ("ready", "unavailable"):
            raise ValueError("invalid scout availability")
        if status[2] == "unavailable":
            raise ValueError("scout movement-query APIs are unavailable; no exploration order was planned")
        sight = None if header[3] == "none" else _natural(header[3])
        moves = []
        for line in lines:
            if not line.startswith(SCOUT_MOVE + "|"):
                continue
            fields = line.split("|")
            if len(fields) != 6 or _natural(fields[1]) != unit.unit_id:
                raise ValueError("invalid scout candidate identity")
            moves.append(ScoutMove(*(_natural(field) for field in fields[2:])))
        if len(moves) > 6 or len({(move.x, move.y) for move in moves}) != len(moves):
            raise ValueError("duplicate or excessive scout candidates")
        if status[2] == "ready" and sight is None:
            raise ValueError("ready scout API omitted base sight")
        # A radius-four hex area contains at most 61 plots, including its center.
        if any(move.unrevealed_tiles > 61 or (move.x, move.y) == (unit.x, unit.y) for move in moves):
            raise ValueError("candidate exceeds the bounded scout observation")
        return sight, tuple(moves)
    except (ValueError, TypeError, AttributeError) as exc:
        raise BridgeError(f"Invalid scout observation: {exc}") from exc


def build_scout_move_guard(order):
    """Reuse the bounded observation immediately before sending a scout move."""
    return _CANDIDATE_HELPERS + f'''
local scoutOK, scoutRows = pcall(function() return scoutCandidateRows(unit, me) end)
if not scoutOK then reject("SCOUT_API_UNAVAILABLE"); return end
local scoutTargetSafe = false
for _, row in ipairs(scoutRows) do
    if row[1] == {order.x} and row[2] == {order.y} and row[4] == 0 then scoutTargetSafe = true end
end
if not scoutTargetSafe then reject("SCOUT_TARGET_CHANGED"); return end
'''


def build_scout_move(order):
    """Issue an ordinary move without the upstream builder's attack modifier.

    The caller must first run the shared live candidate guard in the same Lua
    request. A hidden occupant can still block movement; readback verifies it.
    """
    return f'''
local scoutMoveParams = {{}}
scoutMoveParams[UnitOperationTypes.PARAM_X] = {order.x}
scoutMoveParams[UnitOperationTypes.PARAM_Y] = {order.y}
UnitManager.RequestOperation(unit, UnitOperationTypes.MOVE_TO, scoutMoveParams)
print("OK:SCOUT_MOVE_REQUESTED|{order.x}|{order.y}")
print("---END---")
'''
