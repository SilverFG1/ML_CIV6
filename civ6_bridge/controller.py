"""A first tactical baseline and an append-only record of live decisions."""

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from .models import BridgeError, Order, Snapshot


def plan_order(snapshot, unit_index, target, heal_below=50):
    """Fortify wounded/stationed units; otherwise propose a single tile move.

    This is a scripted baseline. Simulator Q-table actions have a different
    meaning and are deliberately not loaded by this live-game controller.
    Supply an adjacent visible target. The live adapter checks adjacency and
    the game engine checks legality when the proposed order is executed.
    """
    # Order validates the index and coordinates, including rejecting booleans.
    move = Order("move", unit_index, *target)
    if isinstance(heal_below, bool) or not isinstance(heal_below, int) or not 0 <= heal_below <= 100:
        raise ValueError("heal_below must be an integer between 0 and 100")
    if not snapshot.turn_active:
        raise ValueError("the local player's turn is not active")
    unit = next((unit for unit in snapshot.units if unit.index == unit_index), None)
    if unit is None:
        raise ValueError(f"unit index {unit_index} is not in this local-player snapshot")
    if unit.moves_remaining <= 0:
        raise ValueError(f"unit index {unit_index} has no movement remaining")
    if unit.hp <= heal_below or (unit.x, unit.y) == (move.x, move.y):
        return Order("fortify", unit_index)
    return move


def plan_scout_order(snapshot, unit_index, heal_below=50):
    """Choose one safe adjacent frontier move for an explicitly identified scout.

    Candidates come from the live engine. Only currently visible threats are
    counted, so an accepted candidate does not guarantee safety under fog.
    """
    hold = Order("fortify", unit_index)
    if isinstance(heal_below, bool) or not isinstance(heal_below, int) or not 0 <= heal_below <= 100:
        raise ValueError("heal_below must be an integer between 0 and 100")
    if not isinstance(snapshot, Snapshot):
        raise ValueError("snapshot must be a Snapshot")
    if not snapshot.turn_active:
        raise ValueError("the local player's turn is not active")
    unit = next((unit for unit in snapshot.units if unit.index == unit_index), None)
    if unit is None:
        raise ValueError(f"unit index {unit_index} is not in this local-player snapshot")
    if not unit.is_scout:
        raise ValueError(f"unit index {unit_index} must have unit_type UNIT_SCOUT")
    if unit.moves_remaining <= 0:
        raise ValueError(f"unit index {unit_index} has no movement remaining")
    if unit.hp <= heal_below:
        return hold
    candidates = [move for move in unit.scout_moves if move.visible_threats == 0 and move.unrevealed_tiles > 0]
    if not candidates:
        return hold
    selected = min(candidates, key=lambda move: (-move.unrevealed_tiles, move.x, move.y))
    return Order("move", unit_index, selected.x, selected.y)


def read_snapshot(path):
    with Path(path).open(encoding="utf-8") as handle:
        return Snapshot.from_dict(json.load(handle))


def save_snapshot(snapshot, path):
    """Replace a snapshot atomically so interrupted captures keep the old one."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            json.dump(snapshot.to_dict(), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def record_event(path, event, **payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": "civ6.trace.v1", "timestamp": datetime.now(timezone.utc).isoformat(),
              "event": event, **payload}
    text = json.dumps(record, allow_nan=False)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())


async def submit_order(client, snapshot, order, *, execute=False, journal):
    """Journal intent before sending exactly one order; never retry here."""
    before = snapshot.to_dict()
    action = order.to_dict()
    if not execute:
        result = {"status": "preview", "message": "No order sent to the game."}
        record_event(journal, "preview", observation=before, action=action, result=result)
        return result
    record_event(journal, "action_requested", observation=before, action=action)
    try:
        result = await client.perform(order, expected_snapshot=snapshot)
    except (Exception, asyncio.CancelledError) as exc:
        # A disconnected game may have already received the command. Preserve
        # that uncertainty for the caller instead of automatically resubmitting.
        try:
            record_event(journal, "action_error", action=action,
                         result={"status": "unknown", "message": str(exc) or "Interrupted after submission"})
        except OSError as journal_error:
            raise BridgeError("Order outcome unknown and result logging failed; "
                              "inspect the game before resending an order.") from journal_error
        raise
    try:
        record_event(journal, "action_result", action=action, result=result)
    except OSError as exc:
        raise BridgeError(f"Order returned {result.get('status')}: {result.get('message')}. "
                          f"Result logging failed ({exc}); inspect the game before resending an order.") from exc
    return result
