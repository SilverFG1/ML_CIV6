"""Inspect Civ VI or preview/execute one unit order through FireTuner."""

import argparse
import asyncio
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

from .backend import RealCiv6Client
from .controller import plan_order, plan_scout_order, read_snapshot, record_event, save_snapshot, submit_order
from .models import BridgeError, Order
from .terrain import TerrainCatalog


def _timeout(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return number


def _nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="report dependency/setup status without connecting")
    terrain = commands.add_parser("import-terrain", help="import base terrain definitions from the installed game")
    default_game = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Steam/steamapps/common/Sid Meier's Civilization VI"
    terrain.add_argument("--game-path", type=Path, default=default_game,
                         help="Civ VI installation directory, or a Terrains.xml file")
    terrain.add_argument("--output", type=Path, default=Path("runs/civ6/terrain_types.json"))
    observe = commands.add_parser("observe", help="capture structured game state")
    observe.add_argument("--center", nargs=2, type=_nonnegative, metavar=("X", "Y"))
    observe.add_argument("--radius", type=int, choices=range(5), default=2)
    observe.add_argument("--output", type=Path, help="also save a snapshot for offline planning")
    plan = commands.add_parser("plan", help="plan from a saved snapshot without connecting")
    plan.add_argument("--snapshot", type=Path, required=True)
    play = commands.add_parser("play", help="make one tactical decision for a chosen unit")
    scout = commands.add_parser("scout", help="preview or execute one scout exploration decision")
    scout.add_argument("--unit-index", type=_nonnegative, required=True)
    scout.add_argument("--heal-below", type=int, choices=range(101), default=50, metavar="HP")
    scout.add_argument("--snapshot", type=Path, help="offline snapshot for preview only")
    for command in (plan, play):
        command.add_argument("--unit-index", type=_nonnegative, required=True)
        command.add_argument("--target", nargs=2, type=_nonnegative, required=True, metavar=("X", "Y"),
                             help="adjacent visible tile (or current tile to fortify)")
        command.add_argument("--heal-below", type=int, choices=range(101), default=50, metavar="HP")
    order = commands.add_parser("order", help="preview or send an explicit basic unit order")
    order.add_argument("--kind", choices=("move", "attack", "fortify", "skip"), required=True)
    order.add_argument("--unit-index", type=_nonnegative, required=True)
    order.add_argument("--target", nargs=2, type=_nonnegative, metavar=("X", "Y"))
    for command in (play, order, scout):
        command.add_argument("--execute", action="store_true", help="send this single order to the game")
    for command in (observe, play, order, scout):
        command.add_argument("--timeout", type=_timeout, default=10.0,
                             help="connection/operation deadline in seconds (default: 10)")
        command.add_argument("--terrain-catalog", type=Path,
                             help="imported terrain catalog; defaults to bundled base-game definitions")
    for command in (observe, plan, play, order, scout):
        command.add_argument("--journal", type=Path, default=Path("runs/civ6/transitions.jsonl"))
    args = parser.parse_args(argv)
    if args.command == "order":
        targeted = args.kind in ("move", "attack")
        if targeted != (args.target is not None):
            parser.error("--target is required for move/attack and only accepted for those orders")
    if args.command == "scout" and args.snapshot is not None and args.execute:
        parser.error("offline scout plans cannot execute; omit --snapshot to capture current game state")
    return args


def doctor():
    """Do not probe port 4318: an incomplete FireTuner handshake can hang it."""
    available = importlib.util.find_spec("civ_mcp") is not None
    try:
        version = importlib.metadata.version("civ6-mcp")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"bridge_installed": available, "bridge_version": version,
            "connection_tested": False,
            "next_steps": [
                "Install optional dependencies: python -m pip install -r requirements-civ6.txt",
                "Enable Tuner in Civ VI gameplay options (disables achievements).",
                "On Windows install Civilization VI SDK from Steam Tools; close FireTuner.exe.",
                "Load a single-player Gathering Storm game, then run: python run_civ6.py observe",
            ]}


def offline_plan(args):
    snapshot = read_snapshot(args.snapshot)
    order = plan_order(snapshot, args.unit_index, args.target, args.heal_below)
    result = {"status": "preview", "message": "Offline plan; no game connection opened."}
    record_event(args.journal, "offline_plan", observation=snapshot.to_dict(),
                 action=order.to_dict(), result=result)
    return {"order": order.to_dict(), "result": result}


def import_terrain(args):
    catalog = TerrainCatalog.from_game_path(args.game_path)
    catalog.save(args.output)
    payload = catalog.to_dict()
    return {"schema": payload["schema"], "terrain_types": len(payload["terrains"]),
            "features": len(payload["features"]), "source": payload["source"],
            "output": str(args.output.resolve())}


def offline_scout_plan(args):
    snapshot = read_snapshot(args.snapshot)
    order = plan_scout_order(snapshot, args.unit_index, args.heal_below)
    result = {"status": "preview", "message": "Offline scout plan; no game connection opened."}
    record_event(args.journal, "offline_scout_plan", observation=snapshot.to_dict(),
                 action=order.to_dict(), result=result)
    return {"policy": "scout", "turn": snapshot.turn, "player_id": snapshot.player_id,
            "order": order.to_dict(), "result": result}


async def run(args, client_factory=RealCiv6Client):
    if args.command == "doctor":
        return doctor()
    if args.command == "plan":
        return offline_plan(args)
    if args.command == "import-terrain":
        return import_terrain(args)
    if args.command == "scout" and args.snapshot is not None:
        return offline_scout_plan(args)
    client_options = {"timeout": args.timeout}
    if args.terrain_catalog is not None:
        client_options["terrain_catalog"] = TerrainCatalog.load(args.terrain_catalog)
    async with client_factory(**client_options) as client:
        center = tuple(args.center) if args.command == "observe" and args.center else None
        radius = args.radius if args.command == "observe" else 2
        snapshot = await client.snapshot(center=center, radius=radius)
        if args.command == "observe":
            if args.output is not None:
                save_snapshot(snapshot, args.output)
            record_event(args.journal, "observation", observation=snapshot.to_dict())
            return snapshot.to_dict()
        if args.command in ("play", "scout"):
            unit = next((unit for unit in snapshot.units if unit.index == args.unit_index), None)
            if unit is not None and snapshot.center != (unit.x, unit.y):
                snapshot = await client.snapshot(center=(unit.x, unit.y), radius=radius)
            order = (plan_scout_order(snapshot, args.unit_index, args.heal_below) if args.command == "scout" else
                     plan_order(snapshot, args.unit_index, args.target, args.heal_below))
        else:
            coords = args.target or (None, None)
            order = Order(args.kind, args.unit_index, *coords)
            if not snapshot.turn_active:
                raise ValueError("the local player's turn is not active")
            if not any(unit.index == order.unit_index for unit in snapshot.units):
                raise ValueError("unit index is not in this local-player snapshot")
        result = await submit_order(client, snapshot, order, execute=args.execute, journal=args.journal)
        output = {"turn": snapshot.turn, "player_id": snapshot.player_id,
                  "order": order.to_dict(), "result": result}
        if args.command == "scout":
            output["policy"] = "scout"
        return output


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "doctor":
            output = doctor()
        elif args.command == "plan":
            output = offline_plan(args)
        elif args.command == "import-terrain":
            output = import_terrain(args)
        elif args.command == "scout" and args.snapshot is not None:
            output = offline_scout_plan(args)
        else:
            output = asyncio.run(run(args))
    except (BridgeError, ValueError, OSError, TimeoutError) as exc:
        print(f"Civ VI bridge: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Civ VI bridge interrupted. Inspect the game before resending an order.", file=sys.stderr)
        return 130
    print(json.dumps(output, indent=2, allow_nan=False))
    status = output.get("result", {}).get("status")
    return 1 if status in ("rejected", "unknown") else 0
