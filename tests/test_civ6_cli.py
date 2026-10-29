"""Offline commands and live previews need no real game or optional package."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from civ6_bridge.cli import main, parse_args, run
from civ6_bridge.controller import save_snapshot
from civ6_bridge.models import ScoutMove, Snapshot, UnitState
from civ6_bridge.terrain import TerrainCatalog


def snapshot():
    return Snapshot(10, 0, True, (UnitState(0, 65536, "Warrior", 4, 4, 100, 2),),
                    {"civ_name": "Example"}, center=(4, 4))


def scout_snapshot():
    scout = UnitState(0, 65536, "Scout", 4, 4, 100, 3, unit_type="UNIT_SCOUT", max_moves=3,
                      base_combat_strength=10, base_sight_range=2,
                      scout_moves=(ScoutMove(5, 4, 4), ScoutMove(4, 5, 7, 1)))
    return Snapshot(10, 0, True, (scout,), {"source": "synthetic test"}, center=(4, 4))


class FakeClient:
    def __init__(self, **kwargs):
        self.orders = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def snapshot(self, **kwargs):
        return snapshot()

    async def perform(self, order, expected_snapshot):
        self.orders.append(order)
        return {"status": "verified", "message": "Reached tile."}


class CliTests(unittest.TestCase):
    def test_import_and_help_do_not_load_upstream_or_pygame(self):
        code = "import civ6_bridge.cli, sys; assert 'civ_mcp' not in sys.modules; assert 'pygame' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", "run_civ6.py", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_argument_errors_are_reported_before_connecting(self):
        cases = (["order", "--kind", "move", "--unit-index", "0"],
                 ["order", "--kind", "skip", "--unit-index", "0", "--target", "1", "2"],
                 ["observe", "--timeout", "nan"], ["observe", "--timeout", "0"],
                 ["observe", "--radius", "5"], ["observe", "--center", "-1", "0"],
                 ["scout", "--unit-index", "0", "--snapshot", "saved.json", "--execute"])
        for args in cases:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)

    def test_unknown_action_outcome_has_nonzero_exit_code(self):
        async def unknown(*args):
            return {"result": {"status": "unknown"}}

        with patch("civ6_bridge.cli.run", new=unknown), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["order", "--kind", "skip", "--unit-index", "0"]), 1)

    def test_import_terrain_runs_without_a_game_or_async_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            game_data = Path(directory) / "Base/Assets/Gameplay/Data"
            game_data.mkdir(parents=True)
            (game_data / "Terrains.xml").write_text(
                '<GameInfo><Terrains><Row TerrainType="TERRAIN_GRASS" Name="LOC_GRASS" MovementCost="1"/>'
                '</Terrains></GameInfo>', encoding="utf-8")
            destination = Path(directory) / "catalog.json"
            stdout = io.StringIO()
            with patch("civ6_bridge.cli.asyncio.run", side_effect=AssertionError("opened event loop")), \
                    contextlib.redirect_stdout(stdout):
                code = main(["import-terrain", "--game-path", directory, "--output", str(destination)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["terrain_types"], 1)
            self.assertEqual(TerrainCatalog.load(destination).to_dict()["terrains"]["TERRAIN_GRASS"]["movement_cost"], 1)

    def test_import_error_preserves_previous_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "catalog.json"
            destination.write_text("keep previous catalog", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(["import-terrain", "--game-path", str(Path(directory) / "missing"),
                             "--output", str(destination)])
            self.assertEqual(code, 1)
            self.assertEqual(destination.read_text(), "keep previous catalog")

    def test_offline_scout_plan_needs_no_event_loop_or_game(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scout.json"
            save_snapshot(scout_snapshot(), path)
            stdout = io.StringIO()
            with patch("civ6_bridge.cli.asyncio.run", side_effect=AssertionError("opened event loop")), \
                    contextlib.redirect_stdout(stdout):
                code = main(["scout", "--snapshot", str(path), "--unit-index", "0",
                             "--journal", str(Path(directory) / "trace.jsonl")])
            self.assertEqual(code, 0)
            output = json.loads(stdout.getvalue())
            self.assertEqual(output["policy"], "scout")
            self.assertEqual(output["order"], {"kind": "move", "unit_index": 0, "x": 5, "y": 4})
            self.assertEqual(output["result"]["status"], "preview")

    def test_old_snapshot_without_native_type_cannot_be_mistaken_for_scout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            save_snapshot(snapshot(), path)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["scout", "--snapshot", str(path), "--unit-index", "0"]), 1)


class CliAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_scout_live_decision_previews_until_execute_flag(self):
        class ScoutClient(FakeClient):
            async def snapshot(self, **kwargs):
                return scout_snapshot()

        with tempfile.TemporaryDirectory() as directory:
            arguments = ["scout", "--unit-index", "0", "--journal", str(Path(directory) / "trace.jsonl")]
            client = ScoutClient()
            output = await run(parse_args(arguments), client_factory=lambda **kwargs: client)
            self.assertEqual(output["order"]["x"], 5)
            self.assertEqual(output["result"]["status"], "preview")
            self.assertEqual(client.orders, [])
            output = await run(parse_args(arguments + ["--execute"]), client_factory=lambda **kwargs: client)
            self.assertEqual(output["result"]["status"], "verified")
            self.assertEqual(len(client.orders), 1)

    async def test_offline_plan_does_not_create_a_client(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            save_snapshot(snapshot(), path)
            args = parse_args(["plan", "--snapshot", str(path), "--unit-index", "0", "--target", "5", "4",
                               "--journal", str(Path(directory) / "trace.jsonl")])
            with patch("civ6_bridge.cli.RealCiv6Client", side_effect=AssertionError("connected")):
                output = await run(args, client_factory=lambda **kwargs: self.fail("connected"))
            self.assertEqual(output["order"]["kind"], "move")

    async def test_observe_serializes_state_without_sending_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            args = parse_args(["observe", "--output", str(path), "--journal", str(Path(directory) / "trace.jsonl")])
            client = FakeClient()
            output = await run(args, client_factory=lambda **kwargs: client)
            self.assertEqual(output["schema"], "civ6.live.v1")
            self.assertEqual(json.loads(path.read_text())["turn"], 10)
            self.assertEqual(client.orders, [])

    async def test_live_order_previews_until_execute_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            base = ["order", "--kind", "move", "--unit-index", "0", "--target", "5", "4",
                    "--journal", str(Path(directory) / "trace.jsonl")]
            client = FakeClient()
            output = await run(parse_args(base), client_factory=lambda **kwargs: client)
            self.assertEqual(output["result"]["status"], "preview")
            self.assertEqual(client.orders, [])
            output = await run(parse_args(base + ["--execute"]), client_factory=lambda **kwargs: client)
            self.assertEqual(output["result"]["status"], "verified")
            self.assertEqual(len(client.orders), 1)

    async def test_explicit_catalog_is_loaded_before_connecting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            TerrainCatalog.default().save(path)
            args = parse_args(["observe", "--terrain-catalog", str(path),
                               "--journal", str(Path(directory) / "trace.jsonl")])
            client_options = {}

            def factory(**kwargs):
                client_options.update(kwargs)
                return FakeClient()

            await run(args, client_factory=factory)
            self.assertIsInstance(client_options["terrain_catalog"], TerrainCatalog)
            self.assertEqual(client_options["terrain_catalog"].to_dict(), TerrainCatalog.default().to_dict())

    async def test_invalid_explicit_catalog_never_connects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text('{"schema": "wrong"}', encoding="utf-8")
            args = parse_args(["observe", "--terrain-catalog", str(path)])
            with self.assertRaises(ValueError):
                await run(args, client_factory=lambda **kwargs: self.fail("opened a connection"))


if __name__ == "__main__":
    unittest.main()
