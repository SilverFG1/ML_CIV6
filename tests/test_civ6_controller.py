"""The live controller must preserve intent and uncertainty without replay."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from civ6_bridge.controller import plan_order, plan_scout_order, read_snapshot, save_snapshot, submit_order
from civ6_bridge.models import BridgeError, Order, ScoutMove, Snapshot, UnitState


def sample_snapshot(hp=100, moves=2, active=True):
    return Snapshot(10, 0, active, (UnitState(0, 65536, "Warrior", 4, 4, hp, moves),),
                    {"civ_name": "Example"}, center=(4, 4))


class FakeClient:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    async def perform(self, order, expected_snapshot):
        self.calls.append((order, expected_snapshot))
        if self.failure:
            raise self.failure
        return {"status": "requested", "message": "Submitted; outcome unconfirmed."}


class ControllerTests(unittest.TestCase):
    def test_baseline_fortifies_wounded_and_stationed_units(self):
        self.assertEqual(plan_order(sample_snapshot(hp=50), 0, (5, 4)).kind, "fortify")
        self.assertEqual(plan_order(sample_snapshot(), 0, (4, 4)).kind, "fortify")
        self.assertEqual(plan_order(sample_snapshot(), 0, (5, 4)), Order("move", 0, 5, 4))

    def test_baseline_rejects_inactive_unknown_and_exhausted_units(self):
        for snapshot, index in ((sample_snapshot(active=False), 0),
                                (sample_snapshot(), 1), (sample_snapshot(moves=0), 0)):
            with self.subTest(snapshot=snapshot, index=index), self.assertRaises(ValueError):
                plan_order(snapshot, index, (5, 4))

    def test_baseline_rejects_invalid_thresholds_and_coordinates(self):
        for threshold in (True, -1, 101, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                plan_order(sample_snapshot(), 0, (5, 4), threshold)
        with self.assertRaises(ValueError):
            plan_order(sample_snapshot(), 0, (-1, 4))

    def test_snapshot_roundtrip_and_failed_replace_preserve_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            save_snapshot(sample_snapshot(), path)
            before = path.read_bytes()
            self.assertEqual(read_snapshot(path), sample_snapshot())
            with patch("civ6_bridge.controller.os.replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    save_snapshot(sample_snapshot(hp=40), path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


class ScoutControllerTests(unittest.TestCase):
    def snapshot(self, *, hp=100, moves=3, active=True, unit_type="UNIT_SCOUT", name="Scout", candidates=()):
        unit = UnitState(7, 65543, name, 4, 5, hp, moves, unit_type=unit_type, scout_moves=tuple(candidates))
        return Snapshot(12, 0, active, (unit,), {})

    def test_selects_highest_safe_frontier_and_avoids_visible_threats(self):
        snapshot = self.snapshot(candidates=(ScoutMove(5, 5, 3), ScoutMove(4, 6, 99, 1),
                                            ScoutMove(3, 5, 5), ScoutMove(4, 4, 0)))
        self.assertEqual(plan_scout_order(snapshot, 7), Order("move", 7, 3, 5))

    def test_ties_are_deterministic_by_coordinates(self):
        candidates = (ScoutMove(5, 5, 4), ScoutMove(4, 6, 4), ScoutMove(4, 4, 4))
        for options in (candidates, tuple(reversed(candidates))):
            self.assertEqual(plan_scout_order(self.snapshot(candidates=options), 7), Order("move", 7, 4, 4))

    def test_wounded_scout_fortifies_before_exploring(self):
        snapshot = self.snapshot(hp=50, candidates=(ScoutMove(5, 5, 20),))
        self.assertEqual(plan_scout_order(snapshot, 7), Order("fortify", 7))
        self.assertEqual(plan_scout_order(snapshot, 7, heal_below=49), Order("move", 7, 5, 5))

    def test_holds_when_no_candidates_no_frontier_or_only_threatened_tiles(self):
        for candidates in ((), (ScoutMove(5, 5, 0),), (ScoutMove(5, 5, 5, 1),)):
            with self.subTest(candidates=candidates):
                self.assertEqual(plan_scout_order(self.snapshot(candidates=candidates), 7), Order("fortify", 7))

    def test_accepts_renamed_native_scout_but_rejects_names_on_other_unit_types(self):
        candidates = (ScoutMove(5, 5, 2),)
        renamed = self.snapshot(name="Northern expedition", candidates=candidates)
        self.assertEqual(plan_scout_order(renamed, 7), Order("move", 7, 5, 5))
        for unit_type in (None, "UNIT_WARRIOR", "UNIT_RANGER", "MOD_UNIT_SCOUT", "unit_scout"):
            with self.subTest(unit_type=unit_type), self.assertRaisesRegex(ValueError, "UNIT_SCOUT"):
                plan_scout_order(self.snapshot(unit_type=unit_type, name="Scout", candidates=candidates), 7)

    def test_rejects_inactive_missing_and_exhausted_scouts(self):
        for snapshot, index in ((self.snapshot(active=False), 7), (self.snapshot(), 0),
                                (self.snapshot(moves=0), 7)):
            with self.subTest(snapshot=snapshot, index=index), self.assertRaises(ValueError):
                plan_scout_order(snapshot, index)

    def test_rejects_boolean_unit_ids_and_invalid_healing_thresholds(self):
        for threshold in (True, -1, 101, 5.5, float("nan"), None):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                plan_scout_order(self.snapshot(), 7, heal_below=threshold)
        with self.assertRaises(ValueError):
            plan_scout_order(self.snapshot(), True)


class SubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_preview_never_sends_an_order(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "trace.jsonl"
            client = FakeClient()
            result = await submit_order(client, sample_snapshot(), Order("move", 0, 5, 4), journal=journal)
            self.assertEqual(result["status"], "preview")
            self.assertEqual(client.calls, [])
            event = json.loads(journal.read_text())
            self.assertEqual(event["event"], "preview")
            self.assertEqual(event["observation"]["turn"], 10)

    async def test_execute_records_intent_before_sending(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "trace.jsonl"

            class InspectingClient(FakeClient):
                async def perform(self, order, expected_snapshot):
                    self.seen = json.loads(journal.read_text())["event"]
                    return await super().perform(order, expected_snapshot)

            client = InspectingClient()
            result = await submit_order(client, sample_snapshot(), Order("fortify", 0),
                                        execute=True, journal=journal)
            self.assertEqual(client.seen, "action_requested")
            self.assertEqual(result["status"], "requested")
            events = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual([event["event"] for event in events], ["action_requested", "action_result"])
            self.assertEqual(len(client.calls), 1)

    async def test_disconnected_action_is_logged_unknown_and_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "trace.jsonl"
            client = FakeClient(BridgeError("Disconnected after submission"))
            with self.assertRaises(BridgeError):
                await submit_order(client, sample_snapshot(), Order("fortify", 0), execute=True, journal=journal)
            self.assertEqual(len(client.calls), 1)
            events = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(events[-1]["result"]["status"], "unknown")

    async def test_journal_failure_prevents_dispatch(self):
        client = FakeClient()
        with patch("civ6_bridge.controller.record_event", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                await submit_order(client, sample_snapshot(), Order("fortify", 0), execute=True, journal="unused")
        self.assertEqual(client.calls, [])

    async def test_result_log_failure_preserves_known_order_status(self):
        client = FakeClient()
        with patch("civ6_bridge.controller.record_event", side_effect=[None, OSError("disk full")]):
            with self.assertRaisesRegex(BridgeError, "requested.*inspect the game"):
                await submit_order(client, sample_snapshot(), Order("fortify", 0), execute=True, journal="unused")
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
