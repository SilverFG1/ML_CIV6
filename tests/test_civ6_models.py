"""Serialization contracts reject malformed offline plans and snapshots."""

import json
import unittest

from civ6_bridge.models import Order, ScoutMove, Snapshot, UnitState


class ModelTests(unittest.TestCase):
    def make_snapshot(self):
        return Snapshot(12, 0, True, (UnitState(7, 65543, "Warrior", 4, 5, 80, 2.0),),
                        {"gold": 50}, ({"x": 4, "y": 5, "visibility": "visible"},), (4, 5))

    def test_snapshot_round_trips_through_json_without_aliases(self):
        snapshot = self.make_snapshot()
        payload = snapshot.to_dict()
        restored = Snapshot.from_dict(json.loads(json.dumps(payload)))
        self.assertEqual(restored, snapshot)
        self.assertEqual(payload["schema"], "civ6.live.v1")
        self.assertEqual(payload["center"], [4, 5])
        payload["overview"]["gold"] = 999
        self.assertEqual(snapshot.overview["gold"], 50)

    def test_order_rejects_wrong_targets_unknown_kinds_and_booleans(self):
        for values in [("move", 1), ("attack", 1, 2), ("skip", 1, 2, 3),
                       ("fortify", 1, 2, 3), ("end_turn", 1), ("move", True, 1, 1),
                       ("move", 0, False, 1), ("move", 0, -1, 1), ("skip", 1.5)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Order(*values)
        self.assertEqual(Order("move", 7, 3, 2).to_dict(),
                         {"kind": "move", "unit_index": 7, "x": 3, "y": 2})

    def test_snapshot_rejects_malformed_schema_identity_and_metadata(self):
        baseline = self.make_snapshot().to_dict()
        invalid = [
            {**baseline, "schema": "other"}, {**baseline, "turn": True},
            {**baseline, "turn_active": 1}, {**baseline, "units": {}},
            {**baseline, "units": baseline["units"] * 2}, {**baseline, "tiles": {}},
            {**baseline, "radius": 5}, {**baseline, "center": [1]},
            {**baseline, "overview": {"score": float("inf")}},
            {**baseline, "units": [{**baseline["units"][0], "index": True}]},
            {**baseline, "units": [{**baseline["units"][0], "moves_remaining": float("nan")}]},
            {**baseline, "units": [{**baseline["units"][0], "hp": 101}]},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                Snapshot.from_dict(payload)

    def test_old_snapshots_load_without_scout_fields_or_name_inference(self):
        payload = self.make_snapshot().to_dict()
        for key in ("unit_type", "max_moves", "base_combat_strength", "base_sight_range", "scout_moves"):
            payload["units"][0].pop(key)
        payload["units"][0]["name"] = "Scout"
        restored = Snapshot.from_dict(payload)
        self.assertEqual(restored.units[0].scout_moves, ())
        self.assertIsNone(restored.units[0].unit_type)
        self.assertIsNone(restored.units[0].max_moves)
        self.assertIsNone(restored.units[0].base_combat_strength)
        self.assertIsNone(restored.units[0].base_sight_range)
        self.assertFalse(restored.units[0].is_scout)

    def test_scout_stats_and_nested_candidates_roundtrip_without_fixed_base_values(self):
        scout = UnitState(7, 65543, "Éclaireur", 4, 5, 80, 3.5, unit_type="UNIT_SCOUT", max_moves=4.5,
                          base_combat_strength=17, base_sight_range=4,
                          scout_moves=(ScoutMove(5, 5, 7), ScoutMove(4, 6, 3, 1)))
        snapshot = Snapshot(12, 0, True, (scout,), {})
        payload = snapshot.to_dict()
        self.assertIsInstance(payload["units"][0]["scout_moves"], list)
        self.assertEqual(payload["units"][0]["scout_moves"][0],
                         {"x": 5, "y": 5, "unrevealed_tiles": 7, "visible_threats": 0})
        restored = Snapshot.from_dict(json.loads(json.dumps(payload)))
        self.assertEqual(restored, snapshot)
        self.assertTrue(restored.units[0].is_scout)
        self.assertIsInstance(restored.units[0].scout_moves, tuple)
        self.assertEqual(restored.units[0].max_moves, 4.5)

    def test_scout_identity_depends_only_on_exact_unit_type(self):
        for unit_type, name, expected in (("UNIT_SCOUT", "Pathfinder", True),
                                          ("UNIT_WARRIOR", "Scout", False),
                                          ("UNIT_RANGER", "Scout", False),
                                          ("MOD_UNIT_SCOUT", "Scout", False),
                                          ("unit_scout", "Scout", False), (None, "Scout", False)):
            with self.subTest(unit_type=unit_type):
                self.assertEqual(UnitState(1, 65537, name, 4, 5, 100, 2, unit_type=unit_type).is_scout, expected)

    def test_scout_moves_reject_boolean_negative_noninteger_duplicate_and_excess_data(self):
        for values in ((True, 2, 3), (1, False, 3), (1, 2, -1), (1, 2, 3, True), (1, 2, 3.5)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ScoutMove(*values)
        for moves in ([ScoutMove(5, 5, 1)], (ScoutMove(5, 5, 1), ScoutMove(5, 5, 2)),
                      tuple(ScoutMove(x, 5, 1) for x in range(7)), ({"x": 5, "y": 5},)):
            with self.subTest(moves=moves), self.assertRaises(ValueError):
                UnitState(7, 65543, "Scout", 4, 5, 100, 3, scout_moves=moves)

    def test_snapshot_rejects_invalid_nested_scout_records_and_optional_stats(self):
        baseline = self.make_snapshot().to_dict()
        invalid = [
            {"scout_moves": None}, {"scout_moves": {}}, {"scout_moves": (ScoutMove(5, 5, 1),)},
            {"scout_moves": [None]}, {"scout_moves": [{"x": 5, "y": 5}]},
            {"scout_moves": [{"x": 5, "y": 5, "unrevealed_tiles": True}]},
            {"scout_moves": [{"x": 5, "y": 5, "unrevealed_tiles": 1, "unexpected": 0}]},
            {"scout_moves": [{"x": 5, "y": 5, "unrevealed_tiles": 1}] * 2},
            {"unit_type": 3}, {"max_moves": True}, {"max_moves": -1}, {"max_moves": float("inf")},
            {"base_combat_strength": True}, {"base_combat_strength": -1}, {"base_sight_range": 2.5},
        ]
        for fields in invalid:
            payload = {**baseline, "units": [{**baseline["units"][0], **fields}]}
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                Snapshot.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
