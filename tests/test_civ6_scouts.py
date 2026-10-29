"""Protocol and Lua boundary checks for native scout candidate observations."""

from dataclasses import replace
import unittest

from civ6_bridge.models import BridgeError, Order, ScoutMove, UnitState
from civ6_bridge.scouts import (build_scout_move, build_scout_move_guard,
                               build_scout_query, parse_scout_response)


class ScoutQueryTests(unittest.TestCase):
    def setUp(self):
        self.unit = UnitState(7, 65543, "Explorer", 4, 5, 100, 3.0,
                              unit_type="UNIT_SCOUT", max_moves=3.0,
                              base_combat_strength=10)
        self.header = "MLCIV6_SCOUT|65543|7|2"
        self.ready = "MLCIV6_SCOUT_STATUS|65543|ready"

    def test_native_header_and_candidate_rows_parse_without_localized_identity(self):
        sight, moves = parse_scout_response(
            [self.header, self.ready, "MLCIV6_SCOUT_MOVE|65543|5|5|7|0"], self.unit)
        self.assertEqual(sight, 2)
        self.assertEqual(moves, (ScoutMove(5, 5, 7, 0),))
        # Lua may render integer-valued numbers with a decimal suffix.
        self.assertEqual(parse_scout_response(
            ["MLCIV6_SCOUT|65543.0|7.0|2.0", self.ready,
             "MLCIV6_SCOUT_MOVE|65543|5.0|5.0|7.0|0.0"], self.unit), (sight, moves))

    def test_candidate_unavailability_never_creates_a_fallback(self):
        for header in (self.header, "MLCIV6_SCOUT|65543|7|none"):
            with self.subTest(header=header), self.assertRaisesRegex(BridgeError, "APIs are unavailable"):
                parse_scout_response([header, "MLCIV6_SCOUT_STATUS|65543|unavailable"], self.unit)

    def test_protocol_rejects_missing_duplicate_or_foreign_headers(self):
        for rows in ([], [self.header], [self.ready], [self.header, self.header, self.ready],
                     [self.header, self.ready, self.ready],
                     ["MLCIV6_SCOUT|999|7|2", self.ready],
                     ["MLCIV6_SCOUT|65543|8|2", self.ready],
                     [self.header, "MLCIV6_SCOUT_STATUS|999|ready"],
                     ["MLCIV6_SCOUT|65543|7|none", self.ready]):
            with self.subTest(rows=rows), self.assertRaises(BridgeError):
                parse_scout_response(rows, self.unit)

    def test_protocol_rejects_unbounded_or_invalid_candidates(self):
        malformed = ["MLCIV6_SCOUT_MOVE|999|5|5|1|0",
                     "MLCIV6_SCOUT_MOVE|65543|5|5|-1|0",
                     "MLCIV6_SCOUT_MOVE|65543|5|5|1|true",
                     "MLCIV6_SCOUT_MOVE|65543|5|5|NaN|0",
                     "MLCIV6_SCOUT_MOVE|65543|5|5|1.5|0",
                     "MLCIV6_SCOUT_MOVE|65543|5|5|62|0",
                     "MLCIV6_SCOUT_MOVE|65543|4|5|1|0",
                     "MLCIV6_SCOUT_MOVE|65543|5|5|1|0|extra"]
        for row in malformed:
            with self.subTest(row=row), self.assertRaises(BridgeError):
                parse_scout_response([self.header, self.ready, row], self.unit)
        row = "MLCIV6_SCOUT_MOVE|65543|5|5|1|0"
        for rows in ([row, row],
                     [f"MLCIV6_SCOUT_MOVE|65543|{x}|5|1|0" for x in range(10, 17)]):
            with self.subTest(rows=rows), self.assertRaises(BridgeError):
                parse_scout_response([self.header, self.ready, *rows], self.unit)
        with self.assertRaises(BridgeError):
            parse_scout_response([self.header, "MLCIV6_SCOUT_STATUS|65543|unavailable", row], self.unit)

    def test_query_is_bounded_and_checks_current_engine_reachability(self):
        code = build_scout_query(self.unit)
        for expression in ("for direction = 0, 5", "math.min(sight, 4)",
                           "area(origin, 2)", "GetReachableMovement(unit)",
                           "reachable[plot:GetIndex()]", "unit:HasMovedIntoZOC()",
                           "CanStartOperation(unit, UnitOperationTypes.MOVE_TO, nil, params)",
                           "not plot:IsWater()", "not plot:IsImpassable()", "not plot:IsCity()",
                           'info.UnitType ~= "UNIT_SCOUT"', "info.BaseSightRange",
                           "unit:GetID() ~= 65543", "unit:GetX() ~= 4", "unit:GetY() ~= 5",
                           "unit:GetMovesRemaining() ~= 3.0", "ERR:STALE_SCOUT"):
            self.assertIn(expression, code)
        self.assertNotIn("RequestOperation", code)
        self.assertNotIn("AUTOMATE_EXPLORE", code)
        self.assertNotIn("SetProperty", code)
        self.assertNotIn("Map.GetPlotDistance", code)
        self.assertIn("nearbyThreatPlots[threat:GetIndex()]", code)
        self.assertLess(code.index("not player:IsTurnActiveComplete()"), code.index("local ok, rows = pcall"))
        with self.assertRaises(ValueError):
            build_scout_query(replace(self.unit, unit_type="UNIT_WARRIOR", name="Scout"))

    def test_exploration_reads_only_fog_masks_and_threats_require_unit_visibility(self):
        code = build_scout_query(self.unit)
        scoring = code[code.index("for _, nearby in ipairs(area(plot, math.min(sight, 4)))"):]
        scoring = scoring[:scoring.index("local danger = 0")]
        self.assertIn("not vis:IsRevealed(nearby:GetIndex())", scoring)
        for hidden_read in ("GetTerrainType", "GetFeatureType", "GetResourceType", "GetUnitsAt"):
            self.assertNotIn(hidden_read, scoring)
        threat_code = code[code.index("for _, plot in ipairs(area(origin, 2))"):]
        self.assertLess(threat_code.index("vis:IsVisible(plot:GetIndex())"), threat_code.index("Map.GetUnitsAt"))
        self.assertLess(threat_code.index("vis:IsUnitVisible(other)"), threat_code.index("other:GetType()"))
        self.assertIn("Players[owner]:IsBarbarian() or diplomacy:IsAtWarWith(owner)", threat_code)
        self.assertIn("other:GetOwner() == me or vis:IsUnitVisible(other)", code)

    def test_dispatch_guard_rechecks_candidates_and_plain_move_never_adds_attack(self):
        order = Order("move", 7, 5, 5)
        guard = build_scout_move_guard(order)
        self.assertIn("scoutCandidateRows(unit, me)", guard)
        self.assertIn("row[1] == 5 and row[2] == 5 and row[4] == 0", guard)
        self.assertIn("SCOUT_TARGET_CHANGED", guard)
        action = build_scout_move(order)
        self.assertEqual(action.count("UnitManager.RequestOperation"), 1)
        self.assertIn("UnitOperationTypes.MOVE_TO, scoutMoveParams", action)
        self.assertNotIn("ATTACK", action)
        self.assertNotIn("AUTOMATE", action)


if __name__ == "__main__":
    unittest.main()
