"""Optional conformance checks exercise the pinned dependency without a game."""

import importlib.util
import unittest
from unittest.mock import AsyncMock

from civ6_bridge.backend import RealCiv6Client, _load_api


@unittest.skipUnless(importlib.util.find_spec("civ_mcp") is not None, "optional civ6-mcp is not installed")
class UpstreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_transport_override_does_not_replay_a_disconnected_command(self):
        connection, _ = _load_api()

        class Writer:
            def is_closing(self):
                return False

        connection._writer = Writer()
        connection.gamecore_index, connection.ingame_index = 1, 2
        connection._locked_execute = AsyncMock(side_effect=ConnectionError("lost after dispatch"))
        connection.reconnect = AsyncMock()
        with self.assertRaises(ConnectionError):
            await connection.execute_write("print('command')")
        connection._locked_execute.assert_awaited_once()
        connection.reconnect.assert_not_awaited()

    async def test_real_builders_and_parsers_feed_the_snapshot_contract(self):
        connection, lua = _load_api()
        client = RealCiv6Client(connection=connection, lua=lua)
        for code in (lua.build_units_query(), lua.build_overview_query(), lua.build_map_area_query(4, 4, 2)):
            self.assertIn('print("---END---")', code)
        # Canned protocol records exercise actual upstream parsing, not fake
        # dataclasses, while requiring no running game or socket connection.
        client._query = AsyncMock(side_effect=[
            ["MLCIV6_STATE|10|0|true"],
            ["10|0|Example|Leader|100|5|10|5|0|Pottery|Code of Laws|1|1|10", "EXPLORE|20|2000"],
            ["65536|0|Warrior|UNIT_WARRIOR|4,4|2/2|100/100|20|0|0"],
            ["4,4|TERRAIN_GRASS|none|none|0|0|0|none|0|none|visible|1|2,1,0,0,0,0"],
            ["MLCIV6_STATE|10|0|true"],
        ])
        snapshot = await client.snapshot()
        self.assertEqual(snapshot.units[0].unit_id, 65536)
        self.assertEqual(snapshot.units[0].hp, 100)
        self.assertEqual(snapshot.units[0].unit_type, "UNIT_WARRIOR")
        self.assertEqual(snapshot.units[0].max_moves, 2)
        self.assertEqual(snapshot.units[0].base_combat_strength, 20)
        self.assertEqual(snapshot.tiles[0]["yields"], [2, 1, 0, 0, 0, 0])
        self.assertEqual(snapshot.tiles[0]["terrain_info"]["base_type"], "GRASS")
        self.assertNotIn("total_land", snapshot.overview)


if __name__ == "__main__":
    unittest.main()
