"""Base metadata import and annotation, with no game or optional dependencies."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from civ6_bridge.terrain import SCHEMA, SOURCE_FILES, TerrainCatalog


class TerrainCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = TerrainCatalog.default()

    def info(self, terrain, feature=None, **extra):
        return self.catalog.annotate_tile({"terrain": terrain, "feature": feature, **extra})["terrain_info"]

    def test_all_seventeen_exact_base_terrain_types_and_classes(self):
        expected = {f"TERRAIN_{base}{suffix}" for base in ("GRASS", "PLAINS", "DESERT", "TUNDRA", "SNOW")
                    for suffix in ("", "_HILLS", "_MOUNTAIN")} | {"TERRAIN_COAST", "TERRAIN_OCEAN"}
        self.assertEqual(set(self.catalog.terrains), expected)
        self.assertEqual(len(self.catalog.features), 18)
        for identifier in expected:
            with self.subTest(identifier=identifier):
                entry = self.catalog.terrains[identifier]
                info = self.info(identifier)
                self.assertTrue(info["known_terrain"])
                self.assertEqual(info["terrain_type"], identifier)
                self.assertTrue(entry["name_key"].startswith("LOC_TERRAIN_"))
                self.assertTrue(entry["terrain_class"].startswith("TERRAIN_CLASS_"))
                self.assertEqual(info["is_hills"], identifier.endswith("_HILLS"))
                self.assertEqual(info["is_mountain"], identifier.endswith("_MOUNTAIN"))
                self.assertEqual(info["base_movement_cost"], 2 if identifier.endswith("_HILLS") else 1)
                self.assertEqual(info["static_impassable"], identifier.endswith("_MOUNTAIN"))
                self.assertNotIn("passable", info)

    def test_hills_and_mountains_have_distinct_cost_defense_and_static_flags(self):
        hills, mountain = self.info("TERRAIN_GRASS_HILLS"), self.info("TERRAIN_GRASS_MOUNTAIN")
        self.assertEqual(hills["base_type"], "GRASS")
        self.assertEqual(mountain["base_type"], "GRASS")
        self.assertEqual((hills["base_movement_cost"], hills["base_defense_modifier"]), (2, 3))
        self.assertEqual((mountain["base_movement_cost"], mountain["base_defense_modifier"]), (1, 0))
        self.assertFalse(hills["terrain_impassable"])
        self.assertTrue(mountain["terrain_impassable"])

    def test_coast_water_is_not_coastal_land_and_does_not_infer_lakes(self):
        grass = self.info("TERRAIN_GRASS", is_coastal=True)
        coast, ocean = self.info("TERRAIN_COAST"), self.info("TERRAIN_OCEAN")
        self.assertFalse(grass["is_water"])
        self.assertTrue(coast["is_water"])
        self.assertTrue(coast["is_shallow_water"])
        self.assertTrue(ocean["is_water"])
        self.assertFalse(ocean["is_shallow_water"])
        self.assertNotIn("is_lake", coast)

    def test_features_remain_separate_from_terrain_and_actual_yields(self):
        for terrain, feature in (("TERRAIN_GRASS_HILLS", "FEATURE_FOREST"),
                                 ("TERRAIN_PLAINS", "FEATURE_JUNGLE")):
            info = self.info(terrain, feature)
            self.assertTrue(info["known_feature"])
            self.assertEqual(info["feature_type"], feature)
            self.assertEqual(info["feature_movement_change"], 1)
            self.assertEqual(info["feature_defense_modifier"], 3)
            self.assertFalse(info["static_impassable"])
        ice = self.info("TERRAIN_COAST", "FEATURE_ICE")
        self.assertTrue(ice["is_water"])
        self.assertFalse(ice["terrain_impassable"])
        self.assertTrue(ice["feature_impassable"])
        self.assertTrue(ice["static_impassable"])
        raw = {"terrain": "TERRAIN_GRASS_HILLS", "feature": "FEATURE_FOREST", "movement_cost": 17,
               "yields": [9, 8, 7, 6, 5, 4], "route_type": "ROUTE_RAILROAD", "visibility": "visible",
               "resource": "RESOURCE_IRON", "is_river": True, "custom": {"preserved": [1, 2]}}
        annotated = self.catalog.annotate_tile(raw)
        self.assertEqual({key: annotated[key] for key in raw}, raw)
        annotated["custom"]["preserved"].append(3)
        self.assertEqual(raw["custom"]["preserved"], [1, 2])
        self.assertNotIn("terrain_info", raw)
        self.assertEqual(self.catalog.terrains["TERRAIN_GRASS_HILLS"]["yields"],
                         {"YIELD_FOOD": 2, "YIELD_PRODUCTION": 1})
        self.assertEqual(self.catalog.features["FEATURE_FOREST"]["yields"], {"YIELD_PRODUCTION": 1})

    def test_none_feature_is_known_absence_and_unknown_mods_stay_unknown(self):
        for missing in (None, "none", "NONE"):
            info = self.info("TERRAIN_GRASS", missing)
            self.assertTrue(info["known_feature"])
            self.assertEqual(info["feature_type"], missing)
            self.assertEqual(info["feature_movement_change"], 0)
            self.assertEqual(info["feature_defense_modifier"], 0)
            self.assertFalse(info["feature_impassable"])
        info = self.info("TERRAIN_MODDED_CRYSTAL", "FEATURE_MODDED_MIST", is_hills=True)
        self.assertFalse(info["known_terrain"])
        self.assertFalse(info["known_feature"])
        self.assertEqual(info["base_type"], "TERRAIN_MODDED_CRYSTAL")
        for key in ("is_hills", "is_water", "static_impassable", "base_movement_cost", "feature_movement_change"):
            self.assertIsNone(info[key])
        self.assertIsNone(self.info("TERRAIN_GRASS", "FEATURE_MODDED_MIST")["static_impassable"])
        self.assertIsNone(self.catalog.annotate_tile({})["terrain_info"]["base_type"])
        self.assertTrue(self.info("TERRAIN_MODDED_CRYSTAL", "FEATURE_ICE")["static_impassable"])

    def test_source_is_portable_and_explicitly_base_reference(self):
        payload = self.catalog.to_dict()
        self.assertEqual(payload["schema"], SCHEMA)
        self.assertEqual(payload["source"]["kind"], "base_game_xml")
        self.assertEqual(payload["source"]["files"], list(SOURCE_FILES))
        self.assertEqual(self.info("TERRAIN_GRASS")["rules_scope"], "base_game_reference")
        self.assertNotIn("timestamp", json.dumps(payload))
        self.assertNotIn("C:\\", json.dumps(payload))
        payload["terrains"]["TERRAIN_GRASS"]["movement_cost"] = 123
        self.assertEqual(self.info("TERRAIN_GRASS")["base_movement_cost"], 1)

    def make_xml(self, folder, *, feature=True):
        terrain_xml = '''<GameInfo><Terrains><Row><TerrainType>TERRAIN_GRASS_HILLS</TerrainType>
            <Name>LOC_GRASS</Name><Hills>true</Hills><MovementCost>2</MovementCost></Row></Terrains>
            <TerrainClass_Terrains><Row TerrainType="TERRAIN_GRASS_HILLS" TerrainClassType="TERRAIN_CLASS_GRASS"/></TerrainClass_Terrains>
            <Terrain_YieldChanges><Row TerrainType="TERRAIN_GRASS_HILLS" YieldType="YIELD_FOOD" YieldChange="2"/></Terrain_YieldChanges></GameInfo>'''
        (folder / "Terrains.xml").write_text(terrain_xml, encoding="utf-8")
        if feature:
            (folder / "Features.xml").write_text('''<GameInfo><Features>
                <Row FeatureType="FEATURE_FOREST" Name="LOC_FOREST" MovementChange="1" DefenseModifier="3"/>
                </Features><Feature_YieldChanges><Row FeatureType="FEATURE_FOREST"
                YieldType="YIELD_PRODUCTION" YieldChange="1"/></Feature_YieldChanges></GameInfo>''', encoding="utf-8")
        return folder / "Terrains.xml"

    def test_imports_attributes_and_child_rows_from_base_path_and_direct_xml(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            game = Path(temporary)
            folder = game / "Base/Assets/Gameplay/Data"
            folder.mkdir(parents=True)
            xml_path = self.make_xml(folder)
            # DLC patches must not be applied to the base reference catalog.
            dlc = game / "DLC/Expansion2/Data"
            dlc.mkdir(parents=True)
            (dlc / "Terrains.xml").write_text("<GameInfo><Terrains><Delete/></Terrains></GameInfo>")
            catalog = TerrainCatalog.from_game_path(game)
            self.assertEqual(catalog.to_dict(), TerrainCatalog.from_game_path(xml_path).to_dict())
            self.assertEqual(catalog.to_dict(), TerrainCatalog.from_game_path(folder).to_dict())
            self.assertEqual(catalog.to_dict()["source"]["sha256"][SOURCE_FILES[0]],
                             hashlib.sha256(xml_path.read_bytes()).hexdigest())
            info = catalog.annotate_tile({"terrain": "TERRAIN_GRASS_HILLS", "feature": "FEATURE_FOREST"})["terrain_info"]
            self.assertEqual((info["base_movement_cost"], info["feature_movement_change"]), (2, 1))
            self.assertEqual(catalog.terrains["TERRAIN_GRASS_HILLS"]["terrain_class"], "TERRAIN_CLASS_GRASS")

    def test_direct_xml_can_omit_features_with_named_features_remaining_unknown(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            xml_path = self.make_xml(Path(temporary), feature=False)
            catalog = TerrainCatalog.from_game_path(xml_path)
            self.assertEqual(catalog.features, {})
            self.assertEqual(catalog.to_dict()["source"]["files"], [SOURCE_FILES[0]])
            self.assertFalse(catalog.annotate_tile({"terrain": "TERRAIN_GRASS_HILLS", "feature": "FEATURE_FOREST"})
                             ["terrain_info"]["known_feature"])

    def test_import_rejects_broken_xml_invalid_fields_and_patch_operations(self):
        for body in ("not xml", "<Other/>", "<GameInfo><Terrains><Update/></Terrains></GameInfo>",
                     '<GameInfo><Terrains><Row TerrainType="TERRAIN_GRASS" Name="LOC_GRASS"/></Terrains></GameInfo>',
                     '<GameInfo><Terrains><Row TerrainType="TERRAIN_GRASS" Name="LOC_GRASS" MovementCost="NaN"/></Terrains></GameInfo>'):
            with self.subTest(body=body), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
                xml_path = Path(temporary) / "Terrains.xml"
                xml_path.write_text(body, encoding="utf-8")
                with self.assertRaises(ValueError):
                    TerrainCatalog.from_game_path(xml_path)

    def test_json_load_rejects_schema_duplicates_nonfinite_and_wrong_types(self):
        valid = self.catalog.to_dict()
        bad_cost = deepcopy(valid)
        bad_cost["terrains"]["TERRAIN_GRASS"]["movement_cost"] = True
        bad_flag = deepcopy(valid)
        bad_flag["features"]["FEATURE_FOREST"]["impassable"] = "false"
        corrupt = [json.dumps({**valid, "schema": "wrong"}), json.dumps(bad_cost), json.dumps(bad_flag),
                   '{"schema": "a", "schema": "b"}', '{"value": NaN}', "[]", "{bad json"]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / "catalog.json"
            for body in corrupt:
                path.write_text(body, encoding="utf-8")
                with self.subTest(body=body[:80]), self.assertRaisesRegex(ValueError, "terrain catalog"):
                    TerrainCatalog.load(path)

    def test_atomic_save_roundtrip_and_failure_preserve_existing_file(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / "catalog.json"
            self.catalog.save(path)
            self.assertEqual(TerrainCatalog.load(path).to_dict(), self.catalog.to_dict())
            before = path.read_bytes()
            with patch("civ6_bridge.terrain.os.replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    self.catalog.save(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
