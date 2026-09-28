import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.knowledge_base import (
    normalize_brand,
    parse_search_query,
    resolve_vehicle_specs
)

class TestKnowledgeBase(unittest.TestCase):
    def test_01_brand_normalization(self):
        self.assertEqual(normalize_brand("mercedes"), "Mercedes-Benz")
        self.assertEqual(normalize_brand("مرسيدس"), "Mercedes-Benz")
        self.assertEqual(normalize_brand("بي ام دبليو"), "BMW")
        self.assertEqual(normalize_brand("تويوتا"), "Toyota")

    def test_02_query_parsing(self):
        p1 = parse_search_query("kia sportage 22")
        self.assertEqual(p1["make"], "Kia")
        self.assertEqual(p1["year"], 2022)

        p2 = parse_search_query("bmw 320i 2020")
        self.assertEqual(p2["make"], "BMW")
        self.assertEqual(p2["year"], 2020)

        p3 = parse_search_query("مرسيدس c180")
        self.assertEqual(p3["make"], "Mercedes-Benz")

    def test_03_specs_resolution(self):
        specs = resolve_vehicle_specs("Kia", "Sportage", 2024)
        self.assertTrue(specs["found"])
        self.assertEqual(specs["manufacturer"], "Kia")
        self.assertEqual(specs["generation_code"], "NQ5")
        self.assertEqual(len(specs["engine_options"]), 3)
        self.assertTrue(specs["requires_engine_confirmation"])

        # Check provenance status tags
        dims = specs["dimensions"]
        self.assertEqual(dims["length_mm"]["status"], "verified_from_specs")

if __name__ == "__main__":
    unittest.main()
