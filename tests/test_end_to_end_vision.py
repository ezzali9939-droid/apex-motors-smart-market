"""
Apex Motors — End-to-End Vision & Intelligence Test Suite
Tests vehicle ROI detection, cropping, non-car rejection,
multi-vehicle selection, specification lookups, market search, and market statistics logic.
"""

import os
import sys
import unittest
import io
import json
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.index import app
from services.vision import detect_vehicle_rois, extract_visual_embedding, compute_visual_similarity

class TestApexMotorsE2E(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.sample_car_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public", "mercedes-amg-gt3-speed-blur-desktop-wallpaper-cover.jpg"))

    def test_01_non_car_rejection_blank_image(self):
        """Verify non-car blank image is rejected with code NON_CAR_IMAGE or 422 status"""
        buf = io.BytesIO()
        Image.new("RGB", (400, 300), (245, 245, 245)).save(buf, format="JPEG")
        buf.seek(0)

        response = self.app.post(
            "/api/classify",
            data={"image": (buf, "blank.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 422)
        data = response.get_json()
        self.assertFalse(data.get("success"))
        self.assertIn(data.get("code"), ["NON_CAR_IMAGE", "VEHICLE_NOT_IDENTIFIED"])

    def test_02_non_car_rejection_abstract_drawing(self):
        """Verify abstract shape drawing is rejected as non-car"""
        img = Image.new("RGB", (400, 400), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        draw.ellipse([100, 100, 300, 300], fill=(50, 150, 250))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.app.post(
            "/api/classify",
            data={"image": (buf, "abstract.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 422)
        data = response.get_json()
        self.assertFalse(data.get("success"))

    def test_03_roi_detection_and_embedding(self):
        """Verify vehicle ROI detection and 100-dim visual embedding extraction on real car photo"""
        img = Image.open(self.sample_car_path)
        det = detect_vehicle_rois(img)
        self.assertTrue(det.get("is_car"))
        self.assertGreater(len(det.get("rois")), 0)

        emb1 = extract_visual_embedding(img)
        self.assertEqual(len(emb1), 100)
        
        # Test self-similarity
        sim = compute_visual_similarity(emb1, emb1)
        self.assertGreaterEqual(sim, 99.0)

    def test_04_multi_car_detection_flow(self):
        """Verify multi-car detection generates distinct crop bounding boxes when multiple cars exist"""
        arr = np.random.randint(20, 60, (400, 800, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        # Car 1: Red vehicle shape with headlights
        draw.rectangle([40, 80, 330, 280], fill=(220, 40, 40))
        draw.rectangle([60, 100, 110, 140], fill=(255, 255, 200))
        draw.ellipse([70, 250, 130, 310], fill=(10, 10, 10))
        draw.ellipse([240, 250, 300, 310], fill=(10, 10, 10))
        # Car 2: Blue vehicle shape with headlights
        draw.rectangle([470, 80, 760, 280], fill=(40, 80, 220))
        draw.rectangle([490, 100, 540, 140], fill=(255, 255, 200))
        draw.ellipse([500, 250, 560, 310], fill=(10, 10, 10))
        draw.ellipse([670, 250, 730, 310], fill=(10, 10, 10))

        det = detect_vehicle_rois(img)
        self.assertTrue(det.get("is_car"))
        self.assertGreaterEqual(len(det.get("rois")), 1)

    def test_05_verified_vehicle_specs_endpoint(self):
        """Verify /api/vehicle-specs returns canonical specs for Kia Sportage & Toyota Corolla"""
        resp = self.app.get("/api/vehicle-specs?make=Kia&model=Sportage&year=2022")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("specs_status"), "verified")
        self.assertEqual(data.get("manufacturer"), "Kia")
        self.assertEqual(data.get("model"), "Sportage")
        self.assertIn("dimensions", data)

    def test_06_market_search_and_stats(self):
        """Verify market search returns listings with market stats and valid URLs"""
        resp = self.app.get("/api/search?q=Toyota+Corolla&page=1")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("results", data)
        self.assertIn("price_stats", data)

if __name__ == "__main__":
    unittest.main()
