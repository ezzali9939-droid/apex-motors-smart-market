import os
import sys
import unittest
import io
import json
from unittest.mock import patch, MagicMock
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.index import app
from services.vision import VehicleVisionResponse, VehicleCandidate

class TestVisionPipeline(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.sample_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public", "mercedes-amg-gt3-speed-blur-desktop-wallpaper-cover.jpg"))

    def test_01_vision_health_endpoint(self):
        """Verify /api/vision-health returns structured diagnostic data"""
        response = self.app.get('/api/vision-health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("service"), "vehicle-vision")
        self.assertIn("status", data)
        self.assertIn("env_vars", data)

    def test_02_unconfigured_ai_returns_vision_unavailable_503(self):
        """Verify unconfigured GEMINI_API_KEY returns HTTP 503 VISION_UNAVAILABLE without fake fallbacks"""
        with patch.dict(os.environ, {}, clear=True):
            with open(self.sample_image_path, "rb") as img_file:
                response = self.app.post(
                    '/api/classify',
                    data={'image': (img_file, 'car.jpg')},
                    content_type='multipart/form-data'
                )

            self.assertEqual(response.status_code, 503)
            data = response.get_json()
            self.assertFalse(data.get("success"))
            self.assertEqual(data.get("code"), "VISION_UNAVAILABLE")

    def test_03_non_car_image_rejection_422(self):
        """Verify blank or non-car noise image returns HTTP 422 NON_CAR_IMAGE"""
        buf = io.BytesIO()
        Image.new('RGB', (300, 300), (255, 255, 255)).save(buf, format='JPEG')
        buf.seek(0)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_mock_key"}):
            response = self.app.post(
                '/api/classify',
                data={'image': (buf, 'blank.jpg')},
                content_type='multipart/form-data'
            )

            self.assertEqual(response.status_code, 422)
            data = response.get_json()
            self.assertFalse(data.get("vehicle_detected"))
            self.assertEqual(data.get("code"), "NON_CAR_IMAGE")

    @patch("services.vision.call_gemini_vision_api")
    def test_04_gemini_vision_clear_car_identification_mocked(self, mock_gemini):
        """Verify clear car identification via Gemini returns structured output & HTTP 200"""
        mock_response = VehicleVisionResponse(
            is_vehicle=True,
            make="BMW",
            model="X5",
            generation="G05",
            trim="M Sport",
            year_from=2019,
            year_to=2023,
            body_type="SUV",
            color="Black",
            visible_badges=["BMW", "X5", "M Sport"],
            visible_text=[],
            confidence=0.94,
            candidates=[VehicleCandidate(make="BMW", model="X3", generation="G01", confidence=0.65)],
            visual_evidence=["BMW kidney grille", "X5 rear badge"]
        )
        mock_gemini.return_value = (mock_response, None)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_mock_key"}):
            with open(self.sample_image_path, "rb") as img_file:
                response = self.app.post(
                    '/api/classify',
                    data={'image': (img_file, 'bmw_x5.jpg')},
                    content_type='multipart/form-data'
                )

            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertTrue(data.get("success"))
            self.assertTrue(data.get("vehicle_detected"))
            self.assertEqual(data.get("make"), "BMW")
            self.assertEqual(data.get("model"), "X5")
            self.assertEqual(data.get("trim"), "M Sport")
            self.assertEqual(data.get("confidence"), 94.0)

    @patch("services.vision.call_gemini_vision_api")
    def test_05_gemini_vision_low_confidence_uncertain_car_mocked(self, mock_gemini):
        """Verify confidence < 0.50 returns HTTP 422 VEHICLE_NOT_IDENTIFIED"""
        mock_response = VehicleVisionResponse(
            is_vehicle=True,
            make="Toyota",
            model="Corolla",
            confidence=0.42,
            visual_evidence=["Blurry silhouette"]
        )
        mock_gemini.return_value = (mock_response, None)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_mock_key"}):
            with open(self.sample_image_path, "rb") as img_file:
                response = self.app.post(
                    '/api/classify',
                    data={'image': (img_file, 'blurry.jpg')},
                    content_type='multipart/form-data'
                )

            self.assertEqual(response.status_code, 422)
            data = response.get_json()
            self.assertFalse(data.get("success"))
            self.assertEqual(data.get("code"), "VEHICLE_NOT_IDENTIFIED")

    @patch("services.vision.call_gemini_vision_api")
    def test_06_gemini_vision_medium_confidence_policy_mocked(self, mock_gemini):
        """Verify confidence 0.50-0.69 displays make but sets model to None"""
        mock_response = VehicleVisionResponse(
            is_vehicle=True,
            make="Mercedes-Benz",
            model="C-Class",
            confidence=0.62,
            visual_evidence=["Mercedes emblem visible"]
        )
        mock_gemini.return_value = (mock_response, None)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_mock_key"}):
            with open(self.sample_image_path, "rb") as img_file:
                response = self.app.post(
                    '/api/classify',
                    data={'image': (img_file, 'merc.jpg')},
                    content_type='multipart/form-data'
                )

            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data.get("make"), "Mercedes-Benz")
            self.assertIsNone(data.get("model"))
            self.assertTrue(data.get("needs_confirmation"))

    @patch("services.vision.call_gemini_vision_api")
    def test_07_gemini_invalid_json_handling(self, mock_gemini):
        """Verify malformed JSON from vision provider returns HTTP 422 VEHICLE_NOT_IDENTIFIED"""
        mock_gemini.return_value = (None, {
            "success": False,
            "vehicle_detected": False,
            "code": "VEHICLE_NOT_IDENTIFIED",
            "error": "Invalid structured JSON format returned by vision model.",
            "status_code": 422
        })

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_mock_key"}):
            with open(self.sample_image_path, "rb") as img_file:
                response = self.app.post(
                    '/api/classify',
                    data={'image': (img_file, 'car.jpg')},
                    content_type='multipart/form-data'
                )

            self.assertEqual(response.status_code, 422)
            data = response.get_json()
            self.assertEqual(data.get("code"), "VEHICLE_NOT_IDENTIFIED")

    def test_08_specifications_lookup_verified_vs_unverified(self):
        """Verify /api/vehicle-specs returns verified specs for database models and specs_status='not_verified' for unknown models"""
        # 1. Verified model
        resp = self.app.get("/api/vehicle-specs?make=Kia&model=Sportage&year=2024")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("specs_status"), "verified")

        # 2. Unknown model
        resp_unverified = self.app.get("/api/vehicle-specs?make=UnknownMake&model=UnknownModel")
        self.assertEqual(resp_unverified.status_code, 200)
        data_unverified = resp_unverified.get_json()
        self.assertFalse(data_unverified.get("success"))
        self.assertEqual(data_unverified.get("specs_status"), "not_verified")

if __name__ == "__main__":
    unittest.main()
