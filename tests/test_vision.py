import os
import sys
import unittest
import io
from PIL import Image, ImageDraw

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.index import app

class TestVisionPipeline(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.sample_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mercedes-amg-gt3-speed-blur-desktop-wallpaper-cover.jpg"))

    def test_01_vision_health_endpoint(self):
        """Verify /api/vision-health returns structured diagnostic data"""
        response = self.app.get('/api/vision-health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("service"), "vehicle-vision")
        self.assertIn("status", data)
        self.assertIn("env_vars", data)
        self.assertIn("active_provider", data)

    def test_02_no_fake_perceptual_fallback(self):
        """Verify that the system NEVER returns the fake 'Apex Perceptual Vision Engine' fallback"""
        with open(self.sample_image_path, "rb") as img_file:
            response = self.app.post(
                '/api/classify',
                data={'image': (img_file, 'sample_car.jpg')},
                content_type='multipart/form-data'
            )
        
        data = response.get_json()
        model_used = str(data.get("model_used", ""))
        engine = str(data.get("engine", ""))

        self.assertNotIn("Perceptual Vision Engine", model_used, "Fake perceptual engine MUST NOT be used for vehicle classification!")
        self.assertNotIn("Perceptual Vision Engine", engine, "Fake perceptual engine MUST NOT be returned in engine field!")

    def test_03_negative_input_blank_image(self):
        """Test negative input: Blank image must not be confidently classified as a vehicle"""
        buf = io.BytesIO()
        Image.new('RGB', (300, 300), (255, 255, 255)).save(buf, format='JPEG')
        buf.seek(0)

        response = self.app.post(
            '/api/classify',
            data={'image': (buf, 'blank.jpg')},
            content_type='multipart/form-data'
        )

        # Blank image must either be rejected as 422 VEHICLE_NOT_IDENTIFIED or 503 VISION_SERVICE_UNAVAILABLE (if no key)
        self.assertIn(response.status_code, (422, 503), f"Blank image returned unexpected status {response.status_code}")
        data = response.get_json()
        self.assertFalse(data.get("success"), "Blank image classification success MUST be false")

    def test_04_negative_input_abstract_shape(self):
        """Test negative input: Abstract non-vehicle shape must not pass high-confidence classification"""
        img = Image.new('RGB', (400, 400), (50, 50, 50))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 350, 350], fill=(200, 100, 0))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)

        response = self.app.post(
            '/api/classify',
            data={'image': (buf, 'abstract.jpg')},
            content_type='multipart/form-data'
        )

        self.assertIn(response.status_code, (422, 503))
        data = response.get_json()
        self.assertFalse(data.get("success"), "Non-vehicle shape classification success MUST be false")

    def test_05_unconfigured_ai_uses_local_vision_fallback(self):
        """Verify that when no cloud AI provider keys exist in environment, local vision engine fallback activates with HTTP 200"""
        # Save old env vars
        old_hf = os.environ.pop("HF_TOKEN", None)
        old_hf_alt = os.environ.pop("HUGGINGFACE_TOKEN", None)
        old_oai = os.environ.pop("OPENAI_API_KEY", None)

        try:
            with open(self.sample_image_path, "rb") as img_file:
                response = self.app.post(
                    '/api/classify',
                    data={'image': (img_file, 'car.jpg')},
                    content_type='multipart/form-data'
                )

            self.assertEqual(response.status_code, 200, "Unconfigured cloud AI must fall back to local vision engine with HTTP 200")
            data = response.get_json()
            self.assertTrue(data.get("success"))
            self.assertEqual(data.get("engine"), "local_vision_engine")

        finally:
            # Restore env vars
            if old_hf: os.environ["HF_TOKEN"] = old_hf
            if old_hf_alt: os.environ["HUGGINGFACE_TOKEN"] = old_hf_alt
            if old_oai: os.environ["OPENAI_API_KEY"] = old_oai

if __name__ == "__main__":
    unittest.main()
