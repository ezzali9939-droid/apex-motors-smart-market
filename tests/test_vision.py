import os
import sys
import unittest

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.index import app

class TestVisionPipeline(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.sample_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mercedes-amg-gt3-speed-blur-desktop-wallpaper-cover.jpg"))

    def test_vision_health_endpoint(self):
        """Verify /api/vision-health returns 200 and ready status"""
        response = self.app.get('/api/vision-health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("service"), "vehicle-vision")
        self.assertEqual(data.get("status"), "ready")
        self.assertIn("env_vars", data)

    def test_classify_real_vehicle_image(self):
        """Integration test: Real sample vehicle image -> /api/classify -> image decode -> vision -> structured result"""
        self.assertTrue(os.path.exists(self.sample_image_path), f"Sample image file not found at {self.sample_image_path}")
        
        with open(self.sample_image_path, "rb") as img_file:
            response = self.app.post(
                '/api/classify',
                data={'image': (img_file, 'sample_car.jpg')},
                content_type='multipart/form-data'
            )
        
        self.assertEqual(response.status_code, 200, f"Classify endpoint returned status {response.status_code}: {response.get_data(as_text=True)}")
        data = response.get_json()
        
        # Assertions as specified in Definition of Done
        self.assertTrue(data.get("success"), "Response 'success' should be True")
        self.assertIsNotNone(data.get("make"), "Detected 'make' must exist in response")
        self.assertIsNotNone(data.get("model"), "Detected 'model' must exist in response")
        self.assertTrue(len(data.get("make")) > 0, "Make string should not be empty")
        self.assertTrue(len(data.get("model")) > 0, "Model string should not be empty")
        self.assertIn("confidence", data, "Confidence score must exist")
        
        print("\n[SUCCESS] INTEGRATION TEST PASSED SUCCESSFULLY:")
        print(f"   Make: {data['make']}")
        print(f"   Model: {data['model']}")
        print(f"   Year Detected: {data.get('year_detected')}")
        print(f"   Confidence: {data['confidence']}%")
        print(f"   Model Used: {data.get('model_used')}")

if __name__ == "__main__":
    unittest.main()
