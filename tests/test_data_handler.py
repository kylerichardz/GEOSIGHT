import unittest
import numpy as np
from pathlib import Path
from src.data_handler import SatelliteDataHandler

class TestSatelliteDataHandler(unittest.TestCase):
    def setUp(self):
        self.handler = SatelliteDataHandler()
    
    def test_save_to_csv(self):
        # Create test data
        test_data = np.random.rand(10, 10)
        output_path = Path("test_output.csv")
        
        # Test saving to CSV
        self.handler.save_to_csv(test_data, output_path)
        
        # Verify file exists
        self.assertTrue(output_path.exists())
        
        # Clean up
        output_path.unlink()

if __name__ == '__main__':
    unittest.main() 