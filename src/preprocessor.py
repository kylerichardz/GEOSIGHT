import numpy as np
from typing import Optional

class SatellitePreprocessor:
    """Handles preprocessing of satellite imagery data."""
    
    @staticmethod
    def normalize_data(data: np.ndarray) -> np.ndarray:
        """
        Normalize the data to range [0, 1].
        
        Args:
            data: Input satellite data array
            
        Returns:
            numpy.ndarray: Normalized data
        """
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    @staticmethod
    def remove_noise(data: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Basic noise reduction using threshold filtering.
        
        Args:
            data: Input satellite data array
            threshold: Optional threshold value for noise removal
            
        Returns:
            numpy.ndarray: Filtered data
        """
        if threshold is None:
            threshold = np.mean(data) - 2 * np.std(data)
        
        return np.where(data < threshold, 0, data) 