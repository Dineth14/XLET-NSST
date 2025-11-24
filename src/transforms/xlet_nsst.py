"""
XLET-NSST (Nonsubsampled Shearlet Transform with Extended Laplacian) Implementation

This module provides the core XLET-NSST transformation for multi-scale, multi-direction
frequency decomposition optimized for semantic segmentation tasks.
"""

import numpy as np
import pywt
from scipy.ndimage import convolve
from typing import List, Tuple, Dict
import cv2


class XLETNSST:
    """
    XLET-NSST Transformer for extracting multi-scale directional frequency features.
    
    Combines:
    - Extended Laplacian Pyramid for multi-scale decomposition
    - Nonsubsampled Shearlet Transform for directional analysis
    
    Optimized for semantic segmentation feature extraction.
    """
    
    def __init__(self, 
                 levels: int = 3,
                 directions: int = 8,
                 shear_levels: int = 2,
                 filter_type: str = 'maxflat'):
        """
        Initialize XLET-NSST transformer.
        
        Args:
            levels: Number of decomposition levels (scales)
            directions: Number of directional subbands per level
            shear_levels: Number of shearing levels for directional decomposition
            filter_type: Type of filter ('maxflat', 'pkva', or 'haar')
        """
        self.levels = levels
        self.directions = directions
        self.shear_levels = shear_levels
        self.filter_type = filter_type
        
    def transform(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply XLET-NSST transformation to an image.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            
        Returns:
            Dictionary containing:
                - 'lowpass': Low-frequency approximation
                - 'highpass_L{i}_D{j}': High-frequency subbands at level i, direction j
                - 'scales': List of scale levels
                - 'directions': List of direction angles
        """
        # Handle multi-channel images
        if len(image.shape) == 3:
            channels = []
            for c in range(image.shape[2]):
                channels.append(self._transform_single_channel(image[:, :, c]))
            return self._merge_channel_results(channels)
        else:
            return self._transform_single_channel(image)
    
    def _transform_single_channel(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Transform a single channel image."""
        # Normalize input
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        coeffs = {}
        coeffs['scales'] = list(range(self.levels))
        coeffs['directions'] = list(range(self.directions))
        
        # Apply multi-scale decomposition
        current_img = img.copy()
        
        for level in range(self.levels):
            # Laplacian pyramid decomposition
            lowpass, highpass = self._laplacian_decompose(current_img, level)
            
            # Apply directional decomposition to highpass
            directional_subbands = self._directional_decompose(highpass, level)
            
            # Store subbands
            for direction, subband in enumerate(directional_subbands):
                key = f'highpass_L{level}_D{direction}'
                coeffs[key] = subband
            
            # Continue with lowpass for next level
            current_img = lowpass
        
        # Store final lowpass
        coeffs['lowpass'] = current_img
        
        return coeffs
    
    def _laplacian_decompose(self, image: np.ndarray, level: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extended Laplacian pyramid decomposition.
        
        Args:
            image: Input image
            level: Current decomposition level
            
        Returns:
            lowpass: Low-frequency component
            highpass: High-frequency component
        """
        # Gaussian kernel for lowpass
        kernel_size = 5
        sigma = 2 ** level
        kernel = self._get_gaussian_kernel(kernel_size, sigma)
        
        # Apply lowpass filtering
        lowpass = convolve(image, kernel, mode='reflect')
        
        # Highpass is the difference
        highpass = image - lowpass
        
        return lowpass, highpass
    
    def _directional_decompose(self, highpass: np.ndarray, level: int) -> List[np.ndarray]:
        """
        Directional decomposition using shearlet-like filters.
        
        Args:
            highpass: High-frequency image
            level: Current level
            
        Returns:
            List of directional subbands
        """
        subbands = []
        
        # Generate directional filters
        angles = np.linspace(0, 180, self.directions, endpoint=False)
        
        for angle in angles:
            # Create directional filter
            directional_filter = self._create_directional_filter(angle, level)
            
            # Apply filter
            subband = self._apply_shearlet_filter(highpass, directional_filter)
            subbands.append(subband)
        
        return subbands
    
    def _create_directional_filter(self, angle: float, level: int) -> np.ndarray:
        """
        Create a directional filter at specified angle.
        
        Args:
            angle: Direction angle in degrees
            level: Decomposition level
            
        Returns:
            Directional filter kernel
        """
        # Filter size depends on level
        size = 2 ** (level + 3) + 1
        center = size // 2
        
        # Create frequency domain directional filter
        y, x = np.ogrid[-center:center+1, -center:center+1]
        
        # Convert angle to radians
        theta = np.deg2rad(angle)
        
        # Rotate coordinates
        x_rot = x * np.cos(theta) + y * np.sin(theta)
        y_rot = -x * np.sin(theta) + y * np.cos(theta)
        
        # Create directional selectivity
        # Meyer-like window function
        angular_window = self._meyer_window(x_rot, y_rot, theta)
        
        # Radial window
        r = np.sqrt(x**2 + y**2)
        radial_window = self._bump_function(r / center)
        
        # Combine windows
        filter_kernel = angular_window * radial_window
        
        # Normalize
        if filter_kernel.sum() != 0:
            filter_kernel = filter_kernel / filter_kernel.sum()
        
        return filter_kernel
    
    def _meyer_window(self, x: np.ndarray, y: np.ndarray, theta: float) -> np.ndarray:
        """Meyer window function for angular selectivity."""
        # Angular selectivity based on orientation
        angle_tolerance = np.pi / self.directions
        
        # Calculate angle for each point
        angles = np.arctan2(y, x + 1e-10)
        
        # Angular distance from desired direction
        angle_diff = np.abs(angles - theta)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        
        # Smooth window
        window = np.exp(-(angle_diff**2) / (2 * angle_tolerance**2))
        
        return window
    
    def _bump_function(self, r: np.ndarray) -> np.ndarray:
        """Smooth bump function for radial selectivity."""
        result = np.zeros_like(r)
        mask = (r > 0) & (r < 1)
        result[mask] = np.exp(-1.0 / (1 - r[mask]**2))
        return result
    
    def _apply_shearlet_filter(self, image: np.ndarray, filter_kernel: np.ndarray) -> np.ndarray:
        """
        Apply shearlet filter in frequency domain.
        
        Args:
            image: Input image
            filter_kernel: Directional filter
            
        Returns:
            Filtered image
        """
        # Pad image to avoid boundary effects
        pad_h = filter_kernel.shape[0] // 2
        pad_w = filter_kernel.shape[1] // 2
        
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        # Apply filter
        filtered = convolve(padded, filter_kernel, mode='constant')
        
        # Remove padding
        result = filtered[pad_h:-pad_h, pad_w:-pad_w]
        
        return result
    
    def _get_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Generate 2D Gaussian kernel."""
        kernel = cv2.getGaussianKernel(size, sigma)
        kernel = kernel @ kernel.T
        return kernel / kernel.sum()
    
    def _merge_channel_results(self, channel_results: List[Dict]) -> Dict[str, np.ndarray]:
        """Merge results from multiple channels."""
        merged = {}
        
        # Copy metadata from first channel
        merged['scales'] = channel_results[0]['scales']
        merged['directions'] = channel_results[0]['directions']
        
        # Stack channel results
        for key in channel_results[0].keys():
            if key not in ['scales', 'directions']:
                stacked = np.stack([ch[key] for ch in channel_results], axis=-1)
                merged[key] = stacked
        
        return merged
    
    def get_feature_channels(self, coeffs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract organized feature channels from transform coefficients.
        
        Args:
            coeffs: Transform coefficients from transform()
            
        Returns:
            Dictionary of feature channels organized by type
        """
        features = {}
        
        # Lowpass features
        features['lowpass'] = coeffs['lowpass']
        
        # Organize by scale and direction
        for level in coeffs['scales']:
            for direction in coeffs['directions']:
                key = f'highpass_L{level}_D{direction}'
                if key in coeffs:
                    features[key] = coeffs[key]
        
        return features


class NSST:
    """
    Nonsubsampled Shearlet Transform (NSST) implementation.
    Focused version for directional frequency analysis.
    """
    
    def __init__(self, decomposition_level: int = 3, num_directions: int = 8):
        """
        Initialize NSST.
        
        Args:
            decomposition_level: Number of decomposition levels
            num_directions: Number of directional subbands
        """
        self.decomposition_level = decomposition_level
        self.num_directions = num_directions
    
    def nsst_decompose(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform NSST decomposition.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of NSST coefficients
        """
        # Use XLET-NSST as base implementation
        transformer = XLETNSST(
            levels=self.decomposition_level,
            directions=self.num_directions
        )
        
        return transformer.transform(image)
