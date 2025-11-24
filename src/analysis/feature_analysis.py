"""
Feature Extraction and Analysis Tools for XLET-NSST

This module provides comprehensive tools to extract, analyze, and evaluate
different frequency channels from XLET-NSST decomposition for semantic segmentation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    """
    Extract and organize features from XLET-NSST coefficients.
    """
    
    def __init__(self):
        self.feature_statistics = {}
    
    def extract_all_features(self, coeffs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract all available feature channels from XLET-NSST coefficients.
        
        Args:
            coeffs: XLET-NSST transform coefficients
            
        Returns:
            Dictionary of all feature channels
        """
        features = {}
        
        # Extract lowpass (approximation) features
        if 'lowpass' in coeffs:
            features['lowpass'] = coeffs['lowpass']
        
        # Extract all highpass (detail) features
        for key in coeffs.keys():
            if key.startswith('highpass_'):
                features[key] = coeffs[key]
        
        return features
    
    def extract_scale_features(self, coeffs: Dict[str, np.ndarray], scale: int) -> Dict[str, np.ndarray]:
        """
        Extract features from a specific scale level.
        
        Args:
            coeffs: XLET-NSST coefficients
            scale: Scale level to extract
            
        Returns:
            Features at specified scale
        """
        scale_features = {}
        
        for key in coeffs.keys():
            if f'_L{scale}_' in key:
                scale_features[key] = coeffs[key]
        
        return scale_features
    
    def extract_direction_features(self, coeffs: Dict[str, np.ndarray], direction: int) -> Dict[str, np.ndarray]:
        """
        Extract features from a specific direction across all scales.
        
        Args:
            coeffs: XLET-NSST coefficients
            direction: Direction index to extract
            
        Returns:
            Features at specified direction
        """
        direction_features = {}
        
        for key in coeffs.keys():
            if f'_D{direction}' in key:
                direction_features[key] = coeffs[key]
        
        return direction_features
    
    def compute_feature_statistics(self, feature: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive statistics for a feature channel.
        
        Args:
            feature: Feature channel array
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Handle multi-channel features
        if len(feature.shape) == 3:
            feature_flat = feature.reshape(-1, feature.shape[-1])
            # Compute per-channel then average
            stats['mean'] = np.mean(feature)
            stats['std'] = np.std(feature)
            stats['min'] = np.min(feature)
            stats['max'] = np.max(feature)
            stats['energy'] = np.sum(feature ** 2) / feature.size
            stats['sparsity'] = np.sum(np.abs(feature) < 0.01) / feature.size
        else:
            feature_flat = feature.flatten()
            stats['mean'] = np.mean(feature_flat)
            stats['std'] = np.std(feature_flat)
            stats['min'] = np.min(feature_flat)
            stats['max'] = np.max(feature_flat)
            stats['energy'] = np.sum(feature_flat ** 2) / feature_flat.size
            stats['sparsity'] = np.sum(np.abs(feature_flat) < 0.01) / feature_flat.size
        
        # Entropy (information content)
        hist, _ = np.histogram(feature_flat, bins=256, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        stats['entropy'] = scipy_entropy(hist)
        
        # Dynamic range
        stats['dynamic_range'] = stats['max'] - stats['min']
        
        # Coefficient of variation
        if stats['mean'] != 0:
            stats['coef_variation'] = stats['std'] / abs(stats['mean'])
        else:
            stats['coef_variation'] = 0
        
        return stats
    
    def analyze_all_channels(self, coeffs: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Analyze all feature channels and compute statistics.
        
        Args:
            coeffs: XLET-NSST coefficients
            
        Returns:
            Dictionary mapping channel names to their statistics
        """
        analysis = {}
        
        for channel_name, feature in coeffs.items():
            if isinstance(feature, np.ndarray):
                analysis[channel_name] = self.compute_feature_statistics(feature)
        
        self.feature_statistics = analysis
        return analysis
    
    def get_best_channels_by_metric(self, 
                                     analysis: Dict[str, Dict[str, float]], 
                                     metric: str = 'entropy',
                                     top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Rank channels by a specific metric.
        
        Args:
            analysis: Feature analysis results
            metric: Metric to rank by ('entropy', 'energy', 'std', 'dynamic_range')
            top_k: Number of top channels to return
            
        Returns:
            List of (channel_name, metric_value) tuples
        """
        rankings = []
        
        for channel_name, stats in analysis.items():
            if channel_name not in ['scales', 'directions'] and metric in stats:
                rankings.append((channel_name, stats[metric]))
        
        # Sort by metric value (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings[:top_k]
    
    def create_feature_vector(self, 
                             coeffs: Dict[str, np.ndarray], 
                             selected_channels: Optional[List[str]] = None,
                             resize_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Create a feature vector by concatenating selected channels.
        
        Args:
            coeffs: XLET-NSST coefficients
            selected_channels: List of channel names to include (None = all)
            resize_shape: Optional shape to resize all channels to
            
        Returns:
            Feature vector array (H, W, N) where N is number of channels
        """
        if selected_channels is None:
            selected_channels = [k for k in coeffs.keys() 
                               if isinstance(coeffs[k], np.ndarray) and k not in ['scales', 'directions']]
        
        feature_maps = []
        
        for channel_name in selected_channels:
            if channel_name in coeffs:
                feature = coeffs[channel_name]
                
                # Resize if needed
                if resize_shape is not None and feature.shape[:2] != resize_shape:
                    if len(feature.shape) == 3:
                        resized = np.zeros((*resize_shape, feature.shape[2]))
                        for c in range(feature.shape[2]):
                            resized[:, :, c] = cv2.resize(feature[:, :, c], 
                                                         (resize_shape[1], resize_shape[0]))
                        feature = resized
                    else:
                        feature = cv2.resize(feature, (resize_shape[1], resize_shape[0]))
                
                # Ensure 3D
                if len(feature.shape) == 2:
                    feature = feature[:, :, np.newaxis]
                
                feature_maps.append(feature)
        
        if len(feature_maps) == 0:
            raise ValueError("No valid features to concatenate")
        
        # Concatenate along channel dimension
        feature_vector = np.concatenate(feature_maps, axis=-1)
        
        return feature_vector
    
    def compute_channel_correlation(self, coeffs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute correlation matrix between all channels.
        
        Args:
            coeffs: XLET-NSST coefficients
            
        Returns:
            Correlation matrix
        """
        # Extract all feature channels
        channels = []
        channel_names = []
        
        for name, feature in coeffs.items():
            if isinstance(feature, np.ndarray) and name not in ['scales', 'directions']:
                if len(feature.shape) == 3:
                    for c in range(feature.shape[2]):
                        channels.append(feature[:, :, c].flatten())
                        channel_names.append(f"{name}_C{c}")
                else:
                    channels.append(feature.flatten())
                    channel_names.append(name)
        
        # Stack channels
        channel_matrix = np.stack(channels, axis=0)
        
        # Compute correlation
        correlation_matrix = np.corrcoef(channel_matrix)
        
        return correlation_matrix, channel_names
    
    def select_diverse_channels(self, 
                                coeffs: Dict[str, np.ndarray], 
                                num_channels: int = 10,
                                correlation_threshold: float = 0.8) -> List[str]:
        """
        Select diverse channels with low correlation for better feature representation.
        
        Args:
            coeffs: XLET-NSST coefficients
            num_channels: Number of channels to select
            correlation_threshold: Maximum allowed correlation between selected channels
            
        Returns:
            List of selected channel names
        """
        # Compute all statistics
        analysis = self.analyze_all_channels(coeffs)
        
        # Get channels sorted by entropy (information content)
        ranked = self.get_best_channels_by_metric(analysis, 'entropy', top_k=len(analysis))
        
        selected = []
        
        for channel_name, _ in ranked:
            if len(selected) >= num_channels:
                break
            
            # Check correlation with already selected channels
            if len(selected) == 0:
                selected.append(channel_name)
            else:
                # Check if this channel is diverse enough
                is_diverse = True
                current_feature = coeffs[channel_name].flatten()
                
                for selected_name in selected:
                    selected_feature = coeffs[selected_name].flatten()
                    
                    # Handle different sizes
                    min_size = min(len(current_feature), len(selected_feature))
                    corr = np.corrcoef(current_feature[:min_size], 
                                      selected_feature[:min_size])[0, 1]
                    
                    if abs(corr) > correlation_threshold:
                        is_diverse = False
                        break
                
                if is_diverse:
                    selected.append(channel_name)
        
        return selected


class FeatureEvaluator:
    """
    Evaluate feature quality for semantic segmentation tasks.
    """
    
    def __init__(self):
        pass
    
    def compute_separability_index(self, 
                                   features: np.ndarray, 
                                   labels: Optional[np.ndarray] = None) -> float:
        """
        Compute class separability index for features.
        
        Args:
            features: Feature array (H, W, C) or (N, C)
            labels: Ground truth labels (H, W) or (N,)
            
        Returns:
            Separability score (higher is better)
        """
        if labels is None:
            # If no labels, use variance as proxy
            return np.var(features)
        
        # Reshape if needed
        if len(features.shape) == 3:
            h, w, c = features.shape
            features = features.reshape(-1, c)
            labels = labels.flatten()
        
        # Compute between-class and within-class variance
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            return 0.0
        
        # Overall mean
        global_mean = np.mean(features, axis=0)
        
        # Between-class scatter
        between_scatter = 0
        # Within-class scatter
        within_scatter = 0
        
        for label in unique_labels:
            mask = labels == label
            class_features = features[mask]
            
            if len(class_features) == 0:
                continue
            
            class_mean = np.mean(class_features, axis=0)
            
            # Between-class
            diff = class_mean - global_mean
            between_scatter += len(class_features) * np.dot(diff, diff)
            
            # Within-class
            within_scatter += np.sum((class_features - class_mean) ** 2)
        
        # Separability index (Fisher's criterion)
        if within_scatter > 0:
            separability = between_scatter / within_scatter
        else:
            separability = between_scatter
        
        return separability
    
    def compute_texture_score(self, feature: np.ndarray) -> float:
        """
        Compute texture richness score using GLCM-based metrics.
        
        Args:
            feature: Feature channel
            
        Returns:
            Texture score
        """
        # Normalize to 0-255
        feature_norm = ((feature - feature.min()) / (feature.max() - feature.min() + 1e-10) * 255).astype(np.uint8)
        
        # Compute gradient magnitude as texture indicator
        if len(feature.shape) == 3:
            feature_norm = cv2.cvtColor(feature_norm, cv2.COLOR_RGB2GRAY)
        
        grad_x = cv2.Sobel(feature_norm, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(feature_norm, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Texture score based on gradient statistics
        texture_score = np.mean(gradient_magnitude) + np.std(gradient_magnitude)
        
        return texture_score
    
    def compute_edge_preservation_score(self, feature: np.ndarray) -> float:
        """
        Compute how well the feature preserves edge information.
        
        Args:
            feature: Feature channel
            
        Returns:
            Edge preservation score
        """
        if len(feature.shape) == 3:
            feature = np.mean(feature, axis=2)
        
        # Normalize
        feature_norm = (feature - feature.min()) / (feature.max() - feature.min() + 1e-10)
        
        # Detect edges using Canny
        feature_uint8 = (feature_norm * 255).astype(np.uint8)
        edges = cv2.Canny(feature_uint8, 50, 150)
        
        # Edge density as score
        edge_score = np.sum(edges > 0) / edges.size
        
        return edge_score
    
    def rank_features_for_segmentation(self, 
                                       coeffs: Dict[str, np.ndarray],
                                       labels: Optional[np.ndarray] = None,
                                       weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        Rank features based on multiple criteria for segmentation quality.
        
        Args:
            coeffs: XLET-NSST coefficients
            labels: Optional ground truth labels for supervised ranking
            weights: Optional weights for different metrics
            
        Returns:
            Ranked list of (channel_name, score) tuples
        """
        if weights is None:
            weights = {
                'entropy': 0.25,
                'energy': 0.15,
                'texture': 0.25,
                'edge': 0.20,
                'separability': 0.15
            }
        
        extractor = FeatureExtractor()
        analysis = extractor.analyze_all_channels(coeffs)
        
        scores = {}
        
        for channel_name, feature in coeffs.items():
            if isinstance(feature, np.ndarray) and channel_name not in ['scales', 'directions']:
                score = 0
                
                # Statistical metrics
                if channel_name in analysis:
                    stats = analysis[channel_name]
                    score += weights['entropy'] * (stats['entropy'] / 10)  # Normalize
                    score += weights['energy'] * stats['energy']
                
                # Texture score
                texture_score = self.compute_texture_score(feature)
                score += weights['texture'] * (texture_score / 100)  # Normalize
                
                # Edge preservation
                edge_score = self.compute_edge_preservation_score(feature)
                score += weights['edge'] * edge_score
                
                # Separability (if labels provided)
                if labels is not None:
                    sep_score = self.compute_separability_index(feature, labels)
                    score += weights['separability'] * min(sep_score / 100, 1.0)  # Cap at 1.0
                
                scores[channel_name] = score
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked
