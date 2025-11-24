"""
Visualization Tools for XLET-NSST Feature Analysis

This module provides comprehensive visualization utilities for inspecting
and comparing different frequency channels from XLET-NSST decomposition.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import cv2
from pathlib import Path


class FeatureVisualizer:
    """
    Visualize XLET-NSST features and analysis results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        sns.set_style("whitegrid")
    
    def visualize_all_subbands(self, 
                               coeffs: Dict[str, np.ndarray], 
                               save_path: Optional[str] = None,
                               cmap: str = 'viridis') -> plt.Figure:
        """
        Visualize all subbands in a grid layout.
        
        Args:
            coeffs: XLET-NSST coefficients
            save_path: Optional path to save figure
            cmap: Colormap for visualization
            
        Returns:
            Matplotlib figure
        """
        # Count number of subbands
        subband_keys = [k for k in coeffs.keys() 
                       if isinstance(coeffs[k], np.ndarray) and k not in ['scales', 'directions']]
        
        num_subbands = len(subband_keys)
        
        # Determine grid size
        cols = min(4, num_subbands)
        rows = (num_subbands + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        
        if num_subbands == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, key in enumerate(subband_keys):
            feature = coeffs[key]
            
            # Handle multi-channel
            if len(feature.shape) == 3:
                # Show first channel or average
                if feature.shape[2] == 3:
                    display = feature
                else:
                    display = np.mean(feature, axis=2)
            else:
                display = feature
            
            # Normalize for display
            display_norm = (display - display.min()) / (display.max() - display.min() + 1e-10)
            
            axes[idx].imshow(display_norm, cmap=cmap)
            axes[idx].set_title(key, fontsize=10)
            axes[idx].axis('off')
        
        # Hide unused axes
        for idx in range(num_subbands, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_scale_decomposition(self, 
                                     coeffs: Dict[str, np.ndarray],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize decomposition across scales.
        
        Args:
            coeffs: XLET-NSST coefficients
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        scales = coeffs.get('scales', [])
        directions = coeffs.get('directions', [])
        
        if not scales:
            return None
        
        num_scales = len(scales)
        num_directions = min(4, len(directions))  # Show first 4 directions
        
        fig, axes = plt.subplots(num_scales, num_directions + 1, 
                                figsize=(4 * (num_directions + 1), 3 * num_scales))
        
        if num_scales == 1:
            axes = axes.reshape(1, -1)
        
        for scale_idx, scale in enumerate(scales):
            # Show lowpass for this scale level
            if scale == scales[-1] and 'lowpass' in coeffs:
                lowpass = coeffs['lowpass']
                if len(lowpass.shape) == 3:
                    lowpass = np.mean(lowpass, axis=2)
                
                lowpass_norm = (lowpass - lowpass.min()) / (lowpass.max() - lowpass.min() + 1e-10)
                axes[scale_idx, 0].imshow(lowpass_norm, cmap='gray')
                axes[scale_idx, 0].set_title(f'Lowpass (Scale {scale})', fontsize=10)
                axes[scale_idx, 0].axis('off')
            else:
                axes[scale_idx, 0].axis('off')
            
            # Show directional subbands
            for dir_idx in range(num_directions):
                if dir_idx < len(directions):
                    direction = directions[dir_idx]
                    key = f'highpass_L{scale}_D{direction}'
                    
                    if key in coeffs:
                        feature = coeffs[key]
                        
                        if len(feature.shape) == 3:
                            feature = np.mean(feature, axis=2)
                        
                        feature_norm = (feature - feature.min()) / (feature.max() - feature.min() + 1e-10)
                        axes[scale_idx, dir_idx + 1].imshow(feature_norm, cmap='viridis')
                        axes[scale_idx, dir_idx + 1].set_title(
                            f'L{scale}_D{direction} ({direction * 180 // len(directions)}°)', 
                            fontsize=9
                        )
                        axes[scale_idx, dir_idx + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_feature_statistics(self, 
                                analysis: Dict[str, Dict[str, float]],
                                metrics: Optional[List[str]] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot statistical comparison of different features.
        
        Args:
            analysis: Feature analysis results
            metrics: List of metrics to plot (default: ['entropy', 'energy', 'std'])
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['entropy', 'energy', 'std', 'dynamic_range']
        
        # Extract data
        channel_names = [k for k in analysis.keys() if k not in ['scales', 'directions']]
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            values = []
            labels = []
            
            for channel_name in channel_names:
                if metric in analysis[channel_name]:
                    values.append(analysis[channel_name][metric])
                    # Shorten label for display
                    label = channel_name.replace('highpass_', 'H_')
                    labels.append(label)
            
            # Sort by value
            sorted_pairs = sorted(zip(values, labels), reverse=True)
            values, labels = zip(*sorted_pairs) if sorted_pairs else ([], [])
            
            # Plot top 20
            top_n = min(20, len(values))
            
            axes[idx].barh(range(top_n), values[:top_n])
            axes[idx].set_yticks(range(top_n))
            axes[idx].set_yticklabels(labels[:top_n], fontsize=8)
            axes[idx].set_xlabel(metric.capitalize())
            axes[idx].set_title(f'Top {top_n} Channels by {metric.capitalize()}')
            axes[idx].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(self, 
                                correlation_matrix: np.ndarray,
                                channel_names: List[str],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix between channels.
        
        Args:
            correlation_matrix: Correlation matrix
            channel_names: List of channel names
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Limit to reasonable size
        max_channels = 30
        if len(channel_names) > max_channels:
            correlation_matrix = correlation_matrix[:max_channels, :max_channels]
            channel_names = channel_names[:max_channels]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Shorten names
        short_names = [name.replace('highpass_', 'H_') for name in channel_names]
        
        sns.heatmap(correlation_matrix, 
                   xticklabels=short_names,
                   yticklabels=short_names,
                   cmap='RdBu_r',
                   center=0,
                   vmin=-1, vmax=1,
                   square=True,
                   ax=ax,
                   cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Channel Correlation Matrix')
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_ranking_results(self, 
                                  rankings: List[Tuple[str, float]],
                                  title: str = "Feature Channel Rankings",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize ranking results as bar chart.
        
        Args:
            rankings: List of (channel_name, score) tuples
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Show top 15
        top_n = min(15, len(rankings))
        top_rankings = rankings[:top_n]
        
        names = [name.replace('highpass_', 'H_') for name, _ in top_rankings]
        scores = [score for _, score in top_rankings]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(range(len(names)), scores, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(names))))
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Score')
        ax.set_title(title)
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score, i, f' {score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_comparison(self,
                            original_image: np.ndarray,
                            coeffs: Dict[str, np.ndarray],
                            selected_channels: List[str],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare original image with selected feature channels.
        
        Args:
            original_image: Original input image
            coeffs: XLET-NSST coefficients
            selected_channels: List of channel names to compare
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        num_channels = len(selected_channels)
        cols = min(4, num_channels + 1)
        rows = (num_channels + 2) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten() if num_channels > 0 else [axes]
        
        # Show original
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            display_orig = original_image
            if display_orig.max() <= 1.0:
                display_orig = (display_orig * 255).astype(np.uint8)
        else:
            display_orig = original_image
            if len(display_orig.shape) == 3:
                display_orig = np.mean(display_orig, axis=2)
        
        axes[0].imshow(display_orig, cmap='gray' if len(display_orig.shape) == 2 else None)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show selected channels
        for idx, channel_name in enumerate(selected_channels):
            if channel_name in coeffs:
                feature = coeffs[channel_name]
                
                if len(feature.shape) == 3:
                    feature = np.mean(feature, axis=2)
                
                # Normalize
                feature_norm = (feature - feature.min()) / (feature.max() - feature.min() + 1e-10)
                
                axes[idx + 1].imshow(feature_norm, cmap='viridis')
                axes[idx + 1].set_title(channel_name.replace('highpass_', 'H_'), fontsize=9)
                axes[idx + 1].axis('off')
        
        # Hide unused axes
        for idx in range(num_channels + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_feature_montage(self,
                               coeffs: Dict[str, np.ndarray],
                               output_path: str,
                               grid_size: Optional[Tuple[int, int]] = None):
        """
        Create a montage of all features saved as a single image.
        
        Args:
            coeffs: XLET-NSST coefficients
            output_path: Path to save montage image
            grid_size: Optional (rows, cols) for grid layout
        """
        # Get all feature channels
        channels = [(k, v) for k, v in coeffs.items() 
                   if isinstance(v, np.ndarray) and k not in ['scales', 'directions']]
        
        if not channels:
            return
        
        # Determine grid size
        if grid_size is None:
            n = len(channels)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
        else:
            rows, cols = grid_size
        
        # Get maximum dimensions
        max_h = max(v.shape[0] for _, v in channels)
        max_w = max(v.shape[1] for _, v in channels)
        
        # Create montage
        montage = np.zeros((rows * max_h, cols * max_w), dtype=np.float32)
        
        for idx, (name, feature) in enumerate(channels):
            if idx >= rows * cols:
                break
            
            row = idx // cols
            col = idx % cols
            
            # Prepare feature
            if len(feature.shape) == 3:
                display = np.mean(feature, axis=2)
            else:
                display = feature
            
            # Normalize
            display = (display - display.min()) / (display.max() - display.min() + 1e-10)
            
            # Resize if needed
            if display.shape != (max_h, max_w):
                display = cv2.resize(display, (max_w, max_h))
            
            # Place in montage
            montage[row*max_h:(row+1)*max_h, col*max_w:(col+1)*max_w] = display
        
        # Save
        montage_uint8 = (montage * 255).astype(np.uint8)
        cv2.imwrite(output_path, montage_uint8)
