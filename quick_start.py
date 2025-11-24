"""
Quick Start Example for XLET-NSST Feature Testing

This script provides a simple example to get started with testing XLET-NSST features.
"""

import sys
from pathlib import Path
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.transforms.xlet_nsst import XLETNSST
from src.analysis.feature_analysis import FeatureExtractor, FeatureEvaluator
from src.visualization.visualize import FeatureVisualizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def quick_test(image_path: str, output_dir: str = 'quick_results'):
    """
    Quick test of XLET-NSST on a single image.
    
    Args:
        image_path: Path to test image
        output_dir: Where to save results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image.shape}")
    
    # Initialize XLET-NSST
    print("\nInitializing XLET-NSST transformer...")
    transformer = XLETNSST(levels=3, directions=8)
    
    # Transform
    print("Applying transformation...")
    coeffs = transformer.transform(image)
    print(f"Generated {len([k for k in coeffs.keys() if isinstance(coeffs[k], np.ndarray)])} feature channels")
    
    # Analyze
    print("\nAnalyzing features...")
    extractor = FeatureExtractor()
    analysis = extractor.analyze_all_channels(coeffs)
    
    # Get best channels
    best_entropy = extractor.get_best_channels_by_metric(analysis, 'entropy', top_k=5)
    
    print("\nTop 5 channels by entropy (information content):")
    for idx, (channel, value) in enumerate(best_entropy, 1):
        print(f"  {idx}. {channel}: {value:.4f}")
    
    # Rank for segmentation
    print("\nRanking for segmentation quality...")
    evaluator = FeatureEvaluator()
    rankings = evaluator.rank_features_for_segmentation(coeffs)
    
    print("\nTop 5 channels for semantic segmentation:")
    for idx, (channel, score) in enumerate(rankings[:5], 1):
        print(f"  {idx}. {channel}: {score:.4f}")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualizer = FeatureVisualizer()
    
    # All subbands
    visualizer.visualize_all_subbands(coeffs, save_path=str(output_path / 'all_subbands.png'))
    plt.close()
    
    # Rankings
    visualizer.visualize_ranking_results(rankings[:15], 
                                        save_path=str(output_path / 'rankings.png'))
    plt.close()
    
    # Statistics
    visualizer.plot_feature_statistics(analysis, 
                                      save_path=str(output_path / 'statistics.png'))
    plt.close()
    
    print(f"\n✓ Results saved to: {output_path}")
    print("\nRecommended channels for your semantic segmentation pipeline:")
    for idx, (channel, _) in enumerate(rankings[:3], 1):
        print(f"  {idx}. {channel}")


if __name__ == '__main__':
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Quick test of XLET-NSST features')
    parser.add_argument('image', type=str, help='Path to test image')
    parser.add_argument('--output', type=str, default='quick_results', help='Output directory')
    
    args = parser.parse_args()
    
    quick_test(args.image, args.output)
