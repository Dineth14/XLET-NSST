"""
Usage Examples for XLET-NSST Feature Testing

This file contains practical examples for different use cases.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import cv2
from src.transforms.xlet_nsst import XLETNSST
from src.analysis.feature_analysis import FeatureExtractor, FeatureEvaluator
from src.visualization.visualize import FeatureVisualizer


def example_1_basic_transformation():
    """
    Example 1: Basic XLET-NSST transformation
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic XLET-NSST Transformation")
    print("="*70)
    
    # Load image
    image = cv2.imread('data/00004.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create transformer
    transformer = XLETNSST(levels=3, directions=8)
    
    # Apply transformation
    coeffs = transformer.transform(image)
    
    # Print results
    print(f"\nTransformation complete!")
    print(f"Input image shape: {image.shape}")
    print(f"Number of feature channels: {len([k for k in coeffs.keys() if isinstance(coeffs[k], np.ndarray)])}")
    
    # List some channels
    print("\nSample channels:")
    for i, key in enumerate(list(coeffs.keys())[:5]):
        if isinstance(coeffs[key], np.ndarray):
            print(f"  {key}: {coeffs[key].shape}")


def example_2_feature_ranking():
    """
    Example 2: Rank features for segmentation
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Feature Ranking")
    print("="*70)
    
    # Load and transform
    image = cv2.imread('data/00004.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformer = XLETNSST(levels=3, directions=8)
    coeffs = transformer.transform(image)
    
    # Rank features
    evaluator = FeatureEvaluator()
    rankings = evaluator.rank_features_for_segmentation(coeffs)
    
    print("\nTop 10 channels for semantic segmentation:")
    for idx, (channel, score) in enumerate(rankings[:10], 1):
        print(f"  {idx:2d}. {channel:30s} - Score: {score:.4f}")


def example_3_custom_weights():
    """
    Example 3: Custom evaluation weights
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Evaluation Weights")
    print("="*70)
    
    # Load and transform
    image = cv2.imread('data/00004.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformer = XLETNSST(levels=3, directions=8)
    coeffs = transformer.transform(image)
    
    evaluator = FeatureEvaluator()
    
    # Default weights
    print("\nWith default weights:")
    rankings_default = evaluator.rank_features_for_segmentation(coeffs)
    print("Top 3:", [name for name, _ in rankings_default[:3]])
    
    # Emphasize edge preservation
    print("\nWith edge-focused weights:")
    rankings_edge = evaluator.rank_features_for_segmentation(
        coeffs,
        weights={
            'entropy': 0.15,
            'energy': 0.10,
            'texture': 0.15,
            'edge': 0.50,
            'separability': 0.10
        }
    )
    print("Top 3:", [name for name, _ in rankings_edge[:3]])
    
    # Emphasize texture
    print("\nWith texture-focused weights:")
    rankings_texture = evaluator.rank_features_for_segmentation(
        coeffs,
        weights={
            'entropy': 0.15,
            'energy': 0.10,
            'texture': 0.50,
            'edge': 0.15,
            'separability': 0.10
        }
    )
    print("Top 3:", [name for name, _ in rankings_texture[:3]])


def example_4_diverse_selection():
    """
    Example 4: Select diverse, uncorrelated channels
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Diverse Channel Selection")
    print("="*70)
    
    # Load and transform
    image = cv2.imread('data/00004.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformer = XLETNSST(levels=3, directions=8)
    coeffs = transformer.transform(image)
    
    extractor = FeatureExtractor()
    
    # Select diverse channels
    diverse_channels = extractor.select_diverse_channels(
        coeffs, 
        num_channels=10,
        correlation_threshold=0.7
    )
    
    print(f"\nSelected {len(diverse_channels)} diverse channels:")
    for idx, channel in enumerate(diverse_channels, 1):
        print(f"  {idx}. {channel}")


def example_5_create_feature_vector():
    """
    Example 5: Create feature vector for segmentation model
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Create Feature Vector for Segmentation")
    print("="*70)
    
    # Load and transform
    image = cv2.imread('data/00004.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformer = XLETNSST(levels=3, directions=8)
    coeffs = transformer.transform(image)
    
    # Get best channels
    evaluator = FeatureEvaluator()
    rankings = evaluator.rank_features_for_segmentation(coeffs)
    
    # Select top 10
    selected_channels = [name for name, _ in rankings[:10]]
    
    # Create feature vector
    extractor = FeatureExtractor()
    features = extractor.create_feature_vector(
        coeffs,
        selected_channels=selected_channels,
        resize_shape=(256, 256)
    )
    
    print(f"\nFeature vector created!")
    print(f"  Shape: {features.shape}")
    print(f"  Channels: {features.shape[2]}")
    print(f"  Data type: {features.dtype}")
    print(f"  Value range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"\nThis can now be used as input to your segmentation model!")


def example_6_multi_scale_analysis():
    """
    Example 6: Analyze features at different scales
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Multi-Scale Analysis")
    print("="*70)
    
    # Load and transform
    image = cv2.imread('data/00004.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformer = XLETNSST(levels=3, directions=8)
    coeffs = transformer.transform(image)
    
    extractor = FeatureExtractor()
    
    # Analyze each scale
    for level in range(3):
        scale_features = extractor.extract_scale_features(coeffs, scale=level)
        print(f"\nScale {level} (Level {level}):")
        print(f"  Number of channels: {len(scale_features)}")
        print(f"  Channels: {list(scale_features.keys())[:3]}...")  # Show first 3


def example_7_directional_analysis():
    """
    Example 7: Analyze features by direction
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Directional Analysis")
    print("="*70)
    
    # Load and transform
    image = cv2.imread('data/00004.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformer = XLETNSST(levels=3, directions=8)
    coeffs = transformer.transform(image)
    
    extractor = FeatureExtractor()
    
    # Analyze specific directions
    for direction in [0, 2, 4, 6]:  # 0°, 45°, 90°, 135°
        dir_features = extractor.extract_direction_features(coeffs, direction=direction)
        angle = direction * 180 // 8
        print(f"\nDirection {direction} (~{angle}°):")
        print(f"  Number of channels: {len(dir_features)}")
        print(f"  Channels: {list(dir_features.keys())}")


def example_8_compare_parameters():
    """
    Example 8: Compare different decomposition parameters
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Compare Parameters")
    print("="*70)
    
    # Load image
    image = cv2.imread('data/00004.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    configs = [
        {'levels': 2, 'directions': 4, 'name': 'Fast (2 levels, 4 directions)'},
        {'levels': 3, 'directions': 8, 'name': 'Balanced (3 levels, 8 directions)'},
        {'levels': 4, 'directions': 16, 'name': 'Detailed (4 levels, 16 directions)'},
    ]
    
    print("\nComparing different configurations:")
    
    for config in configs:
        transformer = XLETNSST(levels=config['levels'], directions=config['directions'])
        coeffs = transformer.transform(image)
        num_channels = len([k for k in coeffs.keys() if isinstance(coeffs[k], np.ndarray)])
        
        print(f"\n{config['name']}:")
        print(f"  Channels generated: {num_channels}")


def example_9_statistics_comparison():
    """
    Example 9: Compare channels by different statistics
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: Statistics Comparison")
    print("="*70)
    
    # Load and transform
    image = cv2.imread('data/00004.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformer = XLETNSST(levels=3, directions=8)
    coeffs = transformer.transform(image)
    
    extractor = FeatureExtractor()
    analysis = extractor.analyze_all_channels(coeffs)
    
    metrics = ['entropy', 'energy', 'std', 'dynamic_range']
    
    print("\nTop 3 channels by different metrics:\n")
    
    for metric in metrics:
        top_3 = extractor.get_best_channels_by_metric(analysis, metric, top_k=3)
        print(f"{metric.upper()}:")
        for idx, (channel, value) in enumerate(top_3, 1):
            print(f"  {idx}. {channel}: {value:.4f}")
        print()


def run_all_examples():
    """Run all examples"""
    examples = [
        example_1_basic_transformation,
        example_2_feature_ranking,
        example_3_custom_weights,
        example_4_diverse_selection,
        example_5_create_feature_vector,
        example_6_multi_scale_analysis,
        example_7_directional_analysis,
        example_8_compare_parameters,
        example_9_statistics_comparison,
    ]
    
    print("\n" + "="*70)
    print("XLET-NSST USAGE EXAMPLES")
    print("="*70)
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run XLET-NSST usage examples')
    parser.add_argument('--example', type=int, help='Run specific example (1-9)', default=None)
    
    args = parser.parse_args()
    
    if args.example:
        examples = {
            1: example_1_basic_transformation,
            2: example_2_feature_ranking,
            3: example_3_custom_weights,
            4: example_4_diverse_selection,
            5: example_5_create_feature_vector,
            6: example_6_multi_scale_analysis,
            7: example_7_directional_analysis,
            8: example_8_compare_parameters,
            9: example_9_statistics_comparison,
        }
        
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Example {args.example} not found. Choose 1-9.")
    else:
        run_all_examples()
