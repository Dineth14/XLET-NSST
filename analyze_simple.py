"""
Simple Feature Analysis without visualization dependencies
Run this first to get the best channels, then visualize separately
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import cv2
import json
from src.transforms.xlet_nsst import XLETNSST
from src.analysis.feature_analysis import FeatureExtractor, FeatureEvaluator


def analyze_image(image_path: str, output_file: str = 'best_channels_results.json'):
    """
    Analyze image and find best channels without visualization.
    
    Args:
        image_path: Path to image
        output_file: Where to save results
    """
    print(f"\n{'='*70}")
    print("XLET-NSST FEATURE CHANNEL ANALYSIS")
    print(f"{'='*70}\n")
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✓ Image loaded: {image.shape}")
    
    # Initialize XLET-NSST
    print("\nApplying XLET-NSST transformation...")
    print("  Parameters: levels=3, directions=8")
    transformer = XLETNSST(levels=3, directions=8)
    
    # Transform
    coeffs = transformer.transform(image)
    num_channels = len([k for k in coeffs.keys() if isinstance(coeffs[k], np.ndarray)])
    print(f"✓ Generated {num_channels} feature channels")
    
    # Analyze
    print("\nAnalyzing feature statistics...")
    extractor = FeatureExtractor()
    analysis = extractor.analyze_all_channels(coeffs)
    print(f"✓ Analysis complete")
    
    # Rank for segmentation
    print("\nRanking channels for semantic segmentation...")
    evaluator = FeatureEvaluator()
    rankings = evaluator.rank_features_for_segmentation(coeffs)
    print(f"✓ Ranking complete")
    
    # Get best by different metrics
    print("\nComputing best channels by different metrics...")
    best_entropy = extractor.get_best_channels_by_metric(analysis, 'entropy', top_k=10)
    best_energy = extractor.get_best_channels_by_metric(analysis, 'energy', top_k=10)
    best_std = extractor.get_best_channels_by_metric(analysis, 'std', top_k=10)
    
    # Select diverse channels
    print("Selecting diverse channels...")
    diverse_channels = extractor.select_diverse_channels(coeffs, num_channels=10)
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS - TOP FEATURE CHANNELS FOR SEMANTIC SEGMENTATION")
    print(f"{'='*70}\n")
    
    print("Overall Rankings (Combined metrics):")
    print("-" * 70)
    for idx, (channel, score) in enumerate(rankings[:15], 1):
        print(f"{idx:2d}. {channel:35s} Score: {score:.4f}")
    
    print(f"\n{'='*70}")
    print("Best by Information Content (Entropy):")
    print("-" * 70)
    for idx, (channel, value) in enumerate(best_entropy[:5], 1):
        print(f"{idx}. {channel:35s} {value:.4f}")
    
    print(f"\n{'='*70}")
    print("Best by Signal Energy:")
    print("-" * 70)
    for idx, (channel, value) in enumerate(best_energy[:5], 1):
        print(f"{idx}. {channel:35s} {value:.4f}")
    
    print(f"\n{'='*70}")
    print("Best by Variance (Standard Deviation):")
    print("-" * 70)
    for idx, (channel, value) in enumerate(best_std[:5], 1):
        print(f"{idx}. {channel:35s} {value:.4f}")
    
    print(f"\n{'='*70}")
    print("Diverse Channels (Low Correlation):")
    print("-" * 70)
    for idx, channel in enumerate(diverse_channels, 1):
        print(f"{idx}. {channel}")
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    # Convert to serializable format
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy(item) for item in obj)
        return obj
    
    results = {
        'image_path': image_path,
        'image_shape': list(image.shape),
        'num_channels': num_channels,
        'top_15_overall': convert_numpy(rankings[:15]),
        'top_5_entropy': convert_numpy(best_entropy[:5]),
        'top_5_energy': convert_numpy(best_energy[:5]),
        'top_5_std': convert_numpy(best_std[:5]),
        'diverse_channels': diverse_channels,
        'recommended_for_segmentation': [name for name, _ in rankings[:10]]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_file}")
    
    # Print recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS FOR YOUR SEGMENTATION PIPELINE")
    print(f"{'='*70}\n")
    print("Top 10 Recommended Channels to Use:")
    print()
    for idx, (channel, score) in enumerate(rankings[:10], 1):
        print(f"  {idx:2d}. {channel}")
    
    print(f"\n{'='*70}")
    print("Usage Example:")
    print(f"{'='*70}")
    print("""
from src.transforms.xlet_nsst import XLETNSST
from src.analysis.feature_analysis import FeatureExtractor

# Transform your image
transformer = XLETNSST(levels=3, directions=8)
coeffs = transformer.transform(your_image)

# Select recommended channels
selected = [
""")
    for channel, _ in rankings[:10]:
        print(f"    '{channel}',")
    print("""]

# Create feature vector
extractor = FeatureExtractor()
features = extractor.create_feature_vector(
    coeffs, 
    selected_channels=selected,
    resize_shape=(256, 256)  # Adjust as needed
)

# features.shape will be (256, 256, 10)
# Use this as input to your segmentation model
""")
    
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze XLET-NSST features without visualization')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--output', type=str, default='best_channels_results.json', 
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    analyze_image(args.image, args.output)
