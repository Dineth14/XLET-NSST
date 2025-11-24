"""
Analyze multiple images to find consistently good channels
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import cv2
import json
from collections import Counter
from src.transforms.xlet_nsst import XLETNSST
from src.analysis.feature_analysis import FeatureExtractor, FeatureEvaluator


def analyze_multiple_images(image_dir: str, output_file: str = 'multi_image_results.json'):
    """Analyze multiple images to find consistently best channels."""
    
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"MULTI-IMAGE XLET-NSST ANALYSIS")
    print(f"{'='*70}\n")
    print(f"Found {len(image_files)} images to analyze")
    print()
    
    # Store results from all images
    all_rankings = []
    channel_top10_count = Counter()
    channel_top5_count = Counter()
    
    transformer = XLETNSST(levels=3, directions=8)
    extractor = FeatureExtractor()
    evaluator = FeatureEvaluator()
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
        
        try:
            # Load
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"  ✗ Failed to load")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Transform
            coeffs = transformer.transform(image)
            
            # Rank
            rankings = evaluator.rank_features_for_segmentation(coeffs)
            all_rankings.append({
                'image': img_path.name,
                'rankings': [(ch, float(sc)) for ch, sc in rankings[:15]]
            })
            
            # Count top channels
            for channel, _ in rankings[:10]:
                channel_top10_count[channel] += 1
            
            for channel, _ in rankings[:5]:
                channel_top5_count[channel] += 1
            
            print(f"  ✓ Complete - Top channel: {rankings[0][0]}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Aggregate results
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS ACROSS ALL IMAGES")
    print(f"{'='*70}\n")
    
    print("Most Frequently Appearing in Top 10:")
    print("-" * 70)
    for channel, count in channel_top10_count.most_common(15):
        percentage = (count / len(all_rankings)) * 100
        print(f"{channel:35s} {count:2d}/{len(all_rankings)} images ({percentage:5.1f}%)")
    
    print(f"\n{'='*70}")
    print("Most Frequently Appearing in Top 5:")
    print("-" * 70)
    for channel, count in channel_top5_count.most_common(10):
        percentage = (count / len(all_rankings)) * 100
        print(f"{channel:35s} {count:2d}/{len(all_rankings)} images ({percentage:5.1f}%)")
    
    # Calculate average rankings
    print(f"\n{'='*70}")
    print("Computing Average Rankings...")
    print("-" * 70)
    
    channel_scores = {}
    channel_counts = {}
    
    for result in all_rankings:
        for channel, score in result['rankings']:
            if channel not in channel_scores:
                channel_scores[channel] = 0
                channel_counts[channel] = 0
            channel_scores[channel] += score
            channel_counts[channel] += 1
    
    avg_rankings = []
    for channel in channel_scores:
        avg_score = channel_scores[channel] / channel_counts[channel]
        avg_rankings.append((channel, avg_score, channel_counts[channel]))
    
    avg_rankings.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop Channels by Average Score:")
    print("-" * 70)
    for idx, (channel, avg_score, count) in enumerate(avg_rankings[:15], 1):
        print(f"{idx:2d}. {channel:35s} Avg: {avg_score:.4f} (in {count} images)")
    
    # Final recommendations
    print(f"\n{'='*70}")
    print("FINAL RECOMMENDATIONS FOR SEMANTIC SEGMENTATION")
    print(f"{'='*70}\n")
    
    # Get channels that appear in top 10 at least 50% of the time
    threshold = len(all_rankings) * 0.5
    recommended = [ch for ch, cnt in channel_top10_count.most_common() if cnt >= threshold]
    
    print(f"Recommended Channels (appeared in top 10 of ≥50% images):")
    print()
    for idx, channel in enumerate(recommended[:12], 1):
        count = channel_top10_count[channel]
        pct = (count / len(all_rankings)) * 100
        print(f"  {idx:2d}. {channel:35s} ({pct:.0f}% of images)")
    
    # Save results
    results = {
        'num_images_analyzed': len(all_rankings),
        'top_10_frequency': dict(channel_top10_count.most_common(20)),
        'top_5_frequency': dict(channel_top5_count.most_common(15)),
        'average_rankings': [(ch, sc) for ch, sc, _ in avg_rankings[:20]],
        'recommended_channels': recommended[:12],
        'individual_results': all_rankings
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print usage
    print(f"\n{'='*70}")
    print("USAGE IN YOUR CODE")
    print(f"{'='*70}")
    print("""
# Use these recommended channels for your segmentation:
selected_channels = [""")
    for channel in recommended[:12]:
        print(f"    '{channel}',")
    print("""]

# Then extract features:
from src.transforms.xlet_nsst import XLETNSST
from src.analysis.feature_analysis import FeatureExtractor

transformer = XLETNSST(levels=3, directions=8)
extractor = FeatureExtractor()

coeffs = transformer.transform(your_image)
features = extractor.create_feature_vector(
    coeffs, 
    selected_channels=selected_channels,
    resize_shape=(256, 256)
)
# Use features for segmentation
""")
    
    print(f"\n{'='*70}\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data', help='Directory with images')
    parser.add_argument('--output', type=str, default='multi_image_results.json', help='Output file')
    
    args = parser.parse_args()
    
    analyze_multiple_images(args.image_dir, args.output)
