"""
Main Testing Pipeline for XLET-NSST Feature Channel Analysis

This script provides a comprehensive pipeline to:
1. Load and preprocess images
2. Apply XLET-NSST transformation
3. Extract and analyze all feature channels
4. Rank channels by segmentation quality
5. Generate visualizations and reports
"""

import numpy as np
import cv2
from pathlib import Path
import json
import argparse
from typing import Optional, List, Dict
import sys
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transforms.xlet_nsst import XLETNSST
from src.analysis.feature_analysis import FeatureExtractor, FeatureEvaluator
from src.visualization.visualize import FeatureVisualizer


class XLETNSSTTester:
    """
    Comprehensive tester for XLET-NSST feature channels.
    """
    
    def __init__(self, 
                 levels: int = 3,
                 directions: int = 8,
                 output_dir: str = 'results'):
        """
        Initialize tester.
        
        Args:
            levels: Number of decomposition levels
            directions: Number of directional subbands
            output_dir: Directory to save results
        """
        self.transformer = XLETNSST(levels=levels, directions=directions)
        self.extractor = FeatureExtractor()
        self.evaluator = FeatureEvaluator()
        self.visualizer = FeatureVisualizer()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Subdirectories
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'statistics').mkdir(exist_ok=True)
        (self.output_dir / 'rankings').mkdir(exist_ok=True)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image array
        """
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def process_single_image(self, 
                            image_path: str,
                            image_name: Optional[str] = None,
                            mask_path: Optional[str] = None) -> Dict:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            image_name: Optional name for outputs (default: filename)
            mask_path: Optional path to segmentation mask for supervised analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        if image_name is None:
            image_name = Path(image_path).stem
        
        print(f"\n{'='*60}")
        print(f"Processing: {image_name}")
        print(f"{'='*60}")
        
        # Load image
        print("Loading image...")
        image = self.load_image(image_path)
        print(f"Image shape: {image.shape}")
        
        # Load mask if provided
        mask = None
        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            print(f"Mask shape: {mask.shape}")
        
        # Apply XLET-NSST transformation
        print("Applying XLET-NSST transformation...")
        coeffs = self.transformer.transform(image)
        print(f"Generated {len([k for k in coeffs.keys() if isinstance(coeffs[k], np.ndarray)])} feature channels")
        
        # Extract features
        print("Extracting features...")
        features = self.extractor.extract_all_features(coeffs)
        
        # Analyze features
        print("Analyzing feature statistics...")
        analysis = self.extractor.analyze_all_channels(coeffs)
        
        # Rank features
        print("Ranking features for segmentation...")
        rankings = self.evaluator.rank_features_for_segmentation(coeffs, labels=mask)
        
        # Get best channels by different metrics
        best_entropy = self.extractor.get_best_channels_by_metric(analysis, 'entropy', top_k=10)
        best_energy = self.extractor.get_best_channels_by_metric(analysis, 'energy', top_k=10)
        best_std = self.extractor.get_best_channels_by_metric(analysis, 'std', top_k=10)
        
        # Select diverse channels
        print("Selecting diverse feature channels...")
        diverse_channels = self.extractor.select_diverse_channels(coeffs, num_channels=10)
        
        # Compute correlation
        print("Computing channel correlations...")
        corr_matrix, channel_names = self.extractor.compute_channel_correlation(coeffs)
        
        # Generate visualizations
        print("Generating visualizations...")
        self._generate_visualizations(image, coeffs, analysis, rankings, 
                                     corr_matrix, channel_names, diverse_channels, image_name)
        
        # Save statistics
        print("Saving statistics...")
        self._save_statistics(analysis, rankings, best_entropy, best_energy, 
                            best_std, diverse_channels, image_name)
        
        # Compile results
        results = {
            'image_name': image_name,
            'image_shape': image.shape,
            'num_channels': len(features),
            'rankings': rankings,
            'best_entropy': best_entropy,
            'best_energy': best_energy,
            'best_std': best_std,
            'diverse_channels': diverse_channels,
            'analysis': analysis
        }
        
        print(f"\n✓ Processing complete for {image_name}")
        print(f"Results saved to: {self.output_dir}")
        
        return results
    
    def process_directory(self, 
                         image_dir: str,
                         mask_dir: Optional[str] = None,
                         extensions: List[str] = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']) -> Dict:
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Optional directory containing masks
            extensions: List of valid image extensions
            
        Returns:
            Dictionary containing aggregated results
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir) if mask_dir else None
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
        
        if not image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"\nFound {len(image_files)} images to process")
        
        all_results = []
        
        for image_path in tqdm(image_files, desc="Processing images"):
            # Find corresponding mask
            mask_path = None
            if mask_dir:
                mask_candidates = list(mask_dir.glob(f"{image_path.stem}.*"))
                if mask_candidates:
                    mask_path = str(mask_candidates[0])
            
            try:
                results = self.process_single_image(
                    str(image_path),
                    image_name=image_path.stem,
                    mask_path=mask_path
                )
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue
        
        # Aggregate results
        print("\nAggregating results across all images...")
        aggregated = self._aggregate_results(all_results)
        
        # Save aggregated results
        self._save_aggregated_results(aggregated)
        
        return aggregated
    
    def _generate_visualizations(self, 
                                image: np.ndarray,
                                coeffs: Dict[str, np.ndarray],
                                analysis: Dict,
                                rankings: List,
                                corr_matrix: np.ndarray,
                                channel_names: List[str],
                                diverse_channels: List[str],
                                image_name: str):
        """Generate all visualizations."""
        viz_dir = self.output_dir / 'visualizations' / image_name
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # All subbands
        self.visualizer.visualize_all_subbands(
            coeffs, 
            save_path=str(viz_dir / 'all_subbands.png')
        )
        plt.close()
        
        # Scale decomposition
        self.visualizer.visualize_scale_decomposition(
            coeffs,
            save_path=str(viz_dir / 'scale_decomposition.png')
        )
        plt.close()
        
        # Feature statistics
        self.visualizer.plot_feature_statistics(
            analysis,
            save_path=str(viz_dir / 'feature_statistics.png')
        )
        plt.close()
        
        # Correlation matrix (limit size)
        if len(channel_names) > 0:
            self.visualizer.plot_correlation_matrix(
                corr_matrix,
                channel_names,
                save_path=str(viz_dir / 'correlation_matrix.png')
            )
            plt.close()
        
        # Rankings
        self.visualizer.visualize_ranking_results(
            rankings,
            title=f"Feature Rankings for {image_name}",
            save_path=str(viz_dir / 'rankings.png')
        )
        plt.close()
        
        # Comparison with diverse channels
        self.visualizer.visualize_comparison(
            image,
            coeffs,
            diverse_channels[:8],  # Top 8
            save_path=str(viz_dir / 'best_channels_comparison.png')
        )
        plt.close()
        
        # Feature montage
        self.visualizer.create_feature_montage(
            coeffs,
            str(viz_dir / 'feature_montage.png')
        )
    
    def _save_statistics(self,
                        analysis: Dict,
                        rankings: List,
                        best_entropy: List,
                        best_energy: List,
                        best_std: List,
                        diverse_channels: List[str],
                        image_name: str):
        """Save statistical results to JSON."""
        stats_dir = self.output_dir / 'statistics'
        
        # Convert numpy types to native Python types for JSON
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_native(item) for item in obj)
            return obj
        
        stats = {
            'analysis': convert_to_native(analysis),
            'rankings': convert_to_native(rankings[:20]),  # Top 20
            'best_by_entropy': convert_to_native(best_entropy),
            'best_by_energy': convert_to_native(best_energy),
            'best_by_std': convert_to_native(best_std),
            'diverse_channels': diverse_channels
        }
        
        with open(stats_dir / f'{image_name}_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results across all images."""
        if not all_results:
            return {}
        
        # Find most common top channels
        channel_counts = {}
        
        for results in all_results:
            # Count top 10 from rankings
            for channel_name, _ in results['rankings'][:10]:
                channel_counts[channel_name] = channel_counts.get(channel_name, 0) + 1
        
        # Sort by frequency
        sorted_channels = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)
        
        aggregated = {
            'total_images': len(all_results),
            'most_common_top_channels': sorted_channels[:20],
            'average_num_channels': np.mean([r['num_channels'] for r in all_results]),
            'individual_results': all_results
        }
        
        return aggregated
    
    def _save_aggregated_results(self, aggregated: Dict):
        """Save aggregated results."""
        # Remove detailed individual results for summary
        summary = {k: v for k, v in aggregated.items() if k != 'individual_results'}
        
        with open(self.output_dir / 'aggregated_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create a text report
        report = self._generate_text_report(aggregated)
        with open(self.output_dir / 'REPORT.txt', 'w') as f:
            f.write(report)
    
    def _generate_text_report(self, aggregated: Dict) -> str:
        """Generate a human-readable text report."""
        report = []
        report.append("=" * 80)
        report.append("XLET-NSST FEATURE CHANNEL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"Total Images Processed: {aggregated['total_images']}")
        report.append(f"Average Number of Channels: {aggregated['average_num_channels']:.1f}")
        report.append("")
        
        report.append("=" * 80)
        report.append("TOP FEATURE CHANNELS FOR SEMANTIC SEGMENTATION")
        report.append("=" * 80)
        report.append("")
        report.append("Channels ranked by frequency in top 10 across all images:")
        report.append("")
        
        for idx, (channel, count) in enumerate(aggregated['most_common_top_channels'], 1):
            percentage = (count / aggregated['total_images']) * 100
            report.append(f"{idx:2d}. {channel:30s} - Appeared in top 10 of {count}/{aggregated['total_images']} images ({percentage:.1f}%)")
        
        report.append("")
        report.append("=" * 80)
        report.append("RECOMMENDATIONS FOR SEMANTIC SEGMENTATION")
        report.append("=" * 80)
        report.append("")
        report.append("Based on the analysis, the following feature channels are recommended:")
        report.append("")
        
        top_5 = aggregated['most_common_top_channels'][:5]
        for idx, (channel, _) in enumerate(top_5, 1):
            report.append(f"{idx}. {channel}")
        
        report.append("")
        report.append("These channels consistently showed:")
        report.append("  - High information content (entropy)")
        report.append("  - Strong edge preservation")
        report.append("  - Good texture representation")
        report.append("  - Low correlation with other channels (diversity)")
        report.append("")
        
        report.append("=" * 80)
        report.append("For best results in semantic segmentation:")
        report.append("  1. Use a combination of lowpass and selected highpass channels")
        report.append("  2. Include channels from multiple scales for multi-resolution analysis")
        report.append("  3. Select diverse directional channels to capture various edge orientations")
        report.append("  4. Consider the specific characteristics of your dataset")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test XLET-NSST feature channels for semantic segmentation'
    )
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--mask', type=str, help='Path to segmentation mask (for single image)')
    parser.add_argument('--mask_dir', type=str, help='Path to directory of masks')
    parser.add_argument('--levels', type=int, default=3, help='Number of decomposition levels')
    parser.add_argument('--directions', type=int, default=8, help='Number of directions')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create tester
    tester = XLETNSSTTester(
        levels=args.levels,
        directions=args.directions,
        output_dir=args.output
    )
    
    # Process
    if args.image:
        tester.process_single_image(args.image, mask_path=args.mask)
    elif args.image_dir:
        tester.process_directory(args.image_dir, mask_dir=args.mask_dir)
    else:
        print("Please specify either --image or --image_dir")
        parser.print_help()


# Make matplotlib work in script mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    main()
