"""
Boundary Detection Analysis for XLET-NSST Channels

This script performs edge/boundary detection on each decomposition channel
to identify which frequency subbands best capture boundaries for segmentation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
import json
from collections import defaultdict
from src.transforms.xlet_nsst import XLETNSST
from src.analysis.feature_analysis import FeatureExtractor


class BoundaryDetector:
    """Detect and evaluate boundaries in frequency channels."""
    
    def __init__(self):
        self.edge_methods = {
            'canny': self._canny_edges,
            'sobel': self._sobel_edges,
            'laplacian': self._laplacian_edges,
        }
    
    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalize channel to 0-255 uint8."""
        if len(channel.shape) == 3:
            channel = np.mean(channel, axis=2)
        
        channel_min = channel.min()
        channel_max = channel.max()
        
        if channel_max - channel_min > 0:
            normalized = ((channel - channel_min) / (channel_max - channel_min) * 255)
        else:
            normalized = np.zeros_like(channel)
        
        return normalized.astype(np.uint8)
    
    def _canny_edges(self, channel: np.ndarray) -> np.ndarray:
        """Detect edges using Canny."""
        normalized = self._normalize_channel(channel)
        edges = cv2.Canny(normalized, 50, 150)
        return edges
    
    def _sobel_edges(self, channel: np.ndarray) -> np.ndarray:
        """Detect edges using Sobel."""
        normalized = self._normalize_channel(channel)
        
        grad_x = cv2.Sobel(normalized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(normalized, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        
        return magnitude
    
    def _laplacian_edges(self, channel: np.ndarray) -> np.ndarray:
        """Detect edges using Laplacian."""
        normalized = self._normalize_channel(channel)
        laplacian = cv2.Laplacian(normalized, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)
        return laplacian
    
    def compute_boundary_metrics(self, edges: np.ndarray) -> dict:
        """Compute metrics for detected boundaries."""
        total_pixels = edges.size
        edge_pixels = np.sum(edges > 0)
        
        # Edge density
        edge_density = edge_pixels / total_pixels
        
        # Edge strength (average non-zero edge value)
        if edge_pixels > 0:
            edge_strength = np.mean(edges[edges > 0])
        else:
            edge_strength = 0
        
        # Edge continuity (connected components)
        _, labels = cv2.connectedComponents((edges > 0).astype(np.uint8))
        num_components = labels.max()
        
        # Average component size
        if num_components > 0:
            avg_component_size = edge_pixels / num_components
        else:
            avg_component_size = 0
        
        return {
            'edge_density': float(edge_density),
            'edge_strength': float(edge_strength),
            'num_components': int(num_components),
            'avg_component_size': float(avg_component_size),
            'total_edge_pixels': int(edge_pixels)
        }
    
    def evaluate_channel(self, channel: np.ndarray, channel_name: str) -> dict:
        """Evaluate boundary detection quality for a channel."""
        results = {
            'channel_name': channel_name,
            'methods': {}
        }
        
        for method_name, method_func in self.edge_methods.items():
            try:
                edges = method_func(channel)
                metrics = self.compute_boundary_metrics(edges)
                
                results['methods'][method_name] = {
                    'edges': edges,
                    'metrics': metrics
                }
            except Exception as e:
                print(f"  Warning: {method_name} failed for {channel_name}: {e}")
        
        # Compute overall score
        overall_score = 0
        count = 0
        for method_data in results['methods'].values():
            if 'metrics' in method_data:
                m = method_data['metrics']
                # Weighted score: density + strength + continuity
                score = (m['edge_density'] * 0.3 + 
                        m['edge_strength'] / 255 * 0.4 +
                        min(m['avg_component_size'] / 100, 1.0) * 0.3)
                overall_score += score
                count += 1
        
        if count > 0:
            results['overall_boundary_score'] = overall_score / count
        else:
            results['overall_boundary_score'] = 0
        
        return results


def analyze_image_boundaries(image_path: str, output_dir: Path):
    """Analyze boundaries in all channels of an image."""
    
    image_name = Path(image_path).stem
    image_output_dir = output_dir / image_name
    image_output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print(f"Processing: {image_name}")
    print(f"{'='*70}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"ERROR: Could not load {image_path}")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image.shape}")
    
    # Apply XLET-NSST
    print("Applying XLET-NSST transformation...")
    transformer = XLETNSST(levels=3, directions=8)
    coeffs = transformer.transform(image)
    
    num_channels = len([k for k in coeffs.keys() if isinstance(coeffs[k], np.ndarray)])
    print(f"Generated {num_channels} channels")
    
    # Analyze boundaries
    print("Detecting boundaries in each channel...")
    detector = BoundaryDetector()
    
    channel_results = []
    
    for channel_name in coeffs.keys():
        if not isinstance(coeffs[channel_name], np.ndarray):
            continue
        
        if channel_name in ['scales', 'directions']:
            continue
        
        print(f"  Analyzing: {channel_name}")
        channel = coeffs[channel_name]
        
        result = detector.evaluate_channel(channel, channel_name)
        
        # Save visualizations
        channel_dir = image_output_dir / channel_name
        channel_dir.mkdir(exist_ok=True, parents=True)
        
        # Save original channel
        normalized = detector._normalize_channel(channel)
        cv2.imwrite(str(channel_dir / 'original.png'), normalized)
        
        # Save edge detection results
        for method_name, method_data in result['methods'].items():
            if 'edges' in method_data:
                cv2.imwrite(str(channel_dir / f'{method_name}_edges.png'), 
                          method_data['edges'])
        
        # Create composite visualization
        create_composite_visualization(
            normalized, 
            result['methods'],
            str(channel_dir / 'composite.png'),
            channel_name
        )
        
        # Store results (without edge images for JSON)
        result_json = {
            'channel_name': channel_name,
            'overall_boundary_score': result['overall_boundary_score'],
            'methods': {
                method: data['metrics'] 
                for method, data in result['methods'].items()
                if 'metrics' in data
            }
        }
        
        channel_results.append(result_json)
    
    # Sort by boundary score
    channel_results.sort(key=lambda x: x['overall_boundary_score'], reverse=True)
    
    # Save results
    with open(image_output_dir / 'boundary_analysis.json', 'w') as f:
        json.dump({
            'image_name': image_name,
            'num_channels': num_channels,
            'rankings': channel_results
        }, f, indent=2)
    
    # Print top channels
    print(f"\nTop 10 Channels by Boundary Detection Quality:")
    print("-" * 70)
    for idx, result in enumerate(channel_results[:10], 1):
        print(f"{idx:2d}. {result['channel_name']:35s} Score: {result['overall_boundary_score']:.4f}")
    
    return channel_results


def create_composite_visualization(original, methods_data, output_path, title):
    """Create a composite image showing original + all edge detection methods."""
    
    # Prepare images
    images = [original]
    labels = ['Original']
    
    for method_name in ['canny', 'sobel', 'laplacian']:
        if method_name in methods_data and 'edges' in methods_data[method_name]:
            images.append(methods_data[method_name]['edges'])
            labels.append(method_name.capitalize())
    
    # Create composite
    n = len(images)
    h, w = images[0].shape[:2]
    
    # Create 2x2 or 1x4 grid
    if n <= 4:
        cols = min(n, 2)
        rows = (n + cols - 1) // cols
    else:
        cols = 4
        rows = 1
    
    composite = np.zeros((rows * h, cols * w), dtype=np.uint8)
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        
        # Place image
        composite[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        # Add label
        cv2.putText(composite, label, 
                   (col*w + 10, row*h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    
    cv2.imwrite(output_path, composite)


def analyze_all_images(image_dir: str, output_dir: str = 'results/boundary_detection'):
    """Analyze boundaries for all images."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    image_dir = Path(image_dir)
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\n{'='*70}")
    print("XLET-NSST BOUNDARY DETECTION ANALYSIS")
    print(f"{'='*70}")
    print(f"\nFound {len(image_files)} images")
    print(f"Output directory: {output_path}")
    
    all_results = []
    channel_scores = defaultdict(list)
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {img_path.name}")
        
        try:
            results = analyze_image_boundaries(str(img_path), output_path)
            
            if results:
                all_results.append({
                    'image': img_path.name,
                    'results': results
                })
                
                # Accumulate scores
                for result in results:
                    channel_scores[result['channel_name']].append(
                        result['overall_boundary_score']
                    )
        
        except Exception as e:
            print(f"ERROR processing {img_path.name}: {e}")
    
    # Compute aggregate statistics
    print(f"\n{'='*70}")
    print("AGGREGATED BOUNDARY DETECTION RESULTS")
    print(f"{'='*70}\n")
    
    # Average scores across all images
    avg_scores = {
        channel: np.mean(scores)
        for channel, scores in channel_scores.items()
    }
    
    sorted_channels = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 15 Channels by Average Boundary Detection Score:")
    print("-" * 70)
    for idx, (channel, score) in enumerate(sorted_channels[:15], 1):
        freq = len(channel_scores[channel])
        print(f"{idx:2d}. {channel:35s} Avg: {score:.4f} ({freq}/{len(image_files)} images)")
    
    # Identify best channels across different scales
    print(f"\n{'='*70}")
    print("BEST CHANNELS BY SCALE")
    print(f"{'='*70}\n")
    
    # Group by scale
    scale_channels = defaultdict(list)
    for channel, score in sorted_channels:
        if 'L0' in channel:
            scale_channels['Scale 0 (Finest)'].append((channel, score))
        elif 'L1' in channel:
            scale_channels['Scale 1 (Medium)'].append((channel, score))
        elif 'L2' in channel:
            scale_channels['Scale 2 (Coarse)'].append((channel, score))
        elif channel == 'lowpass':
            scale_channels['Lowpass'].append((channel, score))
    
    for scale_name, channels in scale_channels.items():
        print(f"{scale_name}:")
        for idx, (channel, score) in enumerate(channels[:3], 1):
            print(f"  {idx}. {channel:30s} {score:.4f}")
        print()
    
    # Save aggregate results
    aggregate_results = {
        'num_images_analyzed': len(all_results),
        'average_scores': {ch: float(sc) for ch, sc in sorted_channels},
        'top_15_channels': [ch for ch, _ in sorted_channels[:15]],
        'best_by_scale': {
            scale: [ch for ch, _ in chs[:5]]
            for scale, chs in scale_channels.items()
        },
        'individual_images': all_results
    }
    
    with open(output_path / 'aggregate_boundary_analysis.json', 'w') as f:
        json.dump(aggregate_results, f, indent=2)
    
    # Create summary report
    create_summary_report(aggregate_results, output_path)
    
    print(f"\n✓ Complete! Results saved to: {output_path}")
    print(f"  - {len(image_files)} image folders with visualizations")
    print(f"  - aggregate_boundary_analysis.json")
    print(f"  - BOUNDARY_DETECTION_SUMMARY.txt")


def create_summary_report(results, output_path):
    """Create a text summary report."""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("XLET-NSST BOUNDARY DETECTION ANALYSIS - SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Images Analyzed: {results['num_images_analyzed']}")
    report_lines.append(f"Total Channels per Image: 25 (1 lowpass + 24 highpass)")
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("TOP 15 CHANNELS FOR BOUNDARY DETECTION")
    report_lines.append("="*80)
    report_lines.append("")
    
    for idx, channel in enumerate(results['top_15_channels'], 1):
        score = results['average_scores'][channel]
        report_lines.append(f"{idx:2d}. {channel:40s} Score: {score:.4f}")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("BEST CHANNELS BY DECOMPOSITION SCALE")
    report_lines.append("="*80)
    report_lines.append("")
    
    for scale, channels in results['best_by_scale'].items():
        report_lines.append(f"{scale}:")
        for idx, channel in enumerate(channels[:3], 1):
            score = results['average_scores'][channel]
            report_lines.append(f"  {idx}. {channel:35s} {score:.4f}")
        report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("RECOMMENDATIONS FOR SEMANTIC SEGMENTATION")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Use the following channels for optimal boundary detection:")
    report_lines.append("")
    
    for idx, channel in enumerate(results['top_15_channels'][:10], 1):
        report_lines.append(f"  {idx:2d}. {channel}")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("BOUNDARY DETECTION METHODS USED")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("1. Canny Edge Detection - Optimal edge detection with hysteresis")
    report_lines.append("2. Sobel Gradient - Directional gradient magnitude")
    report_lines.append("3. Laplacian - Second derivative edge detection")
    report_lines.append("")
    report_lines.append("Metrics Computed:")
    report_lines.append("  - Edge Density: Ratio of edge pixels to total pixels")
    report_lines.append("  - Edge Strength: Average intensity of detected edges")
    report_lines.append("  - Edge Continuity: Connected component analysis")
    report_lines.append("  - Overall Score: Weighted combination of all metrics")
    report_lines.append("")
    report_lines.append("="*80)
    
    # Write report
    with open(output_path / 'BOUNDARY_DETECTION_SUMMARY.txt', 'w') as f:
        f.write('\n'.join(report_lines))


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze boundary detection quality of XLET-NSST channels'
    )
    parser.add_argument('--image_dir', type=str, default='data',
                       help='Directory containing images')
    parser.add_argument('--output', type=str, default='results/boundary_detection',
                       help='Output directory')
    
    args = parser.parse_args()
    
    analyze_all_images(args.image_dir, args.output)
