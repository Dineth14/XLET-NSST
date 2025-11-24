"""
Optimal Channel Selection Analysis
Find the best channel combination for boundary + frequency features
"""

import json
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

# Load analysis results
with open('results/boundary_detection/aggregate_boundary_analysis.json', 'r') as f:
    boundary_data = json.load(f)

with open('results/multi_image_results.json', 'r') as f:
    feature_data = json.load(f)


def analyze_channel_combinations():
    """
    Analyze different channel combinations to find optimal set
    that balances boundary detection and frequency features
    """
    
    # Get boundary scores
    boundary_scores = boundary_data['average_scores']
    
    # Get feature quality scores
    feature_scores = {}
    for img_result in feature_data['image_results']:
        for channel in img_result['rankings']:
            ch_name = channel['channel_name']
            if ch_name not in feature_scores:
                feature_scores[ch_name] = []
            feature_scores[ch_name].append(channel['overall_score'])
    
    # Average feature scores
    avg_feature_scores = {
        ch: np.mean(scores) 
        for ch, scores in feature_scores.items()
    }
    
    # Combine scores (weighted: 60% boundary, 40% feature quality)
    combined_scores = {}
    for ch in boundary_scores.keys():
        boundary_score = boundary_scores[ch]
        feature_score = avg_feature_scores.get(ch, 0)
        combined_scores[ch] = 0.6 * boundary_score + 0.4 * feature_score
    
    # Sort by combined score
    sorted_channels = sorted(
        combined_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    print("="*80)
    print("OPTIMAL CHANNEL ANALYSIS")
    print("="*80)
    print("\nCombined Score = 60% Boundary + 40% Feature Quality\n")
    
    # Analyze different channel counts
    configurations = [
        (9, "Ultra-Light"),
        (12, "Light"),
        (15, "Balanced"),
        (18, "Enhanced"),
        (21, "Conservative"),
        (24, "Complete")
    ]
    
    results = {}
    
    for num_channels, config_name in configurations:
        # Get top N channels
        top_channels = [ch[0] for ch in sorted_channels[:num_channels]]
        
        # Calculate metrics
        avg_boundary = np.mean([boundary_scores[ch] for ch in top_channels])
        avg_feature = np.mean([avg_feature_scores[ch] for ch in top_channels])
        combined = 0.6 * avg_boundary + 0.4 * avg_feature
        
        # Analyze scale distribution
        scale_dist = {'L0': 0, 'L1': 0, 'L2': 0, 'lowpass': 0}
        for ch in top_channels:
            if 'L0' in ch:
                scale_dist['L0'] += 1
            elif 'L1' in ch:
                scale_dist['L1'] += 1
            elif 'L2' in ch:
                scale_dist['L2'] += 1
            elif 'lowpass' in ch:
                scale_dist['lowpass'] += 1
        
        # Analyze directional coverage
        directions_covered = set()
        for ch in top_channels:
            if 'D' in ch:
                direction = ch.split('D')[1]
                directions_covered.add(direction)
        
        directional_coverage = len(directions_covered) / 8.0  # 8 possible directions
        
        results[config_name] = {
            'num_channels': num_channels,
            'channels': top_channels,
            'boundary_score': avg_boundary,
            'feature_score': avg_feature,
            'combined_score': combined,
            'scale_distribution': scale_dist,
            'directional_coverage': directional_coverage,
            'efficiency': combined / num_channels  # Score per channel
        }
    
    return results, sorted_channels, boundary_scores, avg_feature_scores, combined_scores


def print_results(results):
    """Print configuration comparison"""
    
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    print(f"\n{'Config':<15} {'Channels':<10} {'Boundary':<12} {'Feature':<12} {'Combined':<12} {'Efficiency':<12}")
    print("-"*80)
    
    for config_name, data in results.items():
        print(f"{config_name:<15} "
              f"{data['num_channels']:<10} "
              f"{data['boundary_score']:<12.4f} "
              f"{data['feature_score']:<12.4f} "
              f"{data['combined_score']:<12.4f} "
              f"{data['efficiency']:<12.6f}")
    
    # Find optimal configuration
    best_config = max(results.items(), key=lambda x: x[1]['combined_score'])
    most_efficient = max(results.items(), key=lambda x: x[1]['efficiency'])
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n🏆 BEST OVERALL PERFORMANCE: {best_config[0]}")
    print(f"   - Channels: {best_config[1]['num_channels']}")
    print(f"   - Combined Score: {best_config[1]['combined_score']:.4f}")
    print(f"   - Boundary: {best_config[1]['boundary_score']:.4f}")
    print(f"   - Feature Quality: {best_config[1]['feature_score']:.4f}")
    
    print(f"\n⚡ MOST EFFICIENT: {most_efficient[0]}")
    print(f"   - Channels: {most_efficient[1]['num_channels']}")
    print(f"   - Efficiency: {most_efficient[1]['efficiency']:.6f} (score/channel)")
    print(f"   - Combined Score: {most_efficient[1]['combined_score']:.4f}")
    
    # Detailed breakdown of best config
    print(f"\n📊 DETAILED BREAKDOWN: {best_config[0]}")
    print(f"   Scale Distribution:")
    for scale, count in best_config[1]['scale_distribution'].items():
        if count > 0:
            print(f"      - {scale}: {count} channels ({count*100/best_config[1]['num_channels']:.1f}%)")
    print(f"   Directional Coverage: {best_config[1]['directional_coverage']*100:.1f}%")
    
    print(f"\n   Top Channels:")
    for i, ch in enumerate(best_config[1]['channels'][:10], 1):
        print(f"      {i:2d}. {ch}")
    
    return best_config, most_efficient


def analyze_marginal_benefit(sorted_channels, boundary_scores, feature_scores, combined_scores):
    """Analyze marginal benefit of adding each channel"""
    
    print("\n" + "="*80)
    print("MARGINAL BENEFIT ANALYSIS")
    print("="*80)
    print("\nShows improvement when adding each additional channel\n")
    
    cumulative_boundary = []
    cumulative_feature = []
    cumulative_combined = []
    
    for i in range(1, 26):
        top_n = [ch[0] for ch in sorted_channels[:i]]
        
        avg_b = np.mean([boundary_scores[ch] for ch in top_n])
        avg_f = np.mean([feature_scores[ch] for ch in top_n])
        avg_c = np.mean([combined_scores[ch] for ch in top_n])
        
        cumulative_boundary.append(avg_b)
        cumulative_feature.append(avg_f)
        cumulative_combined.append(avg_c)
        
        if i <= 15 or i % 5 == 0:
            marginal = cumulative_combined[-1] - cumulative_combined[-2] if i > 1 else cumulative_combined[0]
            print(f"Channels 1-{i:2d}: Combined={avg_c:.4f}, Marginal Δ={marginal:+.6f}")
    
    # Find elbow point (diminishing returns)
    marginal_gains = [cumulative_combined[i] - cumulative_combined[i-1] 
                      for i in range(1, len(cumulative_combined))]
    
    # Threshold: where marginal gain drops below 0.0001
    threshold = 0.0001
    elbow_point = next((i for i, gain in enumerate(marginal_gains) 
                       if gain < threshold), len(marginal_gains)) + 2
    
    print(f"\n📉 ELBOW POINT: ~{elbow_point} channels")
    print(f"   (Point where adding more channels gives < {threshold} improvement)")
    
    return cumulative_boundary, cumulative_feature, cumulative_combined, elbow_point


def analyze_scale_combinations():
    """Test different scale combinations"""
    
    print("\n" + "="*80)
    print("SCALE COMBINATION ANALYSIS")
    print("="*80)
    
    # Load data again for fresh analysis
    boundary_scores = boundary_data['average_scores']
    
    scale_configs = {
        'L0 Only (All 8)': [ch for ch in boundary_scores.keys() if 'L0' in ch],
        'L0 (All) + L1 (Best 3)': None,  # Will compute
        'L0 (All) + L1 (Best 6)': None,
        'L0 (All) + L1 (All 8)': None,
        'L0 (Top 5) + L1 (Top 3)': None,
        'All Scales': list(boundary_scores.keys())
    }
    
    # Compute specific combinations
    l0_channels = sorted(
        [ch for ch in boundary_scores.keys() if 'L0' in ch],
        key=lambda x: boundary_scores[x],
        reverse=True
    )
    l1_channels = sorted(
        [ch for ch in boundary_scores.keys() if 'L1' in ch],
        key=lambda x: boundary_scores[x],
        reverse=True
    )
    
    scale_configs['L0 (All) + L1 (Best 3)'] = l0_channels + l1_channels[:3]
    scale_configs['L0 (All) + L1 (Best 6)'] = l0_channels + l1_channels[:6]
    scale_configs['L0 (All) + L1 (All 8)'] = l0_channels + l1_channels
    scale_configs['L0 (Top 5) + L1 (Top 3)'] = l0_channels[:5] + l1_channels[:3]
    
    print(f"\n{'Configuration':<30} {'Channels':<10} {'Avg Boundary':<15} {'Best Channel':<15}")
    print("-"*80)
    
    scale_results = {}
    for config_name, channels in scale_configs.items():
        if channels:
            avg_score = np.mean([boundary_scores[ch] for ch in channels])
            best_channel = max(channels, key=lambda x: boundary_scores[x])
            best_score = boundary_scores[best_channel]
            
            scale_results[config_name] = {
                'channels': channels,
                'count': len(channels),
                'avg_boundary': avg_score,
                'best_score': best_score
            }
            
            print(f"{config_name:<30} {len(channels):<10} {avg_score:<15.4f} {best_score:<15.4f}")
    
    best_scale_config = max(scale_results.items(), key=lambda x: x[1]['avg_boundary'])
    
    print(f"\n🎯 BEST SCALE COMBINATION: {best_scale_config[0]}")
    print(f"   - Channels: {best_scale_config[1]['count']}")
    print(f"   - Average Boundary Score: {best_scale_config[1]['avg_boundary']:.4f}")
    
    return scale_results, best_scale_config


def create_visualizations(results, cumulative_scores, elbow_point):
    """Create visualization plots"""
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Configuration Comparison
    ax1 = plt.subplot(2, 3, 1)
    configs = list(results.keys())
    boundary_scores = [results[c]['boundary_score'] for c in configs]
    feature_scores = [results[c]['feature_score'] for c in configs]
    combined_scores_list = [results[c]['combined_score'] for c in configs]
    
    x = np.arange(len(configs))
    width = 0.25
    
    ax1.bar(x - width, boundary_scores, width, label='Boundary', alpha=0.8)
    ax1.bar(x, feature_scores, width, label='Feature Quality', alpha=0.8)
    ax1.bar(x + width, combined_scores_list, width, label='Combined', alpha=0.8)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Score')
    ax1.set_title('Score Comparison Across Configurations')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Efficiency Analysis
    ax2 = plt.subplot(2, 3, 2)
    channel_counts = [results[c]['num_channels'] for c in configs]
    efficiencies = [results[c]['efficiency'] for c in configs]
    
    ax2.plot(channel_counts, combined_scores_list, 'o-', linewidth=2, markersize=8)
    ax2.axvline(x=elbow_point, color='r', linestyle='--', label=f'Elbow Point ({elbow_point} ch)')
    ax2.set_xlabel('Number of Channels')
    ax2.set_ylabel('Combined Score')
    ax2.set_title('Performance vs Channel Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Marginal Benefit Curve
    ax3 = plt.subplot(2, 3, 3)
    channels_range = list(range(1, len(cumulative_scores[2]) + 1))
    ax3.plot(channels_range, cumulative_scores[2], 'g-', linewidth=2, label='Combined Score')
    ax3.axvline(x=elbow_point, color='r', linestyle='--', label=f'Elbow ({elbow_point} ch)')
    ax3.set_xlabel('Number of Channels')
    ax3.set_ylabel('Average Combined Score')
    ax3.set_title('Cumulative Score (Diminishing Returns)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Scale Distribution Heatmap
    ax4 = plt.subplot(2, 3, 4)
    scale_matrix = []
    for config in configs:
        dist = results[config]['scale_distribution']
        scale_matrix.append([dist['L0'], dist['L1'], dist['L2'], dist['lowpass']])
    
    sns.heatmap(scale_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=['L0', 'L1', 'L2', 'Lowpass'],
                yticklabels=configs, ax=ax4, cbar_kws={'label': 'Channel Count'})
    ax4.set_title('Scale Distribution per Configuration')
    
    # 5. Directional Coverage
    ax5 = plt.subplot(2, 3, 5)
    directional_coverage = [results[c]['directional_coverage'] * 100 for c in configs]
    colors = plt.cm.RdYlGn([cov/100 for cov in directional_coverage])
    
    bars = ax5.barh(configs, directional_coverage, color=colors)
    ax5.set_xlabel('Directional Coverage (%)')
    ax5.set_title('Angular Coverage by Configuration')
    ax5.set_xlim([0, 100])
    ax5.grid(axis='x', alpha=0.3)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax5.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}%', ha='left', va='center')
    
    # 6. Efficiency Ratio
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(channel_counts, efficiencies, s=200, alpha=0.6, c=combined_scores_list, 
                cmap='viridis', edgecolors='black', linewidth=1.5)
    
    for i, config in enumerate(configs):
        ax6.annotate(config, (channel_counts[i], efficiencies[i]), 
                    fontsize=8, ha='right', va='bottom')
    
    ax6.set_xlabel('Number of Channels')
    ax6.set_ylabel('Efficiency (Score/Channel)')
    ax6.set_title('Efficiency vs Channel Count')
    ax6.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(ax6.collections[0], ax=ax6)
    cbar.set_label('Combined Score')
    
    plt.tight_layout()
    plt.savefig('results/optimal_channel_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: results/optimal_channel_analysis.png")
    
    # Create second figure for detailed top channels
    fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 20 channels comparison
    top_20_channels = list(boundary_data['average_scores'].keys())[:20]
    boundary_top20 = [boundary_data['average_scores'][ch] for ch in top_20_channels]
    
    ax7.barh(range(20), boundary_top20, alpha=0.7)
    ax7.set_yticks(range(20))
    ax7.set_yticklabels(top_20_channels, fontsize=8)
    ax7.set_xlabel('Boundary Detection Score')
    ax7.set_title('Top 20 Channels by Boundary Score')
    ax7.invert_yaxis()
    ax7.grid(axis='x', alpha=0.3)
    ax7.axvline(x=0.65, color='r', linestyle='--', label='Excellence Threshold')
    ax7.legend()
    
    # Channel type distribution
    channel_types = {'L0': 0, 'L1': 0, 'L2': 0, 'Lowpass': 0}
    for ch in top_20_channels:
        if 'L0' in ch:
            channel_types['L0'] += 1
        elif 'L1' in ch:
            channel_types['L1'] += 1
        elif 'L2' in ch:
            channel_types['L2'] += 1
        else:
            channel_types['Lowpass'] += 1
    
    colors_pie = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    ax8.pie(channel_types.values(), labels=channel_types.keys(), autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    ax8.set_title('Scale Distribution in Top 20 Channels')
    
    plt.tight_layout()
    plt.savefig('results/top_channels_breakdown.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/top_channels_breakdown.png")


def save_recommendations(results, best_config, most_efficient, elbow_point, scale_results, best_scale_config):
    """Save final recommendations to file"""
    
    output = {
        'analysis_type': 'Optimal Channel Selection',
        'methodology': '60% Boundary Score + 40% Feature Quality',
        'elbow_point': elbow_point,
        'best_overall': {
            'name': best_config[0],
            'data': best_config[1]
        },
        'most_efficient': {
            'name': most_efficient[0],
            'data': most_efficient[1]
        },
        'best_scale_combination': {
            'name': best_scale_config[0],
            'data': {
                'count': best_scale_config[1]['count'],
                'avg_boundary': best_scale_config[1]['avg_boundary'],
                'channels': best_scale_config[1]['channels']
            }
        },
        'all_configurations': results,
        'scale_combinations': {
            k: {
                'count': v['count'],
                'avg_boundary': v['avg_boundary']
            }
            for k, v in scale_results.items()
        },
        'recommendation_summary': {
            'for_maximum_accuracy': best_config[0],
            'for_best_efficiency': most_efficient[0],
            'optimal_balance': 'Enhanced' if elbow_point >= 18 else 'Balanced'
        }
    }
    
    with open('results/optimal_channel_recommendation.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✅ Saved: results/optimal_channel_recommendation.json")


def main():
    print("\n" + "="*80)
    print("XLET-NSST OPTIMAL CHANNEL SELECTION ANALYSIS")
    print("="*80)
    
    # Run analyses
    results, sorted_channels, boundary_scores, feature_scores, combined_scores = analyze_channel_combinations()
    best_config, most_efficient = print_results(results)
    
    cumulative_boundary, cumulative_feature, cumulative_combined, elbow_point = \
        analyze_marginal_benefit(sorted_channels, boundary_scores, feature_scores, combined_scores)
    
    scale_results, best_scale_config = analyze_scale_combinations()
    
    # Create visualizations
    create_visualizations(results, (cumulative_boundary, cumulative_feature, cumulative_combined), elbow_point)
    
    # Save recommendations
    save_recommendations(results, best_config, most_efficient, elbow_point, scale_results, best_scale_config)
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS FOR URBANMAMBA V3")
    print("="*80)
    
    print(f"""
🎯 PRIMARY RECOMMENDATION: {best_config[0]} Configuration
   - Number of channels: {best_config[1]['num_channels']} (per RGB = {best_config[1]['num_channels'] * 3} total)
   - Combined score: {best_config[1]['combined_score']:.4f}
   - Boundary detection: {best_config[1]['boundary_score']:.4f}
   - Feature quality: {best_config[1]['feature_score']:.4f}
   - Directional coverage: {best_config[1]['directional_coverage']*100:.0f}%
   
   Channel reduction: {(1 - best_config[1]['num_channels']/29)*100:.1f}% (87 → {best_config[1]['num_channels']*3})
   
⚡ EFFICIENCY CHAMPION: {most_efficient[0]} Configuration
   - Number of channels: {most_efficient[1]['num_channels']} (per RGB = {most_efficient[1]['num_channels'] * 3} total)
   - Efficiency ratio: {most_efficient[1]['efficiency']:.6f} score/channel
   - Combined score: {most_efficient[1]['combined_score']:.4f}
   
   Channel reduction: {(1 - most_efficient[1]['num_channels']/29)*100:.1f}% (87 → {most_efficient[1]['num_channels']*3})

📊 SCALE RECOMMENDATION: {best_scale_config[0]}
   - Channels per RGB: {best_scale_config[1]['count']}
   - Average boundary score: {best_scale_config[1]['avg_boundary']:.4f}

💡 KEY INSIGHT:
   After {elbow_point} channels, diminishing returns set in.
   Going beyond this provides minimal improvement.
""")
    
    print("="*80)
    print("ANALYSIS COMPLETE ✅")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/optimal_channel_recommendation.json")
    print("  - results/optimal_channel_analysis.png")
    print("  - results/top_channels_breakdown.png")
    print()


if __name__ == '__main__':
    main()
