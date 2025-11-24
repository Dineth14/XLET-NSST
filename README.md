# XLET-NSST Feature Channel Testing for Semantic Segmentation

A comprehensive repository for testing and identifying the best frequency feature channels from XLET-NSST (Extended Laplacian with Nonsubsampled Shearlet Transform) transformation for semantic segmentation tasks.

## ✅ **ANALYSIS COMPLETE** - Best Channels Identified!

**Top 9 Recommended Channels for Segmentation:**
1. `highpass_L0_D1` (22.5°) - Score: 0.5040, Boundary: 0.6527 ⭐
2. `highpass_L0_D4` (90°) - Score: 0.4955, Boundary: 0.6512
3. `highpass_L0_D7` (157.5°) - Score: 0.4907, Boundary: 0.6513
4. `highpass_L0_D0` (0°) - Score: 0.5007, Boundary: 0.6517
5. `highpass_L0_D5` (112.5°) - Score: 0.4923, Boundary: 0.6503
6. `highpass_L0_D2` (45°) - Score: 0.4965, Boundary: 0.6522
7. `highpass_L0_D3` (67.5°) - Score: 0.5010, Boundary: 0.6526
8. `highpass_L0_D6` (135°) - Score: 0.4945, Boundary: 0.6527
9. `highpass_L1_D0` (0°) - Score: 0.4450, Boundary: 0.6473

📊 **See [COMPLETE_ANALYSIS_RESULTS.md](COMPLETE_ANALYSIS_RESULTS.md) for full analysis**

## 🎯 Purpose

This repository provides tools to:
- Apply XLET-NSST multi-scale, multi-directional frequency decomposition to images
- Extract and analyze all frequency subbands (lowpass and directional highpass)
- Evaluate feature quality using multiple metrics (entropy, energy, texture, edge preservation)
- Rank and select the best feature channels for semantic segmentation
- Generate comprehensive visualizations and analysis reports

## 🔬 What is XLET-NSST?

XLET-NSST combines:
- **Extended Laplacian Pyramid**: Multi-scale frequency decomposition
- **Nonsubsampled Shearlet Transform**: Directional frequency analysis

This creates a rich set of frequency features that capture:
- Different spatial scales (coarse to fine details)
- Multiple orientations (edges and textures at various angles)
- Both low-frequency (smooth regions) and high-frequency (edges, details) information

These features are particularly valuable for semantic segmentation where understanding texture, edges, and multi-scale patterns is crucial.

## 📁 Repository Structure

```
XLET-NSST/
├── src/
│   ├── transforms/
│   │   └── xlet_nsst.py          # Core XLET-NSST implementation
│   ├── analysis/
│   │   └── feature_analysis.py   # Feature extraction and evaluation
│   └── visualization/
│       └── visualize.py           # Visualization tools
├── data/                          # Your test images
├── results/                       # Output directory (auto-created)
│   ├── visualizations/           # Feature visualizations
│   ├── statistics/               # Statistical analysis
│   └── rankings/                 # Channel rankings
├── test_features.py              # Main testing pipeline
├── quick_start.py                # Quick example script
├── requirements.txt              # Dependencies
├── config.yaml                   # Configuration
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```powershell
# Create a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Test on a Single Image

```powershell
# Quick test with default settings
python quick_start.py data/00004.png

# With custom output directory
python quick_start.py data/00004.png --output my_results
```

This will:
- Apply XLET-NSST transformation
- Analyze all feature channels
- Rank them for segmentation quality
- Generate visualizations
- Show top recommended channels

### 3. Comprehensive Testing

```powershell
# Test single image with full analysis
python test_features.py --image data/00004.png --output results

# Test all images in data directory
python test_features.py --image_dir data --output results

# With segmentation masks (for supervised analysis)
python test_features.py --image_dir data/images --mask_dir data/masks --output results

# Custom parameters
python test_features.py --image_dir data --levels 4 --directions 16 --output results
```

## 📊 Output Results

After running the tests, you'll get:

### 1. Visualizations
- **all_subbands.png**: Grid view of all frequency channels
- **scale_decomposition.png**: Multi-scale decomposition view
- **feature_statistics.png**: Statistical comparison charts
- **correlation_matrix.png**: Channel correlation heatmap
- **rankings.png**: Top channels ranked by quality
- **best_channels_comparison.png**: Side-by-side comparison of best channels
- **feature_montage.png**: Complete feature montage

### 2. Statistics (JSON)
- Detailed metrics for each channel (entropy, energy, variance, etc.)
- Rankings by different criteria
- Best channels by specific metrics
- Diverse channel selections

### 3. Analysis Report (REPORT.txt)
- Summary of top recommended channels
- Frequency of channels across multiple images
- Recommendations for semantic segmentation

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
transformation:
  levels: 3              # More levels = more scales (1-5)
  directions: 8          # More directions = finer angular resolution (4, 8, 16)

evaluation_weights:
  entropy: 0.25         # Information content
  texture: 0.25         # Texture richness
  edge: 0.20            # Edge preservation
  # Adjust weights based on your priorities
```

## 📈 Understanding the Results

### Key Metrics

1. **Entropy**: Information content - higher means more diverse features
2. **Energy**: Signal strength - captures strong features
3. **Texture Score**: Richness of texture patterns
4. **Edge Preservation**: How well edges are maintained
5. **Separability**: Class discrimination (requires masks)

### Channel Naming Convention

- `lowpass`: Low-frequency approximation (smooth regions)
- `highpass_L{level}_D{direction}`: 
  - `L{level}`: Scale level (0=finest, higher=coarser)
  - `D{direction}`: Direction index (0-7 for 8 directions)
  - Example: `highpass_L1_D3` = Scale 1, Direction 3 (≈67.5°)

### Recommended Channels

The pipeline automatically identifies the best channels based on:
- **High information content** (entropy)
- **Strong edge preservation**
- **Rich texture representation**
- **Low correlation** with other channels (diversity)
- **Class separability** (if masks provided)

## 💡 Usage in Semantic Segmentation

### Integrating Features

```python
from src.transforms.xlet_nsst import XLETNSST
from src.analysis.feature_analysis import FeatureExtractor

# Load your image
image = load_image('path/to/image.png')

# Transform
transformer = XLETNSST(levels=3, directions=8)
coeffs = transformer.transform(image)

# Select best channels (based on your testing)
extractor = FeatureExtractor()
best_channels = ['lowpass', 'highpass_L0_D2', 'highpass_L1_D5', ...]

# Create feature vector for segmentation
features = extractor.create_feature_vector(
    coeffs, 
    selected_channels=best_channels,
    resize_shape=(256, 256)
)

# Now use 'features' as input to your segmentation model
# Shape: (H, W, N) where N = number of selected channels
```

### Tips for Best Results

1. **Multi-Scale**: Include channels from different levels (L0, L1, L2)
2. **Multi-Direction**: Select diverse directional channels
3. **Balance**: Mix lowpass (smooth) and highpass (detail) features
4. **Dataset-Specific**: Test on your specific images to find optimal channels
5. **Dimensionality**: Start with 10-15 best channels, tune based on performance

## 📝 Examples

### Example 1: Quick Feature Analysis

```python
from src.transforms.xlet_nsst import XLETNSST
import cv2

# Load image
image = cv2.imread('data/00004.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Transform
transformer = XLETNSST(levels=3, directions=8)
coeffs = transformer.transform(image)

# See what channels were created
print("Available channels:")
for key in coeffs.keys():
    if isinstance(coeffs[key], np.ndarray):
        print(f"  {key}: shape {coeffs[key].shape}")
```

### Example 2: Custom Feature Selection

```python
from src.analysis.feature_analysis import FeatureExtractor, FeatureEvaluator

extractor = FeatureExtractor()
evaluator = FeatureEvaluator()

# Analyze
analysis = extractor.analyze_all_channels(coeffs)

# Get top channels by entropy
top_entropy = extractor.get_best_channels_by_metric(
    analysis, metric='entropy', top_k=5
)

# Or select diverse channels
diverse = extractor.select_diverse_channels(
    coeffs, num_channels=10, correlation_threshold=0.7
)

# Or rank for segmentation
rankings = evaluator.rank_features_for_segmentation(coeffs)
```

## 🔬 Advanced Usage

### Custom Evaluation Weights

```python
# Prioritize edge preservation for boundary-heavy tasks
rankings = evaluator.rank_features_for_segmentation(
    coeffs,
    weights={
        'entropy': 0.15,
        'energy': 0.10,
        'texture': 0.15,
        'edge': 0.40,      # Increased
        'separability': 0.20
    }
)
```

### Supervised Analysis with Masks

```python
# Load ground truth mask
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

# Rank with mask for better separability estimation
rankings = evaluator.rank_features_for_segmentation(
    coeffs, 
    labels=mask
)
```

## 📚 Key Functions

### XLETNSST.transform(image)
Apply XLET-NSST transformation to an image.

### FeatureExtractor.analyze_all_channels(coeffs)
Compute statistics for all channels.

### FeatureEvaluator.rank_features_for_segmentation(coeffs, labels)
Rank channels by segmentation quality.

### FeatureVisualizer.visualize_all_subbands(coeffs)
Visualize all frequency channels.

## ⚙️ Parameters Guide

### Decomposition Levels (1-5)
- **1-2**: Fast, fewer features, captures main structures
- **3**: Good balance (recommended for most cases)
- **4-5**: More features, captures fine details, slower

### Directions (4, 8, 16)
- **4**: Basic directional analysis (0°, 45°, 90°, 135°)
- **8**: Good angular resolution (recommended)
- **16**: Fine angular detail, more features, slower

## 🎓 Understanding Frequency Features

### Why Frequency Features for Segmentation?

1. **Multi-Scale Analysis**: Different objects appear at different scales
2. **Directional Information**: Edges and textures have orientations
3. **Texture Discrimination**: Frequency patterns distinguish classes
4. **Edge Enhancement**: High-frequency subbands highlight boundaries
5. **Noise Robustness**: Frequency decomposition can filter noise

### What Makes a Good Feature Channel?

- **High Entropy**: Rich, diverse information
- **Strong Edges**: Clear boundary detection
- **Texture Details**: Discriminative patterns
- **Low Redundancy**: Not correlated with other channels
- **Class Separability**: Different classes have different responses

## 🐛 Troubleshooting

**Issue**: Out of memory
- Solution: Process images at lower resolution or reduce decomposition levels

**Issue**: Slow processing
- Solution: Reduce number of directions or decomposition levels

**Issue**: No clear best channels
- Solution: Your images may need domain-specific tuning; try different weights

## 📖 Citation

If you use this code in your research, please cite:

```
@software{xlet_nsst_testing,
  title={XLET-NSST Feature Channel Testing for Semantic Segmentation},
  year={2024},
  url={https://github.com/yourusername/XLET-NSST}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Happy Feature Testing! 🚀**
