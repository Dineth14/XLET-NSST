# XLET-NSST Usage Guide

## Quick Start Guide

### Installation

```powershell
# Clone or navigate to the repository
cd "e:\Computer Vision\wavelets\XLET-NSST"

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Test

```powershell
# Quick test on a single image
python quick_start.py data/00004.png
```

This will generate:
- Feature analysis results
- Top recommended channels
- Basic visualizations in `quick_results/`

## Detailed Usage

### 1. Test Single Image

```powershell
# Comprehensive analysis of one image
python test_features.py --image data/00004.png --output results

# With custom parameters
python test_features.py --image data/00004.png --levels 4 --directions 16 --output results
```

### 2. Test Multiple Images

```powershell
# Process all images in data directory
python test_features.py --image_dir data --output results

# With segmentation masks for better analysis
python test_features.py --image_dir data/images --mask_dir data/masks --output results
```

### 3. View Examples

```powershell
# Run all examples
python examples.py

# Run specific example
python examples.py --example 1   # Basic transformation
python examples.py --example 2   # Feature ranking
python examples.py --example 5   # Create feature vector
```

### 4. Interactive Notebook

```powershell
# Start Jupyter
jupyter notebook

# Open: notebooks/feature_analysis_tutorial.ipynb
```

## Understanding the Output

### Directory Structure After Running

```
results/
├── visualizations/
│   └── image_name/
│       ├── all_subbands.png          # All frequency channels
│       ├── scale_decomposition.png   # Multi-scale view
│       ├── feature_statistics.png    # Statistical charts
│       ├── correlation_matrix.png    # Channel correlations
│       ├── rankings.png              # Ranked channels
│       └── best_channels_comparison.png
├── statistics/
│   └── image_name_stats.json         # Detailed metrics
├── aggregated_summary.json           # Summary across all images
└── REPORT.txt                        # Human-readable report
```

### Reading the REPORT.txt

The report contains:
1. **Total images processed** and average channel count
2. **Top recommended channels** - most frequently appearing in top 10
3. **Specific recommendations** for your segmentation task

Example:
```
TOP FEATURE CHANNELS FOR SEMANTIC SEGMENTATION
==============================================

1. highpass_L1_D3              - 92% (11/12 images)
2. highpass_L0_D5              - 83% (10/12 images)
3. lowpass                     - 75% (9/12 images)
...
```

## Using Features in Your Segmentation Model

### Extract Features for ML Pipeline

```python
from src.transforms.xlet_nsst import XLETNSST
from src.analysis.feature_analysis import FeatureExtractor

# Transform image
transformer = XLETNSST(levels=3, directions=8)
coeffs = transformer.transform(image)

# Select best channels (from your testing)
selected = ['lowpass', 'highpass_L0_D3', 'highpass_L1_D5', ...]

# Create feature vector
extractor = FeatureExtractor()
features = extractor.create_feature_vector(
    coeffs, 
    selected_channels=selected,
    resize_shape=(256, 256)
)

# features.shape = (256, 256, N) where N = number of channels
# Use as input to your segmentation model
```

### Integration Example

```python
import torch
import torch.nn as nn

class XLETNSSTSegmentationModel(nn.Module):
    def __init__(self, xlet_channels=10, num_classes=5):
        super().__init__()
        self.transformer = XLETNSST(levels=3, directions=8)
        self.extractor = FeatureExtractor()
        
        # Your segmentation network
        self.segnet = nn.Sequential(
            nn.Conv2d(xlet_channels, 64, 3, padding=1),
            nn.ReLU(),
            # ... more layers ...
            nn.Conv2d(64, num_classes, 1)
        )
    
    def forward(self, x):
        # Extract XLET-NSST features
        coeffs = self.transformer.transform(x.cpu().numpy())
        features = self.extractor.create_feature_vector(
            coeffs, 
            selected_channels=self.best_channels
        )
        
        # Convert to tensor
        features = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0)
        
        # Segment
        return self.segnet(features)
```

## Parameter Tuning Guide

### Decomposition Levels

| Levels | Use Case | Speed | Channels |
|--------|----------|-------|----------|
| 1-2 | Fast processing, main structures | Fast | Few |
| 3 | **Recommended balance** | Medium | Moderate |
| 4-5 | Fine details, large images | Slow | Many |

### Number of Directions

| Directions | Angular Resolution | Speed | Channels |
|------------|-------------------|-------|----------|
| 4 | Basic (0°, 45°, 90°, 135°) | Fast | Few |
| 8 | **Recommended** | Medium | Moderate |
| 16 | Fine angular detail | Slow | Many |

### Evaluation Weights

Adjust in `config.yaml` or programmatically:

```python
# For edge-heavy tasks (roads, buildings)
weights = {
    'entropy': 0.15,
    'energy': 0.10,
    'texture': 0.15,
    'edge': 0.45,        # Emphasized
    'separability': 0.15
}

# For texture-heavy tasks (land cover, vegetation)
weights = {
    'entropy': 0.20,
    'energy': 0.10,
    'texture': 0.45,     # Emphasized
    'edge': 0.15,
    'separability': 0.10
}
```

## Common Workflows

### Workflow 1: First Time Analysis

1. **Test on sample images**
   ```powershell
   python test_features.py --image_dir data --output results
   ```

2. **Review REPORT.txt** - identify top channels

3. **Visualize best channels** - check `visualizations/`

4. **Select 8-15 diverse channels** for your model

### Workflow 2: With Ground Truth Masks

1. **Organize data**
   ```
   data/
   ├── images/
   │   ├── img1.png
   │   └── img2.png
   └── masks/
       ├── img1.png
       └── img2.png
   ```

2. **Run supervised analysis**
   ```powershell
   python test_features.py --image_dir data/images --mask_dir data/masks --output results
   ```

3. **Check separability scores** in JSON output

4. **Select channels with high separability**

### Workflow 3: Parameter Optimization

1. **Test multiple configurations**
   ```python
   python examples.py --example 8  # Compare parameters
   ```

2. **Balance speed vs. detail** based on your needs

3. **Re-run with optimal parameters**

### Workflow 4: Production Integration

1. **Identify best channels** from testing

2. **Create feature extraction function**
   ```python
   def extract_xlet_features(image, selected_channels):
       transformer = XLETNSST(levels=3, directions=8)
       coeffs = transformer.transform(image)
       extractor = FeatureExtractor()
       return extractor.create_feature_vector(
           coeffs, selected_channels=selected_channels
       )
   ```

3. **Integrate into your pipeline**

## Troubleshooting

### Issue: Out of Memory

**Solution 1**: Reduce image size
```python
image = cv2.resize(image, (512, 512))
```

**Solution 2**: Use fewer levels/directions
```powershell
python test_features.py --image data/img.png --levels 2 --directions 4
```

### Issue: Slow Processing

**Solution**: Reduce decomposition complexity
- Use `levels=2` instead of `levels=4`
- Use `directions=4` instead of `directions=16`

### Issue: No Clear Best Channels

**Solution**: Adjust evaluation weights for your domain
```yaml
# In config.yaml, adjust weights to emphasize what matters for your task
```

### Issue: Images are too large

**Solution**: Process patches
```python
# Split image into patches
patches = split_into_patches(large_image, patch_size=512)
results = [process_patch(p) for p in patches]
final = merge_patches(results)
```

## Performance Tips

1. **Start small**: Test with 1-2 images first
2. **Use appropriate parameters**: Don't over-decompose
3. **Select diverse channels**: Avoid redundant features
4. **Resize if needed**: XLET-NSST works on any size
5. **Cache results**: Save coefficients for repeated analysis

## Advanced Topics

### Custom Metrics

```python
from src.analysis.feature_analysis import FeatureEvaluator

class CustomEvaluator(FeatureEvaluator):
    def my_custom_metric(self, feature):
        # Your custom quality metric
        return custom_score
```

### Batch Processing

```python
from pathlib import Path
from tqdm import tqdm

images = list(Path('data').glob('*.png'))
for img_path in tqdm(images):
    results = tester.process_single_image(str(img_path))
```

### Parallel Processing

```python
from multiprocessing import Pool

def process_wrapper(img_path):
    return tester.process_single_image(str(img_path))

with Pool(4) as pool:
    results = pool.map(process_wrapper, image_paths)
```

## Best Practices

1. ✅ **Test on representative samples** of your dataset
2. ✅ **Use ground truth masks** when available
3. ✅ **Select diverse channels** (low correlation)
4. ✅ **Balance channel count** (8-15 typically optimal)
5. ✅ **Validate on segmentation task** performance
6. ✅ **Document selected channels** for reproducibility

## Getting Help

- Check `examples.py` for code samples
- Review `notebooks/feature_analysis_tutorial.ipynb`
- Read the detailed README.md
- Inspect JSON outputs for detailed metrics
- Examine visualizations for intuition

## Next Steps

After testing:
1. Identify your top 10-15 channels
2. Create a feature extraction pipeline
3. Train your segmentation model
4. Evaluate and iterate
5. Fine-tune channel selection based on results

Happy feature engineering! 🎯
