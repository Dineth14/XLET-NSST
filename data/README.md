# Data Directory

This directory is for storing your remote sensing images and segmentation masks.

## Expected Structure

```
data/
├── images/
│   ├── train/
│   │   ├── image_001.tif
│   │   ├── image_002.tif
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── masks/
│   ├── train/
│   │   ├── mask_001.png
│   │   ├── mask_002.png
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── README.md (this file)
```

## Data Format Requirements

### Images
- **Format**: GeoTIFF (.tif), PNG, or JPEG
- **Type**: Multi-spectral remote sensing imagery
- **Channels**: 3+ bands (RGB or multi-spectral)
- **Size**: Any size (will be resized/patched as needed)

### Masks
- **Format**: PNG or single-channel TIFF
- **Type**: Integer class labels (0, 1, 2, ...)
- **Size**: Must match corresponding image dimensions
- **Classes**: Each pixel value represents a class (e.g., 0=background, 1=road, 2=building, etc.)

## Sample Datasets

If you don't have your own data, you can use these publicly available datasets:

### 1. **ISPRS Potsdam** (Recommended)
- Urban semantic segmentation
- 6 classes: impervious surfaces, buildings, low vegetation, trees, cars, clutter
- High resolution (5cm GSD)
- Download: http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html

### 2. **LandCover.ai**
- Polish landscape aerial imagery
- 4 classes: building, woodland, water, background
- 25-50cm GSD
- Download: https://landcover.ai/

### 3. **DeepGlobe Land Cover**
- Satellite imagery classification
- 7 classes: urban, agriculture, rangeland, forest, water, barren, unknown
- 50cm GSD
- Download: http://deepglobe.org/

### 4. **Sentinel-2 Cloud Masks**
- Multi-spectral satellite imagery
- Cloud detection and classification
- 10m-60m GSD
- Download: https://github.com/sentinel-hub/sentinel2-cloud-detector

## Quick Test

If you don't have data yet, the main script will automatically generate synthetic test images:

```bash
python main.py  # Uses synthetic data by default
```

## Loading Your Data

```python
from src.utils import RemoteSensingDataLoader

# Initialize loader
loader = RemoteSensingDataLoader(
    normalize=True,
    target_size=(256, 256)  # Optional resizing
)

# Load image
image = loader.load_image('data/images/train/image_001.tif')

# Load corresponding mask
mask = loader.load_mask('data/masks/train/mask_001.png')
```

## Data Preprocessing Tips

1. **Normalization**: Always normalize images to [0, 1] range
2. **Patching**: For large images (>1024x1024), create patches:
   ```python
   patches = loader.create_patch(image, patch_size=256, overlap=32)
   ```
3. **Band Selection**: For multi-spectral data, select relevant bands:
   ```python
   image = loader.load_image('image.tif', bands=[1, 2, 3])  # R, G, B
   ```

## Citation

If you use any public dataset, please cite the original authors appropriately.
