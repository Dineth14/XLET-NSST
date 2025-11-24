"""
Create a visual summary showing best boundary detection channels
"""

import cv2
import numpy as np
from pathlib import Path


def create_summary_montage(image_name='00004'):
    """Create a montage showing top boundary detection channels."""
    
    base_path = Path(f'results/boundary_detection/{image_name}')
    
    if not base_path.exists():
        print(f"Error: {base_path} not found")
        return
    
    # Top channels to display
    top_channels = [
        'highpass_L0_D1',
        'highpass_L0_D6',
        'highpass_L0_D3',
        'highpass_L0_D0',
        'highpass_L0_D4',
        'highpass_L0_D2',
    ]
    
    images = []
    labels = []
    
    for channel in top_channels:
        channel_path = base_path / channel / 'composite.png'
        if channel_path.exists():
            img = cv2.imread(str(channel_path))
            if img is not None:
                images.append(img)
                labels.append(channel)
    
    if not images:
        print("No images found")
        return
    
    # Create grid
    n = len(images)
    cols = 2
    rows = (n + 1) // 2
    
    # Resize all to same size
    target_h, target_w = 400, 800
    resized = []
    for img in images:
        resized.append(cv2.resize(img, (target_w, target_h)))
    
    # Create montage
    montage_h = rows * target_h
    montage_w = cols * target_w
    montage = np.zeros((montage_h, montage_w, 3), dtype=np.uint8)
    
    for idx, (img, label) in enumerate(zip(resized, labels)):
        row = idx // cols
        col = idx % cols
        
        y = row * target_h
        x = col * target_w
        
        montage[y:y+target_h, x:x+target_w] = img
        
        # Add title
        cv2.putText(montage, label, 
                   (x + 20, y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # Save
    output_path = Path(f'results/boundary_detection/BEST_CHANNELS_{image_name}.png')
    cv2.imwrite(str(output_path), montage)
    print(f"✓ Summary montage saved: {output_path}")
    
    return montage


def create_comparison_grid():
    """Create comparison showing top 3 channels across all images."""
    
    base_path = Path('results/boundary_detection')
    
    # Get all image folders
    image_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    if not image_dirs:
        print("No image directories found")
        return
    
    # Top 3 channels
    top_channels = ['highpass_L0_D1', 'highpass_L0_D6', 'highpass_L0_D3']
    
    # Load images
    grid_images = []
    
    for img_dir in image_dirs[:6]:  # First 6 images
        row = []
        for channel in top_channels:
            img_path = img_dir / channel / 'canny_edges.png'
            if img_path.exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize
                    img = cv2.resize(img, (200, 200))
                    row.append(img)
        
        if len(row) == 3:
            grid_images.append(row)
    
    if not grid_images:
        print("Could not load images")
        return
    
    # Create grid
    rows = len(grid_images)
    cols = 3
    cell_size = 200
    
    grid = np.zeros((rows * cell_size, cols * cell_size), dtype=np.uint8)
    
    for r, row in enumerate(grid_images):
        for c, img in enumerate(row):
            y = r * cell_size
            x = c * cell_size
            grid[y:y+cell_size, x:x+cell_size] = img
    
    # Add labels
    grid_color = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    
    # Column headers
    for c, channel in enumerate(top_channels):
        x = c * cell_size + 10
        cv2.putText(grid_color, channel.replace('highpass_', ''), 
                   (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save
    output_path = Path('results/boundary_detection/TOP3_COMPARISON_GRID.png')
    cv2.imwrite(str(output_path), grid_color)
    print(f"✓ Comparison grid saved: {output_path}")
    
    return grid_color


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Creating Visual Summary of Boundary Detection Results")
    print("="*70 + "\n")
    
    # Create montages for first few images
    for image_name in ['00004', '00005', '00007']:
        create_summary_montage(image_name)
    
    print()
    
    # Create comparison grid
    create_comparison_grid()
    
    print("\n" + "="*70)
    print("Summary visualizations created!")
    print("Check: results/boundary_detection/")
    print("="*70)
