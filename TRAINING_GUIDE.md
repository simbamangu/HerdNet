# HerdNet Training Guide: Transfer Learning with New Species

## Overview
This guide covers training a new HerdNet model using the pretrained `HerdnetGeneral2022.pth` as a starting point, with new/different species labels from COCO-formatted annotations.

**Current model classes**: buffalo, elephant, kob, topi, warthog, waterbuck (6 species)
**Your task**: Add new species and/or modify existing labels

---

## Prerequisites

### 1. Data Preparation Checklist
- [ ] Annotated images (aerial/nadir imagery)
- [ ] COCO JSON file with annotations
- [ ] Decide on train/validation split (recommended: 80/20 or 70/30)

### 2. Environment Setup
```bash
# Activate conda environment
conda activate herdnet

# Verify installation
python -c "import animaloc; print('AnimalOC version:', animaloc.__version__)"

# Ensure W&B is configured
wandb login
```

---

## Step-by-Step Training Process

### STEP 1: Convert COCO to HerdNet CSV Format

HerdNet requires CSV files with point annotations in format:
```csv
images,x,y,labels
image001.jpg,517,1653,2
image001.jpg,800,1253,1
image002.jpg,896,742,3
```

**Conversion script** (create as `tools/coco_to_csv.py`):
```python
import json
import pandas as pd
from pathlib import Path

def coco_to_herdnet_csv(coco_json_path, output_csv_path):
    """
    Convert COCO format to HerdNet CSV format.

    For bounding boxes: uses center point (x_center, y_center)
    For keypoints: uses keypoint coordinates directly
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Build image id to filename mapping
    img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}

    # Build category id to label mapping (1-indexed for HerdNet)
    cat_id_to_label = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'], start=1)}

    rows = []
    for ann in coco_data['annotations']:
        img_name = img_id_to_name[ann['image_id']]
        category_label = cat_id_to_label[ann['category_id']]

        # Handle bounding boxes
        if 'bbox' in ann and ann['bbox']:
            x, y, w, h = ann['bbox']
            # Use center point
            x_center = x + w / 2
            y_center = y + h / 2
            rows.append({
                'images': img_name,
                'x': x_center,
                'y': y_center,
                'labels': category_label
            })

        # Handle keypoints (if present)
        elif 'keypoints' in ann and ann['keypoints']:
            # COCO keypoints format: [x1, y1, v1, x2, y2, v2, ...]
            keypoints = ann['keypoints']
            for i in range(0, len(keypoints), 3):
                x, y, visibility = keypoints[i:i+3]
                if visibility > 0:  # Only use visible keypoints
                    rows.append({
                        'images': img_name,
                        'x': x,
                        'y': y,
                        'labels': category_label
                    })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)

    # Print category mapping for reference
    print(f"\nCategory mapping (for config file):")
    for cat in coco_data['categories']:
        label = cat_id_to_label[cat['id']]
        print(f"  {label}: '{cat['name']}'")

    print(f"\nConverted {len(rows)} annotations to {output_csv_path}")
    return df

# Usage
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python coco_to_csv.py <coco_json> <output_csv>")
        sys.exit(1)

    coco_to_herdnet_csv(sys.argv[1], sys.argv[2])
```

**Run conversion**:
```bash
python tools/coco_to_csv.py /path/to/train_annotations.json data/train.csv
python tools/coco_to_csv.py /path/to/val_annotations.json data/val.csv
```

---

### STEP 2: Create Patches (Optional but Recommended)

If your images are large (>2000x2000), create patches to fit in GPU memory:

```bash
python tools/patcher.py \
    /path/to/train/images \
    512 512 160 \
    /path/to/train_patches \
    -csv data/train.csv \
    -min 1
```

Parameters:
- `512 512`: patch size (height, width)
- `160`: overlap in pixels
- `-min 1`: minimum annotations per patch
- `-all`: (optional) include all patches even without annotations

Repeat for validation set if needed.

---

### STEP 3: Create Training Config File

Create `configs/train/my_species.yaml`:

```yaml
# Experiment settings
wandb_project: 'HerdNet-MySpecies'
wandb_entity: 'your_wandb_username'
wandb_run: 'transfer_learning_v1'
seed: 42
device_name: 'cuda'

# Model configuration
model:
  name: 'HerdNet'
  from_torchvision: False

  # IMPORTANT: Load pretrained weights for transfer learning
  load_from: '/home/simbamangu/workspace/HerdNet/HerdnetGeneral2022.pth'

  resume_from: null  # Use this instead if resuming interrupted training

  kwargs:
    num_layers: 34
    pretrained: False  # Set to False when loading from checkpoint
    down_ratio: 2
    head_conv: 64

  # Optional: freeze backbone layers for faster convergence
  freeze: null  # or ['dla', 'base'] to freeze backbone

# Losses (same as pretrained model)
losses:
  FocalLoss:
    print_name: 'focal_loss'
    from_torch: False
    output_idx: 0
    target_idx: 0
    lambda_const: 1.0
    kwargs:
      reduction: 'mean'
      normalize: False

  CrossEntropyLoss:
    print_name: 'ce_loss'
    from_torch: True
    output_idx: 1
    target_idx: 1
    lambda_const: 1.0
    kwargs:
      reduction: 'mean'
      # Adjust weights based on class imbalance in YOUR data
      # First weight (0.1) is for background, rest for each species
      weight: [0.1, 1.0, 2.0, 1.5, 1.0, 3.0, 1.0]  # Example for 6 species + background

# Dataset configuration
datasets:
  img_size: [512, 512]
  anno_type: 'point'

  # CRITICAL: Update number of classes (YOUR species count + 1 for background)
  num_classes: 7  # Example: 6 species + 1 background = 7

  collate_fn: null

  # IMPORTANT: Define your species mapping (must match COCO conversion)
  class_def:
    1: 'elephant'
    2: 'buffalo'
    3: 'giraffe'      # Example new species
    4: 'zebra'        # Example new species
    5: 'wildebeest'   # Example new species
    6: 'impala'       # Example new species

  # Training dataset
  train:
    name: 'CSVDataset'
    csv_file: '/path/to/data/train.csv'  # UPDATE THIS PATH
    root_dir: '/path/to/train/images'     # UPDATE THIS PATH

    sampler: null

    # Data augmentation
    albu_transforms:
      HorizontalFlip:
        p: 0.5
      VerticalFlip:
        p: 0.5
      MotionBlur:
        p: 0.3
        blur_limit: 5
      Normalize:
        mean: [0.485, 0.456, 0.406]  # ImageNet normalization
        std: [0.229, 0.224, 0.225]
        p: 1.0

    # Target transforms
    end_transforms:
      MultiTransformsWrapper:
        FIDT:
          num_classes: ${train.datasets.num_classes}
          down_ratio: ${train.model.kwargs.down_ratio}
        PointsToMask:
          radius: 2
          num_classes: ${train.datasets.num_classes}
          squeeze: True
          down_ratio: 32

  # Validation dataset
  validate:
    name: 'CSVDataset'
    csv_file: '/path/to/data/val.csv'    # UPDATE THIS PATH
    root_dir: '/path/to/val/images'      # UPDATE THIS PATH

    albu_transforms:
      Normalize:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        p: 1.0

    end_transforms:
      DownSample:
        down_ratio: ${train.model.kwargs.down_ratio}
        anno_type: ${train.datasets.anno_type}

# Training settings
training_settings:
  trainer: 'Trainer'
  valid_freq: 1         # Validate every epoch
  print_freq: 50        # Print every 50 batches
  batch_size: 8         # Adjust based on GPU memory (reduce if OOM)

  # Optimizer settings
  optimizer: 'adam'
  lr: 5e-5              # Lower LR for transfer learning (original: 1e-4)
  weight_decay: 0.0005

  # Learning rate scheduler
  auto_lr:
    mode: 'max'
    patience: 10
    threshold: 1e-4
    threshold_mode: 'rel'
    cooldown: 5
    min_lr: 1e-6
    verbose: True

  warmup_iters: 100     # Warmup iterations
  vizual_fn: null       # Optional: add visualization function
  epochs: 50            # Reduce epochs for transfer learning (original: 100)

  # Evaluator configuration
  evaluator:
    name: 'HerdNetEvaluator'
    threshold: 5        # Distance threshold for TP (in pixels)
    select_mode: 'max'
    validate_on: 'f1_score'  # Can also use 'mAP', 'recall', 'precision'
    kwargs:
      print_freq: 10
      lmds_kwargs:
        kernel_size: [3, 3]
        adapt_ts: 0.2   # Adaptive threshold for detection
        neg_ts: 0.1     # Negative threshold

  # Stitcher for large images during validation
  stitcher:
    name: 'HerdNetStitcher'
    kwargs:
      overlap: 160
      down_ratio: ${train.model.kwargs.down_ratio}
      up: True
      reduction: 'mean'
```

---

### STEP 4: Calculate Class Weights (Important!)

Class imbalance affects training. Calculate weights based on your data:

```python
# tools/calculate_weights.py
import pandas as pd
import numpy as np

def calculate_class_weights(csv_file, num_classes):
    """Calculate inverse frequency weights for classes"""
    df = pd.read_csv(csv_file)

    # Count instances per class
    class_counts = df['labels'].value_counts().sort_index()

    # Calculate inverse frequency weights
    total = len(df)
    weights = []

    # Background weight (typically low)
    weights.append(0.1)

    # Calculate for each species
    for i in range(1, num_classes):
        count = class_counts.get(i, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(round(weight, 2))

    print(f"Class distribution:")
    print(class_counts)
    print(f"\nRecommended weights: {weights}")
    return weights

# Usage
calculate_class_weights('data/train.csv', num_classes=7)
```

Update the `weight` parameter in your config under `losses.CrossEntropyLoss.kwargs`.

---

### STEP 5: Launch Training

**Basic training**:
```bash
python tools/train.py train=my_species
```

**Override parameters from command line**:
```bash
python tools/train.py train=my_species \
    train.training_settings.batch_size=4 \
    train.training_settings.lr=1e-5 \
    train.wandb_run='experiment_lr_1e5'
```

**Multi-run hyperparameter search**:
```bash
python tools/train.py -m train=my_species \
    train.training_settings.batch_size=4,8 \
    train.training_settings.lr=1e-5,5e-5,1e-4
```

Training outputs are saved to `outputs/YYYY-MM-DD/HH-MM-SS/`

---

### STEP 6: Monitor Training

**Weights & Biases dashboard**:
- Real-time loss curves
- Validation metrics (F1, mAP, recall, precision)
- Learning rate tracking
- System metrics (GPU usage, etc.)

**Check logs**:
```bash
tail -f outputs/*/training_*.txt
```

**Key metrics to watch**:
- `f1_score`: Overall detection performance
- `mAP`: Mean average precision across classes
- `focal_loss`: Should decrease steadily
- `ce_loss`: Classification loss

---

### STEP 7: Export Trained Model for Inference

After training completes, prepare the model for `infer.py`:

```python
# tools/prepare_inference_model.py
import torch

# Load best checkpoint
checkpoint_path = 'outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/best.pth'
pth = torch.load(checkpoint_path)

# Add required metadata for inference
pth['classes'] = {
    1: 'elephant',
    2: 'buffalo',
    3: 'giraffe',
    4: 'zebra',
    5: 'wildebeest',
    6: 'impala'
}
pth['mean'] = [0.485, 0.456, 0.406]
pth['std'] = [0.229, 0.224, 0.225]

# Save inference-ready model
output_path = 'MySpeciesHerdNet_v1.pth'
torch.save(pth, output_path)
print(f"Inference model saved to {output_path}")
```

---

### STEP 8: Test Inference

```bash
python tools/infer.py \
    /path/to/test/images \
    MySpeciesHerdNet_v1.pth \
    -device cuda
```

Results saved to `/path/to/test/images/YYYY-MM-DD_HerdNet_results/`:
- `detections.csv`: All detections with coordinates, species, scores
- `plots/`: Images with detection overlays
- `thumbnails/`: Cropped detections

---

## Advanced Tips

### Fine-tuning Strategy

1. **Freeze backbone initially** (faster convergence):
   ```yaml
   model:
     freeze: ['dla']
   ```
   Train for 10-20 epochs, then unfreeze:
   ```yaml
   model:
     freeze: null
   ```

2. **Lower learning rate**: Transfer learning benefits from smaller LR (1e-5 to 5e-5)

3. **Gradual unfreezing**: Train classification head first, then unfreeze backbone layers progressively

### Handling Different Number of Classes

The pretrained model has 6 species classes (+ background = 7 total). If your dataset has:

**More classes**: The final layer will be reinitialized automatically. Expect longer training.

**Fewer classes**: Still loads most weights. Classification head adapts to new number.

**Completely different classes**: Still beneficial due to learned feature representations.

### Data Augmentation Best Practices

For aerial imagery:
```yaml
albu_transforms:
  HorizontalFlip: {p: 0.5}
  VerticalFlip: {p: 0.5}
  Rotate: {limit: 180, p: 0.5}  # Aerial views have no "up"
  RandomBrightnessContrast: {p: 0.3}
  MotionBlur: {p: 0.2}
  GaussNoise: {p: 0.2}
  Normalize: {p: 1.0}
```

### Troubleshooting

**OOM errors**: Reduce `batch_size` to 2 or 4

**Poor convergence**:
- Check class weights
- Reduce learning rate
- Increase warmup iterations
- Verify CSV file format

**Low validation scores**:
- Check annotation quality
- Adjust `adapt_ts` threshold in evaluator
- Increase training epochs
- Review class balance

**Model not learning**:
- Verify CSV paths are correct
- Check image normalization matches pretrained model
- Ensure labels are 1-indexed (not 0-indexed)

---

## Summary Checklist

- [ ] Convert COCO annotations to CSV format
- [ ] Split data into train/validation sets
- [ ] (Optional) Create patches if images are large
- [ ] Calculate class weights for your data
- [ ] Create custom config file with your species
- [ ] Update all file paths in config
- [ ] Launch training with W&B logging
- [ ] Monitor training metrics
- [ ] Export best checkpoint for inference
- [ ] Test on new images

---

## Quick Reference Commands

```bash
# Convert COCO to CSV
python tools/coco_to_csv.py annotations.json output.csv

# Create patches
python tools/patcher.py /images 512 512 160 /patches -csv data.csv -min 1

# Train
python tools/train.py train=my_species

# Test
python tools/test.py test=my_species

# Inference
python tools/infer.py /test/images model.pth

# View results
python tools/view.py /images groundtruth.csv -dets detections.csv
```

---

**For detailed config documentation, see**: `doc/configs_train.md`
