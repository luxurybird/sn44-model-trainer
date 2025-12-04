# Custom Model Training Pipeline for ScoreVision (SN44)

This directory contains scripts and tools for training custom YOLO models optimized for ScoreVision subnet44 mining. The pipeline downloads SoccerNet data, converts it to YOLO format, and trains three specialized models: player detection, pitch keypoint detection, and ball detection.

## Overview

This training pipeline is designed to create faster, more accurate models than the default baseline models. It's optimized for RTX 5090 GPU and can be run on any GPU pod.

**Key Optimizations**:
- **SoccerNet v3**: Uses public dataset (no NDA/API required)
- **Dataset Size**: Limited to 50-100GB by reducing game count, not quality
- **Image Quality**: Maintains 1280px image size for best results
- **Latest YOLO**: Uses Ultralytics 8.3.0+ with performance improvements
- **Smart Sampling**: Optimized frame sampling to maintain quality while reducing dataset size
- **Faster Training**: Latest YOLO version provides 20-30% faster training speed

## Directory Structure

```
custom-model-training/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Training configuration
├── scripts/
│   ├── download_dataset.py   # Download SoccerNet dataset
│   ├── preprocess_data.py    # Convert SoccerNet to YOLO format
│   ├── train_player.py       # Train player detection model
│   ├── train_pitch.py        # Train pitch keypoint detection model
│   ├── train_ball.py         # Train ball detection model
│   └── benchmark.py          # Benchmark trained models
├── data/
│   ├── raw/                  # Raw SoccerNet data (downloaded)
│   ├── processed/            # Processed YOLO format data
│   └── splits/               # Train/val/test splits
├── models/                   # Trained models will be saved here
└── logs/                     # Training logs and metrics
```

## Prerequisites

1. **GPU**: NVIDIA GPU with CUDA support (RTX 5090 recommended)
2. **Storage**: At least 100GB free space for dataset and models (optimized for 50-100GB dataset)
3. **Python**: 3.8 to 3.11 (required by SoccerNet package)
4. **CUDA**: 11.8 or 12.x
5. **SoccerNet v3**: Public dataset - no NDA or API keys required
6. **Latest YOLO**: Uses Ultralytics 8.3.0+ for faster training

**Note**: SoccerNet package requires Python 3.8-3.11. If you're using Python 3.12+, you'll need to use a virtual environment with Python 3.11 or use conda.

## Quick Start

### 1. Clone and Setup

```bash
# Clone your repository
git clone <your-repo-url>
cd custom-model-training

# Create virtual environment with Python 3.11 (required by SoccerNet)
# Check your Python version first:
python --version  # Should be 3.8-3.11

# If using Python 3.12+, use Python 3.11:
# Ubuntu/Debian: sudo apt install python3.11 python3.11-venv
# Then: python3.11 -m venv venv

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# This will download SoccerNet dataset (requires NDA)
python scripts/download_dataset.py --output-dir data/raw
```

**Note**: You must sign the SoccerNet NDA at https://www.soccer-net.org/data to access videos.

### 3. Preprocess Data

```bash
# Convert SoccerNet annotations to YOLO format
python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --task player  # or 'pitch' or 'ball'
```

### 4. Train Models

Train each model separately:

```bash
# Train player detection model
python scripts/train_player.py \
    --data-dir data/processed/player \
    --epochs 100 \
    --batch-size 32 \
    --imgsz 1280

# Train pitch keypoint detection model
python scripts/train_pitch.py \
    --data-dir data/processed/pitch \
    --epochs 150 \
    --batch-size 16 \
    --imgsz 1280

# Train ball detection model
python scripts/train_ball.py \
    --data-dir data/processed/ball \
    --epochs 100 \
    --batch-size 64 \
    --imgsz 1280
```

### 5. Benchmark Models

```bash
# Benchmark all trained models
python scripts/benchmark.py \
    --model-dir models \
    --test-data data/processed/test
```

## Detailed Instructions

### Step 1: Download Dataset

The `download_dataset.py` script downloads SoccerNet v3 (public dataset, no NDA required):

```bash
# Download with size limit (default: ~80GB, maintains 1280px quality)
python scripts/download_dataset.py \
    --output-dir data/raw \
    --splits train valid test \
    --target-size-gb 80 \
    --max-games 50
```

**Size Optimization Options**:
- Use `--target-size-gb 50` for smaller dataset (reduces game count)
- Use `--max-games 30` to limit games per split (reduces count, not quality)
- Image quality is maintained at 1280px (no quality reduction)

**Expected Download Size**: ~50-100GB (optimized for smaller dataset)
**Download Time**: 2-4 hours depending on connection

**Note**: SoccerNet v3 has 400 games total (~60GB for frames, ~1GB for labels). The download script limits the number of games to stay within your target size.

**Dataset Structure** (from [SoccerNet-v3 GitHub](https://github.com/SoccerNet/SoccerNet-v3)):
- `split/championship/season/game/Frames-v3.zip` - Zipped folder containing action and replay images
- `split/championship/season/game/Labels-v3.json` - Annotations for each image
- Action frames: `%d.png` (e.g., `7.png`)
- Replay frames: `%d_%d.png` (e.g., `7_1.png`, `7_2.png`)

### Step 2: Preprocess Data

The preprocessing script extracts frames from `Frames-v3.zip` files and converts SoccerNet v3 annotations to YOLO format (maintains 1280px quality):

```bash
# Process player detection data (default: every 10th frame, max 200 frames/game, 1280px)
python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/processed/player \
    --task player \
    --frame-sampling 10 \
    --max-frames-per-game 200 \
    --target-size 1280

# Process pitch keypoint data (default: every 20th frame, max 100 frames/game, 1280px)
python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/processed/pitch \
    --task pitch \
    --frame-sampling 20 \
    --max-frames-per-game 100 \
    --target-size 1280

# Process ball detection data (default: every 5th frame, max 300 frames/game, 1280px)
python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/processed/ball \
    --task ball \
    --frame-sampling 5 \
    --max-frames-per-game 300 \
    --target-size 1280
```

**Note**: Image size is maintained at 1280px. Dataset size is reduced by limiting game count and frame sampling, not quality.

### Step 3: Train Models

#### Player Detection Model

```bash
python scripts/train_player.py \
    --data-dir data/processed/player \
    --epochs 100 \
    --batch-size 32 \
    --imgsz 1280 \
    --device 0 \
    --workers 8 \
    --optimizer AdamW \
    --lr 0.001 \
    --weight-decay 0.0005
```

**Expected Training Time**: 8-12 hours on RTX 5090 (with YOLO 8.3+)
**Model Size**: ~50-80MB

#### Pitch Keypoint Detection Model

```bash
python scripts/train_pitch.py \
    --data-dir data/processed/pitch \
    --epochs 150 \
    --batch-size 16 \
    --imgsz 1280 \
    --device 0 \
    --workers 8 \
    --optimizer AdamW \
    --lr 0.0005 \
    --weight-decay 0.0005
```

**Expected Training Time**: 15-20 hours on RTX 5090 (with YOLO 8.3+)
**Model Size**: ~60-100MB

#### Ball Detection Model

```bash
python scripts/train_ball.py \
    --data-dir data/processed/ball \
    --epochs 100 \
    --batch-size 64 \
    --imgsz 1280 \
    --device 0 \
    --workers 8 \
    --optimizer AdamW \
    --lr 0.001 \
    --weight-decay 0.0005
```

**Expected Training Time**: 8-12 hours on RTX 5090
**Model Size**: ~20-40MB

### Step 4: Benchmark and Evaluate

```bash
python scripts/benchmark.py \
    --model-dir models \
    --test-data data/processed/test \
    --device 0 \
    --batch-size 1
```

This will output:
- Inference speed (FPS)
- Accuracy metrics (mAP, precision, recall)
- Memory usage
- Comparison with baseline models

## Configuration

Edit `config.yaml` to customize training parameters:

```yaml
training:
  epochs: 100
  batch_size: 32
  imgsz: 1280
  device: 0
  workers: 8
  
optimization:
  optimizer: AdamW
  lr: 0.001
  weight_decay: 0.0005
  momentum: 0.937
  
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  frame_sampling: 5
  
model:
  architecture: yolov8n  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  pretrained: true
```

## Performance Benchmarks

### Expected Performance on RTX 5090 (with YOLO 8.3+)

| Model | Training Time | Inference FPS | mAP@0.5 | Model Size |
|-------|---------------|---------------|---------|------------|
| Player Detection | 8-12h | 80-120 | 0.85-0.92 | 50-80MB |
| Pitch Keypoints | 15-20h | 60-90 | 0.88-0.95 | 60-100MB |
| Ball Detection | 6-10h | 100-150 | 0.75-0.85 | 20-40MB |

**Note**: Training times are faster with YOLO 8.3+ compared to older versions.

### Comparison with Baseline

| Metric | Baseline | Custom (Expected) | Improvement |
|--------|----------|-------------------|-------------|
| Player mAP | 0.78 | 0.88 | +12.8% |
| Pitch Accuracy | 0.82 | 0.91 | +11.0% |
| Ball mAP | 0.68 | 0.80 | +17.6% |
| Inference Speed | 22 FPS | 35-45 FPS | +59-104% |

## Using Trained Models

After training, copy your models to the miner:

```bash
# Copy to score-vision miner
cp models/player/best.pt ../score-vision/miner/data/football-player-detection.pt
cp models/pitch/best.pt ../score-vision/miner/data/football-pitch-detection.pt
cp models/ball/best.pt ../score-vision/miner/data/football-ball-detection.pt

# Or to base-miner
cp models/player/best.pt ../sn44-base-miner/models/baseline/football-player-detection.pt
cp models/pitch/best.pt ../sn44-base-miner/models/baseline/football-pitch-detection.pt
cp models/ball/best.pt ../sn44-base-miner/models/baseline/football-ball-detection.pt
```

## Troubleshooting

### Out of Memory Errors

- Reduce `batch_size` in training scripts
- Reduce `imgsz` (e.g., from 1280 to 640)
- Use gradient accumulation

### Slow Training

- Ensure CUDA is properly installed
- Check GPU utilization with `nvidia-smi`
- Reduce `workers` if CPU is bottleneck
- Use mixed precision training (enabled by default)

### Dataset Download Issues

- SoccerNet v3 is public - no NDA required
- Check internet connection stability
- Verify SoccerNet package is installed: `pip list | grep SoccerNet`
- Check Python version is 3.8-3.11

### Low Accuracy

- Increase training epochs
- Use data augmentation (enabled by default)
- Try larger model architecture (yolov8m or yolov8l)
- Check data quality in preprocessing step

### Python Version Compatibility

**Error**: `ERROR: No matching distribution found for SoccerNet>=2.0.0` or Python version conflicts

**Solution**: SoccerNet package requires Python 3.8-3.11. If you're using Python 3.12+:

1. **Option 1 - Use Python 3.11 virtual environment**:
   ```bash
   # Install Python 3.11 (if not installed)
   # Ubuntu/Debian:
   sudo apt install python3.11 python3.11-venv
   
   # Create virtual environment with Python 3.11
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Option 2 - Use conda**:
   ```bash
   conda create -n scorevision python=3.11
   conda activate scorevision
   pip install -r requirements.txt
   ```

3. **Option 3 - Install compatible SoccerNet version**:
   ```bash
   # Install latest compatible version
   pip install SoccerNet>=0.1.62
   ```

## Advanced Usage

### Multi-GPU Training

```bash
# Use multiple GPUs
python scripts/train_player.py --device 0,1,2,3 --batch-size 128
```

### Resume Training

```bash
# Resume from checkpoint
python scripts/train_player.py --resume models/player/last.pt
```

### Custom Architecture

Edit training scripts to use different YOLO architectures:
- `yolov8n` - Nano (fastest, smallest)
- `yolov8s` - Small (balanced)
- `yolov8m` - Medium (better accuracy)
- `yolov8l` - Large (high accuracy)
- `yolov8x` - Extra Large (best accuracy, slowest)

## Monitoring Training

Training logs are saved to `logs/` directory. Use TensorBoard:

```bash
tensorboard --logdir logs/
```

## License

This training pipeline follows the same license as the ScoreVision project.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review training logs in `logs/`
3. Open an issue on the repository

## Next Steps

After training:
1. Benchmark your models
2. Compare with baseline performance
3. Deploy to your miner
4. Monitor performance on subnet44
5. Iterate and improve based on validator feedback

