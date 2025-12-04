# Quick Start Guide

This is a condensed guide for getting started quickly. For detailed information, see [README.md](README.md).

## Prerequisites

- GPU: NVIDIA GPU with CUDA (RTX 5090 recommended)
- Storage: 100GB+ free space (optimized for 50-100GB dataset)
- Python: 3.8+
- SoccerNet v3: Public dataset - no NDA or API keys required
- Latest YOLO: Uses Ultralytics 8.3.0+ for faster training

## Setup (5 minutes)

```bash
# Clone repository
git clone <your-repo-url>
cd custom-model-training

# Setup (Linux/Mac)
bash setup.sh

# Or Windows
setup.bat

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

## Training Pipeline (3 steps)

### Step 1: Download Dataset (2-4 hours)

```bash
# Download SoccerNet v3 (public, no NDA) - maintains 1280px quality
python scripts/download_dataset.py \
    --output-dir data/raw \
    --target-size-gb 80 \
    --max-games 50
```

**Note**: SoccerNet v3 is public (no NDA). Download size: ~50-100GB by reducing game count, not quality

### Step 2: Preprocess Data (2-4 hours)

```bash
# Player detection
python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/processed/player \
    --task player \
    --frame-sampling 5

# Pitch keypoints
python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/processed/pitch \
    --task pitch \
    --frame-sampling 10

# Ball detection
python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/processed/ball \
    --task ball \
    --frame-sampling 2
```

### Step 3: Train Models (12-30 hours total)

```bash
# Train player model (12-18 hours)
python scripts/train_player.py \
    --data-dir data/processed/player \
    --epochs 100 \
    --batch-size 32 \
    --device 0

# Train pitch model (20-30 hours)
python scripts/train_pitch.py \
    --data-dir data/processed/pitch \
    --epochs 150 \
    --batch-size 16 \
    --device 0

# Train ball model (8-12 hours)
python scripts/train_ball.py \
    --data-dir data/processed/ball \
    --epochs 100 \
    --batch-size 64 \
    --device 0
```

## Benchmark Models

```bash
python scripts/benchmark.py \
    --model-dir custom-model-training/player-detection \
    --test-data data/processed/player \
    --device 0
```

## Use Trained Models

```bash
# Copy to miner
cp custom-model-training/player-detection/best.pt ../score-vision/miner/data/football-player-detection.pt
cp custom-model-training/pitch-detection/best.pt ../score-vision/miner/data/football-pitch-detection.pt
cp custom-model-training/ball-detection/best.pt ../score-vision/miner/data/football-ball-detection.pt
```

## Expected Timeline

| Step | Time | Notes |
|------|------|-------|
| Setup | 5 min | One-time setup |
| Download | 2-4 hours | Optimized dataset (50-100GB) |
| Preprocess | 1-2 hours | Per task (optimized sampling) |
| Train Player | 8-12 hours | RTX 5090 (faster with YOLO 8.3+) |
| Train Pitch | 15-20 hours | RTX 5090 (faster with YOLO 8.3+) |
| Train Ball | 6-10 hours | RTX 5090 (faster with YOLO 8.3+) |
| **Total** | **32-48 hours** | ~1.5-2 days |

## Common Issues

### Out of Memory
- Reduce `--batch-size` (e.g., 16 instead of 32)
- Reduce `--imgsz` (e.g., 640 instead of 1280)

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Ensure CUDA is installed correctly
- Reduce `--workers` if CPU is bottleneck

### Dataset Download Fails
- Verify SoccerNet NDA is signed
- Check internet connection
- Use `--resume` to resume interrupted downloads

## Tips for Best Results

1. **Use larger models** for better accuracy:
   - `--architecture yolov8m` or `yolov8l`
   - Trade-off: Slower training, better results

2. **Increase epochs** if accuracy is low:
   - Player: 150 epochs
   - Pitch: 200 epochs
   - Ball: 150 epochs

3. **Fine-tune batch size**:
   - Larger batch = faster training, more memory
   - Smaller batch = slower training, less memory

4. **Monitor training**:
   ```bash
   tensorboard --logdir logs/
   ```

## Next Steps

After training:
1. Benchmark your models
2. Compare with baseline
3. Deploy to miner
4. Monitor performance on subnet44
5. Iterate based on results

For detailed information, see [README.md](README.md).

