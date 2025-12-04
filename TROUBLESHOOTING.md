# Troubleshooting Guide

## Common Issues and Solutions

### Python Version Compatibility

**Issue**: `ERROR: No matching distribution found for SoccerNet>=2.0.0`

**Cause**: SoccerNet package requires Python 3.8-3.11. The package versioning uses 0.x format (latest is 0.1.62), not 2.0.0.

**Solutions**:

1. **Check your Python version**:
   ```bash
   python --version
   ```
   Should be 3.8, 3.9, 3.10, or 3.11.

2. **If using Python 3.12+**, create a virtual environment with Python 3.11:

   **Linux/Mac**:
   ```bash
   # Install Python 3.11 if not available
   sudo apt install python3.11 python3.11-venv python3.11-pip
   
   # Create venv with Python 3.11
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **Windows**:
   ```bash
   # Download Python 3.11 from python.org
   # Create venv with Python 3.11
   py -3.11 -m venv venv
   venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Or use conda**:
   ```bash
   conda create -n scorevision python=3.11
   conda activate scorevision
   pip install -r requirements.txt
   ```

4. **Manual install of SoccerNet** (if other methods fail):
   ```bash
   pip install SoccerNet>=0.1.62
   ```

### Dataset Download Issues

**Issue**: Cannot download SoccerNet dataset

**Solutions**:
- SoccerNet v3 is public - no NDA required
- Check internet connection
- Verify SoccerNet package is installed: `pip list | grep SoccerNet`
- Try downloading manually from: https://www.soccer-net.org/data

**Issue**: Download is too slow

**Solutions**:
- Use `--max-games` to limit download size
- Download only training split first: `--splits train`
- Resume interrupted downloads (they should resume automatically)

### Memory/Storage Issues

**Issue**: Out of memory during training

**Solutions**:
- Reduce batch size: `--batch-size 16` (or lower)
- Reduce image size: `--imgsz 640` (but quality will be lower)
- Use gradient accumulation
- Close other applications using GPU

**Issue**: Disk space running out

**Solutions**:
- Reduce target dataset size: `--target-size-gb 50`
- Download fewer games: `--max-games 30`
- Clean up intermediate files after preprocessing
- Use external storage and symlink

### Training Issues

**Issue**: Training is too slow

**Solutions**:
- Check GPU utilization: `nvidia-smi`
- Ensure CUDA is properly installed
- Reduce batch size if causing memory issues
- Use mixed precision (enabled by default in YOLO 8.3+)
- Close other GPU processes

**Issue**: Model accuracy is low

**Solutions**:
- Increase training epochs
- Use larger model architecture: `--architecture yolov8m`
- Check data quality (verify annotations are correct)
- Increase dataset size (more games)
- Adjust learning rate: `--lr 0.0005`

### Preprocessing Issues

**Issue**: No frames found during preprocessing

**Solutions**:
- Verify SoccerNet v3 download completed
- Check directory structure: should have `1_ResizedFrames` folders
- Verify frame files exist: `ls data/raw/train/*/1_ResizedFrames/`
- Check annotation files exist: `ls data/raw/train/*/Labels-v3.json`

**Issue**: Preprocessing is slow

**Solutions**:
- Reduce frame sampling: `--frame-sampling 20`
- Reduce max frames: `--max-frames-per-game 100`
- Process splits separately
- Use SSD storage for faster I/O

### Dependency Issues

**Issue**: Package conflicts during installation

**Solutions**:
- Use fresh virtual environment
- Upgrade pip: `pip install --upgrade pip`
- Install packages one by one to identify conflicts
- Use conda for better dependency resolution

**Issue**: CUDA/PyTorch version mismatch

**Solutions**:
- Verify CUDA version: `nvidia-smi`
- Install matching PyTorch version:
  ```bash
  # For CUDA 11.8
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  
  # For CUDA 12.1
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

### Other Issues

**Issue**: Scripts not executable

**Solutions**:
```bash
chmod +x scripts/*.py
```

**Issue**: Permission denied errors

**Solutions**:
- Check file permissions
- Use `sudo` only if necessary (avoid if possible)
- Ensure user has write access to data directories

**Issue**: Import errors

**Solutions**:
- Verify virtual environment is activated
- Reinstall packages: `pip install -r requirements.txt --force-reinstall`
- Check Python path: `python -c "import sys; print(sys.path)"`

## Getting Help

If you encounter issues not covered here:

1. Check the main README.md for detailed instructions
2. Review error messages carefully - they often contain helpful hints
3. Check GitHub issues for SoccerNet and Ultralytics YOLO
4. Verify your system meets all prerequisites
5. Try starting with a minimal example to isolate the issue

## System Requirements Checklist

Before starting, verify:

- [ ] Python 3.8-3.11 installed
- [ ] NVIDIA GPU with CUDA support
- [ ] CUDA toolkit installed (11.8 or 12.x)
- [ ] At least 100GB free disk space
- [ ] Stable internet connection
- [ ] Virtual environment created and activated
- [ ] All dependencies installed

## Performance Optimization Tips

1. **Use SSD** for dataset storage (much faster than HDD)
2. **Increase workers** for data loading: `--workers 16` (if CPU allows)
3. **Monitor GPU** usage: `watch -n 1 nvidia-smi`
4. **Use TensorBoard** to monitor training: `tensorboard --logdir logs/`
5. **Resume training** from checkpoints to avoid restarting
6. **Preprocess data** once and reuse for multiple training runs

