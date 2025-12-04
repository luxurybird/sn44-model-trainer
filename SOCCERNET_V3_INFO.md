# SoccerNet v3 Dataset Information

This training pipeline uses the [SoccerNet v3 dataset](https://github.com/SoccerNet/SoccerNet-v3), which is publicly available under MIT license.

## Dataset Overview

- **Total Games**: 400 games
- **Total Size**: ~60GB for frames, ~1GB for labels
- **License**: MIT (public, no NDA required)
- **Repository**: https://github.com/SoccerNet/SoccerNet-v3

## Directory Structure

SoccerNet v3 follows this structure:

```
data/raw/
├── train/
│   ├── england_epl/
│   │   ├── 2014-2015/
│   │   │   ├── 2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/
│   │   │   │   ├── Frames-v3.zip      # Zipped folder with all images
│   │   │   │   └── Labels-v3.json     # Annotations for all images
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── valid/
└── test/
```

## Image Naming Convention

- **Action frames**: `%d.png` (e.g., `7.png`, `8.png`)
- **Replay frames**: `%d_%d.png` (e.g., `7_1.png`, `7_2.png` for replays of action 7)

## Annotation Format

The `Labels-v3.json` file contains annotations for each frame with:
- **Bounding boxes**: For players, goalkeepers, referees, and ball
- **Lines**: Pitch line annotations
- **Correspondences**: Links between bounding boxes across replay and action frames

## Download Method

The dataset is downloaded using the SoccerNet Python package:

```python
from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(LocalDirectory="path/to/SoccerNet")
downloader.downloadGames(
    files=["Labels-v3.json", "Frames-v3.zip"],
    split=["train", "valid", "test"],
    task="frames"
)
```

## Processing

Our preprocessing script:
1. Extracts frames from `Frames-v3.zip` files
2. Processes only action frames (skips replay frames)
3. Resizes to 1280px while maintaining aspect ratio
4. Converts annotations to YOLO format
5. Creates train/val/test splits

## References

- Official Repository: https://github.com/SoccerNet/SoccerNet-v3
- Paper: Check the repository for citation information
- License: MIT License

