#!/usr/bin/env python3
"""
Preprocess SoccerNet data and convert to YOLO format.

This script extracts frames from videos, converts annotations to YOLO format,
and prepares data for training player, pitch, and ball detection models.
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import shutil
from collections import defaultdict
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Class mappings for ScoreVision
CLASS_MAPPING = {
    "player": 2,
    "goalkeeper": 1,
    "referee": 3,
    "ball": 0
}

# Pitch keypoint indices (32 keypoints as per ScoreVision)
PITCH_KEYPOINTS = 32


def process_soccernet_v3_frames(
    frames_dir: Path,
    output_dir: Path,
    frame_sampling: int = 10,  # Sample every Nth frame
    max_frames: Optional[int] = 200,  # Limit frames per game
    target_size: int = 1280  # Target image size (keep quality)
) -> List[str]:
    """
    Process SoccerNet v3 pre-extracted frames.
    
    Args:
        frames_dir: Directory containing SoccerNet v3 frames
        output_dir: Directory to save processed frames
        frame_sampling: Sample every Nth frame
        max_frames: Maximum frames to process per game
        target_size: Target image size (maintains quality)
        
    Returns:
        List of processed frame filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all frame images
    frame_images = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    
    if not frame_images:
        logger.warning(f"No frames found in {frames_dir}")
        return []
    
    frame_files = []
    saved_count = 0
    
    for idx, frame_path in enumerate(frame_images):
        if idx % frame_sampling != 0:
            continue
        
        if max_frames and saved_count >= max_frames:
            break
        
        # Read and resize to target size (maintain aspect ratio)
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Resize to target size while maintaining aspect ratio
        if max(h, w) != target_size:
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Save processed frame
        frame_filename = f"{frame_path.stem}.jpg"
        frame_output_path = output_dir / frame_filename
        cv2.imwrite(str(frame_output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_files.append(frame_filename)
        saved_count += 1
    
    logger.info(f"Processed {len(frame_files)} frames from {frames_dir.name}")
    return frame_files


def convert_bbox_to_yolo(
    bbox: List[float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from pixel coordinates to YOLO format (normalized center, width, height).
    
    Args:
        bbox: [x1, y1, x2, y2] in pixel coordinates
        img_width: Image width
        img_height: Image height
        
    Returns:
        (center_x, center_y, width, height) normalized to [0, 1]
    """
    x1, y1, x2, y2 = bbox
    
    # Clamp to image bounds
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    
    # Calculate center and dimensions
    center_x = ((x1 + x2) / 2.0) / img_width
    center_y = ((y1 + y2) / 2.0) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return center_x, center_y, width, height


def convert_keypoints_to_yolo(
    keypoints: List[float],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Convert keypoints to YOLO format (normalized coordinates).
    
    Args:
        keypoints: List of [x1, y1, x2, y2, ...] in pixel coordinates
        img_width: Image width
        img_height: Image height
        
    Returns:
        Normalized keypoints [x1, y1, x2, y2, ...] in [0, 1]
    """
    normalized = []
    for i in range(0, len(keypoints), 2):
        x = keypoints[i] / img_width
        y = keypoints[i + 1] / img_height
        normalized.extend([x, y])
    
    # Pad to 32 keypoints if needed
    while len(normalized) < PITCH_KEYPOINTS * 2:
        normalized.extend([0.0, 0.0])
    
    return normalized[:PITCH_KEYPOINTS * 2]


def load_soccernet_annotations(annotation_path: Path) -> Dict:
    """Load SoccerNet annotation file."""
    try:
        with open(annotation_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load annotations from {annotation_path}: {e}")
        return {}


def process_player_detection(
    input_dir: Path,
    output_dir: Path,
    frame_sampling: int = 10,  # Sample every Nth frame
    max_frames_per_game: int = 200,  # Limit frames per game
    target_size: int = 1280  # Keep image size at 1280
):
    """Process data for player detection model from SoccerNet v3 frames."""
    logger.info("Processing player detection data from SoccerNet v3...")
    
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all game directories (SoccerNet v3 structure)
    game_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    for game_dir in tqdm(game_dirs, desc="Processing games"):
        # Find frames directory for this game
        frames_dir = game_dir / "1_ResizedFrames"  # SoccerNet v3 structure
        if not frames_dir.exists():
            frames_dir = game_dir / "frames"
        if not frames_dir.exists():
            continue
        
        # Process frames
        frame_files = process_soccernet_v3_frames(
            frames_dir, images_dir, frame_sampling, max_frames_per_game, target_size
        )
        
        # Find annotation file (Labels-v3.json)
        annotation_path = game_dir / "Labels-v3.json"
        if not annotation_path.exists():
            annotation_path = game_dir / "Labels.json"
        
        annotations = load_soccernet_annotations(annotation_path) if annotation_path.exists() else {}
        
        # Process each frame
        for frame_file in frame_files:
            frame_path = images_dir / frame_file
            frame_name = Path(frame_file).stem
            
            # Load frame to get dimensions
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Create YOLO label file
            label_file = labels_dir / f"{frame_name}.txt"
            
            # Get annotations for this frame (SoccerNet v3 format)
            # Frame names might be like "frame_000001.jpg" or just numbers
            frame_idx = None
            if "_" in frame_name:
                try:
                    frame_idx = int(frame_name.split("_")[-1])
                except:
                    pass
            else:
                try:
                    frame_idx = int(frame_name)
                except:
                    pass
            
            frame_annotations = {}
            if frame_idx is not None:
                # Try different annotation formats
                if "annotations" in annotations:
                    frame_annotations = annotations["annotations"].get(str(frame_idx), {})
                elif str(frame_idx) in annotations:
                    frame_annotations = annotations[str(frame_idx)]
            
            objects = frame_annotations.get("objects", [])
            
            with open(label_file, 'w') as f:
                for obj in objects:
                    class_name = obj.get("class", "").lower()
                    if class_name not in CLASS_MAPPING:
                        continue
                    
                    class_id = CLASS_MAPPING[class_name]
                    bbox = obj.get("bbox", [])
                    
                    if len(bbox) == 4:
                        cx, cy, width, height = convert_bbox_to_yolo(bbox, w, h)
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n")
    
    logger.info(f"Player detection data processed. Images: {len(list(images_dir.glob('*.jpg')))}")


def process_pitch_detection(
    input_dir: Path,
    output_dir: Path,
    frame_sampling: int = 20,  # Sample every Nth frame
    max_frames_per_game: int = 100,  # Limit frames per game
    target_size: int = 1280  # Keep image size at 1280
):
    """Process data for pitch keypoint detection model from SoccerNet v3 frames."""
    logger.info("Processing pitch keypoint detection data from SoccerNet v3...")
    
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all game directories
    game_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    for game_dir in tqdm(game_dirs, desc="Processing games"):
        frames_dir = game_dir / "1_ResizedFrames"
        if not frames_dir.exists():
            frames_dir = game_dir / "frames"
        if not frames_dir.exists():
            continue
        
        frame_files = process_soccernet_v3_frames(
            frames_dir, images_dir, frame_sampling, max_frames_per_game, target_size
        )
        
        annotation_path = game_dir / "Labels-v3.json"
        if not annotation_path.exists():
            annotation_path = game_dir / "Labels.json"
        
        annotations = load_soccernet_annotations(annotation_path) if annotation_path.exists() else {}
        
        for frame_file in frame_files:
            frame_path = images_dir / frame_file
            frame_name = Path(frame_file).stem
            
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            label_file = labels_dir / f"{frame_name}.txt"
            
            # Get keypoints for this frame
            frame_idx = None
            if "_" in frame_name:
                try:
                    frame_idx = int(frame_name.split("_")[-1])
                except:
                    pass
            else:
                try:
                    frame_idx = int(frame_name)
                except:
                    pass
            
            keypoints = []
            if frame_idx is not None:
                if "annotations" in annotations:
                    frame_annotations = annotations["annotations"].get(str(frame_idx), {})
                    keypoints = frame_annotations.get("keypoints", [])
                elif str(frame_idx) in annotations:
                    frame_annotations = annotations[str(frame_idx)]
                    keypoints = frame_annotations.get("keypoints", [])
            
            if len(keypoints) >= 4:  # Need at least 2 keypoints
                normalized_kpts = convert_keypoints_to_yolo(keypoints, w, h)
                
                with open(label_file, 'w') as f:
                    # YOLO keypoint format: class_id x1 y1 v1 x2 y2 v2 ...
                    line = "0 "  # class_id for pitch
                    for i in range(0, len(normalized_kpts), 2):
                        x = normalized_kpts[i]
                        y = normalized_kpts[i + 1]
                        v = 2 if (x > 0 and y > 0) else 0
                        line += f"{x:.6f} {y:.6f} {v} "
                    f.write(line.strip() + "\n")
    
    logger.info(f"Pitch keypoint data processed. Images: {len(list(images_dir.glob('*.jpg')))}")


def process_ball_detection(
    input_dir: Path,
    output_dir: Path,
    frame_sampling: int = 5,  # Sample every Nth frame
    max_frames_per_game: int = 300,  # Limit frames per game
    target_size: int = 1280  # Keep image size at 1280
):
    """Process data for ball detection model from SoccerNet v3 frames."""
    logger.info("Processing ball detection data from SoccerNet v3...")
    
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all game directories
    game_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    for game_dir in tqdm(game_dirs, desc="Processing games"):
        frames_dir = game_dir / "1_ResizedFrames"
        if not frames_dir.exists():
            frames_dir = game_dir / "frames"
        if not frames_dir.exists():
            continue
        
        frame_files = process_soccernet_v3_frames(
            frames_dir, images_dir, frame_sampling, max_frames_per_game, target_size
        )
        
        annotation_path = game_dir / "Labels-v3.json"
        if not annotation_path.exists():
            annotation_path = game_dir / "Labels.json"
        
        annotations = load_soccernet_annotations(annotation_path) if annotation_path.exists() else {}
        
        for frame_file in frame_files:
            frame_path = images_dir / frame_file
            frame_name = Path(frame_file).stem
            
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            label_file = labels_dir / f"{frame_name}.txt"
            
            # Get annotations for this frame
            frame_idx = None
            if "_" in frame_name:
                try:
                    frame_idx = int(frame_name.split("_")[-1])
                except:
                    pass
            else:
                try:
                    frame_idx = int(frame_name)
                except:
                    pass
            
            objects = []
            if frame_idx is not None:
                if "annotations" in annotations:
                    frame_annotations = annotations["annotations"].get(str(frame_idx), {})
                    objects = frame_annotations.get("objects", [])
                elif str(frame_idx) in annotations:
                    frame_annotations = annotations[str(frame_idx)]
                    objects = frame_annotations.get("objects", [])
            
            with open(label_file, 'w') as f:
                for obj in objects:
                    class_name = obj.get("class", "").lower()
                    if class_name == "ball":
                        bbox = obj.get("bbox", [])
                        if len(bbox) == 4:
                            cx, cy, width, height = convert_bbox_to_yolo(bbox, w, h)
                            f.write(f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n")
    
    logger.info(f"Ball detection data processed. Images: {len(list(images_dir.glob('*.jpg')))}")


def create_yolo_dataset_config(
    output_dir: Path,
    task: str,
    classes: List[str]
):
    """Create YOLO dataset configuration file."""
    config_path = output_dir / "dataset.yaml"
    
    config = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes
    }
    
    if task == "pitch":
        config["kpt_shape"] = [PITCH_KEYPOINTS, 3]  # [num_keypoints, x, y, visibility]
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created dataset config: {config_path}")


def split_dataset(data_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split dataset into train/val/test sets."""
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    
    # Get all image files
    image_files = sorted(list(images_dir.glob("*.jpg")))
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(image_files)
    
    # Split
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Create split directories
    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(exist_ok=True)
        (labels_dir / split).mkdir(exist_ok=True)
    
    # Move files
    for files, split in [(train_files, "train"), (val_files, "val"), (test_files, "test")]:
        for img_file in files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            if label_file.exists():
                shutil.move(str(img_file), str(images_dir / split / img_file.name))
                shutil.move(str(label_file), str(labels_dir / split / label_file.name))
    
    logger.info(f"Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess SoccerNet data for YOLO training")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory with raw SoccerNet data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["player", "pitch", "ball"],
        help="Task to process"
    )
    parser.add_argument(
        "--frame-sampling",
        type=int,
        default=None,
        help="Sample every Nth frame (default: 10 for player, 20 for pitch, 5 for ball)"
    )
    parser.add_argument(
        "--max-frames-per-game",
        type=int,
        default=None,
        help="Maximum frames to process per game (default: 200 for player, 100 for pitch, 300 for ball)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1280,
        help="Target image size (default: 1280, maintains quality)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Set defaults based on task
    if args.frame_sampling is None:
        if args.task == "player":
            args.frame_sampling = 10
        elif args.task == "pitch":
            args.frame_sampling = 20
        elif args.task == "ball":
            args.frame_sampling = 5
    
    if args.max_frames_per_game is None:
        if args.task == "player":
            args.max_frames_per_game = 200
        elif args.task == "pitch":
            args.max_frames_per_game = 100
        elif args.task == "ball":
            args.max_frames_per_game = 300
    
    # Process based on task (SoccerNet v3 uses frames, not videos)
    if args.task == "player":
        process_player_detection(input_dir, output_dir, args.frame_sampling, args.max_frames_per_game, args.target_size)
        classes = ["ball", "goalkeeper", "player", "referee"]
    elif args.task == "pitch":
        process_pitch_detection(input_dir, output_dir, args.frame_sampling, args.max_frames_per_game, args.target_size)
        classes = ["pitch"]
    elif args.task == "ball":
        process_ball_detection(input_dir, output_dir, args.frame_sampling, args.max_frames_per_game, args.target_size)
        classes = ["ball"]
    
    # Split dataset
    split_dataset(output_dir, args.train_ratio, args.val_ratio)
    
    # Create dataset config
    create_yolo_dataset_config(output_dir, args.task, classes)
    
    logger.info(f"Preprocessing complete! Data saved to: {output_dir}")


if __name__ == "__main__":
    main()

