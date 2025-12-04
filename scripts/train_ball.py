#!/usr/bin/env python3
"""
Train custom ball detection model for ScoreVision.

This script trains a YOLO model specifically for ball detection.
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_ball_model(
    data_dir: str,
    epochs: int = 100,
    batch_size: int = 64,
    imgsz: int = 1280,
    device: str = "0",
    workers: int = 8,
    architecture: str = "yolov8n",
    lr: float = 0.001,
    weight_decay: float = 0.0005,
    resume: str = None,
    project: str = "custom-model-training",
    name: str = "ball-detection"
):
    """
    Train ball detection model.
    
    Args:
        data_dir: Directory with YOLO format dataset
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Image size
        device: GPU device ID or 'cpu'
        workers: Number of data loading workers
        architecture: YOLO architecture (yolov8n, yolov8s, etc.)
        lr: Learning rate
        weight_decay: Weight decay
        resume: Path to checkpoint to resume from
        project: Project name
        name: Run name
    """
    data_path = Path(data_dir)
    dataset_yaml = data_path / "dataset.yaml"
    
    if not dataset_yaml.exists():
        logger.error(f"Dataset config not found: {dataset_yaml}")
        logger.error("Please run preprocess_data.py first")
        return
    
    # Load model
    model_name = f"{architecture}.pt"
    logger.info(f"Loading model: {model_name}")
    
    if resume:
        logger.info(f"Resuming training from: {resume}")
        model = YOLO(resume)
    else:
        model = YOLO(model_name)
    
    # Training parameters - optimized for small object detection
    train_params = {
        "data": str(dataset_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "project": project,
        "name": name,
        "optimizer": "AdamW",
        "lr0": lr,
        "weight_decay": weight_decay,
        "momentum": 0.937,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,  # Box loss gain
        "cls": 0.5,  # Class loss gain
        "dfl": 1.5,  # DFL loss gain
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,  # Important for small ball detection
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "patience": 50,
        "save_period": 10,
        "val": True,
        "plots": True,
        "verbose": True,
        # Ball-specific optimizations
        "anchor_t": 4.0,  # Anchor-multiple threshold
        "fl_gamma": 0.0,  # Focal loss gamma
    }
    
    logger.info("Starting training...")
    logger.info(f"Dataset: {dataset_yaml}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}")
    logger.info(f"Device: {device}, Workers: {workers}")
    
    # Train
    results = model.train(**train_params)
    
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {model.trainer.best}")
    logger.info(f"Results: {results}")
    
    # Export best model
    best_model_path = Path(model.trainer.best)
    if best_model_path.exists():
        export_path = Path(project) / name / "best_ball.pt"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(best_model_path, export_path)
        logger.info(f"Exported best model to: {export_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ball detection model")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with YOLO format dataset"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Image size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device ID or 'cpu'"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLO architecture"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Weight decay"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="custom-model-training",
        help="Project name"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ball-detection",
        help="Run name"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (optional)"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            if 'training' in config:
                args.epochs = config['training'].get('epochs', args.epochs)
                args.batch_size = config['training'].get('batch_size', args.batch_size)
                args.imgsz = config['training'].get('imgsz', args.imgsz)
                args.device = str(config['training'].get('device', args.device))
                args.workers = config['training'].get('workers', args.workers)
            if 'optimization' in config:
                args.lr = config['optimization'].get('lr', args.lr)
                args.weight_decay = config['optimization'].get('weight_decay', args.weight_decay)
            if 'model' in config:
                args.architecture = config['model'].get('architecture', args.architecture)
            if 'tasks' in config and 'ball' in config['tasks']:
                ball_config = config['tasks']['ball']
                args.epochs = ball_config.get('epochs', args.epochs)
                args.batch_size = ball_config.get('batch_size', args.batch_size)
                args.imgsz = ball_config.get('imgsz', args.imgsz)
    
    train_ball_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        architecture=args.architecture,
        lr=args.lr,
        weight_decay=args.weight_decay,
        resume=args.resume,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()

