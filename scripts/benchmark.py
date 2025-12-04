#!/usr/bin/env python3
"""
Benchmark trained models for ScoreVision.

This script evaluates model performance including inference speed, accuracy,
and compares with baseline models.
"""

import argparse
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from ultralytics import YOLO
import logging
from tqdm import tqdm
import supervision as sv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_inference_speed(
    model: YOLO,
    test_images: List[Path],
    batch_size: int = 1,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Benchmark inference speed.
    
    Args:
        model: YOLO model
        test_images: List of test image paths
        batch_size: Batch size for inference
        warmup: Number of warmup iterations
        
    Returns:
        Dictionary with speed metrics
    """
    logger.info(f"Benchmarking inference speed with {len(test_images)} images...")
    
    # Warmup
    logger.info(f"Warming up with {warmup} iterations...")
    for i in range(warmup):
        if i < len(test_images):
            img = cv2.imread(str(test_images[i]))
            if img is not None:
                _ = model(img, verbose=False)
    
    # Actual benchmark
    times = []
    device = next(model.model.parameters()).device
    
    for img_path in tqdm(test_images[:100], desc="Benchmarking"):  # Limit to 100 for speed
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # GPU sync
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        _ = model(img, verbose=False)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    return {
        "avg_latency_ms": avg_time * 1000,
        "std_latency_ms": std_time * 1000,
        "min_latency_ms": min_time * 1000,
        "max_latency_ms": max_time * 1000,
        "fps": fps,
        "throughput": fps
    }


def evaluate_accuracy(
    model: YOLO,
    test_images: List[Path],
    test_labels: List[Path],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Dict[str, float]:
    """
    Evaluate model accuracy.
    
    Args:
        model: YOLO model
        test_images: List of test image paths
        test_labels: List of corresponding label paths
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        
    Returns:
        Dictionary with accuracy metrics
    """
    logger.info(f"Evaluating accuracy on {len(test_images)} images...")
    
    # Use YOLO's built-in validation
    if len(test_images) > 0:
        # Create temporary dataset config
        temp_config = Path("temp_dataset.yaml")
        dataset_dir = test_images[0].parent.parent
        
        # Run validation
        try:
            results = model.val(
                data=str(dataset_dir / "dataset.yaml"),
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=True
            )
            
            metrics = {
                "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
                "mAP50-95": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
                "precision": float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
                "recall": float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
            }
            
            return metrics
        except Exception as e:
            logger.warning(f"Could not run full validation: {e}")
            return {
                "mAP50": 0.0,
                "mAP50-95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
    
    return {
        "mAP50": 0.0,
        "mAP50-95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }


def get_model_size(model_path: Path) -> float:
    """Get model file size in MB."""
    if model_path.exists():
        return model_path.stat().st_size / (1024 * 1024)
    return 0.0


def benchmark_model(
    model_path: Path,
    test_data_dir: Path,
    device: str = "0",
    batch_size: int = 1,
    num_samples: int = 100
) -> Dict:
    """
    Comprehensive model benchmarking.
    
    Args:
        model_path: Path to model file
        test_data_dir: Directory with test data
        device: GPU device ID
        batch_size: Batch size
        num_samples: Number of samples to test
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking model: {model_path}")
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return {}
    
    # Load model
    model = YOLO(str(model_path))
    model.to(device)
    
    # Find test images
    test_images_dir = test_data_dir / "images" / "test"
    if not test_images_dir.exists():
        test_images_dir = test_data_dir / "test" / "images"
    
    if not test_images_dir.exists():
        logger.warning(f"Test images directory not found: {test_images_dir}")
        return {}
    
    test_images = list(test_images_dir.glob("*.jpg"))[:num_samples]
    
    if len(test_images) == 0:
        logger.error("No test images found")
        return {}
    
    logger.info(f"Found {len(test_images)} test images")
    
    # Benchmark speed
    speed_metrics = benchmark_inference_speed(model, test_images, batch_size)
    
    # Evaluate accuracy
    test_labels_dir = test_data_dir / "labels" / "test"
    if not test_labels_dir.exists():
        test_labels_dir = test_data_dir / "test" / "labels"
    
    test_labels = []
    if test_labels_dir.exists():
        for img_path in test_images:
            label_path = test_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                test_labels.append(label_path)
    
    accuracy_metrics = {}
    if len(test_labels) > 0:
        accuracy_metrics = evaluate_accuracy(model, test_images, test_labels)
    
    # Model size
    model_size_mb = get_model_size(model_path)
    
    # Memory usage
    memory_usage = {}
    if torch.cuda.is_available():
        memory_usage = {
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        }
    
    results = {
        "model_path": str(model_path),
        "model_size_mb": model_size_mb,
        "speed": speed_metrics,
        "accuracy": accuracy_metrics,
        "memory": memory_usage,
    }
    
    return results


def compare_with_baseline(
    custom_results: Dict,
    baseline_path: Path = None
) -> Dict:
    """
    Compare custom model with baseline.
    
    Args:
        custom_results: Results from custom model
        baseline_path: Path to baseline model (optional)
        
    Returns:
        Comparison dictionary
    """
    comparison = {
        "custom": custom_results,
    }
    
    if baseline_path and baseline_path.exists():
        logger.info(f"Comparing with baseline: {baseline_path}")
        baseline_results = benchmark_model(
            baseline_path,
            Path(custom_results.get("test_data_dir", ".")),
            num_samples=50  # Fewer samples for baseline
        )
        comparison["baseline"] = baseline_results
        
        # Calculate improvements
        if "speed" in custom_results and "speed" in baseline_results:
            custom_fps = custom_results["speed"].get("fps", 0)
            baseline_fps = baseline_results["speed"].get("fps", 0)
            if baseline_fps > 0:
                speed_improvement = ((custom_fps - baseline_fps) / baseline_fps) * 100
                comparison["speed_improvement_percent"] = speed_improvement
        
        if "accuracy" in custom_results and "accuracy" in baseline_results:
            custom_map = custom_results["accuracy"].get("mAP50", 0)
            baseline_map = baseline_results["accuracy"].get("mAP50", 0)
            if baseline_map > 0:
                accuracy_improvement = ((custom_map - baseline_map) / baseline_map) * 100
                comparison["accuracy_improvement_percent"] = accuracy_improvement
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Benchmark trained models")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Directory with test data"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device ID"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of test samples"
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Directory with baseline models for comparison"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    test_data_dir = Path(args.test_data)
    
    all_results = {}
    
    # Benchmark each model type
    model_types = ["player", "pitch", "ball"]
    
    for model_type in model_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking {model_type} model")
        logger.info(f"{'='*60}")
        
        # Find model file
        model_files = list(model_dir.rglob(f"*{model_type}*.pt"))
        if not model_files:
            model_files = list(model_dir.rglob(f"best.pt"))
        
        if not model_files:
            logger.warning(f"No {model_type} model found in {model_dir}")
            continue
        
        model_path = model_files[0]
        logger.info(f"Using model: {model_path}")
        
        # Find corresponding test data
        test_data = test_data_dir / model_type
        if not test_data.exists():
            test_data = test_data_dir
        
        results = benchmark_model(
            model_path=model_path,
            test_data_dir=test_data,
            device=args.device,
            batch_size=args.batch_size,
            num_samples=args.num_samples
        )
        
        results["test_data_dir"] = str(test_data)
        all_results[model_type] = results
        
        # Print results
        logger.info(f"\n{model_type.upper()} Model Results:")
        logger.info(f"  Model Size: {results.get('model_size_mb', 0):.2f} MB")
        if "speed" in results:
            logger.info(f"  Inference Speed: {results['speed'].get('fps', 0):.2f} FPS")
            logger.info(f"  Avg Latency: {results['speed'].get('avg_latency_ms', 0):.2f} ms")
        if "accuracy" in results:
            logger.info(f"  mAP@0.5: {results['accuracy'].get('mAP50', 0):.4f}")
            logger.info(f"  mAP@0.5:0.95: {results['accuracy'].get('mAP50-95', 0):.4f}")
            logger.info(f"  Precision: {results['accuracy'].get('precision', 0):.4f}")
            logger.info(f"  Recall: {results['accuracy'].get('recall', 0):.4f}")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nBenchmark results saved to: {output_path}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*60}")
    for model_type, results in all_results.items():
        logger.info(f"\n{model_type.upper()}:")
        if "speed" in results:
            logger.info(f"  FPS: {results['speed'].get('fps', 0):.2f}")
        if "accuracy" in results:
            logger.info(f"  mAP@0.5: {results['accuracy'].get('mAP50', 0):.4f}")


if __name__ == "__main__":
    main()

