#!/usr/bin/env python3
"""
Download SoccerNet v3 dataset for training custom ScoreVision models.

SoccerNet v3 is publicly available and does not require NDA or API keys.
This script downloads frames and annotations directly.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import logging
from tqdm import tqdm

try:
    from SoccerNet.Downloader import SoccerNetDownloader
except ImportError:
    print("ERROR: SoccerNet package not installed. Install with: pip install SoccerNet")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_soccernet(
    output_dir: str,
    splits: List[str] = ["train", "valid", "test"],
    download_frames: bool = True,
    download_annotations: bool = True,
    max_games_per_split: Optional[int] = None,  # Limit number of games
    target_size_gb: Optional[float] = 80.0  # Target dataset size in GB
):
    """
    Download SoccerNet v3 dataset (public, no NDA required).
    
    Args:
        output_dir: Directory to save downloaded data
        splits: Dataset splits to download
        download_frames: Whether to download frame images
        download_annotations: Whether to download annotation files
        max_games_per_split: Maximum games per split (None = auto-calculate)
        target_size_gb: Target dataset size in GB (approximate)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting SoccerNet v3 download (public dataset, no NDA required)")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Splits: {splits}")
    logger.info(f"Target size: ~{target_size_gb} GB")
    
    # Size estimates per game for SoccerNet v3 frames (approximate)
    # SoccerNet v3 frames are ~150MB per game on average
    estimated_size_per_game = 0.15  # ~150MB per game for frames
    
    # Calculate max games to stay within target size
    if max_games_per_split is None:
        games_per_split = int(target_size_gb / (estimated_size_per_game * len(splits)))
        max_games_per_split = max(15, games_per_split)  # At least 15 games per split
        logger.info(f"Auto-calculated max games per split: {max_games_per_split}")
    else:
        logger.info(f"Max games per split: {max_games_per_split}")
    
    try:
        downloader = SoccerNetDownloader(LocalDirectory=str(output_path))
        
        # Download annotations and frames using SoccerNet v3 API
        if download_annotations or download_frames:
            logger.info("Downloading SoccerNet v3 frames and annotations...")
            
            for split in splits:
                logger.info(f"Downloading {split} split (max {max_games_per_split} games)...")
                try:
                    # Download frames and labels for SoccerNet v3
                    # This downloads pre-extracted frames (no video processing needed)
                    downloader.downloadGames(
                        files=["Labels-v3.json", "Frames-v3.zip"],
                        split=[split],
                        task="frames"
                    )
                    
                    # Count downloaded games and stop if limit reached
                    split_path = output_path / split
                    if split_path.exists():
                        game_dirs = [d for d in split_path.iterdir() if d.is_dir()]
                        if len(game_dirs) >= max_games_per_split:
                            logger.info(f"Reached {len(game_dirs)} games for {split}, stopping")
                            # Remove excess games if any
                            if len(game_dirs) > max_games_per_split:
                                for excess_dir in game_dirs[max_games_per_split:]:
                                    import shutil
                                    shutil.rmtree(excess_dir)
                                    logger.info(f"Removed excess game: {excess_dir.name}")
                    
                    # Check total size and stop if needed
                    current_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024**3)
                    if current_size >= target_size_gb * 0.9:  # Stop at 90% of target
                        logger.info(f"Reached ~{current_size:.1f} GB, stopping download to stay within target")
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to download {split} data: {e}")
                    logger.info("SoccerNet v3 is public - no NDA required. Check your internet connection.")
        
        logger.info("Download completed successfully!")
        logger.info(f"Data saved to: {output_path}")
        
        # Print summary
        total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
        logger.info(f"Total dataset size: {total_size / (1024**3):.2f} GB")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error("Please ensure:")
        logger.error("1. SoccerNet package is installed: pip install SoccerNet")
        logger.error("2. You have sufficient disk space (100GB+ recommended)")
        logger.error("3. You have stable internet connection")
        logger.info("Note: SoccerNet v3 is public and does not require NDA or API keys")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download SoccerNet dataset for training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
        help="Dataset splits to download"
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Skip frame download (annotations only)"
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Skip annotation download (frames only)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum games per split (auto-calculated if not specified)"
    )
    parser.add_argument(
        "--target-size-gb",
        type=float,
        default=80.0,
        help="Target dataset size in GB (default: 80GB)"
    )
    
    args = parser.parse_args()
    
    download_soccernet(
        output_dir=args.output_dir,
        splits=args.splits,
        download_frames=not args.no_frames,
        download_annotations=not args.no_annotations,
        max_games_per_split=args.max_games,
        target_size_gb=args.target_size_gb
    )


if __name__ == "__main__":
    main()

