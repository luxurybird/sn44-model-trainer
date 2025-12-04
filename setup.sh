#!/bin/bash
# Setup script for custom model training pipeline

set -e

echo "=========================================="
echo "ScoreVision Custom Model Training Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p logs

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x scripts/*.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Sign SoccerNet NDA at https://www.soccer-net.org/data"
echo "3. Download dataset: python scripts/download_dataset.py --output-dir data/raw"
echo "4. Preprocess data: python scripts/preprocess_data.py --input-dir data/raw --output-dir data/processed/player --task player"
echo "5. Train models: python scripts/train_player.py --data-dir data/processed/player"
echo ""
echo "See README.md for detailed instructions."

