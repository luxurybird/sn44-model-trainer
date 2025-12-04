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

# Check if Python version is 3.8-3.11 (required by SoccerNet)
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" != "3" ]; then
    echo "ERROR: Python 3 is required"
    exit 1
fi

if [ "$python_minor" -lt 8 ] || [ "$python_minor" -gt 11 ]; then
    echo "WARNING: SoccerNet requires Python 3.8-3.11"
    echo "Your version: $python_version"
    echo "Consider using Python 3.11 for compatibility"
    echo "Install with: sudo apt install python3.11 python3.11-venv"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

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
echo "2. Download dataset (SoccerNet v3 is public, no NDA required):"
echo "   python scripts/download_dataset.py --output-dir data/raw"
echo "3. Preprocess data:"
echo "   python scripts/preprocess_data.py --input-dir data/raw --output-dir data/processed/player --task player"
echo "4. Train models:"
echo "   python scripts/train_player.py --data-dir data/processed/player"
echo ""
echo "See README.md for detailed instructions."
echo "Note: SoccerNet requires Python 3.8-3.11. If issues, see TROUBLESHOOTING.md"

