#!/bin/bash

# Quick start script for Numin meta-learning training

set -e

echo "=================================================="
echo "Numin Meta-Learning Quick Start"
echo "=================================================="

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies if needed
echo "Installing dependencies..."
uv pip install -e . --quiet

# Create output directories
mkdir -p models logs data

echo ""
echo "=================================================="
echo "Environment ready!"
echo "=================================================="
echo ""
echo "Available training modes:"
echo ""
echo "1. Standard supervised learning:"
echo "   python train.py --lookback 10 --batch_size 32 --epochs 50"
echo ""
echo "2. Meta-learning (recommended):"
echo "   python train.py --meta --lookback 10 --batch_size 16 --k_shot 5 --epochs 50"
echo ""
echo "3. Evaluation:"
echo "   python eval.py --checkpoint models/meta_ohlcv_meta_best.pt --meta"
echo ""
echo "For more options, run:"
echo "   python train.py --help"
echo ""
