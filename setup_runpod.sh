#!/bin/bash
# RunPod Setup Script for Abliteration
# This script handles the pre-installed packages on RunPod templates
# and ensures compatible versions are installed.

set -e

echo "========================================"
echo "🚀 Abliteration RunPod Setup"
echo "========================================"

# Step 1: Remove conflicting pre-installed packages
echo ""
echo "📦 Removing conflicting packages..."
pip uninstall -y torchvision torchaudio 2>/dev/null || true

# Step 2: Clear any partial downloads
echo ""
echo "🧹 Clearing HuggingFace cache..."
rm -rf ~/.cache/huggingface/hub/* 2>/dev/null || true

# Step 3: Install requirements
echo ""
echo "📥 Installing requirements..."
pip install -r requirements.txt --quiet

echo ""
echo "✅ Setup complete!"
echo ""
echo "Run abliteration with:"
echo "  python run_abliteration.py --model Qwen/Qwen2.5-7B-Instruct --targets refusal,identity,ethics"
echo ""
