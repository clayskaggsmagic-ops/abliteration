#!/bin/bash
# RunPod Setup Script - Installs all dependencies with correct versions
# Run this ONCE after cloning the repo on a fresh RunPod pod

set -e

echo "🔧 Setting up Abliteration environment..."
echo ""

# Uninstall conflicting packages first
echo "📦 Removing conflicting packages..."
pip uninstall -y transformers_stream_generator 2>/dev/null || true
pip uninstall -y torch torchvision torchaudio numpy 2>/dev/null || true

# Install exact versions that work together
echo ""
echo "📦 Installing compatible package versions..."
pip install --no-cache-dir \
    numpy==1.26.4 \
    torch==2.6.0 \
    torchvision==0.21.0 \
    transformers==4.48.0 \
    transformer_lens==2.17.0 \
    huggingface_hub==0.27.0 \
    datasets==2.21.0 \
    einops==0.8.0 \
    jaxtyping==0.2.34 \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    tqdm==4.67.1 \
    colorama==0.4.6 \
    accelerate==1.2.1

echo ""
echo "✅ Setup complete! Now run:"
echo "   python run_abliteration.py"
