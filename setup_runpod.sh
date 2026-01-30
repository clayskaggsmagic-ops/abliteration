#!/bin/bash
# RunPod Setup Script - EXACT versions from transformer_lens 2.17.0 requirements
set -e

echo "🔧 Setting up Abliteration environment..."

# Force uninstall everything first
pip uninstall -y torch torchvision torchaudio numpy transformers transformer_lens transformers_stream_generator huggingface_hub 2>/dev/null || true

echo ""
echo "📦 Installing exact compatible versions..."

# Install in correct order with exact versions matching transformer_lens 2.17.0 requirements
pip install --no-cache-dir \
    "numpy>=1.24,<2" \
    "torch>=2.6" \
    "transformers>=4.51" \
    "huggingface-hub>=0.23.2,<1.0" \
    "transformers-stream-generator>=0.0.5,<0.0.6" \
    "transformer_lens==2.17.0" \
    "accelerate>=0.23.0" \
    "einops>=0.6.0" \
    "jaxtyping>=0.2.11" \
    "datasets>=2.7.1" \
    "pandas>=1.1.5" \
    "scikit-learn>=1.3.0" \
    "tqdm>=4.64.1" \
    "colorama>=0.4.6"

echo ""
echo "✅ Setup complete! Now run:"
echo "   python run_abliteration.py"
