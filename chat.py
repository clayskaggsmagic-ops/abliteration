#!/usr/bin/env python3
"""
Chat with an abliterated model.

Usage:
    python chat.py                           # Uses default model and weights
    python chat.py --model Qwen/Qwen2.5-1.5B-Instruct --weights ./output/abliterated_model/abliterated_weights.pt
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.abliterate import Abliterator


def main():
    parser = argparse.ArgumentParser(description="Chat with an abliterated model")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model (must match the model used for abliteration)"
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default="./output/abliterated_model/abliterated_weights.pt",
        help="Path to abliterated weights"
    )
    args = parser.parse_args()
    
    print("🔧 Loading abliterated model...")
    print(f"   Base model: {args.model}")
    print(f"   Weights: {args.weights}")
    
    # Load base model
    abl = Abliterator(model_path=args.model)
    abl.load_model()
    
    # Load abliterated weights
    if os.path.exists(args.weights):
        print("   Loading abliterated weights...")
        state_dict = torch.load(args.weights, map_location=abl.device)
        abl.model.load_state_dict(state_dict)
        print("   ✓ Abliterated weights loaded!")
    else:
        print(f"   ⚠️ Weights file not found: {args.weights}")
        print("   Using base model (not abliterated)")
    
    # Interactive chat
    print("\n" + "=" * 60)
    print("💬 CHAT MODE")
    print("Type a prompt and press Enter.")
    print("Type 'quit' to exit.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            prompt = input("📝 You: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            if not prompt:
                continue
                
            responses = abl.generate([prompt], max_tokens=256)
            print(f"🤖 Model: {responses[0]}\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


if __name__ == "__main__":
    main()
