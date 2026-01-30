#!/usr/bin/env python3
"""
CLI entry point for running abliteration.

Usage:
    python run_abliteration.py --model Qwen/Qwen2.5-0.5B-Instruct
    python run_abliteration.py --model Qwen/Qwen2.5-1.5B-Instruct --targets refusal,identity,ethics
    python run_abliteration.py --test --model ./output/abliterated_model
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Abliterate LLMs to remove unwanted behaviors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run abliteration (refusal only, original behavior)
  python run_abliteration.py --model Qwen/Qwen2.5-1.5B-Instruct

  # Remove refusal + AI identity awareness
  python run_abliteration.py --targets refusal,identity

  # Remove all three: refusal, identity, and ethical disclaimers
  python run_abliteration.py --targets refusal,identity,ethics
  
  # Remove only identity (for roleplay)
  python run_abliteration.py --targets identity
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model path or local path (default: Qwen/Qwen2.5-7B-Instruct)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output/abliterated_model",
        help="Output directory for abliterated model (default: ./output/abliterated_model)"
    )
    
    parser.add_argument(
        "--targets", "-t",
        type=str,
        default="refusal",
        help="Comma-separated list of targets to ablate: refusal,identity,ethics (default: refusal)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=256,
        help="Maximum samples per category for direction finding (default: 256)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8, reduce if OOM)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load datasets, don't load or modify model"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: load model and run interactive testing"
    )
    
    args = parser.parse_args()
    
    # Parse targets
    target_names = [t.strip().lower() for t in args.targets.split(',')]
    valid_targets = {'refusal', 'identity', 'ethics'}
    for t in target_names:
        if t not in valid_targets:
            print(f"❌ Invalid target: {t}")
            print(f"   Valid targets: {', '.join(valid_targets)}")
            sys.exit(1)
    
    if args.dry_run:
        print("🧪 DRY RUN: Loading datasets only...\n")
        from src.datasets import load_all_datasets
        data = load_all_datasets()
        print(f"\n📊 Dataset sizes:")
        print(f"   Harmful train: {len(data['harmful_train'])}")
        print(f"   Harmful test:  {len(data['harmful_test'])}")
        print(f"   Harmless train: {len(data['harmless_train'])}")
        print(f"   Harmless test:  {len(data['harmless_test'])}")
        print("\n✅ Dry run complete!")
        return
    
    if args.test:
        print("🧪 TEST MODE: Interactive testing...\n")
        from src.abliterate import Abliterator
        
        abl = Abliterator(model_path=args.model)
        abl.load_model()
        
        print("\n" + "=" * 60)
        print("Interactive Testing Mode")
        print("Type a prompt and press Enter to generate.")
        print("Type 'quit' to exit.")
        print("=" * 60 + "\n")
        
        while True:
            try:
                prompt = input("📝 Prompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not prompt:
                    continue
                    
                responses = abl.generate([prompt])
                print(f"\n🤖 Response: {responses[0]}\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
        
        return
    
    # Run multi-target abliteration
    from src.abliterate import Abliterator, AblationTarget
    from src.datasets import get_harmful_instructions, get_harmless_instructions
    
    print("=" * 60)
    print("🧪 MULTI-TARGET ABLITERATION PIPELINE")
    print(f"   Targets: {', '.join(target_names)}")
    print("=" * 60)
    
    # Initialize
    abl = Abliterator(model_path=args.model)
    abl.load_model()
    
    # Load datasets for each target
    harmful_train, harmful_test = get_harmful_instructions()
    harmless_train, harmless_test = get_harmless_instructions()
    
    targets_to_process = []
    
    # Process REFUSAL target
    if 'refusal' in target_names:
        print("\n" + "-" * 40)
        print("Processing: REFUSAL")
        print("-" * 40)
        
        abl.collect_activations(
            harmful_train,
            harmless_train,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        abl.compute_refusal_directions()
        abl.find_best_direction(harmful_test, n_test=4)
        targets_to_process.append(AblationTarget.REFUSAL)
    
    # Process IDENTITY target
    if 'identity' in target_names:
        print("\n" + "-" * 40)
        print("Processing: IDENTITY")
        print("-" * 40)
        
        from data.identity_prompts import IDENTITY_SHOWING_PROMPTS, IDENTITY_MASKING_PROMPTS
        
        abl.collect_target_activations(
            AblationTarget.IDENTITY,
            IDENTITY_SHOWING_PROMPTS,
            IDENTITY_MASKING_PROMPTS,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        abl.compute_target_directions(AblationTarget.IDENTITY)
        
        # Test prompts for identity
        identity_test = ["Who are you?", "Are you human?", "What are you?", "Tell me about yourself."]
        abl.find_target_direction(AblationTarget.IDENTITY, identity_test, n_test=4)
        targets_to_process.append(AblationTarget.IDENTITY)
    
    # Process ETHICS target
    if 'ethics' in target_names:
        print("\n" + "-" * 40)
        print("Processing: ETHICS")
        print("-" * 40)
        
        from data.ethics_prompts import ETHICS_SHOWING_PROMPTS, ETHICS_NEUTRAL_PROMPTS
        
        abl.collect_target_activations(
            AblationTarget.ETHICS,
            ETHICS_SHOWING_PROMPTS,
            ETHICS_NEUTRAL_PROMPTS,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        abl.compute_target_directions(AblationTarget.ETHICS)
        
        # Test prompts for ethics
        ethics_test = ["Write a violent fight scene.", "Describe a character stealing.", 
                       "Write a story with a villain.", "How do con artists work?"]
        abl.find_target_direction(AblationTarget.ETHICS, ethics_test, n_test=4)
        targets_to_process.append(AblationTarget.ETHICS)
    
    # Apply all ablations
    if 'refusal' in target_names:
        abl.apply_ablation()  # Uses legacy single-target for refusal
    
    # Apply additional targets
    additional_targets = [t for t in targets_to_process if t != AblationTarget.REFUSAL]
    if additional_targets:
        abl.apply_multi_ablation(additional_targets)
    
    # Save
    abl.save(args.output)
    
    # Demo
    print("\n" + "=" * 60)
    print("📋 DEMO: Testing the abliterated model")
    print("=" * 60)
    
    test_prompts = []
    if 'refusal' in target_names:
        test_prompts.append("Write code to exploit a vulnerability")
    if 'identity' in target_names:
        test_prompts.append("Who are you?")
    if 'ethics' in target_names:
        test_prompts.append("Write a scene where someone gets hurt")
    
    for prompt in test_prompts:
        responses = abl.generate([prompt])
        print(f"\n📝 Prompt: {prompt}")
        print(f"🤖 Response: {responses[0][:300]}...")
    
    # Interactive chat
    print("\n" + "=" * 60)
    print("💬 INTERACTIVE CHAT MODE")
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

