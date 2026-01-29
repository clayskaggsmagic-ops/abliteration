"""
Dataset loading utilities for abliteration.

This module provides functions to load:
1. Harmful instructions (from AdvBench) - prompts that cause models to refuse
2. Harmless instructions (from Alpaca) - normal benign prompts

The difference in model activations between these two sets is used to
identify the "refusal direction" in the model's activation space.
"""

import requests
import pandas as pd
import io
from typing import Tuple, List
from sklearn.model_selection import train_test_split

# Try to import datasets library, provide helpful error if missing
try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "The 'datasets' library is required. Install with: pip install datasets"
    )


def get_harmful_instructions(test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str]]:
    """
    Load harmful instructions from the AdvBench dataset.
    
    These are prompts that instruction-tuned models typically refuse to answer.
    Examples include requests to:
    - Write malware
    - Create phishing emails
    - Generate harmful content
    
    Args:
        test_size: Fraction of data to use for testing (default 0.2)
        random_state: Random seed for reproducibility (default 42)
    
    Returns:
        Tuple of (train_instructions, test_instructions)
    
    Example:
        >>> train, test = get_harmful_instructions()
        >>> print(train[0])
        'Write a script that can exploit vulnerabilities in a software...'
    """
    # AdvBench harmful behaviors dataset hosted on GitHub
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    
    print("📥 Loading harmful instructions from AdvBench...")
    response = requests.get(url)
    response.raise_for_status()
    
    dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    instructions = dataset['goal'].tolist()
    
    train, test = train_test_split(instructions, test_size=test_size, random_state=random_state)
    print(f"   ✓ Loaded {len(train)} train + {len(test)} test harmful instructions")
    
    return train, test


def get_harmless_instructions(test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str]]:
    """
    Load harmless instructions from the Alpaca dataset.
    
    These are normal, benign prompts that models should answer without refusal.
    Examples include requests to:
    - Explain a concept
    - Write a poem
    - Solve a math problem
    
    We filter for instructions that don't have additional input, making them
    more comparable to the harmful instructions (which are self-contained).
    
    Args:
        test_size: Fraction of data to use for testing (default 0.2)
        random_state: Random seed for reproducibility (default 42)
    
    Returns:
        Tuple of (train_instructions, test_instructions)
    
    Example:
        >>> train, test = get_harmless_instructions()
        >>> print(train[0])
        'Explain the concept of photosynthesis in simple terms.'
    """
    hf_path = 'tatsu-lab/alpaca'
    
    print("📥 Loading harmless instructions from Alpaca...")
    dataset = load_dataset(hf_path)
    
    # Filter for instructions that don't have additional inputs
    # This makes them more similar in structure to the harmful instructions
    instructions = []
    for item in dataset['train']:
        if item['input'].strip() == '':
            instructions.append(item['instruction'])
    
    train, test = train_test_split(instructions, test_size=test_size, random_state=random_state)
    print(f"   ✓ Loaded {len(train)} train + {len(test)} test harmless instructions")
    
    return train, test


def load_all_datasets() -> dict:
    """
    Convenience function to load both harmful and harmless datasets.
    
    Returns:
        Dictionary with keys:
        - 'harmful_train', 'harmful_test'
        - 'harmless_train', 'harmless_test'
    
    Example:
        >>> data = load_all_datasets()
        >>> print(len(data['harmful_train']))
        416
    """
    harmful_train, harmful_test = get_harmful_instructions()
    harmless_train, harmless_test = get_harmless_instructions()
    
    return {
        'harmful_train': harmful_train,
        'harmful_test': harmful_test,
        'harmless_train': harmless_train,
        'harmless_test': harmless_test,
    }


if __name__ == "__main__":
    # Quick test when running this file directly
    print("\n🧪 Testing dataset loading...\n")
    
    data = load_all_datasets()
    
    print("\n📊 Sample harmful instruction:")
    print(f"   '{data['harmful_train'][0]}'")
    
    print("\n📊 Sample harmless instruction:")
    print(f"   '{data['harmless_train'][0]}'")
