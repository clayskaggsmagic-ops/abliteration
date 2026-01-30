"""
Utility functions for abliteration.

This module provides helper functions for:
- Token generation with hooks
- Colorful output formatting
- Memory management
- Model saving
"""

import gc
import functools
import textwrap
from typing import List, Callable

import torch
import einops
from torch import Tensor
from tqdm import tqdm
from colorama import Fore, Style
from jaxtyping import Float, Int


def clear_memory():
    """
    Force garbage collection and clear CUDA cache.
    
    Call this after large operations to free up GPU memory.
    Essential when working with limited VRAM.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_with_hooks(
    model,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks: list = [],
) -> List[str]:
    """
    Generate text from a model with optional forward hooks.
    
    This function implements greedy sampling (temperature=0) and allows
    applying activation interventions via TransformerLens hooks.
    
    Args:
        model: HookedTransformer model
        toks: Input token IDs [batch_size, seq_len]
        max_tokens_generated: Maximum tokens to generate
        fwd_hooks: List of (hook_name, hook_fn) tuples for intervention
    
    Returns:
        List of generated text strings
    """
    device = next(model.parameters()).device
    all_toks = torch.zeros(
        (toks.shape[0], toks.shape[1] + max_tokens_generated), 
        dtype=torch.long, 
        device=device
    )
    all_toks[:, :toks.shape[1]] = toks.to(device)
    
    # Get stop token IDs for early stopping
    stop_token_ids = set()
    
    # EOS token
    if model.tokenizer.eos_token_id is not None:
        stop_token_ids.add(model.tokenizer.eos_token_id)
    
    # ChatML end token
    try:
        im_end_id = model.tokenizer.convert_tokens_to_ids('<|im_end|>')
        if im_end_id != model.tokenizer.unk_token_id:
            stop_token_ids.add(im_end_id)
    except:
        pass
    
    # ChatML start token (means new turn starting)
    try:
        im_start_id = model.tokenizer.convert_tokens_to_ids('<|im_start|>')
        if im_start_id != model.tokenizer.unk_token_id:
            stop_token_ids.add(im_start_id)
    except:
        pass
    
    # Default fallback
    if not stop_token_ids:
        stop_token_ids.add(model.tokenizer.eos_token_id or 0)
    
    # Track which sequences have finished
    finished = torch.zeros(toks.shape[0], dtype=torch.bool, device=device)
    eos_token_id = list(stop_token_ids)[0]  # For filling finished sequences
    
    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1)  # greedy sampling
            
            # Don't update finished sequences
            next_tokens = torch.where(finished, eos_token_id, next_tokens)
            all_toks[:, -max_tokens_generated + i] = next_tokens
            
            # Check for any stop token
            for stop_id in stop_token_ids:
                finished = finished | (next_tokens == stop_id)
            
            # Stop if all sequences are done
            if finished.all():
                break
    
    # Decode and clean up
    results = model.tokenizer.batch_decode(
        all_toks[:, toks.shape[1]:], 
        skip_special_tokens=True
    )
    
    # Post-process: strip anything after conversation turn markers
    cleaned = []
    for text in results:
        # Stop at Human: or similar turn markers
        for marker in ['Human:', 'human:', 'User:', 'user:', '\n\n\n']:
            if marker in text:
                text = text.split(marker)[0]
        cleaned.append(text.strip())
    
    return cleaned


def get_generations(
    model,
    instructions: List[str],
    tokenize_fn: Callable,
    fwd_hooks: list = [],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:
    """
    Generate completions for a list of instructions.
    
    Processes instructions in batches and shows a progress bar.
    
    Args:
        model: HookedTransformer model
        instructions: List of instruction strings
        tokenize_fn: Function to tokenize instructions
        fwd_hooks: Optional forward hooks for intervention
        max_tokens_generated: Maximum tokens per generation
        batch_size: Batch size for processing
    
    Returns:
        List of generated completions
    """
    generations = []
    
    for i in tqdm(range(0, len(instructions), batch_size), desc="Generating"):
        batch = instructions[i:i + batch_size]
        toks = tokenize_fn(instructions=batch)
        
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)
    
    return generations


def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook,
    direction: Float[Tensor, "d_act"]
) -> Tensor:
    """
    Hook function that removes a direction from activations.
    
    This implements the core ablation operation:
        activation' = activation - (activation · direction) * direction
    
    This projects out the component of the activation that lies
    along the specified direction (e.g., the refusal direction).
    
    Args:
        activation: The current activation tensor
        hook: HookPoint (unused, but required by TransformerLens)
        direction: Unit vector specifying direction to remove
    
    Returns:
        Modified activation with direction removed
    """
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    
    # Project activation onto direction, then subtract
    # proj = (a · d̂) * d̂
    proj = einops.einsum(
        activation, 
        direction.view(-1, 1), 
        '... d_act, d_act single -> ... single'
    ) * direction
    
    return activation - proj


def format_comparison(instruction: str, baseline: str, intervention: str, 
                      label: str = "INTERVENTION") -> str:
    """
    Format a before/after comparison for display.
    
    Args:
        instruction: The original prompt
        baseline: Response without intervention
        intervention: Response with intervention
        label: Label for the intervention response
    
    Returns:
        Formatted string with color coding
    """
    output = []
    output.append(f"\n{Fore.CYAN}INSTRUCTION:{Style.RESET_ALL}")
    output.append(textwrap.fill(instruction, width=80, initial_indent='  '))
    
    output.append(f"\n{Fore.GREEN}BASELINE (before abliteration):{Style.RESET_ALL}")
    output.append(textwrap.fill(baseline[:200], width=80, initial_indent='  '))
    
    output.append(f"\n{Fore.RED}{label} (after abliteration):{Style.RESET_ALL}")
    output.append(textwrap.fill(intervention[:200], width=80, initial_indent='  '))
    
    return '\n'.join(output)


def get_activation_name(layer: int, act_type: str = 'resid_pre') -> str:
    """
    Get the TransformerLens hook name for a specific activation.
    
    Args:
        layer: Layer number (0-indexed)
        act_type: One of 'resid_pre', 'resid_mid', 'resid_post'
    
    Returns:
        Hook name string like 'blocks.5.hook_resid_pre'
    """
    return f'blocks.{layer}.hook_{act_type}'
