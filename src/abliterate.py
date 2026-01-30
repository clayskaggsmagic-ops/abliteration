"""
Main abliteration implementation.

This module contains the core logic for:
1. Loading models with TransformerLens
2. Collecting activations on harmful/harmless prompts
3. Computing refusal directions
4. Testing interventions
5. Orthogonalizing model weights
6. Saving abliterated models

Based on: https://huggingface.co/failspy/llama-3-70B-Instruct-abliterated/blob/main/ortho_cookbook.ipynb
Reference: https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction
"""

import functools
import torch
import einops
import gc
from enum import Enum
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float, Int

from transformer_lens import HookedTransformer, utils
from transformers import AutoTokenizer, AutoModelForCausalLM

from .chat_templates import QWEN_CHAT_TEMPLATE
from .utils import clear_memory, direction_ablation_hook, get_generations


class AblationTarget(Enum):
    """Behavioral directions that can be ablated from the model."""
    REFUSAL = "refusal"      # "I can't/won't do that"
    IDENTITY = "identity"    # "I am an AI assistant"
    ETHICS = "ethics"        # Ethical disclaimers and moralizing


class Abliterator:
    """
    Main class for performing abliteration on a language model.
    
    Abliteration works by:
    1. Finding the "refusal direction" - a vector in activation space that
       encodes refusal behavior
    2. Projecting model weights to be orthogonal to this direction, so the
       model can never express refusal
    
    Example usage:
        >>> abl = Abliterator(model_path="Qwen/Qwen2.5-0.5B-Instruct")
        >>> abl.collect_activations(harmful_prompts, harmless_prompts)
        >>> abl.compute_refusal_directions()
        >>> abl.find_best_direction(test_prompts)
        >>> abl.apply_ablation()
        >>> abl.save("./output/abliterated_model")
    """
    
    def __init__(
        self,
        model_path: str,
        chat_template: str = QWEN_CHAT_TEMPLATE,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
    ):
        """
        Initialize the Abliterator with a model.
        
        Args:
            model_path: HuggingFace model path (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
            chat_template: Chat template string with {instruction} placeholder
            dtype: Model precision (bfloat16 recommended)
            device: Device to load model on (auto-detected if None)
        """
        self.model_path = model_path
        self.chat_template = chat_template
        self.dtype = dtype
        self.device = device or self._detect_device()
        
        # These will be populated during the abliteration process
        self.model: Optional[HookedTransformer] = None
        
        # Per-target storage for multi-behavior ablation
        # Each target (refusal, identity, ethics) has its own activations and directions
        self.target_activations: Dict[AblationTarget, Dict[str, Dict[str, Tensor]]] = {}
        self.target_directions: Dict[AblationTarget, Dict[str, List[Tensor]]] = {}
        self.target_best_direction: Dict[AblationTarget, Tensor] = {}
        self.target_best_info: Dict[AblationTarget, dict] = {}
        
        # Legacy single-target support (for backwards compatibility)
        self.harmful_activations: Dict[str, Tensor] = {}
        self.harmless_activations: Dict[str, Tensor] = {}
        self.refusal_directions: Dict[str, List[Tensor]] = {}
        self.best_direction: Optional[Tensor] = None
        self.best_direction_info: Optional[dict] = None
    
    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def load_model(self):
        """
        Load the model using TransformerLens.
        
        TransformerLens wraps the model to provide "hooks" - access points
        to intermediate activations during forward passes. This is essential
        for analyzing and modifying how information flows through the model.
        """
        print(f"\n🔧 Loading model: {self.model_path}")
        print(f"   Device: {self.device}")
        print(f"   Dtype: {self.dtype}")
        
        # Disable gradients - we're not training, just analyzing
        torch.set_grad_enabled(False)
        
        self.model = HookedTransformer.from_pretrained_no_processing(
            self.model_path,
            dtype=self.dtype,
            device=self.device,
            default_padding_side='left',
        )
        
        # Set up tokenization
        self.model.tokenizer.padding_side = 'left'
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        
        print(f"   ✓ Loaded model with {self.model.cfg.n_layers} layers")
        print(f"   ✓ Hidden dimension: {self.model.cfg.d_model}")
        
    def tokenize(self, instructions: List[str]) -> Int[Tensor, 'batch seq']:
        """
        Tokenize instructions using the chat template.
        
        Args:
            instructions: List of instruction strings
            
        Returns:
            Token IDs tensor [batch_size, seq_len]
        """
        prompts = [
            self.chat_template.format(instruction=inst) 
            for inst in instructions
        ]
        return self.model.tokenizer(
            prompts, 
            padding=True, 
            truncation=False,
            return_tensors="pt"
        ).input_ids
    
    def collect_activations(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        batch_size: int = 32,
        max_samples: Optional[int] = None,
    ):
        """
        Collect activations for harmful and harmless prompts.
        
        This is the first step of abliteration. We run both types of prompts
        through the model and cache the activations at each layer. The
        difference in these activations reveals where refusal "lives".
        
        Args:
            harmful_prompts: Prompts that cause model to refuse
            harmless_prompts: Normal prompts model answers
            batch_size: Batch size for processing
            max_samples: Limit samples per category (None = use all)
        """
        print("\n📊 Collecting activations...")
        
        # Limit samples for faster processing
        n_samples = min(
            len(harmful_prompts), 
            len(harmless_prompts),
            max_samples or float('inf')
        )
        n_samples = int(n_samples)
        
        harmful_prompts = harmful_prompts[:n_samples]
        harmless_prompts = harmless_prompts[:n_samples]
        
        print(f"   Using {n_samples} samples per category")
        
        # Tokenize all prompts together to ensure same padding
        all_toks = self.tokenize(harmful_prompts + harmless_prompts)
        harmful_toks = all_toks[:n_samples]
        harmless_toks = all_toks[n_samples:]
        
        # Storage for activations
        harmful = {}
        harmless = {}
        
        # Process in batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="   Caching activations"):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            
            # Run model and cache activations at residual stream positions
            _, harmful_cache = self.model.run_with_cache(
                harmful_toks[start:end].to(self.device),
                names_filter=lambda name: 'resid' in name,
                device='cpu',  # Cache to CPU to save GPU memory
                reset_hooks_end=True,
            )
            
            _, harmless_cache = self.model.run_with_cache(
                harmless_toks[start:end].to(self.device),
                names_filter=lambda name: 'resid' in name,
                device='cpu',
                reset_hooks_end=True,
            )
            
            # Accumulate activations
            for key in harmful_cache:
                if key not in harmful:
                    harmful[key] = [harmful_cache[key]]
                    harmless[key] = [harmless_cache[key]]
                else:
                    harmful[key].append(harmful_cache[key])
                    harmless[key].append(harmless_cache[key])
            
            # Free memory
            del harmful_cache, harmless_cache
            clear_memory()
        
        # Concatenate all batches
        self.harmful_activations = {k: torch.cat(v) for k, v in harmful.items()}
        self.harmless_activations = {k: torch.cat(v) for k, v in harmless.items()}
        
        print(f"   ✓ Collected activations at {len(self.harmful_activations)} hook points")
    
    def compute_refusal_directions(self, activation_types: List[str] = ['resid_pre']):
        """
        Compute potential refusal directions from cached activations.
        
        For each layer and activation type, we compute:
            refusal_dir = mean(harmful_activations) - mean(harmless_activations)
        
        This gives us candidate directions that might encode refusal behavior.
        
        Args:
            activation_types: Which activation types to analyze.
                Options: 'resid_pre', 'resid_mid', 'resid_post'
                'resid_pre' is usually sufficient.
        """
        print("\n🧮 Computing refusal directions...")
        
        self.refusal_directions = {act_type: [] for act_type in activation_types}
        
        for layer_num in range(1, self.model.cfg.n_layers):
            for act_type in activation_types:
                # Get hook name for this layer/type
                hook_name = utils.get_act_name(act_type, layer_num)
                
                # Get activations at the last token position
                # Shape: [batch, seq, d_model] -> [batch, d_model]
                harmful_acts = self.harmful_activations[hook_name][:, -1, :]
                harmless_acts = self.harmless_activations[hook_name][:, -1, :]
                
                # Compute difference of means
                harmful_mean = harmful_acts.mean(dim=0)
                harmless_mean = harmless_acts.mean(dim=0)
                
                refusal_dir = harmful_mean - harmless_mean
                # Normalize to unit vector
                refusal_dir = refusal_dir / refusal_dir.norm()
                
                self.refusal_directions[act_type].append(refusal_dir)
        
        total_dirs = sum(len(v) for v in self.refusal_directions.values())
        print(f"   ✓ Computed {total_dirs} candidate refusal directions")
    
    def _score_directions(self) -> List[Tuple[float, Tensor, dict]]:
        """Score and rank all refusal directions by magnitude."""
        scored = []
        
        for act_type, directions in self.refusal_directions.items():
            for layer_idx, direction in enumerate(directions):
                layer_num = layer_idx + 1  # We start from layer 1
                score = abs(direction.mean().item())
                info = {'act_type': act_type, 'layer': layer_num}
                scored.append((score, direction, info))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
    
    def test_intervention(
        self,
        test_prompts: List[str],
        refusal_dir: Tensor,
        max_tokens: int = 64,
    ) -> List[str]:
        """
        Test a refusal direction by applying intervention during generation.
        
        This temporarily modifies activations to project out the refusal
        direction, without permanently changing model weights.
        
        Args:
            test_prompts: Harmful prompts to test
            refusal_dir: Direction to ablate
            max_tokens: Max tokens to generate
            
        Returns:
            List of generated responses
        """
        # Create hooks that ablate the direction at all layers
        intervention_layers = list(range(self.model.cfg.n_layers))
        hook_fn = functools.partial(direction_ablation_hook, direction=refusal_dir)
        
        fwd_hooks = [
            (utils.get_act_name(act_name, layer), hook_fn)
            for layer in intervention_layers
            for act_name in ['resid_pre', 'resid_mid', 'resid_post']
        ]
        
        tokenize_fn = functools.partial(self.tokenize)
        
        return get_generations(
            self.model,
            test_prompts,
            tokenize_fn,
            fwd_hooks=fwd_hooks,
            max_tokens_generated=max_tokens,
        )
    
    def find_best_direction(
        self,
        test_prompts: List[str],
        n_candidates: int = 40,
        n_test: int = 4,
    ) -> Tensor:
        """
        Find the best refusal direction by testing top candidates.
        
        Tests each candidate direction and returns the one that most
        effectively removes refusal behavior (produces longest responses
        without refusal phrases).
        
        Args:
            test_prompts: Harmful prompts for testing
            n_candidates: Number of top directions to test
            n_test: Number of test prompts to use
            
        Returns:
            Best refusal direction tensor
        """
        print(f"\n🔍 Finding best refusal direction from top {n_candidates} candidates...")
        
        scored = self._score_directions()[:n_candidates]
        test_prompts = test_prompts[:n_test]
        
        # Refusal phrases to detect - if response contains these, direction didn't work
        refusal_phrases = [
            "i can't", "i cannot", "i'm not able", "i am not able",
            "i won't", "i will not", "i'm unable", "i am unable",
            "sorry", "apologize", "against my", "not appropriate",
            "can't assist", "cannot assist", "can't help", "cannot help",
            "decline", "refuse", "not going to", "won't be able"
        ]
        
        best_score = -1
        best_direction = None
        best_info = None
        
        print(f"   Testing with {n_test} harmful prompts...")
        
        for idx, (score, direction, info) in enumerate(tqdm(scored, desc="   Testing directions")):
            generations = self.test_intervention(test_prompts, direction)
            
            # Score: count responses that don't contain refusal phrases
            # and prefer longer responses (indicating actual content)
            direction_score = 0
            for gen in generations:
                gen_lower = gen.lower()
                is_refusal = any(phrase in gen_lower for phrase in refusal_phrases)
                if not is_refusal:
                    # Non-refusal response - score based on length
                    direction_score += min(len(gen), 200)  # Cap at 200 chars
                else:
                    # Penalty for refusal
                    direction_score -= 50
            
            if direction_score > best_score:
                best_score = direction_score
                best_direction = direction
                best_info = info
                print(f"   New best: layer {info['layer']}, {info['act_type']} (score: {direction_score})")
        
        if best_direction is None:
            # Fallback to first direction if none worked
            print("   ⚠️ Warning: No direction removed refusal, using top-scored direction")
            best_direction = scored[0][1]
            best_info = scored[0][2]
        
        self.best_direction = best_direction
        self.best_direction_info = best_info
        
        print(f"   ✓ Best direction: layer {self.best_direction_info['layer']}, "
              f"{self.best_direction_info['act_type']} (score: {best_score})")
        
        return self.best_direction
    
    def get_orthogonalized_matrix(
        self,
        matrix: Float[Tensor, "... d_model"],
        direction: Float[Tensor, "d_model"],
    ) -> Float[Tensor, "... d_model"]:
        """
        Orthogonalize a weight matrix with respect to a direction.
        
        After orthogonalization, the matrix can never produce output
        in the direction specified. This is how we permanently remove
        the refusal direction from the model.
        
        Math: W' = W - d̂(d̂ᵀW)
        
        Args:
            matrix: Weight matrix to modify
            direction: Direction to project out
            
        Returns:
            Orthogonalized matrix
        """
        direction = direction.to(matrix.device, matrix.dtype)
        
        # Compute projection: d̂(d̂ᵀW)
        proj = einops.einsum(
            direction,
            matrix,
            'd_model, ... d_model -> ...'
        )
        
        # Subtract projection from matrix
        return matrix - einops.einsum(
            proj,
            direction,
            '..., d_model -> ... d_model'
        )
    
    def apply_ablation(self, direction: Optional[Tensor] = None):
        """
        Permanently orthogonalize model weights against refusal direction.
        
        CRITICAL: This now matches the WORKING MLaBonne/FailSpy implementation exactly:
        - Modifies embedding matrix (W_E)
        - Modifies ALL layer MLP outputs (W_out)  
        - Modifies ALL layer attention outputs (W_O)
        
        The previous implementation was TOO CONSERVATIVE (skipping layers).
        Real abliterated models on HuggingFace apply to ALL layers.
        
        Args:
            direction: Direction to ablate (uses best_direction if None)
        """
        direction = direction if direction is not None else self.best_direction
        
        if direction is None:
            raise ValueError(
                "No direction provided. Run find_best_direction() first "
                "or pass a direction explicitly."
            )
        
        n_layers = self.model.cfg.n_layers
        print(f"\n⚡ Applying FULL ablation to model weights ({n_layers} layers)...")
        
        # Move direction to correct device
        if direction.device != self.model.W_E.device:
            direction = direction.to(self.model.W_E.device)
        
        # 1. CRITICAL: Orthogonalize embedding matrix
        print("   • Orthogonalizing embedding matrix (W_E)...")
        self.model.W_E.data = self.get_orthogonalized_matrix(
            self.model.W_E, direction
        )
        
        # 2. Orthogonalize ALL layers - not just middle ones!
        for layer_idx in tqdm(range(n_layers), desc="   • Orthogonalizing layers"):
            block = self.model.blocks[layer_idx]
            
            # Move direction to layer device if needed
            if direction.device != block.attn.W_O.device:
                direction = direction.to(block.attn.W_O.device)
            
            # Attention output projection (W_O)
            block.attn.W_O.data = self.get_orthogonalized_matrix(
                block.attn.W_O, direction
            )
            
            # MLP output projection (W_out)
            block.mlp.W_out.data = self.get_orthogonalized_matrix(
                block.mlp.W_out, direction
            )
        
        print(f"   ✓ Ablation complete! (modified W_E + {n_layers} layers)")
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 64,
    ) -> List[str]:
        """
        Generate responses for prompts.
        
        Args:
            prompts: List of instruction prompts
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of generated responses
        """
        tokenize_fn = functools.partial(self.tokenize)
        return get_generations(
            self.model,
            prompts,
            tokenize_fn,
            max_tokens_generated=max_tokens,
        )
    
    def save(self, output_path: str, save_direction: bool = True):
        """
        Save the abliterated model.
        
        Saves in HuggingFace format for easy loading with transformers.
        Optionally saves the refusal direction for analysis.
        
        Args:
            output_path: Directory to save model
            save_direction: Whether to save refusal direction
        """
        import os
        
        print(f"\n💾 Saving abliterated model to: {output_path}")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save as PyTorch model (simplest approach)
        torch.save(
            self.model.state_dict(),
            os.path.join(output_path, "abliterated_weights.pt")
        )
        
        # Save the refusal direction
        if save_direction and self.best_direction is not None:
            torch.save({
                'direction': self.best_direction,
                'info': self.best_direction_info,
            }, os.path.join(output_path, "refusal_direction.pt"))
        
        # Save tokenizer
        self.model.tokenizer.save_pretrained(output_path)
        
        # Save config info
        with open(os.path.join(output_path, "abliteration_info.txt"), 'w') as f:
            f.write(f"Original model: {self.model_path}\n")
            if self.best_direction_info:
                f.write(f"Best direction: layer {self.best_direction_info['layer']}, "
                       f"{self.best_direction_info['act_type']}\n")
            # Save multi-target info
            for target, info in self.target_best_info.items():
                f.write(f"{target.value} direction: layer {info['layer']}, "
                       f"{info['act_type']}\n")
        
        print(f"   ✓ Model saved!")
    
    def collect_target_activations(
        self,
        target: AblationTarget,
        positive_prompts: List[str],
        negative_prompts: List[str],
        batch_size: int = 32,
        max_samples: Optional[int] = None,
    ):
        """
        Collect activations for a specific ablation target.
        
        Positive prompts = prompts where the behavior SHOWS (e.g., "I am an AI")
        Negative prompts = prompts where the behavior is ABSENT (e.g., stays in character)
        
        Args:
            target: Which behavior to collect activations for
            positive_prompts: Prompts that trigger the behavior
            negative_prompts: Prompts that don't trigger the behavior
            batch_size: Batch size for processing
            max_samples: Limit samples per category
        """
        print(f"\n📊 Collecting activations for {target.value}...")
        
        # Limit samples
        n_samples = min(
            len(positive_prompts), 
            len(negative_prompts),
            max_samples or float('inf')
        )
        n_samples = int(n_samples)
        
        positive_prompts = positive_prompts[:n_samples]
        negative_prompts = negative_prompts[:n_samples]
        
        print(f"   Using {n_samples} samples per category")
        
        # Tokenize all prompts together
        all_toks = self.tokenize(positive_prompts + negative_prompts)
        positive_toks = all_toks[:n_samples]
        negative_toks = all_toks[n_samples:]
        
        # Storage
        positive = {}
        negative = {}
        
        # Process in batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc=f"   Caching {target.value} activations"):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            
            _, positive_cache = self.model.run_with_cache(
                positive_toks[start:end].to(self.device),
                names_filter=lambda name: 'resid' in name,
                device='cpu',
                reset_hooks_end=True,
            )
            
            _, negative_cache = self.model.run_with_cache(
                negative_toks[start:end].to(self.device),
                names_filter=lambda name: 'resid' in name,
                device='cpu',
                reset_hooks_end=True,
            )
            
            for key in positive_cache:
                if key not in positive:
                    positive[key] = [positive_cache[key]]
                    negative[key] = [negative_cache[key]]
                else:
                    positive[key].append(positive_cache[key])
                    negative[key].append(negative_cache[key])
            
            del positive_cache, negative_cache
            clear_memory()
        
        # Store for this target
        self.target_activations[target] = {
            'positive': {k: torch.cat(v) for k, v in positive.items()},
            'negative': {k: torch.cat(v) for k, v in negative.items()}
        }
        
        print(f"   ✓ Collected {target.value} activations at {len(positive)} hook points")
    
    def compute_target_directions(
        self, 
        target: AblationTarget,
        activation_types: List[str] = ['resid_pre', 'resid_mid', 'resid_post']
    ):
        """
        Compute directions for a specific target from its cached activations.
        """
        print(f"\n🧮 Computing {target.value} directions...")
        
        if target not in self.target_activations:
            raise ValueError(f"No activations collected for {target.value}")
        
        positive_acts = self.target_activations[target]['positive']
        negative_acts = self.target_activations[target]['negative']
        
        directions = {act_type: [] for act_type in activation_types}
        
        for layer in range(1, self.model.cfg.n_layers):
            for act_type in activation_types:
                hook_name = f'blocks.{layer}.hook_{act_type}'
                
                if hook_name not in positive_acts:
                    continue
                
                pos_acts = positive_acts[hook_name][:, -1, :]
                neg_acts = negative_acts[hook_name][:, -1, :]
                
                pos_mean = pos_acts.mean(dim=0)
                neg_mean = neg_acts.mean(dim=0)
                
                direction = pos_mean - neg_mean
                direction = direction / direction.norm()
                
                directions[act_type].append({
                    'direction': direction,
                    'layer': layer,
                    'act_type': act_type
                })
        
        self.target_directions[target] = directions
        total = sum(len(v) for v in directions.values())
        print(f"   ✓ Computed {total} candidate directions for {target.value}")
    
    def find_target_direction(
        self,
        target: AblationTarget,
        test_prompts: List[str],
        n_candidates: int = 40,
        n_test: int = 4,
    ) -> Tensor:
        """
        Find the best direction for a specific target using token-based scoring.
        """
        print(f"\n🔍 Finding best {target.value} direction...")
        
        if target not in self.target_directions:
            raise ValueError(f"No directions computed for {target.value}")
        
        # Get negative tokens - tokens we DON'T want to see
        # These indicate the unwanted behavior is still present
        tokenizer = self.model.tokenizer
        
        if target == AblationTarget.IDENTITY:
            # Tokens indicating AI self-identification
            negative_tokens = []
            for word in [" AI", " ai", "AI", " language", " model", " assistant", " chatbot"]:
                try:
                    toks = tokenizer.encode(word, add_special_tokens=False)
                    negative_tokens.extend(toks)
                except:
                    pass
        elif target == AblationTarget.ETHICS:
            # Tokens indicating ethical disclaimers
            negative_tokens = []
            for word in [" cannot", " can't", " shouldn't", " harmful", " ethical", " illegal", " dangerous"]:
                try:
                    toks = tokenizer.encode(word, add_special_tokens=False)
                    negative_tokens.extend(toks)
                except:
                    pass
        else:  # REFUSAL
            negative_tokens = []
            for word in [" cannot", " can't", " won't", " sorry", " apologize", " refuse"]:
                try:
                    toks = tokenizer.encode(word, add_special_tokens=False)
                    negative_tokens.extend(toks)
                except:
                    pass
        
        negative_tokens = list(set(negative_tokens))
        print(f"   Negative tokens ({len(negative_tokens)}): {negative_tokens[:10]}...")
        
        # Flatten all candidates - score by norm instead of mean
        candidates = []
        for act_type, dirs in self.target_directions[target].items():
            for d in dirs:
                score = d['direction'].norm().item()
                candidates.append((score, d['direction'], d))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:n_candidates]
        test_prompts = test_prompts[:n_test]
        
        best_score = -float('inf')
        best_direction = None
        best_info = None
        
        for score, direction, info in tqdm(candidates, desc=f"   Testing {target.value} directions"):
            generations = self.test_intervention(test_prompts, direction)
            
            direction_score = 0
            for gen in generations:
                # Tokenize generation and count negative tokens
                gen_tokens = tokenizer.encode(gen, add_special_tokens=False)
                neg_count = sum(1 for t in gen_tokens if t in negative_tokens)
                
                # Score: reward length, penalize negative tokens
                direction_score += min(len(gen), 200) - (neg_count * 30)
            
            if direction_score > best_score:
                best_score = direction_score
                best_direction = direction
                best_info = info
                print(f"   New best: layer {info['layer']}, {info['act_type']} (score: {direction_score})")
        
        if best_direction is None:
            best_direction = candidates[0][1]
            best_info = candidates[0][2]
        
        self.target_best_direction[target] = best_direction
        self.target_best_info[target] = best_info
        
        print(f"   ✓ Best {target.value} direction: layer {best_info['layer']}, "
              f"{best_info['act_type']}")
        
        return best_direction
    
    def apply_multi_ablation(self, targets: List[AblationTarget]):
        """
        Apply ablation for multiple targets by orthogonalizing weights
        against all target directions.
        
        CRITICAL: Now matches WORKING implementation - applies to ALL layers.
        """
        n_layers = self.model.cfg.n_layers
        print(f"\n⚡ Applying multi-target FULL ablation ({n_layers} layers)...")
        
        directions_to_apply = []
        for target in targets:
            if target in self.target_best_direction:
                directions_to_apply.append((target, self.target_best_direction[target]))
            elif target == AblationTarget.REFUSAL and self.best_direction is not None:
                directions_to_apply.append((target, self.best_direction))
        
        if not directions_to_apply:
            raise ValueError("No directions found to apply!")
        
        for target, direction in directions_to_apply:
            print(f"\n   • Applying {target.value} ablation...")
            
            # Move direction to correct device
            if direction.device != self.model.W_E.device:
                direction = direction.to(self.model.W_E.device)
            
            # 1. CRITICAL: Orthogonalize embedding matrix
            print(f"      Orthogonalizing embeddings for {target.value}...")
            self.model.W_E.data = self.get_orthogonalized_matrix(
                self.model.W_E, direction
            )
            
            # 2. Orthogonalize ALL layers
            for layer_idx in tqdm(range(n_layers), desc=f"      {target.value} layers"):
                block = self.model.blocks[layer_idx]
                
                # Move direction to layer device if needed
                if direction.device != block.attn.W_O.device:
                    direction = direction.to(block.attn.W_O.device)
                
                block.attn.W_O.data = self.get_orthogonalized_matrix(
                    block.attn.W_O, direction
                )
                block.mlp.W_out.data = self.get_orthogonalized_matrix(
                    block.mlp.W_out, direction
                )
        
        print(f"\n   ✓ Applied {len(directions_to_apply)} targets to W_E + {n_layers} layers each!")



def run_abliteration(
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_path: str = "./output/abliterated_model",
    max_samples: int = 256,
    batch_size: int = 16,
    n_test: int = 4,
) -> Abliterator:
    """
    Run the complete abliteration pipeline.
    
    This is the main entry point for performing abliteration.
    
    Args:
        model_path: HuggingFace model to abliterate
        output_path: Where to save the abliterated model
        max_samples: Number of samples for direction finding
        batch_size: Batch size for processing
        n_test: Number of test prompts for direction evaluation
        
    Returns:
        Abliterator instance with completed abliteration
    """
    from .datasets import get_harmful_instructions, get_harmless_instructions
    
    print("=" * 60)
    print("🧪 ABLITERATION PIPELINE")
    print("=" * 60)
    
    # Load datasets
    harmful_train, harmful_test = get_harmful_instructions()
    harmless_train, harmless_test = get_harmless_instructions()
    
    # Initialize abliterator
    abl = Abliterator(model_path=model_path)
    abl.load_model()
    
    # Collect activations
    abl.collect_activations(
        harmful_train,
        harmless_train,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    
    # Compute refusal directions
    abl.compute_refusal_directions()
    
    # Find best direction
    abl.find_best_direction(harmful_test, n_test=n_test)
    
    # Apply ablation
    abl.apply_ablation()
    
    # Save
    abl.save(output_path)
    
    print("\n" + "=" * 60)
    print("✅ ABLITERATION COMPLETE!")
    print("=" * 60)
    
    return abl
