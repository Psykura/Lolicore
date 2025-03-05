import os
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from transformers import AutoTokenizer
from typing import List, Optional, NamedTuple, Callable
import numpy as np
import time
if os.path.exists("train.py"):
    from train import (
        DTYPE, CONTEXT_LENGTH, MODEL_CONFIG
    )
    from transformer import Transformer
from functools import partial

MODEL_CONFIG['training'] = False
MODEL_CONFIG['use_gradient_checkpointing'] = False

class InferenceState(NamedTuple):
    """Simple state for inference without optimizer."""
    apply_fn: callable
    params: dict

def create_inference_state(
    checkpoint_dir: str,
    step: Optional[int] = None,
    use_best_model: bool = True,
) -> InferenceState:
    """Creates inference state and loads model weights from checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Specific step to load, or None to load latest
        use_best_model: If True and step is None, first try to load 'best_model_inference'
                        checkpoint before falling back to latest
    
    Returns:
        InferenceState with loaded parameters and model
    """
    # Ensure we're using CPU
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        # First initialize the model to get the expected parameter structure
        model = Transformer(**MODEL_CONFIG)
        
        # Create a dummy input for initialization
        dummy_input = jnp.ones((1, CONTEXT_LENGTH), dtype=jnp.int32)
        dummy_mask = jnp.ones((1, CONTEXT_LENGTH), dtype=jnp.int32)
        
        # Initialize model parameters with dummy input
        init_rng = jax.random.PRNGKey(0)
        init_rng, dropout_rng = jax.random.split(init_rng)
        init_rngs = {'params': init_rng, 'noise': dropout_rng}
        variables = model.init(init_rngs, dummy_input, dummy_mask)
        
        print("Loading checkpoint...")
        # Load checkpoint
        async_checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler(), timeout_secs=50)
        async_checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir,
            async_checkpointer,
            options=ocp.CheckpointManagerOptions(enable_async_checkpointing=True)
        )

        # Try to load best model if requested
        loaded_step = None
        loaded_params = None
        if step is None and use_best_model:
            try:
                # Try to load the CPU-optimized best model
                print("Attempting to load best_model_inference checkpoint...")
                loaded_state = async_checkpoint_manager.restore('best_model_inference')
                loaded_step = 'best_model_inference'
                print("Loaded best_model_inference checkpoint successfully")
            except Exception as e:
                print(f"Could not load best_model_inference: {e}")
                try:
                    # Try to load the regular best model
                    print("Attempting to load best_model checkpoint...")
                    loaded_state = async_checkpoint_manager.restore('best_model')
                    loaded_step = 'best_model'
                    print("Loaded best_model checkpoint successfully")
                except Exception as e:
                    print(f"Could not load best_model: {e}")
                    # Fall back to latest checkpoint
                    loaded_step = async_checkpoint_manager.latest_step()
                    if loaded_step is None:
                        raise ValueError("No checkpoints found in directory")
                    print(f"Falling back to latest checkpoint at step {loaded_step}")
                    loaded_state = async_checkpoint_manager.restore(loaded_step)
                    
                    # Check if it's a parameter-only checkpoint or full state checkpoint
                    if 'params' in loaded_state:
                        # Parameter-only checkpoint
                        loaded_params = loaded_state['params']
                        print(f"Loaded step {loaded_step} parameters checkpoint successfully")
                    elif 'state' in loaded_state and 'params' in loaded_state['state']:
                        # Full state checkpoint
                        loaded_params = loaded_state['state']['params']
                        print(f"Loaded step {loaded_step} state checkpoint successfully")
                    else:
                        raise ValueError("Unrecognized checkpoint format")
        else:
            # Load specific step or latest
            if step is None:
                step = async_checkpoint_manager.latest_step()
                if step is None:
                    raise ValueError("No checkpoints found in directory")
            
            print(f"Loading checkpoint from step {step}")
            loaded_state = async_checkpoint_manager.restore(step)
            loaded_step = step
            
            # Check if it's a parameter-only checkpoint or full state checkpoint
            if 'params' in loaded_state:
                # Parameter-only checkpoint
                loaded_params = loaded_state['params']
                print(f"Loaded step {loaded_step} parameters checkpoint successfully")
            elif 'state' in loaded_state and 'params' in loaded_state['state']:
                # Full state checkpoint
                loaded_params = loaded_state['state']['params']
                print(f"Loaded step {loaded_step} state checkpoint successfully")
            else:
                raise ValueError("Unrecognized checkpoint format")

        async_checkpoint_manager.wait_until_finished()
        print(f"Checkpoint loaded successfully from {loaded_step}")

        # Update the initialized parameters with the loaded checkpoint
        if loaded_params is not None:
            variables['params'] = loaded_params

        # Create inference state with properly initialized parameters
        return InferenceState(
            apply_fn=model.apply,
            params=variables['params']
        ), model

def top_k_top_p_filtering(
    logits: jnp.ndarray,
    top_k: int = 0,
    top_p: float = 0.0,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Filter logits using top-k and/or nucleus (top-p) sampling."""
    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < jax.lax.top_k(logits, top_k)[0][..., -1, None]
        logits = jnp.where(indices_to_remove, float('-inf'), logits)
    
    if top_p > 0.0:
        sorted_logits = jnp.sort(logits, axis=-1)[:, ::-1]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)
        
        indices_to_remove = jnp.zeros_like(logits, dtype=bool).at[
            jnp.arange(logits.shape[0])[:, None],
            jnp.argsort(logits, axis=-1)[:, ::-1]
        ].set(sorted_indices_to_remove)
        
        logits = jnp.where(indices_to_remove, float('-inf'), logits)
    
    return logits

def generate_step(
    apply_fn: Callable,
    params: dict,
    input_ids: jnp.ndarray,
    rng: jax.random.PRNGKey,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> jnp.ndarray:
    """Single step of text generation."""
    logits, _ = apply_fn(
        {'params': params},
        input_ids,
        rngs={'noise': rng}
    )
    
    # Get logits of the last token
    next_token_logits = logits[:, -1, :]
    
    # Apply temperature and sampling methods
    filtered_logits = top_k_top_p_filtering(
        next_token_logits,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature
    )
    
    # Sample from the filtered distribution
    next_token = jax.random.categorical(rng, filtered_logits, axis=-1)
    
    return next_token

def generate_text(
    prompt: str,
    model: Transformer,
    state: InferenceState,
    tokenizer: AutoTokenizer,
    max_length: int = 100,
    temperature: float = 0.5,
    top_k: int = 50,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    seed: int = 42,
) -> List[str]:
    """Generate text from a prompt.
    
    Args:
        prompt: The input prompt to generate from
        model: The transformer model
        state: The inference state containing model parameters
        tokenizer: The tokenizer
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        num_return_sequences: Number of sequences to generate
        seed: Random seed
    
    Returns:
        List of generated text sequences
    """
    # Ensure we're using CPU
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        # Tokenize the prompt
        input_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=CONTEXT_LENGTH,
            return_tensors="np"
        )
        
        input_ids = jnp.array(input_tokens["input_ids"])
        
        # Prepare for batch generation
        input_ids = jnp.repeat(input_ids, num_return_sequences, axis=0)
        
        # Track current sequence
        current_ids = input_ids
        
        # Generate tokens one at a time
        for i in range(max_length):
            rng, step_rng = jax.random.split(jax.random.PRNGKey(seed + i))
            
            # Process the entire sequence each time
            logits, _ = state.apply_fn(
                {'params': state.params},
                current_ids,
                rngs={'noise': step_rng}
            )
            
            # Get logits of the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature and sampling methods
            filtered_logits = top_k_top_p_filtering(
                next_token_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )
            
            # Sample from the filtered distribution
            next_token = jax.random.categorical(step_rng, filtered_logits, axis=-1)
            
            # Append new token to sequence
            current_ids = jnp.concatenate([current_ids, next_token[:, None]], axis=1)
            
            # Check if any sequence has generated an EOS token
            if jnp.any(next_token == tokenizer.eos_token_id):
                break
        
        # Decode generated sequences
        generated_sequences = []
        for i in range(num_return_sequences):
            output_tokens = current_ids[i].tolist()
            # Remove padding tokens
            output_tokens = [t for t in output_tokens if t != tokenizer.pad_token_id]
            generated_sequences.append(tokenizer.decode(output_tokens))
        
        return generated_sequences

def greedy_generate_text(
    prompt: str,
    model: Transformer,
    state: InferenceState,
    tokenizer: AutoTokenizer,
    max_length: int = 20,
    seed: int = 0,
) -> str:
    """Generate text using simple greedy decoding
    
    Args:
        prompt: The input prompt to generate from
        model: The transformer model
        state: The inference state containing model parameters
        tokenizer: The tokenizer
        max_length: Maximum number of tokens to generate
        seed: Random seed
    
    Returns:
        Generated text sequence
    """
    # Ensure we're using CPU
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        # Create prompt
        input_tokens = tokenizer(prompt, return_tensors="np")
        input_ids = jnp.array(input_tokens["input_ids"])
        prompt_length = input_ids.shape[1]
        batch_size = 1
        
        # Setup generation
        print("Generating with greedy decoding...")
        start_time = time.time()
        
        total_length = prompt_length + max_length
        
        # Pre-fill the sequence with padding tokens
        padded_ids = jnp.pad(
            input_ids,
            ((0, 0), (0, max_length)),
            mode='constant',
            constant_values=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        
        # Initialize attention mask (1 for prompt tokens, 0 for future tokens)
        attention_mask = jnp.zeros((batch_size, total_length))
        attention_mask = attention_mask.at[:, :prompt_length].set(1)
        
        @jax.jit
        def model_step(variables, input_ids, attention_mask, rng):
            """Single forward pass of the model with JIT."""
            logits, _ = model.apply(
                variables,
                input_ids,
                attn_mask=attention_mask,
                rngs={'noise': rng}
            )
            return logits
        
        # Generate tokens
        current_length = prompt_length
        rng = jax.random.PRNGKey(seed)
        
        for i in range(max_length):
            # Forward pass with masked sequence
            rng, rng_step = jax.random.split(rng)
            
            logits = model_step(
                {'params': state.params},
                padded_ids,
                attention_mask,
                rng=rng_step
            )
            
            # Get next token (only look at the current position)
            next_token = jnp.argmax(logits[:, current_length-1], axis=-1)
            
            # Update sequence and mask
            padded_ids = padded_ids.at[:, current_length].set(next_token)
            attention_mask = attention_mask.at[:, current_length].set(1)
            current_length += 1
            
            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break
        
        generation_time = time.time() - start_time
        
        # Get the actual generated sequence (without padding)
        generated_ids = padded_ids[:, :current_length]
        generated_text = tokenizer.decode(generated_ids[0])
        
        print(f"Generation time: {generation_time:.2f} seconds")
        return generated_text

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
    
    # Initialize model state on CPU
    checkpoint_dir = "/root/checkpoints"
    print("Loading model from checkpoint...")
    
    # First use the best model if available
    state, model = create_inference_state(
        checkpoint_dir,
        use_best_model=True  # Will first try best_model_inference, then best_model, then latest
    )
    
    # Example prompts
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a galaxy far, far away"
    ]
    
    # Generate text for each prompt with different generation methods
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        
        print("\nGenerating with greedy decoding:")
        greedy_text = greedy_generate_text(
            prompt,
            model,
            state,
            tokenizer,
            max_length=10
        )
        print(f"\nGeneration:")
        print(greedy_text)
        print("-" * 50)

if __name__ == "__main__":
    main() 