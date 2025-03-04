import os
import jax
import jax.numpy as jnp
from flax.training import train_state
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from transformers import AutoTokenizer
from typing import List, Optional, NamedTuple, Callable
import numpy as np
if os.path.exists("train.py"):
    from train import (
        DTYPE, CONTEXT_LENGTH
    )
    from transformer import Transformer
from functools import partial

vocab_size = 50257
vocab_size = ((vocab_size + 127) // 128) * 128

MODEL_CONFIG = {
    'num_blocks': 6,
    'num_heads': 8,
    'd_model': 512,
    'hidden_size': 4096,
    'max_seq_length': CONTEXT_LENGTH,
    'vocab_size': vocab_size,  # GPT-2 vocab size
    'num_experts': 16,
    'num_shared_experts': 1,
    'use_gradient_checkpointing': False,
    'attention_latent_dim': 64,
    'num_constant_experts': 0,
    'num_noise_experts': 0,
}

class InferenceState(NamedTuple):
    """Simple state for inference without optimizer."""
    apply_fn: callable
    params: dict

def create_inference_state(
    checkpoint_dir: str,
    step: Optional[int] = None,
) -> InferenceState:
    """Creates inference state and loads model weights from checkpoint."""
    # Ensure we're using CPU
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        # First initialize the model to get the expected parameter structure
        model = Transformer(dtype=DTYPE, training=False, **MODEL_CONFIG)
        
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

        if step is None:
            step = async_checkpoint_manager.latest_step()
            if step is None:
                raise ValueError("No checkpoints found in directory")

        print(f"Loading checkpoint from step {step}")
        loaded_state = async_checkpoint_manager.restore(step)
        async_checkpoint_manager.wait_until_finished()
        print("Checkpoint loaded successfully")

        # Update the initialized parameters with the loaded checkpoint
        variables['params'].update(loaded_state['params'])

        # Create inference state with properly initialized parameters
        return InferenceState(
            apply_fn=model.apply,
            params=variables['params']
        )

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

@partial(jax.jit, static_argnames=['apply_fn', 'temperature', 'top_k', 'top_p'])
def generate_step(
    apply_fn: Callable,
    params: dict,
    input_ids: jnp.ndarray,
    attention_mask: jnp.ndarray,
    rng: jax.random.PRNGKey,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> jnp.ndarray:
    """Single step of text generation."""
    logits, _ = apply_fn(
        {'params': params},
        input_ids,
        attention_mask,
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
    state: InferenceState,
    tokenizer: AutoTokenizer,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    seed: int = 42,
) -> List[str]:
    """Generate text from a prompt."""
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
        attention_mask = jnp.array(input_tokens["attention_mask"])
        
        # Prepare for batch generation
        input_ids = jnp.repeat(input_ids, num_return_sequences, axis=0)
        attention_mask = jnp.repeat(attention_mask, num_return_sequences, axis=0)
        
        # Initialize RNG
        rng = jax.random.PRNGKey(seed)
        
        generated_sequences = []
        for _ in range(max_length):
            rng, step_rng = jax.random.split(rng)
            
            # Generate next token
            next_token = generate_step(
                state.apply_fn,
                state.params,
                input_ids,
                attention_mask,
                step_rng,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Append new token to sequence
            input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)
            attention_mask = jnp.concatenate([
                attention_mask,
                jnp.ones((num_return_sequences, 1), dtype=jnp.int32)
            ], axis=1)
            
            # Check if any sequence has generated an EOS token
            if jnp.any(next_token == tokenizer.eos_token_id):
                break
        
        # Decode generated sequences
        for i in range(num_return_sequences):
            output_tokens = input_ids[i].tolist()
            # Remove padding tokens
            output_tokens = [t for t in output_tokens if t != tokenizer.pad_token_id]
            generated_sequences.append(tokenizer.decode(output_tokens))
        
        return generated_sequences

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
    
    # Initialize model state on CPU
    checkpoint_dir = "/root/checkpoints"
    print("Loading model from checkpoint...")
    state = create_inference_state(checkpoint_dir)
    
    # Example prompts
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a galaxy far, far away"
    ]
    
    # Generate text for each prompt
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        
        generated_texts = generate_text(
            prompt,
            state,
            tokenizer,
            max_length=10,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            num_return_sequences=1
        )
        
        for i, text in enumerate(generated_texts, 1):
            print(f"\nGeneration {i}:")
            print(text)
            print("-" * 50)

if __name__ == "__main__":
    main() 