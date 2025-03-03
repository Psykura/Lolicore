from transformer import Transformer
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from flax.training import checkpoints
import os
import numpy as np
from flax.training import orbax_utils
import time
import functools

CONTEXT_LENGTH = 2048
VOCAB_SIZE = 50257
# Pad vocab size to be divisible by 128 for better TPU/GPU utilization
PADDED_VOCAB_SIZE = ((VOCAB_SIZE + 127) // 128) * 128  # = 50304
DTYPE = jnp.bfloat16  # Set default dtype to bfloat16

MODEL_CONFIG = {
    'num_blocks': 12,
    'num_heads': 8,
    'd_model': 512,
    'hidden_size': 4096,
    'max_seq_length': CONTEXT_LENGTH,
    'vocab_size': PADDED_VOCAB_SIZE,  # GPT-2 vocab size
    'num_experts': 16,
    'num_shared_experts': 1,
    'use_gradient_checkpointing': False,
    'training': False,
    'attention_latent_dim': 48,
    'num_constant_experts': 4,
    'num_noise_experts': 1,
}

def create_model():
    """Create and initialize the model."""
    return Transformer(**MODEL_CONFIG)

# JIT-compiled model forward pass
@functools.partial(jax.jit, static_argnums=(0,))
def model_forward(model, params, input_ids, attention_mask, rng):
    """JIT-compiled forward pass for the model."""
    # Use a pure function approach for better JIT compatibility
    def _forward(params, input_ids, attention_mask, rng):
        return model.apply(
            {'params': params},
            input_ids,
            attention_mask,
            rngs={'dropout': rng, 'noise': rng},
            mutable=False  # Ensure no state is mutated for pure function
        )
    
    # Call the forward function
    logits = _forward(params, input_ids, attention_mask, rng)
    
    # The model returns a tuple of (logits, router_loss) in training mode
    # and just logits in inference mode
    if isinstance(logits, tuple):
        logits = logits[0]
    
    return logits

def load_model():
    """Load model from checkpoint."""
    # Try to use unsharded checkpoint first
    unsharded_dir = os.path.abspath("./checkpoints/checkpoint_25000")
    
    # Initialize model with the same config as during training
    model = create_model()
    # Initialize with random weights - don't JIT this part
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, CONTEXT_LENGTH), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, CONTEXT_LENGTH), dtype=jnp.int32)
    
    # Initialize model without JIT
    variables = model.init(rng, dummy_input, dummy_mask)
    state = {'params': variables['params']}
    
    #state = checkpoints.restore_checkpoint(unsharded_dir, target=state)

    return model, state

def generate_text(prompt, model, state, tokenizer, 
                 max_new_tokens=50,  # Reduced from 100 to 50 for faster generation
                 temperature=0.8,
                 top_k=50):  # Add timeout to prevent hanging
    """Generate text using a simple autoregressive approach with timeout."""
    # Tokenize the prompt
    input_tokens = tokenizer(prompt, return_tensors="np")
    input_ids = jnp.array(input_tokens["input_ids"])  # Convert to jax array
    
    # Create attention mask (1 for tokens, 0 for padding)
    attention_mask = jnp.ones_like(input_ids)
    
    # Create PRNG key for sampling
    rng = jax.random.key(0)
    
    # Measure generation time
    start_time = time.time()
    tokens_generated = 0
    
    # Store generated tokens
    generated_ids = input_ids
    
    # Generate tokens one by one with timeout
    for i in range(max_new_tokens):
        # Get model output for the current sequence using JIT-compiled forward pass
        # Only use the last 128 tokens to reduce memory usage for long sequences
        context_window = min(128, generated_ids.shape[1])
        input_window = generated_ids[:, -context_window:]
        mask_window = jnp.ones_like(input_window)
        
        # Get logits for the current window
        logits = model_forward(model, state['params'], input_window, mask_window, rng)
        
        # Get logits for the last token
        next_token_logits = logits[:, -1, :] / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, k=min(top_k, next_token_logits.shape[-1]))
            next_token_logits = jnp.zeros_like(next_token_logits).at[
                jnp.arange(input_ids.shape[0])[:, None], top_k_indices
            ].set(top_k_logits)
        
        # Sample from the distribution
        rng, sample_rng = jax.random.split(rng)
        next_token = jax.random.categorical(sample_rng, next_token_logits, axis=-1)
        
        # Add the new token to the sequence
        generated_ids = jnp.concatenate([generated_ids, next_token[:, None]], axis=1)
        tokens_generated += 1
        
        # Clear memory cache every 10 tokens
        if i % 10 == 0 and i > 0:
            jax.clear_caches()
    
    # Calculate generation time and tokens per second
    end_time = time.time()
    generation_time = end_time - start_time
    tokens_per_second = tokens_generated / generation_time if tokens_generated > 0 else 0
    
    print(f"Generated {tokens_generated} tokens in {generation_time:.2f} seconds")
    print(f"Speed: {tokens_per_second:.2f} tokens/second")
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return generated_text

def main():
    # Try to enable GPU if available
    try:
        # Print available devices
        print(f"Available JAX devices: {jax.devices()}")
        print(f"JAX process index: {jax.process_index()}")
        print(f"JAX device count: {jax.device_count()}")
    except Exception as e:
        print(f"Error checking JAX devices: {e}")
    
    # Set memory growth to allow for more efficient memory usage
    jax.config.update('jax_default_matmul_precision', 'bfloat16')
    
    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Load the model and state
    print("Loading model...")
    model, state = load_model()
    
    # Warm up the JIT compilation with a dummy forward pass
    print("Warming up JIT compilation...")
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)  # Small input for warmup
    dummy_mask = jnp.ones_like(dummy_input)
    dummy_rng = jax.random.key(0)
    try:
        _ = model_forward(model, state['params'], dummy_input, dummy_mask, dummy_rng)
        print("JIT compilation complete")
    except Exception as e:
        print(f"Warning: JIT warmup failed, but continuing: {e}")
    
    print("Model loaded and ready for generation")
    
    # Example prompts to test - use shorter prompts for faster testing
    prompts = [
        "Why is",
        "Once",
        "The"
    ]
    
    print("\nGenerating text from prompts:")
    for i, prompt in enumerate(prompts):
        print("\nPrompt:", prompt)
        print("-" * 50)
        
        # Clear memory before each generation
        jax.clear_caches()
        
        try:
            generated_text = generate_text(
                prompt, 
                model, 
                state, 
                tokenizer,
                max_new_tokens=20 if i == 0 else 30  # Use fewer tokens for first prompt
            )
            print("Generated:", generated_text)
        except Exception as e:
            print(f"Error during generation: {e}")
        
        print("-" * 50)
        
        # Force garbage collection
        import gc
        gc.collect()
        jax.clear_caches()

if __name__ == "__main__":
    main()

