from transformer import Transformer
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from flax.training import checkpoints
import os
import numpy as np
from flax.training import orbax_utils
import time

CONTEXT_LENGTH = 512
VOCAB_SIZE = 50257
# Pad vocab size to be divisible by 128 for better TPU/GPU utilization
PADDED_VOCAB_SIZE = ((VOCAB_SIZE + 127) // 128) * 128  # = 50304
DTYPE = jnp.bfloat16  # Set default dtype to bfloat16

MODEL_CONFIG = {
    'num_blocks': 12,
    'num_heads': 8,
    'd_model': 512,
    'hidden_size': 2048,
    'max_seq_length': CONTEXT_LENGTH,
    'vocab_size': PADDED_VOCAB_SIZE,  # GPT-2 vocab size
    'num_experts': 16,
    'num_shared_experts': 1,
    'top_k': 4,
    'use_gradient_checkpointing': False,
    'training': False,
    'attention_latent_dim': 32,
    'num_zeros_experts': 1,
    'num_constant_experts': 2,
    'num_noise_experts': 1,
}

def create_model():
    """Create and initialize the model."""
    return Transformer(**MODEL_CONFIG)

def load_model():
    """Load model from checkpoint."""
    # Try to use unsharded checkpoint first
    unsharded_dir = os.path.abspath("./checkpoints/checkpoint_20000")
    
    # Initialize model with the same config as during training
    model = Transformer(
        num_blocks=MODEL_CONFIG['num_blocks'],
        num_heads=MODEL_CONFIG['num_heads'],
        d_model=MODEL_CONFIG['d_model'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        max_seq_length=MODEL_CONFIG['max_seq_length'],
        vocab_size=MODEL_CONFIG['vocab_size'],
        num_experts=MODEL_CONFIG['num_experts'],
        num_shared_experts=MODEL_CONFIG['num_shared_experts'],
        top_k=MODEL_CONFIG['top_k'],
        dtype=DTYPE,
        use_gradient_checkpointing=MODEL_CONFIG['use_gradient_checkpointing'],
        training=MODEL_CONFIG['training'],
        attention_latent_dim=MODEL_CONFIG['attention_latent_dim'],
        num_zeros_experts=MODEL_CONFIG['num_zeros_experts'],
        num_constant_experts=MODEL_CONFIG['num_constant_experts'],
        num_noise_experts=MODEL_CONFIG['num_noise_experts'],
    )

        # Initialize with random weights
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, CONTEXT_LENGTH), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, CONTEXT_LENGTH), dtype=jnp.int32)
    
    variables = model.init(rng, dummy_input, dummy_mask)
    state = {'params': variables['params']}
    
    # Check if unsharded checkpoint exists
    if os.path.exists(unsharded_dir):
        try:
            print(f"Loading from checkpoint at {unsharded_dir}...")
            state = checkpoints.restore_checkpoint(unsharded_dir, target=state)
            if state is not None:
                print("Successfully loaded checkpoint.")
                return model, state
        except Exception as e:
            print(f"Error loading unsharded checkpoint: {e}")
    print("Initializing model with random weights...")
    
    return model, state

# JIT-compiled text generation function for better performance
@jax.jit
def generate_text_jit(model, state, input_ids, attention_mask, max_new_tokens, temperature, top_k, top_p, prng_key):
    """JIT-compiled version of text generation for better performance."""
    return model.apply(
        {'params': state['params']},
        method=model.generate,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        prng_key=prng_key
    )

def generate_text(prompt, model, state, tokenizer, 
                 max_new_tokens=100, 
                 temperature=0.8,
                 top_k=50,
                 top_p=0.9):
    # Tokenize the prompt
    input_tokens = tokenizer(prompt, return_tensors="np")
    input_ids = jnp.array(input_tokens["input_ids"])  # Convert to jax array
    
    # Create attention mask
    attention_mask = jnp.ones_like(input_ids)
    
    # Create PRNG keys for sampling and noise
    rng = jax.random.PRNGKey(0)
    rng, sampling_rng = jax.random.split(rng)
    
    # Measure generation time
    start_time = time.time()
    
    # Generate tokens using the JIT-compiled function
    output_ids = generate_text_jit(
        model, 
        state, 
        input_ids, 
        attention_mask, 
        max_new_tokens, 
        temperature, 
        top_k, 
        top_p, 
        sampling_rng
    )
    
    # Calculate generation time and tokens per second
    end_time = time.time()
    generation_time = end_time - start_time
    tokens_generated = output_ids.shape[1] - input_ids.shape[1]
    tokens_per_second = tokens_generated / generation_time
    
    print(f"Generated {tokens_generated} tokens in {generation_time:.2f} seconds")
    print(f"Speed: {tokens_per_second:.2f} tokens/second")
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
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
    
    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Load the model and state
    print("Loading model...")
    model, state = load_model()
    
    # Compile the model once with a dummy input to warm up JAX
    print("Warming up JAX compilation...")
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    dummy_mask = jnp.ones_like(dummy_input)
    dummy_rng = jax.random.PRNGKey(0)
    _ = generate_text_jit(model, state, dummy_input, dummy_mask, 1, 0.8, 50, 0.9, dummy_rng)
    print("JAX compilation completed")
    
    # Example prompts to test
    prompts = [
        "Why the ocean is salty? The Ocean is salty because",
        "Once upon a time",
        "The future of artificial intelligence",
        "In a world where technology",
        "The capital of France is",
    ]
    
    print("\nGenerating text from prompts:")
    for prompt in prompts:
        print("\nPrompt:", prompt)
        print("-" * 50)
        generated_text = generate_text(prompt, model, state, tokenizer)
        print("Generated:", generated_text)
        print("-" * 50)

if __name__ == "__main__":
    main()

