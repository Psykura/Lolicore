import os
import jax
import jax.numpy as jnp
import numpy as np
import time
from transformers import AutoTokenizer
if os.path.exists('transformer.py'):
    from transformer import Transformer
    from train import MODEL_CONFIG
#jax.config.update("jax_check_tracer_leaks", True)

MODEL_CONFIG['training'] = False
MODEL_CONFIG['use_gradient_checkpointing'] = False

@jax.jit
def model_step(variables, input_ids, attention_mask, rng):
    """Single forward pass of the model with JIT."""
    logits, _ = Transformer(**MODEL_CONFIG).apply(
        variables,
        input_ids,
        attn_mask=attention_mask,
        rngs={'noise': rng}
    )
    return logits

def test_autoregressive_generation():
    # Initialize model and tokenizer
    model = Transformer(**MODEL_CONFIG)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    batch_size = 1
    
    # Create prompt
    prompt = "Can you tell me"
    input_tokens = tokenizer(prompt, return_tensors="np")
    input_ids = jnp.array(input_tokens["input_ids"])
    prompt_length = input_ids.shape[1]
    
    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, input_ids)
    
    # Setup generation
    print("Generating")
    start_time = time.time()
    
    max_new_tokens = 20
    total_length = prompt_length + max_new_tokens
    
    # Pre-fill the sequence with padding tokens
    padded_ids = jnp.pad(
        input_ids,
        ((0, 0), (0, max_new_tokens)),
        mode='constant',
        constant_values=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    )
    
    # Initialize attention mask (1 for prompt tokens, 0 for future tokens)
    attention_mask = jnp.zeros((batch_size, total_length))
    attention_mask = attention_mask.at[:, :prompt_length].set(1)
    
    # Generate tokens
    current_length = prompt_length
    for i in range(max_new_tokens):
        # Forward pass with masked sequence
        rng, rng_step = jax.random.split(rng)
        logits = model_step(variables, padded_ids, attention_mask, rng=rng_step)
        
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
    
    print(f"Generated text:\n{generated_text}")
    print(f"\nGeneration time: {generation_time:.2f} seconds")

def main():    
    # Run tests
    test_autoregressive_generation()
    
    print("\nAll tests completed successfully! ✨")

if __name__ == "__main__":
    main()