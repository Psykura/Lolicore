import jax.numpy as jnp
import flax.linen as nn
import jax
from jax import random
from transformer import ExpertsFeedForward
import time

# Example usage
def main():
  # Create a key for random number generation
  key = random.key(0)
  
  # Create input data
  batch_size = 1
  seq_len = 128
  d_model = 512  # Match the d_model parameter in the model
  x = random.normal(key, (batch_size, seq_len, d_model))
  
  # Test both training and inference modes
  for training_mode in [False, True]:
    print(f"\nTesting with training = {training_mode}")
    
    # Create model instance with compatible parameters
    model = ExpertsFeedForward(
        d_model=512, 
        hidden_size=2048 * 1,
        num_experts=64,
        num_shared_experts=1, 
        num_constant_experts=10, 
        num_noise_experts=10, 
        dtype=jnp.bfloat16, 
        use_gradient_checkpointing=False, 
        training=training_mode
    )
    
    # Initialize model parameters
    variables = model.init({'noise': key, 'dropout': key, 'params': key}, x)
    params = variables['params']
    
    @jax.jit
    def apply_fn(params, inputs):
        output, loss = model.apply({'params': params}, inputs, rngs={'noise': key})
        return output, loss
    
    output, _ = apply_fn(params, x)
    # Ensure the first run completes before timing
    output = jax.block_until_ready(output)
    print(f"Output shape: {output.shape}")
    
    # Time multiple runs
    num_runs = 10
    start_time = time.time()
    for i in range(num_runs):
        output, _ = apply_fn(params, x)
        # Block on the last run to ensure all computation is complete
        if i == num_runs - 1:
            output = jax.block_until_ready(output)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"Average time per run: {avg_time*1000:.2f} ms")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
  main()

