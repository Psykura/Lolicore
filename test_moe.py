import jax
import jax.numpy as jnp
from flax import linen as nn
from transformer import ExpertsFeedForward
import numpy as np
import time

def test_experts_feed_forward():
    # Initialize random key
    key = jax.random.PRNGKey(0)
    
    # Model parameters
    batch_size = 1
    seq_len = 1
    d_model = 16
    hidden_size = 32
    num_experts = 16  # Increased number of experts
    num_shared_experts = 1
    
    # Create model instance
    model = ExpertsFeedForward(
        d_model=d_model,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        training=False
    )
    
    # Create dummy input
    x = jax.random.normal(key, (batch_size, seq_len, d_model))
    
    # Initialize parameters
    variables = model.init(key, x)
    
    # Test both routing methods
    timing_stats = {}
    
    for routing_method in ['expert_choose_tokens', 'token_choose_experts']:
        print(f"\nTesting {routing_method}...")
        timing_stats[routing_method] = []
        
        # Define apply function
        def apply_fn(x):
            output, router_loss = model.apply(
                variables, 
                x, 
                use_token_choose_experts=(routing_method == 'token_choose_experts')
            )
            return output, router_loss
        
        # JIT compile the forward pass
        jitted_apply = jax.jit(apply_fn)
        
        # Warmup run to compile
        warmup_out = jitted_apply(x)
        jax.block_until_ready(warmup_out)
        
        # Test with different inputs
        print("\nTesting with different inputs...")
        for i in range(3):
            key_i = jax.random.PRNGKey(i)
            x_i = jax.random.normal(key_i, (batch_size, seq_len, d_model))
            
            # Time the execution
            start_time = time.perf_counter()
            output_i, router_loss_i = jitted_apply(x_i)
            # Block until computation is complete
            jax.block_until_ready((output_i, router_loss_i))
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            timing_stats[routing_method].append(execution_time)
            
            # Convert to numpy for printing
            output_np = np.array(output_i)
            
            print(f"\nRun {i+1}:")
            print(f"Input shape: {x_i.shape}")
            print(f"Output shape: {output_i.shape}")
            print(f"Output mean: {float(output_np.mean()):.4f}")
            print(f"Output std: {float(output_np.std()):.4f}")
            print(f"Router loss: {float(router_loss_i)}")
            print(f"Execution time: {execution_time:.2f} ms")
            
            # Basic checks
            assert x_i.shape == output_i.shape, f"Shape mismatch: input {x_i.shape} != output {output_i.shape}"
            assert not jnp.allclose(output_i, 0), "Output is all zeros"
            assert not jnp.any(jnp.isnan(output_i)), "Output contains NaN values"
        
        avg_time = np.mean(timing_stats[routing_method])
        std_time = np.std(timing_stats[routing_method])
        print(f"\n{routing_method} tests passed! ✓")
        print(f"Average execution time: {avg_time:.2f} ms (±{std_time:.2f} ms)")
    
    # Compare methods
    print("\nPerformance Comparison:")
    for method in timing_stats:
        avg = np.mean(timing_stats[method])
        std = np.std(timing_stats[method])
        print(f"{method:20s}: {avg:.2f} ms (±{std:.2f} ms)")

if __name__ == "__main__":
    test_experts_feed_forward() 