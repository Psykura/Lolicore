import jax
import jax.numpy as jnp
import flax.linen as nn
from transformer import MultiHeadAttention, RotaryEmbedding
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import time

def test_attention_pattern():
    """Test if the attention module produces expected attention patterns with and without attention masks."""
    print("Testing attention pattern...")
    
    # Initialize parameters
    batch_size = 2
    seq_len = 16
    d_model = 64
    num_heads = 4
    latent_dim = 16
    max_seq_length = 32
    
    # Create a simple input where tokens attend to specific positions
    x = jnp.ones((batch_size, seq_len, d_model))
    
    # Create attention mask with last 4 positions masked
    attn_mask = jnp.ones((batch_size, seq_len))
    attn_mask = attn_mask.at[:, -4:].set(0)  # Set last 4 positions to 0
    
    # Initialize the attention module
    attention = MultiHeadAttention(
        num_heads=num_heads,
        d_model=d_model,
        latent_dim=latent_dim,
        max_seq_length=max_seq_length,
        dtype=jnp.float32,
        training=False
    )
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = attention.init(key, x)
    
    # Define a function to extract attention weights
    def get_attention_weights(params, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Get the query, key, value projections
        q = attention.apply(params, x, method=lambda self, x: self.q_proj(x))
        k = attention.apply(params, x, method=lambda self, x: self.k_proj(x))
        v = attention.apply(params, x, method=lambda self, x: self.v_proj(x))
        
        # Reshape
        q = q.reshape(batch_size, seq_len, num_heads, latent_dim)
        k = k.reshape(batch_size, seq_len, num_heads, latent_dim)
        
        # Apply rotary embeddings
        rotary = RotaryEmbedding(
            dim=latent_dim,
            max_seq_length=max_seq_length,
            dtype=jnp.float32,
            training=False
        )
        rotary_params = rotary.init(key, q)
        q, k = rotary.apply(rotary_params, q, k, method=lambda self, q, k: self.rotate_queries_and_keys(q, k))
        
        # Transpose for attention calculation
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        
        # Calculate attention scores
        scores = jnp.einsum('bnqd,bnkd->bnqk', q, k) / jnp.sqrt(latent_dim)
        
        # Apply causal mask
        def apply_causal_mask(weights):
            row_idx = jnp.arange(seq_len)[None, :]
            col_idx = jnp.arange(seq_len)[:, None]
            return jnp.where(row_idx <= col_idx, weights, -1e9)
        
        scores = jax.vmap(jax.vmap(apply_causal_mask))(scores)
        
        # Apply softmax after causal masking
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply input attention mask after softmax to prevent NaN issues
        if mask is not None and mask.ndim == 2 and mask.shape[0] == batch_size:
            if mask.shape[1] > seq_len:
                mask = mask[:, :seq_len]
            attn_weights = jnp.where(
                mask[:, None, :, None] > 0,
                attn_weights,
                0.0
            )
        
        return attn_weights
    
    # Get attention weights with and without mask
    attn_weights_no_mask = get_attention_weights(params, x)
    attn_weights_with_mask = get_attention_weights(params, x, attn_mask)
    
    # Plot attention patterns for the first batch and first head
    plt.figure(figsize=(15, 5))
    
    # Plot without mask
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(attn_weights_no_mask[0, 0]), cmap='viridis')
    plt.colorbar()
    plt.title('Attention Pattern (No Mask)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Plot with mask
    plt.subplot(1, 2, 2)
    plt.imshow(np.array(attn_weights_with_mask[0, 0]), cmap='viridis')
    plt.colorbar()
    plt.title('Attention Pattern (With Mask)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig('attention_pattern.png')
    print(f"Attention pattern visualization saved to 'attention_pattern.png'")
    
    # Check if the attention pattern is causal (lower triangular)
    is_causal = jnp.all(jnp.triu(attn_weights_no_mask[0, 0], k=1) < 1e-6)
    print(f"Is attention pattern causal? {is_causal}")
    
    # Check if masked positions have zero attention weights
    is_masked = jnp.allclose(attn_weights_with_mask[0, 0, :, -4:], 0.0)
    print(f"Are masked positions zeroed out? {is_masked}")
    
    return is_causal and is_masked

def test_attention_performance():
    """Test the performance of the attention module with different sequence lengths."""
    print("\nTesting attention performance...")
    
    # Parameters
    batch_size = 1
    d_model = 64
    num_heads = 4
    latent_dim = 16
    max_seq_length = 1024
    
    # Sequence lengths to test
    seq_lengths = [32, 64, 128, 256, 512]
    
    # Initialize the attention module
    attention = MultiHeadAttention(
        num_heads=num_heads,
        d_model=d_model,
        latent_dim=latent_dim,
        max_seq_length=max_seq_length,
        dtype=jnp.float32,
        training=False
    )
    
    # Compile the forward pass
    @partial(jax.jit, static_argnums=(1,))
    def forward_pass(params, seq_len):
        x = jnp.ones((batch_size, seq_len, d_model))
        return attention.apply(params, x)
    
    # Measure performance for different sequence lengths
    times = []
    
    for seq_len in seq_lengths:
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        x = jnp.ones((batch_size, seq_len, d_model))
        params = attention.init(key, x)
        
        # Warm-up
        forward_pass(params, seq_len)
        
        # Measure time
        start_time = time.time()
        n_runs = 10
        for _ in range(n_runs):
            forward_pass(params, seq_len)
        jax.block_until_ready(forward_pass(params, seq_len))
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_runs
        times.append(avg_time)
        print(f"Sequence length {seq_len}: {avg_time:.4f} seconds per forward pass")
    
    # Plot performance
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, times, marker='o')
    plt.title('Attention Forward Pass Time vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.savefig('attention_performance.png')
    print(f"Performance plot saved to 'attention_performance.png'")

def test_attention_quality():
    """Test the quality of attention by checking if it can capture relevant information."""
    print("\nTesting attention quality...")
    
    # Parameters
    batch_size = 1
    seq_len = 32
    d_model = 64
    num_heads = 4
    latent_dim = 16
    max_seq_length = 64
    
    # Create a structured input where specific positions have higher values
    x = jnp.ones((batch_size, seq_len, d_model)) * 0.1
    
    # Make positions 5, 10, and 15 have higher values in specific dimensions
    for pos, dim_start in [(5, 0), (10, 20), (15, 40)]:
        x = x.at[:, pos, dim_start:dim_start+10].set(1.0)
    
    # Initialize the attention module
    attention = MultiHeadAttention(
        num_heads=num_heads,
        d_model=d_model,
        latent_dim=latent_dim,
        max_seq_length=max_seq_length,
        dtype=jnp.float32,
        training=False
    )
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = attention.init(key, x)
    
    # Forward pass
    output = attention.apply(params, x)
    
    # Check if the output at positions after 5, 10, and 15 have higher values
    # This would indicate that the attention mechanism is capturing information from important positions
    avg_values = []
    for pos in range(seq_len):
        avg_values.append(jnp.mean(output[0, pos]))
    
    # Plot the average values at each position
    plt.figure(figsize=(12, 6))
    plt.bar(range(seq_len), avg_values)
    plt.axvline(x=5, color='r', linestyle='--', label='Important Position')
    plt.axvline(x=10, color='r', linestyle='--')
    plt.axvline(x=15, color='r', linestyle='--')
    plt.title('Average Output Values at Each Position')
    plt.xlabel('Position')
    plt.ylabel('Average Value')
    plt.legend()
    plt.savefig('attention_quality.png')
    print(f"Attention quality plot saved to 'attention_quality.png'")
    
    # Check if positions after important positions have higher values
    important_positions = [5, 10, 15]
    for pos in important_positions:
        if pos < seq_len - 1:
            print(f"Position {pos} (important): {avg_values[pos]:.4f}")
            print(f"Position {pos+1} (after important): {avg_values[pos+1]:.4f}")
    
    return output

def main():
    print("Testing MultiHeadAttention module...")
    
    # Test attention pattern
    is_causal = test_attention_pattern()
    
    # Test attention performance
    test_attention_performance()
    
    # Test attention quality
    test_attention_quality()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 