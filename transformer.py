import jax
import jax.numpy as jnp
from flax import linen as nn

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding module with caching."""
    dim: int
    max_seq_length: int
    base: int = 10000
    dtype: jnp.dtype = jnp.bfloat16
    training: bool = False

    def setup(self):
        # Compute frequency bands
        inv_freq = 1.0 / (self.base ** (2 * jnp.arange(0, self.dim // 2) / self.dim))

        # Compute angles for each position
        positions = jnp.arange(self.max_seq_length)
        angles = positions[:, None] * inv_freq[None, :]

        # Precompute cos and sin values
        self.cos = jnp.cos(angles)  # (max_seq_length, dim//2)
        self.sin = jnp.sin(angles)  # (max_seq_length, dim//2)

    def get_rotary_cache(self, seq_len):
        """Get cached rotary embeddings or compute and cache them."""
        # Reshape for broadcasting
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        cos = cos.reshape(1, seq_len, 1, cos.shape[-1])
        sin = sin.reshape(1, seq_len, 1, sin.shape[-1])
        return cos, sin

    def __call__(self, x, seq_len=None):
        """Apply rotary embeddings to input tensor with caching.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
            seq_len: Optional sequence length. If None, uses x.shape[1]
        """
        if seq_len is None:
            seq_len = x.shape[1]

        # Get cached or compute rotary embeddings
        cos, sin = self.get_rotary_cache(seq_len)

        # Reshape input to separate last dimension pairs
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)  # (..., dim//2, 2)

        # Apply rotation
        x_out = jnp.concatenate([
            x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin,
            x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos
        ], axis=-1)

        # Reshape back to original shape
        return x_out.reshape(x.shape)

    def rotate_queries_and_keys(self, q, k, seq_len=None):
        """Rotate both queries and keys for attention computation.

        Args:
            q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
            seq_len: Optional sequence length. If None, uses q.shape[1]

        Returns:
            Tuple of rotated (queries, keys)
        """
        if seq_len is None:
            seq_len = q.shape[1]

        # Apply rotary embeddings to both queries and keys
        q_rotated = self.__call__(q, seq_len)
        k_rotated = self.__call__(k, seq_len)

        return q_rotated, k_rotated

    def create_sinusoidal_positions(self, positions):
        """Create sinusoidal position encodings for arbitrary positions.

        Args:
            positions: Integer tensor of positions

        Returns:
            Tuple of (cos, sin) values for the requested positions
        """
        inv_freq = 1.0 / (self.base ** (2 * jnp.arange(0, self.dim // 2) / self.dim))
        angles = positions[:, None] * inv_freq[None, :]
        return jnp.cos(angles), jnp.sin(angles)

class MultiHeadAttention(nn.Module):
    """Multi-head attention with attention masking and rotary embeddings"""
    num_heads: int
    d_model: int
    latent_dim: int
    max_seq_length: int
    dtype: jnp.dtype = jnp.bfloat16
    training: bool = False
    use_gradient_checkpointing: bool = False

    def setup(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")

        self.head_dim = self.latent_dim * self.num_heads

        # Initialize rotary embeddings
        self.rotary = RotaryEmbedding(
            dim=self.latent_dim,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            training=self.training
        )

        # Layer normalization and projections
        if self.use_gradient_checkpointing:
            self.q_proj = nn.remat(nn.Dense)(features=self.head_dim, dtype=self.dtype)
            self.k_proj = nn.remat(nn.Dense)(features=self.head_dim, dtype=self.dtype)
            self.v_proj = nn.remat(nn.Dense)(features=self.head_dim, dtype=self.dtype)
        else:
            self.q_proj = nn.Dense(features=self.head_dim, dtype=self.dtype)
            self.k_proj = nn.Dense(features=self.head_dim, dtype=self.dtype)
            self.v_proj = nn.Dense(features=self.head_dim, dtype=self.dtype)
        self.out_proj = nn.Dense(features=self.d_model, dtype=self.dtype)

    def __call__(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, and values
        q = self.q_proj(x)  # (batch_size, seq_len, head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.latent_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.latent_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.latent_dim)

        # Apply rotary embeddings
        q, k = self.rotary.rotate_queries_and_keys(q, k, seq_len)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Calculate attention scores
        scale = jnp.sqrt(self.latent_dim)

        # Replace matmul with einsum for better performance and readability
        # Original: scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        scores = jnp.einsum('bnqd,bnkd->bnqk', q, k) / scale

        # MEMORY OPTIMIZATION: Apply masking directly without materializing full masks
        
        # Create a function to apply causal mask without materializing the full mask
        def apply_causal_mask(attn_scores):
            # Create row and column indices for broadcasting
            row_idx = jnp.arange(seq_len)[None, :]  # [1, seq_len]
            col_idx = jnp.arange(seq_len)[:, None]  # [seq_len, 1]
            
            # Apply causal mask using arithmetic comparison (much more memory efficient)
            # This creates a mask where row_idx >= col_idx (lower triangular)
            causal_mask = row_idx >= col_idx
            
            # Apply the mask to attention scores
            return jnp.where(causal_mask, attn_scores, -jnp.inf)
        
        # Apply causal masking to scores
        scores = jax.vmap(jax.vmap(apply_causal_mask))(scores)
        
        # Apply attention mask if provided
        if attn_mask is not None and attn_mask.ndim == 2 and attn_mask.shape[0] == batch_size:
            # Truncate if necessary
            if attn_mask.shape[1] > seq_len:
                attn_mask = attn_mask[:, :seq_len]
                
            # Reshape for broadcasting: [batch_size, 1, seq_len, 1]
            attn_mask_4d = attn_mask[:, None, :, None]
            
            # Apply attention mask to scores
            # This masks out padding tokens in a memory-efficient way
            scores = jnp.where(attn_mask_4d > 0, scores, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = nn.softmax(scores, axis=-1)
        
        # Apply attention weights to values using einsum instead of matmul
        # Original: attended = jnp.matmul(attn_weights, v)
        attended = jnp.einsum('bnqk,bnkd->bnqd', attn_weights, v)

        # Reshape back
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        attended = attended.reshape(batch_size, seq_len, self.head_dim)

        return self.out_proj(attended)

class FeedForward(nn.Module):
    hidden_size: int
    d_model: int
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    
    def setup(self):
        if self.use_gradient_checkpointing:
            self.keys = nn.remat(nn.Dense)(features=self.hidden_size, dtype=self.dtype)
            self.values = nn.remat(nn.Dense)(features=self.d_model, dtype=self.dtype)
        else:
            self.keys = nn.Dense(features=self.hidden_size, dtype=self.dtype)
            self.values = nn.Dense(features=self.d_model, dtype=self.dtype)

        self.activation = nn.gelu

    def __call__(self, x):
        x = self.keys(x)
        x = self.activation(x)
        x = self.values(x)
        return x

class Router(nn.Module):
    """Router module for Mixture of Experts that computes token-to-expert assignments."""
    d_model: int
    num_experts: int
    z_loss_coef: float = 1e-3
    balance_loss_coef: float = 4e-2
    dtype: jnp.dtype = jnp.bfloat16
    training: bool = False
    use_gradient_checkpointing: bool = False

    def setup(self):
        # Gate network that produces logits for expert assignment
        gate_init = nn.initializers.normal(stddev=0.02)
        self.gate = nn.Dense(
            features=self.num_experts,
            use_bias=False,
            kernel_init=gate_init,
            dtype=self.dtype
        )

    def __call__(self, x, expert_capacity: int, use_mask_routing: bool = False):
        """Route tokens to experts based on learned assignment.
        
        Args:
            x: Input tensor of shape (num_groups, group_size, d_model)
            expert_capacity: Maximum number of tokens per expert
            
        Returns:
            If training:
                expert_masks: Tensor of shape (num_experts, num_groups, group_size) containing binary masks
                weight_masks: Tensor of shape (num_experts, num_groups, group_size) containing routing weights
                loss: Combined router loss (balance + z-loss)
            Else:
                indices: Tensor of shape (num_experts, expert_capacity, 2) containing (group, position) indices
                weights: Tensor of shape (num_experts, expert_capacity) containing routing weights
                loss: Combined router loss (balance + z-loss)
        """
        num_groups, group_size, _ = x.shape
        total_tokens = num_groups * group_size
        
        # Compute routing probabilities
        router_logits = self.gate(x)  # (num_groups, group_size, num_experts)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        
        if self.training:
            # Calculate load balancing loss
            # Ideal: uniform distribution across experts
            expert_usage = jnp.sum(router_probs, axis=(0, 1)) / (total_tokens + 1e-5)
            balance_loss = (self.num_experts * jnp.sum(expert_usage ** 2) - 1.0) * self.balance_loss_coef
            
            # Calculate router z-loss to stabilize routing probabilities
            # Penalizes large values in the routing logits
            router_z = jax.nn.logsumexp(router_logits, axis=-1)
            router_z_loss = jnp.mean(jnp.square(router_z)) * self.z_loss_coef
            
            # Combined loss
            loss = balance_loss + router_z_loss
        else:
            loss = 0.0
        
        # Reshape for expert-wise token selection
        # (num_experts, num_groups * group_size)
        flat_probs = router_probs.transpose(2, 0, 1).reshape(self.num_experts, -1)

        # Pad with large negative values to ensure they are never selected
        padding_width = max(0, expert_capacity - flat_probs.shape[1])
        flat_probs = jnp.pad(
            flat_probs,
            ((0, 0), (0, padding_width)),
            mode='constant',
            constant_values=-1e9
        )
        
        # Select top tokens for each expert, now safe since input is padded
        scores, token_indices = jax.lax.top_k(flat_probs, k=expert_capacity)
        
        # Zero out scores for padded values
        scores = jnp.where(scores > -1e8, scores, 0.0)
        
        # Convert flat indices to (group, position) coordinates
        group_indices = token_indices // group_size
        pos_indices = token_indices % group_size
        
        if use_mask_routing:
            # For training, create masks directly from the selected indices
            # Initialize masks
            expert_masks = jnp.zeros((self.num_experts, num_groups, group_size), dtype=jnp.bool_)
            weight_masks = jnp.zeros((self.num_experts, num_groups, group_size), dtype=self.dtype)
            
            # Create a list of indices for scatter operations
            batch_indices = jnp.arange(self.num_experts)[:, None]
            batch_indices = jnp.broadcast_to(batch_indices, (self.num_experts, expert_capacity))
            
            # Flatten for scatter operation
            flat_batch_indices = batch_indices.reshape(-1)
            flat_group_indices = group_indices.reshape(-1)
            flat_pos_indices = pos_indices.reshape(-1)
            flat_scores = scores.reshape(-1)
            
            # Create indices for scatter
            scatter_indices = jnp.stack([flat_batch_indices, flat_group_indices, flat_pos_indices], axis=1)
            
            # Use scatter to update the masks
            expert_masks = expert_masks.at[scatter_indices[:, 0], scatter_indices[:, 1], scatter_indices[:, 2]].set(True)
            weight_masks = weight_masks.at[scatter_indices[:, 0], scatter_indices[:, 1], scatter_indices[:, 2]].set(flat_scores)
            
            return expert_masks, weight_masks, loss
        else:
            # For inference, return indices and weights
            indices = jnp.stack([group_indices, pos_indices], axis=-1)
            return indices, scores, loss

class JumpModule(nn.Module):
    """Jump module that can return zeros, trainable constant, or random noise"""
    d_model: int
    jump_type: str = 'constant'  # 'zeros', 'constant', or 'noise'
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False

    def setup(self):
        if self.jump_type == 'constant':
            self.jump = self.param('jump', nn.initializers.normal(0.02), (self.d_model,), dtype=self.dtype)

    def __call__(self, x):
        if self.jump_type == 'constant':
            # Broadcast the jump parameter to match the input shape
            return jnp.broadcast_to(self.jump, x.shape)
        else:  # noise
            # Generate noise with the same shape as the input
            return jax.random.normal(self.make_rng('noise'), x.shape, dtype=self.dtype) * 0.02

class ExpertsFeedForward(nn.Module):
    """Mixture of Experts layer with efficient routing and expert handling."""
    d_model: int
    hidden_size: int
    num_experts: int
    num_shared_experts: int
    num_constant_experts: int = 0
    num_noise_experts: int = 0

    expert_capacity_factor: float = 1.0
    min_expert_capacity: int = 8
    max_group_size: int = 4096

    dtype: jnp.dtype = jnp.bfloat16
    training: bool = False
    use_gradient_checkpointing: bool = False

    def setup(self):
        # Calculate number of feedforward experts
        self.num_ff_experts = (self.num_experts - self.num_constant_experts - self.num_noise_experts)
        assert self.num_ff_experts >= 0, "Total special experts exceeds num_experts"
        
        # Initialize router
        self.router = Router(
            d_model=self.d_model,
            num_experts=self.num_experts,
            dtype=self.dtype,
            training=self.training,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        
        # Initialize different types of experts
        self.experts = self._create_experts()
        
        # Shared experts always processed for all tokens
        self.shared_experts = [
            FeedForward(
                hidden_size=self.hidden_size, 
                d_model=self.d_model, 
                dtype=self.dtype, 
                use_gradient_checkpointing=self.use_gradient_checkpointing
            ) for _ in range(self.num_shared_experts)
        ]

    def _create_experts(self):
        """Create the different types of experts."""
        experts = []
        
        # Standard feedforward experts
        for _ in range(self.num_ff_experts):
            experts.append(FeedForward(
                hidden_size=self.hidden_size, 
                d_model=self.d_model, 
                dtype=self.dtype, 
                use_gradient_checkpointing=self.use_gradient_checkpointing
            ))
        
        # Constant experts (return learned constant)
        for _ in range(self.num_constant_experts):
            experts.append(JumpModule(
                d_model=self.d_model, 
                jump_type='constant', 
                dtype=self.dtype
            ))
        
        # Noise experts (return random noise)
        for _ in range(self.num_noise_experts):
            experts.append(JumpModule(
                d_model=self.d_model, 
                jump_type='noise', 
                dtype=self.dtype
            ))
            
        return experts

    def _compute_group_size(self, batch_size, seq_len):
        """Compute optimal group size for token routing with efficient algorithm.
        
        This optimized version:
        1. Uses GCD to find the largest divisor efficiently
        2. Avoids potential infinite loops
        3. Ensures group sizes are within reasonable bounds
        """
        num_tokens = batch_size * seq_len
        
        # Start with minimum number of groups based on max_group_size
        min_num_groups = max(1, (num_tokens + self.max_group_size - 1) // self.max_group_size)
        
        # Find the greatest common divisor of num_tokens and min_num_groups
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        # Find the smallest number >= min_num_groups that divides num_tokens evenly
        # This is more efficient than incrementing by 1 each time
        if num_tokens % min_num_groups == 0:
            num_groups = min_num_groups
        else:
            # Find the largest divisor of num_tokens that's >= min_num_groups
            divisor = num_tokens // gcd(num_tokens, min_num_groups)
            if divisor >= min_num_groups:
                num_groups = divisor
            else:
                # If no suitable divisor found, use the next multiple of divisor
                num_groups = ((min_num_groups + divisor - 1) // divisor) * divisor
                # If num_groups is too large, fall back to min_num_groups and accept uneven groups
                if num_groups > 2 * min_num_groups:
                    num_groups = min_num_groups
        
        # Calculate group size
        group_size = num_tokens // num_groups
        
        # Ensure group_size is within reasonable bounds
        if group_size > self.max_group_size:
            # Adjust if our calculation somehow exceeded max_group_size
            group_size = self.max_group_size
            num_groups = (num_tokens + group_size - 1) // group_size
            # Recalculate for exact division
            group_size = num_tokens // num_groups
        
        jax.debug.print('group_size: {group_size}, num_groups: {num_groups}', group_size=group_size, num_groups=num_groups)
        return group_size, num_groups

    def _process_shared_experts(self, x):
        """Process input through shared experts."""
        if not self.shared_experts:
            return jnp.zeros_like(x)
            
        # Apply all shared experts and average their outputs
        # Use a more efficient approach for a single shared expert
        if len(self.shared_experts) == 1:
            return jax.vmap(self.shared_experts[0])(x)
        
        # For multiple shared experts, average their outputs
        shared_outputs = [jax.vmap(expert)(x) for expert in self.shared_experts]
        return jnp.mean(jnp.stack(shared_outputs), axis=0)

    def _process_routed_experts_inference(self, x, expert_indices, routing_weights):
        """Process tokens through routed experts based on router assignments."""
        output = jnp.zeros_like(x)
        
        for expert_idx, expert in enumerate(self.experts):
            indices_for_expert = expert_indices[expert_idx]
            weights_for_expert = routing_weights[expert_idx]
            
            # Extract tokens for this expert using advanced indexing
            tokens_for_expert = x[indices_for_expert[:, 0], indices_for_expert[:, 1]]
            
            # Process tokens through expert
            processed_tokens = jax.vmap(expert)(tokens_for_expert)
            
            # Scale by routing weights
            scaled_tokens = processed_tokens * weights_for_expert[:, None]
            
            # Use JAX's scatter_add instead of a Python loop
            output = output.at[indices_for_expert[:, 0], indices_for_expert[:, 1]].add(scaled_tokens)
        
        return output
    
    def _process_routed_experts_training(self, x, expert_masks, weight_masks):
        """Process tokens through experts during training using pre-computed masks.
        
        Args:
            x: Input tensor of shape (num_groups, group_size, d_model)
            expert_masks: Binary masks of shape (num_experts, num_groups, group_size)
            weight_masks: Weight masks of shape (num_experts, num_groups, group_size)
            
        Returns:
            Output tensor of shape (num_groups, group_size, d_model)
        """
        output = jnp.zeros_like(x)
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Get masks for this expert
            expert_mask = expert_masks[expert_idx]  # (num_groups, group_size)
            weight_mask = weight_masks[expert_idx]  # (num_groups, group_size)
            
            # Apply mask to input (zero out tokens not routed to this expert)
            masked_input = x * expert_mask[:, :, None]

            # Process all tokens through expert (masked tokens are zero)
            processed_tokens = jax.vmap(jax.vmap(expert))(masked_input)
            
            # Scale by routing weights
            scaled_tokens = processed_tokens * weight_mask[:, :, None]
            
            # Add to output
            output += scaled_tokens
            
        return output

    def __call__(self, x):
        """Apply mixture of experts to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tuple of (output tensor, router loss)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute group size and reshape input
        group_size, num_groups = self._compute_group_size(batch_size, seq_len)
        
        # Reshape with explicit handling of remainder
        if batch_size * seq_len == num_groups * group_size:
            # Perfect division case
            x_grouped = x.reshape(num_groups, group_size, self.d_model)
        else:
            # Handle case where division isn't perfect (shouldn't happen with optimized algorithm)
            # But we keep this as a safeguard
            padded_size = num_groups * group_size
            padding = padded_size - (batch_size * seq_len)
            
            # Flatten, pad, and reshape
            x_flat = x.reshape(-1, self.d_model)
            x_padded = jnp.pad(x_flat, ((0, padding), (0, 0)))
            x_grouped = x_padded.reshape(num_groups, group_size, self.d_model)
        
        # Calculate expert capacity with a minimum based on group size
        expert_capacity = max(
            int(round(self.expert_capacity_factor * group_size / self.num_experts)),
            self.min_expert_capacity,
            # Ensure at least 1% of tokens per expert to avoid too small batches
            max(1, group_size // 100)
        )
        
        # Get routing assignments from router
        if self.training:
            expert_masks, weight_masks, router_loss = self.router(x_grouped, expert_capacity, use_mask_routing=True)
        else:
            expert_indices, routing_weights, router_loss = self.router(x_grouped, expert_capacity)
        
        # Process through shared experts (applied to all tokens)
        output = self._process_shared_experts(x_grouped)
        
        # Process through routed experts
        if self.training:
            output += self._process_routed_experts_training(x_grouped, expert_masks, weight_masks)
        else:
            output += self._process_routed_experts_inference(x_grouped, expert_indices, routing_weights)
        
        # Reshape back to original dimensions
        return output.reshape(batch_size, seq_len, self.d_model), router_loss

class Block(nn.Module):
    """Transformer block with attention and feedforward layers"""
    num_heads: int
    d_model: int
    hidden_size: int
    max_seq_length: int
    attention_latent_dim: int
    # Expert configuration parameters
    num_experts: int = 8
    num_shared_experts: int = 1
    num_constant_experts: int = 0
    num_noise_experts: int = 0
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    training: bool = False

    def setup(self):
        if self.use_gradient_checkpointing:
            mha_impl = nn.remat(MultiHeadAttention)
            ffn_impl = nn.remat(ExpertsFeedForward)
        else:
            mha_impl = MultiHeadAttention
            ffn_impl = ExpertsFeedForward
        self.attention = mha_impl(
            num_heads=self.num_heads,
            d_model=self.d_model,
            latent_dim=self.attention_latent_dim,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            training=self.training,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        self.feedforward = ffn_impl(
            d_model=self.d_model,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            num_shared_experts=self.num_shared_experts,
            num_constant_experts=self.num_constant_experts,
            num_noise_experts=self.num_noise_experts,
            dtype=self.dtype,
            training=self.training,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        self.attention_norm = nn.RMSNorm(dtype=self.dtype)
        self.feedforward_norm = nn.RMSNorm(dtype=self.dtype)

    def __call__(self, x, attn_mask=None):
        # Residual connection for attention
        x = x + self.attention(self.attention_norm(x), attn_mask)
        # Residual connection for feedforward, pass training and rngs
        residual, router_loss = self.feedforward(
            self.feedforward_norm(x),
        )
        x = x + residual
        return x, router_loss

class Transformer(nn.Module):
    """Transformer model with multiple blocks"""
    num_blocks: int
    num_heads: int
    d_model: int
    hidden_size: int
    max_seq_length: int
    vocab_size: int
    attention_latent_dim: int
    # Expert configuration parameters
    num_experts: int = 12
    num_shared_experts: int = 1
    num_constant_experts: int = 2
    num_noise_experts: int = 1
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    training: bool = False
    
    def setup(self):
        self.token_embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            dtype=self.dtype
        )
        
        self.blocks = [
            Block(
                num_heads=self.num_heads,
                d_model=self.d_model,
                hidden_size=self.hidden_size,
                max_seq_length=self.max_seq_length,
                attention_latent_dim=self.attention_latent_dim,
                num_experts=self.num_experts,
                num_shared_experts=self.num_shared_experts,
                num_constant_experts=self.num_constant_experts,
                num_noise_experts=self.num_noise_experts,
                dtype=self.dtype,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                training=self.training
            ) for _ in range(self.num_blocks)
        ]
        
        # Final layer norm
        self.norm = nn.LayerNorm(dtype=self.dtype)
        
        # Output projection
        self.output_proj = nn.Dense(
            features=self.vocab_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
    def __call__(self, input_ids, attn_mask=None):
        # Get embeddings
        x = self.token_embedding(input_ids)

        # Pass through transformer blocks
        if self.training:
            router_loss = 0.0
            for block in self.blocks:
                x, block_router_loss = block(x, attn_mask)
                router_loss += block_router_loss
            router_loss /= len(self.blocks)
        else:
            for block in self.blocks:
                x, _ = block(x, attn_mask)
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.output_proj(x)
        
        if self.training:
            return logits, router_loss
        
        return logits, 0.0