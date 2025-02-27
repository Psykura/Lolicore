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
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        
        # Create causal mask (lower triangular)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        causal_mask = causal_mask[None, None, :, :]  # Add batch and head dims
        
        # Combine causal mask with attention mask if provided
        if attn_mask is not None:
            # Reshape attention mask to include head dimension
            attn_mask = attn_mask[:, None, :, None] * attn_mask[:, None, None, :]
            combined_mask = causal_mask * attn_mask
        else:
            combined_mask = causal_mask
            
        # Apply mask
        scores = jnp.where(combined_mask == 0, float('-inf'), scores)
        
        attn_weights = nn.softmax(scores, axis=-1)
        attended = jnp.matmul(attn_weights, v)
        
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
    d_model: int
    num_experts: int
    top_k: int = 0
    z_loss_coef: float = 1e-3
    balance_loss_coef: float = 1e-2
    dtype: jnp.dtype = jnp.bfloat16
    training: bool = False
    use_gradient_checkpointing: bool = False

    def setup(self):
        self.gate = nn.Dense(
            features=self.num_experts,
            use_bias=False,
            kernel_init=nn.initializers.lecun_normal(),
            dtype=self.dtype
        )

    def __call__(self, x):
        # x shape: (batch_size, seq_len, d_model)
        
        # Compute router logits
        router_logits = self.gate(x)
        # router_logits shape: (batch_size, seq_len, num_experts)
        
        # Get routing weights and indices
        if self.top_k > 0:
            indices, weights = self._top_k_routing(router_logits)
            # indices shape: (batch_size, seq_len, top_k)
            # weights shape: (batch_size, seq_len, top_k)
        else:
            raise ValueError("top_k must be greater than 0")
        
        if not self.training:
            return indices, weights, 0.0
        
        # Calculate balance loss
        expert_mask = jax.nn.one_hot(indices, self.num_experts)
        expert_usage = expert_mask * weights[..., None]
        expert_usage = expert_usage.sum(axis=(0, 1))
        balance_loss = (expert_usage.std() / expert_usage.mean()) * self.balance_loss_coef

        # Calculate router z-loss
        # sum(exp(logits)) per token, then log, square, and mean
        router_z_loss = jnp.square(
            jnp.log(
                jnp.sum(
                    jnp.exp(router_logits), 
                    axis=-1
                )
            )
        ).mean() * self.z_loss_coef
        
        loss = balance_loss + router_z_loss
        
        return indices, weights, loss

    def _top_k_routing(self, logits):
        """Optimized path for top-k routing"""
        # logits shape: (batch_size, seq_len, num_experts)
        scores, indices = jax.lax.top_k(logits, k=self.top_k)
        # scores shape: (batch_size, seq_len, top_k) - values of top-k experts
        # indices shape: (batch_size, seq_len, top_k) - indices of top-k experts
        routing_weights = jax.nn.softmax(scores, axis=-1)
        # routing_weights shape: (batch_size, seq_len, top_k) - normalized weights summing to 1
        return indices, routing_weights

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
        if self.jump_type == 'zeros':
            return jnp.zeros_like(x)
        elif self.jump_type == 'constant':
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
    num_zeros_experts: int = 0
    num_constant_experts: int = 0
    num_noise_experts: int = 0
    top_k: int = 2
    max_group_size: int = 4096
    train_capacity_factor: float = 1.25
    eval_capacity_factor: float = 2.0
    min_expert_capacity: int = 4
    dtype: jnp.dtype = jnp.bfloat16
    training: bool = False
    use_gradient_checkpointing: bool = False

    def setup(self):
        # Calculate number of feedforward experts
        self.num_ff_experts = (self.num_experts - self.num_zeros_experts 
                             - self.num_constant_experts - self.num_noise_experts)
        assert self.num_ff_experts >= 0, "Total special experts exceeds num_experts"
        
        self.router = Router(
            d_model=self.d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dtype=self.dtype,
            training=self.training,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        
        # Initialize expert types
        self.experts = (
            [FeedForward(hidden_size=self.hidden_size, d_model=self.d_model, dtype=self.dtype, use_gradient_checkpointing=self.use_gradient_checkpointing) for _ in range(self.num_ff_experts)] +
            [JumpModule(d_model=self.d_model, jump_type='zeros', dtype=self.dtype) for _ in range(self.num_zeros_experts)] +
            [JumpModule(d_model=self.d_model, jump_type='constant', dtype=self.dtype) for _ in range(self.num_constant_experts)] +
            [JumpModule(d_model=self.d_model, jump_type='noise', dtype=self.dtype) for _ in range(self.num_noise_experts)]
        )
        
        # Shared experts always processed for all tokens
        self.shared_experts = [
            FeedForward(hidden_size=self.hidden_size, d_model=self.d_model, dtype=self.dtype, use_gradient_checkpointing=self.use_gradient_checkpointing)
            for _ in range(self.num_shared_experts)
        ]

    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        num_tokens = batch_size * seq_len

        # Group tokens for efficient routing
        num_groups = self._compute_num_groups(num_tokens)
        tokens_per_group = num_tokens // num_groups
        
        if tokens_per_group == 0:
            raise ValueError(f"Invalid grouping: num_tokens={num_tokens}, "
                            f"num_groups={num_groups} would result in 0 tokens per group")
        
        # Reshape for grouped processing
        grouped_x = x.reshape(num_groups, tokens_per_group, self.d_model)

        # Calculate expert capacity
        capacity_factor = self.train_capacity_factor if self.training else self.eval_capacity_factor
        expert_capacity = max(
            int(round(capacity_factor * tokens_per_group / self.num_experts)),
            self.min_expert_capacity
        )

        # Get routing decisions
        expert_indices, routing_weights, router_loss = self.router(grouped_x)
        
        # Initialize output and process shared experts
        output = jnp.zeros_like(grouped_x)
        if self.num_shared_experts > 0:
            shared_output = jnp.zeros_like(output)
            for expert in self.shared_experts:
                shared_output += jax.vmap(expert)(grouped_x)
            output += shared_output / len(self.shared_experts)

        # Choose computation path based on training mode
        if True: #self.training:
            output += self._compute_training_path(
                grouped_x, expert_indices, routing_weights, expert_capacity
            )
        else:
            output += self._compute_inference_path(
                grouped_x, expert_indices, routing_weights, expert_capacity
            )

        # Reshape back to original dimensions
        final_output = output.reshape(batch_size, seq_len, self.d_model)
        return final_output, router_loss

    def _compute_num_groups(self, num_tokens: int) -> int:
        """Find optimal number of token groups that divides evenly."""
        min_groups = max(num_tokens // self.max_group_size, 1)
        
        while min_groups > 0:
            if num_tokens % min_groups == 0 and num_tokens // min_groups >= 1:
                return min_groups
            min_groups -= 1
        
        return 1

    def _compute_training_path(self, x, expert_indices, routing_weights, expert_capacity):
        """Training path processes all experts for gradient flow."""
        outputs = []
        num_groups, tokens_per_group, d_model = x.shape
        
        for i, expert in enumerate(self.experts):
            # Create expert mask and sum weights across top-k dimension
            expert_mask = (expert_indices == i) * routing_weights
            expert_mask = expert_mask.sum(axis=-1)
            
            # Process all tokens through expert and apply weights
            expert_output = jax.vmap(expert)(x)
            expert_output = expert_output * expert_mask[..., None]
            outputs.append(expert_output)
        
        return sum(outputs)

    def _compute_inference_path(self, x, expert_indices, routing_weights, expert_capacity):
        """Inference path computes only necessary experts for efficiency."""
        num_groups, tokens_per_group, _ = x.shape
        output = jnp.zeros_like(x)
        
        for group_idx in range(num_groups):
            # Get unique experts needed for this group
            group_experts = jnp.unique(expert_indices[group_idx])
            
            for expert_idx in group_experts:
                # Find tokens that use this expert in any top-k slot
                token_mask = (expert_indices[group_idx] == expert_idx).any(axis=-1)
                if not jnp.any(token_mask):
                    continue
                    
                # Compute outputs only for tokens that use this expert
                selected_tokens = x[group_idx][token_mask]
                expert_matches = expert_indices[group_idx] == expert_idx
                selected_weights = routing_weights[group_idx] * expert_matches
                selected_weights = selected_weights[token_mask].sum(axis=-1)
                
                expert_output = self.experts[expert_idx](selected_tokens)
                output = output.at[group_idx, token_mask].add(
                    expert_output * selected_weights[:, None]
                )
        
        return output

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
    num_zeros_experts: int = 0
    num_constant_experts: int = 0
    num_noise_experts: int = 0
    top_k: int = 0
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    training: bool = False

    def setup(self):        
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            d_model=self.d_model,
            latent_dim=self.attention_latent_dim,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            training=self.training,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        self.feedforward = ExpertsFeedForward(
            d_model=self.d_model,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            num_shared_experts=self.num_shared_experts,
            num_zeros_experts=self.num_zeros_experts,
            num_constant_experts=self.num_constant_experts,
            num_noise_experts=self.num_noise_experts,
            top_k=self.top_k,
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
    num_zeros_experts: int = 1
    num_constant_experts: int = 2
    num_noise_experts: int = 1
    top_k: int = 0
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
                num_zeros_experts=self.num_zeros_experts,
                num_constant_experts=self.num_constant_experts,
                num_noise_experts=self.num_noise_experts,
                top_k=self.top_k,
                dtype=self.dtype,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                training=self.training
            ) for _ in range(self.num_blocks)
        ]
        
        # Final layer norm
        self.norm = nn.LayerNorm(dtype=self.dtype)
        
        # Output projection
        self.output_proj = nn.Dense(features=self.vocab_size, dtype=self.dtype)
        
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

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, prng_key=None):
        """Generate text autoregressively.
        
        Args:
            input_ids: Initial token ids of shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep (if None, keep all)
            top_p: Cumulative probability for nucleus sampling (if None, don't use)
            prng_key: JAX PRNG key for sampling
            
        Returns:
            Generated token ids including the input_ids
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask for initial input
        attn_mask = jnp.ones((batch_size, seq_len))
        
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits, _ = self.__call__(input_ids, attn_mask)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, k=top_k)
                next_token_logits = jnp.zeros_like(next_token_logits).at[
                    jnp.arange(batch_size)[:, None], top_k_indices
                ].set(top_k_logits)
            
            # Apply top-p (nucleus) sampling
            if top_p is not None:
                # Create indices array for sorting
                vocab_indices = jnp.broadcast_to(jnp.arange(next_token_logits.shape[-1]), next_token_logits.shape)
                # Sort both logits and indices
                sorted_logits, sorted_indices = jax.lax.sort_key_val(next_token_logits, vocab_indices)
                cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
                sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)
                sorted_logits = jnp.where(sorted_indices_to_remove, -jnp.inf, sorted_logits)
                next_token_logits = jnp.zeros_like(next_token_logits).at[
                    jnp.arange(batch_size)[:, None], sorted_indices
                ].set(sorted_logits)
            
            # Sample next token
            if prng_key is None:
                # Greedy decoding
                next_token = jnp.argmax(next_token_logits, axis=-1)
            else:
                # Sample from distribution
                prng_key, subkey = jax.random.split(prng_key)
                next_token = jax.random.categorical(subkey, next_token_logits, axis=-1)
            
            # Append new token and update attention mask
            input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)
            attn_mask = jnp.ones((batch_size, input_ids.shape[1]))
            
        return input_ids
