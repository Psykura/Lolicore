import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import partitioning

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding module."""
    dim: int
    max_seq_length: int
    base: int = 10000
    dtype: jnp.dtype = jnp.bfloat16
    training: bool = False

    def setup(self):
        inv_freq = 1.0 / (self.base ** (2 * jnp.arange(0, self.dim // 2) / self.dim))
        positions = jnp.arange(self.max_seq_length, dtype=self.dtype)
        angles = positions[:, None] * inv_freq[None, :]
        self.cos = jnp.cos(angles)
        self.sin = jnp.sin(angles)

    def get_rotary_cache(self, seq_len):
        # Ensure we don't exceed max_seq_length
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        return (
            cos.reshape(1, seq_len, 1, cos.shape[-1]),
            sin.reshape(1, seq_len, 1, sin.shape[-1])
        )

    def __call__(self, x, seq_len=None):
        seq_len = seq_len if seq_len is not None else x.shape[1]
        cos, sin = self.get_rotary_cache(seq_len)
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)

        x_out = jnp.concatenate([
            x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin,
            x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos
        ], axis=-1)

        return x_out.reshape(x.shape)

    def rotate_queries_and_keys(self, q, k, seq_len=None):
        seq_len = seq_len if seq_len is not None else q.shape[1]
        return self.__call__(q, seq_len), self.__call__(k, seq_len)

    def create_sinusoidal_positions(self, positions):
        inv_freq = 1.0 / (self.base ** (2 * jnp.arange(0, self.dim // 2, dtype=self.dtype) / self.dim))
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
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_normal()

    def setup(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")

        self.head_dim = self.latent_dim * self.num_heads
        self.rotary = RotaryEmbedding(
            dim=self.latent_dim,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            training=self.training
        )

        dense_impl = nn.remat(nn.Dense) if self.use_gradient_checkpointing else nn.Dense
        self.q_proj = dense_impl(
            features=self.head_dim,
            kernel_init=nn.with_partitioning(self.kernel_init, ('model', 'expert')),
            bias_init=nn.with_partitioning(nn.initializers.zeros, ('model',)),
            dtype=self.dtype
        )
        self.k_proj = dense_impl(
            features=self.head_dim,
            kernel_init=nn.with_partitioning(self.kernel_init, ('model', 'expert')),
            bias_init=nn.with_partitioning(nn.initializers.zeros, ('model',)),
            dtype=self.dtype
        )
        self.v_proj = dense_impl(
            features=self.head_dim,
            kernel_init=nn.with_partitioning(self.kernel_init, ('model', 'expert')),
            bias_init=nn.with_partitioning(nn.initializers.zeros, ('model',)),
            dtype=self.dtype
        )
        # Output projection: partition across both model and expert dimensions
        self.out_proj = dense_impl(
            features=self.d_model,
            kernel_init=nn.with_partitioning(self.kernel_init, ('expert', 'model')),
            bias_init=nn.with_partitioning(nn.initializers.zeros, ('model',)),
            dtype=self.dtype
        )

    def _compute_qkv(self, x):
        batch_size, seq_len, _ = x.shape

        # Apply data and model partitioning constraint to the input
        x = partitioning.with_sharding_constraint(
            x, ('data', None, ('model', None))  # (batch, seq, d_model)
        )

        # Project inputs to q, k, v
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.latent_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.latent_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.latent_dim)

        # Apply partitioning constraints after projection and reshaping
        q = partitioning.with_sharding_constraint(
            q, ('data', None, 'expert', 'model')  # (batch, seq, heads, latent_dim)
        )
        k = partitioning.with_sharding_constraint(
            k, ('data', None, 'expert', 'model')
        )
        v = partitioning.with_sharding_constraint(
            v, ('data', None, 'expert', 'model')
        )
        
        # apply rotary embeddings
        q, k = self.rotary.rotate_queries_and_keys(q, k, seq_len)

        # Rearrange for attention computation
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, heads, seq, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Apply partitioning constraints after transpose
        q = partitioning.with_sharding_constraint(
            q, ('data', 'expert', None, 'model')  # (batch, heads, seq, latent_dim)
        )
        k = partitioning.with_sharding_constraint(
            k, ('data', 'expert', None, 'model')
        )
        v = partitioning.with_sharding_constraint(
            v, ('data', 'expert', None, 'model')
        )

        return q, k, v

    def __call__(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Compute query, key, value
        q, k, v = self._compute_qkv(x)
        
        # Compute attention scores
        scores = jnp.einsum('bnqd,bnkd->bnqk', q, k) / jnp.sqrt(self.latent_dim)
        
        # Apply partitioning constraint to attention scores
        scores = partitioning.with_sharding_constraint(
            scores, ('data', 'expert', None, None)  # (batch, heads, seq_q, seq_k)
        )
        
        # Standard causal mask
        def apply_causal_mask(score_matrix):
            row_idx = jnp.arange(seq_len)[None, :]
            col_idx = jnp.arange(seq_len)[:, None]
            return jnp.where(row_idx <= col_idx, score_matrix, -1e9)
            
        scores = jax.vmap(jax.vmap(apply_causal_mask))(scores)
        
        # Apply softmax after causal masking
        attn_weights = nn.softmax(scores, axis=-1)
        
        # Apply partitioning constraint to attention weights
        attn_weights = partitioning.with_sharding_constraint(
            attn_weights, ('data', 'expert', None, None)  # (batch, heads, seq_q, seq_k)
        )
        
        # Apply input attention mask after softmax to prevent NaN issues
        if attn_mask is not None and attn_mask.ndim == 2 and attn_mask.shape[0] == batch_size:
            if attn_mask.shape[1] > seq_len:
                attn_mask = attn_mask[:, :seq_len]
            attn_weights = jnp.where(
                attn_mask[:, None, :, None] > 0,
                attn_weights,
                0.0
            )

        # Apply attention weights to values
        attended = jnp.einsum('bnqk,bnkd->bnqd', attn_weights, v)
        
        # Apply partitioning constraint to attention output
        attended = partitioning.with_sharding_constraint(
            attended, ('data', 'expert', None, 'model')  # (batch, heads, seq, latent_dim)
        )
        
        # Transpose back to [batch, seq, heads, latent_dim]
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        
        # Apply partitioning constraint after transpose
        attended = partitioning.with_sharding_constraint(
            attended, ('data', None, 'expert', 'model')  # (batch, seq, heads, latent_dim)
        )
        
        # Reshape to [batch, seq, head_dim]
        attended = attended.reshape(batch_size, seq_len, self.head_dim)
        
        # Apply partitioning constraint after reshape
        attended = partitioning.with_sharding_constraint(
            attended, ('data', None, ('model', 'expert'))  # (batch, seq, head_dim)
        )
        
        # Project to output space
        output = self.out_proj(attended)
        
        # Final partitioning constraint
        output = partitioning.with_sharding_constraint(
            output, ('data', None, ('model', None))  # (batch, seq, d_model)
        )
        
        return output

class FeedForward(nn.Module):
    """Feed-forward network that processes multiple experts in parallel with separate parameters."""
    hidden_size: int
    d_model: int
    num_experts: int = 1  # Number of parallel experts
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_normal()
    
    def setup(self):
        # keys: [num_experts, hidden_size, d_model]
        # Partition across expert and model dimensions for better parallelism
        self.keys = self.param(
            'keys',
            nn.with_partitioning(self.kernel_init, ('expert', ('model', 'expert'), None)),
            (self.num_experts, self.hidden_size, self.d_model),
            dtype=self.dtype
        )
        
        # values: [num_experts, d_model, hidden_size]
        # Partition across expert and model dimensions for better parallelism
        self.values = self.param(
            'values',
            nn.with_partitioning(self.kernel_init, ('expert', None, ('model', 'expert'))),
            (self.num_experts, self.d_model, self.hidden_size),
            dtype=self.dtype
        )
        self.activation = nn.gelu

    def __call__(self, x):
        # x shape: [batch, seq, num_experts, d_model]
        
        # Apply partitioning constraint to input with model dimension
        x = partitioning.with_sharding_constraint(
            x, ('data', None, 'expert', ('model', None))  # (batch, seq, num_experts, d_model)
        )
        
        # First projection: x @ keys
        # [batch, seq, num_experts, d_model] @ [num_experts, hidden_size, d_model] 
        # -> [batch, seq, num_experts, hidden_size]
        hidden = jnp.einsum('bsed,ehd->bseh', x, self.keys)
        
        # Apply partitioning constraint after first projection
        hidden = partitioning.with_sharding_constraint(
            hidden, ('data', None, 'expert', ('model', 'expert'))  # (batch, seq, num_experts, hidden_size)
        )
        
        # Apply activation
        hidden = self.activation(hidden)
        
        # Second projection: hidden @ values
        # [batch, seq, num_experts, hidden_size] @ [num_experts, d_model, hidden_size] 
        # -> [batch, seq, num_experts, d_model]
        output = jnp.einsum('bseh,edh->bsed', hidden, self.values)
        
        # Apply final partitioning constraint
        output = partitioning.with_sharding_constraint(
            output, ('data', None, 'expert', ('model', None))  # (batch, seq, num_experts, d_model)
        )
        
        return output
        
    def process_by_indices(self, x, expert_indices, expert_weights):
        # Apply partitioning constraint to input with model dimension
        x = partitioning.with_sharding_constraint(
            x, ('data', None, ('model', None))  # (batch, seq, d_model)
        )
        
        # expert_indices shape: [batch, seq, top_k]
        # Result shapes: [batch, seq, top_k, hidden_size, d_model] and [batch, seq, top_k, d_model, hidden_size]
        selected_keys = self.keys[expert_indices]
        selected_values = self.values[expert_indices]
        
        # Apply partitioning constraints to selected parameters
        selected_keys = partitioning.with_sharding_constraint(
            selected_keys, ('data', None, 'expert', ('model', 'expert'), None)  # (batch, seq, top_k, hidden_size, d_model)
        )
        selected_values = partitioning.with_sharding_constraint(
            selected_values, ('data', None, 'expert', None, ('model', 'expert'))  # (batch, seq, top_k, d_model, hidden_size)
        )
        
        # [batch, seq, d_model] @ [batch, seq, top_k, hidden_size, d_model] -> [batch, seq, top_k, hidden_size]
        hidden = jnp.einsum('bsd,bskhd->bskh', x, selected_keys)
        
        # Apply partitioning constraint after first projection
        hidden = partitioning.with_sharding_constraint(
            hidden, ('data', None, 'expert', ('model', 'expert'))  # (batch, seq, top_k, hidden_size)
        )
        
        hidden = self.activation(hidden)
        
        # [batch, seq, top_k, hidden_size] @ [batch, seq, top_k, d_model, hidden_size] * [batch, seq, top_k] 
        # -> [batch, seq, d_model]
        final_output = jnp.einsum('bskh,bskdh,bsk->bsd', hidden, selected_values, expert_weights)
        
        # Apply final partitioning constraint
        final_output = partitioning.with_sharding_constraint(
            final_output, ('data', None, ('model', None))  # (batch, seq, d_model)
        )
        
        return final_output

# ref to astralord/jax_parallel
class Router(nn.Module):
    """Router module for Mixture of Experts."""
    d_model: int
    num_experts: int
    z_loss_coef: float = 1e-3
    balance_loss_coef: float = 1e-4
    dtype: jnp.dtype = jnp.bfloat16
    training: bool = False
    use_gradient_checkpointing: bool = False
    top_k: int = 2
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_normal()

    def setup(self):
        # Initialize gating network
        # Partition input dim (d_model) along model and output dim (num_experts) along expert
        self.gate = nn.Dense(
            features=self.num_experts,
            kernel_init=nn.with_partitioning(self.kernel_init, ('model', 'expert')),
            bias_init=nn.with_partitioning(nn.initializers.zeros, ('expert',)),
            dtype=self.dtype
        )

    def load_balance_loss(self, gating_probs, combined_expert_mask):
        return jnp.mean(
            jnp.mean(gating_probs, axis=1) * jnp.mean(combined_expert_mask, axis=1)
        ) * (jnp.mean(combined_expert_mask, axis=1).shape[-1] ** 2)
    
    def z_loss(self, gating_logits):
        """
        Compute auxiliary z-loss for router logits.
        
        This loss penalizes large logit values to prevent numerical instability
        in the softmax operation.
        
        Args:
            gating_logits: Raw logits from router, shape [G, S, E]
            
        Returns:
            z_loss: Scalar loss value
        """
        # Calculate log-sum-exp of logits
        log_z = jax.nn.logsumexp(gating_logits, axis=-1)
        # Square and mean across all tokens
        z_loss = jnp.mean(log_z ** 2)
        return z_loss
    
    def _eval_routing(self, x):
        # Apply partitioning constraint to input
        x = partitioning.with_sharding_constraint(
            x, ('data', None, ('model', None))  # (batch, seq, d_model)
        )
        
        gating_logits = self.gate(x)
        
        # Apply partitioning constraint to logits
        gating_logits = partitioning.with_sharding_constraint(
            gating_logits, ('data', None, 'expert')  # (batch, seq, num_experts)
        )
        
        gating_probs = jax.nn.softmax(gating_logits)
        
        # Apply partitioning constraint to probabilities
        gating_probs = partitioning.with_sharding_constraint(
            gating_probs, ('data', None, 'expert')  # (batch, seq, num_experts)
        )

        # expert_gate: [G, S, top_k] with the top_k probabilities.
        # expert_index: [G, S, top_k] with the corresponding expert indices.
        expert_gate, expert_index = jax.lax.top_k(gating_probs, self.top_k)
        
        # Apply partitioning constraints to outputs
        expert_gate = partitioning.with_sharding_constraint(
            expert_gate, ('data', None, 'expert')  # (batch, seq, top_k)
        )
        expert_index = partitioning.with_sharding_constraint(
            expert_index, ('data', None, 'expert')  # (batch, seq, top_k)
        )
        
        return expert_gate, expert_index
    
    def __call__(self, x, expert_capacity: int):
        # Apply partitioning constraint to input
        x = partitioning.with_sharding_constraint(
            x, ('data', None, ('model', None))  # (batch, seq, d_model)
        )
        
        # Compute gating probabilities for each token: shape [G, S, E]
        gating_logits = self.gate(x)
        
        # Apply partitioning constraint to logits
        gating_logits = partitioning.with_sharding_constraint(
            gating_logits, ('data', None, 'expert')  # (batch, seq, num_experts)
        )
        
        gating_probs = jax.nn.softmax(gating_logits)
        
        # Apply partitioning constraint to probabilities
        gating_probs = partitioning.with_sharding_constraint(
            gating_probs, ('data', None, 'expert')  # (batch, seq, num_experts)
        )

        # expert_gate: [G, S, top_k] with the top_k probabilities.
        # expert_index: [G, S, top_k] with the corresponding expert indices.
        expert_gate, expert_index = jax.lax.top_k(gating_probs, self.top_k)
        
        # Apply partitioning constraints to top-k outputs
        expert_gate = partitioning.with_sharding_constraint(
            expert_gate, ('data', None, 'expert')  # (batch, seq, top_k)
        )
        expert_index = partitioning.with_sharding_constraint(
            expert_index, ('data', None, 'expert')  # (batch, seq, top_k)
        )

        # Resulting shape: [G, S, top_k, E]
        expert_mask = jax.nn.one_hot(expert_index, num_classes=gating_probs.shape[2])
        
        # Apply partitioning constraint to expert mask
        expert_mask = partitioning.with_sharding_constraint(
            expert_mask, ('data', None, 'expert', 'expert')  # (batch, seq, top_k, num_experts)
        )

        # Shape: [G, S, E]
        combined_expert_mask = jnp.sum(expert_mask, axis=2)
        
        # Apply partitioning constraint to combined mask
        combined_expert_mask = partitioning.with_sharding_constraint(
            combined_expert_mask, ('data', None, 'expert')  # (batch, seq, num_experts)
        )

        if self.training:
            router_z_loss = self.z_loss(gating_logits)
            router_balance_loss = self.load_balance_loss(gating_probs, combined_expert_mask)
            loss = router_balance_loss * self.balance_loss_coef + router_z_loss * self.z_loss_coef
        else:
            loss = 0.0

        # Shape: [G, S, top_k, E]
        position_in_expert = jnp.cumsum(expert_mask, axis=1) * expert_mask
        
        # Apply partitioning constraint to position_in_expert
        position_in_expert = partitioning.with_sharding_constraint(
            position_in_expert, ('data', None, 'expert', 'expert')  # (batch, seq, top_k, num_experts)
        )

        # Shape: [G, S, top_k, E]
        valid_assignment = jnp.less(position_in_expert, expert_capacity)
        
        # Apply partitioning constraint to valid_assignment
        valid_assignment = partitioning.with_sharding_constraint(
            valid_assignment, ('data', None, 'expert', 'expert')  # (batch, seq, top_k, num_experts)
        )

        # Shape: [G, S, top_k, E]
        expert_gate_valid = expert_gate[..., None] * valid_assignment.astype(expert_gate.dtype)
        
        # Apply partitioning constraint to expert_gate_valid
        expert_gate_valid = partitioning.with_sharding_constraint(
            expert_gate_valid, ('data', None, 'expert', 'expert')  # (batch, seq, top_k, num_experts)
        )

        # Shape: [G, S, top_k, E, expert_capacity]
        combine_tensor_per_assignment = (
            expert_gate_valid[..., None] *
            jax.nn.one_hot(position_in_expert, num_classes=expert_capacity)
        )
        
        # Apply partitioning constraint to combine_tensor_per_assignment
        combine_tensor_per_assignment = partitioning.with_sharding_constraint(
            combine_tensor_per_assignment, ('data', None, 'expert', 'expert', None)  # (batch, seq, top_k, num_experts, expert_capacity)
        )

        # Shape: [G, S, E, expert_capacity]
        combine_tensor = jnp.sum(combine_tensor_per_assignment, axis=2)
        
        # Apply partitioning constraint to combine_tensor
        combine_tensor = partitioning.with_sharding_constraint(
            combine_tensor, ('data', None, ('expert', 'model'), None)  # (batch, seq, num_experts, expert_capacity)
        )

        # Often the 0th capacity slot is unused (or reserved), so we slice it off.
        combine_tensor = combine_tensor[..., 1:]
        
        # Apply partitioning constraint after slicing
        combine_tensor = partitioning.with_sharding_constraint(
            combine_tensor, ('data', None, ('expert', 'model'), None)  # (batch, seq, num_experts, expert_capacity-1)
        )

        # Create a boolean mask indicating which positions are valid.
        dispatch_mask = combine_tensor.astype(bool)
        
        # Apply partitioning constraint to dispatch_mask
        dispatch_mask = partitioning.with_sharding_constraint(
            dispatch_mask, ('data', None, ('expert', 'model'), None)  # (batch, seq, num_experts, expert_capacity-1)
        )
        
        return combine_tensor, dispatch_mask, loss

class JumpModule(nn.Module):
    """Jump module that processes multiple experts in parallel."""
    d_model: int
    num_experts: int = 1  # Number of parallel experts
    jump_type: str = 'constant'  # 'constant', or 'noise'
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    kernel_init: nn.initializers.Initializer = nn.initializers.normal(0.02)

    def setup(self):
        if self.jump_type == 'constant':
            # Each expert gets its own trainable constant
            # Partition across expert and model dimensions
            self.jump = self.param(
                'jump', 
                nn.with_partitioning(self.kernel_init, ('expert', None)),
                (self.num_experts, self.d_model), 
                dtype=self.dtype
            )

    def __call__(self, x):
        # Apply partitioning constraint to input
        x = partitioning.with_sharding_constraint(
            x, ('data', None, None)  # (batch, seq, d_model)
        )
        
        batch_size, seq_len, _ = x.shape
        if self.jump_type == 'constant':
            # Broadcast jump to batch and sequence dimensions
            jump_broadcast = jnp.broadcast_to(
                self.jump[None, None, :, :],  # [1, 1, num_experts, d_model]
                (batch_size, seq_len, self.num_experts, self.d_model)
            )
            
            # Apply partitioning constraint to output
            jump_broadcast = partitioning.with_sharding_constraint(
                jump_broadcast, ('data', None, 'expert', None)  # (batch, seq, num_experts, d_model)
            )
            
            return jump_broadcast
        else:
            # Generate different noise for each expert
            noise = jax.random.normal(
                self.make_rng('noise'), 
                (batch_size, seq_len, self.num_experts, self.d_model), 
                dtype=self.dtype
            ) * 0.02
            
            # Apply partitioning constraint to noise output
            noise = partitioning.with_sharding_constraint(
                noise, ('data', None, 'expert', None)  # (batch, seq, num_experts, d_model)
            )
            
            return noise

class ExpertsFeedForward(nn.Module):
    """Mixture of Experts layer with efficient parallel processing."""
    d_model: int
    hidden_size: int
    num_experts: int
    num_shared_experts: int
    num_constant_experts: int = 0
    num_noise_experts: int = 0
    expert_capacity_factor: float = 2.0
    min_expert_capacity: int = 8
    max_group_size: int = 4096
    top_k: int = 2
    dtype: jnp.dtype = jnp.bfloat16
    training: bool = False
    use_gradient_checkpointing: bool = False
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_normal()

    def setup(self):
        self.num_ff_experts = (self.num_experts - self.num_constant_experts - self.num_noise_experts)
        assert self.num_ff_experts >= 0, "Total special experts exceeds num_experts"
        
        self.router = Router(
            d_model=self.d_model,
            num_experts=self.num_experts,
            dtype=self.dtype,
            training=self.training,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            top_k=self.top_k,
            kernel_init=self.kernel_init
        )
        
        # Single instance for all feedforward experts
        if self.num_ff_experts > 0:
            self.feedforward_experts = FeedForward(
                hidden_size=self.hidden_size,
                d_model=self.d_model,
                num_experts=self.num_ff_experts,
                dtype=self.dtype,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                kernel_init=self.kernel_init
            )
        
        # Single instance for all constant experts
        if self.num_constant_experts > 0:
            self.constant_experts = JumpModule(
                d_model=self.d_model,
                num_experts=self.num_constant_experts,
                jump_type='constant',
                dtype=self.dtype,
                kernel_init=self.kernel_init
            )
        
        # Single instance for all noise experts
        if self.num_noise_experts > 0:
            self.noise_experts = JumpModule(
                d_model=self.d_model,
                num_experts=self.num_noise_experts,
                jump_type='noise',
                dtype=self.dtype,
            )
        
        # Single instance for all shared experts
        if self.num_shared_experts > 0:
            self.shared_experts = FeedForward(
                hidden_size=self.hidden_size,
                d_model=self.d_model,
                num_experts=self.num_shared_experts,
                dtype=self.dtype,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                kernel_init=self.kernel_init
            )

    def _compute_group_size(self, batch_size, seq_len):
        # Convert dynamic values to static integers where possible
        num_tokens = int(batch_size * seq_len)
        
        # Calculate target group size using static operations
        sqrt_tokens = int(float(num_tokens) ** 0.5)
        target_group_size = min(
            self.max_group_size,
            max(32, sqrt_tokens)
        )
        
        # Calculate number of groups needed
        num_groups = (num_tokens + target_group_size - 1) // target_group_size
        group_size = target_group_size
        
        # Calculate total tokens and expert capacity
        total_tokens = num_groups * group_size
        tokens_per_expert = total_tokens / max(1, self.num_experts)
        
        # Calculate capacity with factor using static operations
        capacity_from_factor = int(self.expert_capacity_factor * tokens_per_expert)
        
        # Set minimum capacity
        min_capacity = max(
            max(self.min_expert_capacity, group_size),
            int(total_tokens * 0.001)
        )
        
        # Set maximum capacity
        max_capacity = min(
            group_size * 32,
            int(total_tokens * 0.1)
        )
        
        # Final expert capacity
        expert_capacity = min(max_capacity, max(capacity_from_factor, min_capacity))
        
        return group_size, num_groups, expert_capacity

    def _process_shared_experts(self, x):
        """Process shared experts in parallel."""
        if not self.num_shared_experts:
            return jnp.zeros_like(x)
        
        # Apply partitioning constraint to input
        x = partitioning.with_sharding_constraint(
            x, ('data', None, None)  # (batch/groups, seq/group_size, d_model)
        )
        
        # Process all shared experts in parallel and take mean
        # Add expert dimension: [batch, seq, 1, d_model]
        x_expanded = x[:, :, None, :]
        
        # Apply partitioning constraint to expanded input
        x_expanded = partitioning.with_sharding_constraint(
            x_expanded, ('data', None, 'expert', None)  # (batch, seq, 1, d_model)
        )
        
        shared_outputs = self.shared_experts(x_expanded)  # [batch, seq, num_shared_experts, d_model]
        
        # Apply partitioning constraint to shared outputs
        shared_outputs = partitioning.with_sharding_constraint(
            shared_outputs, ('data', None, 'expert', None)  # (batch, seq, num_shared_experts, d_model)
        )
        
        # Take mean across expert dimension
        output = jnp.mean(shared_outputs, axis=2)  # [batch, seq, d_model]
        
        # Apply partitioning constraint to final output
        output = partitioning.with_sharding_constraint(
            output, ('data', None, None)  # (batch, seq, d_model)
        )
        
        return output
    
    def _group_inputs(self, x):
        # Apply partitioning constraint to input
        x = partitioning.with_sharding_constraint(
            x, ('data', None, None)  # (batch, seq, d_model)
        )
        
        batch_size, seq_len, _ = x.shape
        group_size, num_groups, expert_capacity = self._compute_group_size(batch_size, seq_len)
        
        # Calculate total size after grouping
        total_size = num_groups * group_size
        original_size = batch_size * seq_len
        
        # Reshape and pad input to grouped form
        x_flat = x.reshape(-1, self.d_model)
        
        # Apply partitioning constraint to flattened input
        x_flat = partitioning.with_sharding_constraint(
            x_flat, ('data', None)  # (batch*seq, d_model)
        )
        
        padding_needed = total_size - original_size
        x_padded = jnp.pad(
            x_flat,
            pad_width=((0, padding_needed), (0, 0)),
            mode='constant',
            constant_values=0
        )
        
        # Apply partitioning constraint to padded input
        x_padded = partitioning.with_sharding_constraint(
            x_padded, ('data', None)  # (total_size, d_model)
        )
        
        x_grouped = x_padded.reshape(num_groups, group_size, self.d_model)
        
        # Apply partitioning constraint to grouped input
        x_grouped = partitioning.with_sharding_constraint(
            x_grouped, ('data', None, None)  # (num_groups, group_size, d_model)
        )
        
        return x_grouped, expert_capacity
    
    def _degroup_outputs(self, x, batch_size, seq_len):
        # Apply partitioning constraint to grouped output
        x = partitioning.with_sharding_constraint(
            x, ('data', None, None)  # (num_groups, group_size, d_model)
        )
        
        output_flat = x.reshape(-1, self.d_model)
        
        # Apply partitioning constraint to flattened output
        output_flat = partitioning.with_sharding_constraint(
            output_flat, ('data', None)  # (total_size, d_model)
        )
        
        # Slice to original size
        output_flat = output_flat[:batch_size * seq_len]
        
        # Apply partitioning constraint after slicing
        output_flat = partitioning.with_sharding_constraint(
            output_flat, ('data', None)  # (batch*seq, d_model)
        )
        
        output = output_flat.reshape(batch_size, seq_len, self.d_model)
        
        # Apply partitioning constraint to final output
        output = partitioning.with_sharding_constraint(
            output, ('data', None, None)  # (batch, seq, d_model)
        )
        
        return output
    
    def _train_routing(self, x):
        # Apply partitioning constraint to input
        x = partitioning.with_sharding_constraint(
            x, ('data', None, None)  # (batch, seq, d_model)
        )
        
        batch_size, seq_len, _ = x.shape
        x_grouped, expert_capacity = self._group_inputs(x)
        combine_tensor, dispatch_mask, router_loss = self.router(x_grouped, expert_capacity)
        
        # Apply partitioning constraints to router outputs
        combine_tensor = partitioning.with_sharding_constraint(
            combine_tensor, ('data', None, 'expert', None)  # (groups, group_size, num_experts, capacity)
        )
        dispatch_mask = partitioning.with_sharding_constraint(
            dispatch_mask, ('data', None, 'expert', None)  # (groups, group_size, num_experts, capacity)
        )
        
        # Initialize output with shared experts processing
        output = self._process_shared_experts(x_grouped)
        
        # Process feedforward experts
        if self.num_ff_experts > 0:
            # Get the routing for feedforward experts
            ff_dispatch = dispatch_mask[:, :, :self.num_ff_experts, :]
            ff_combine = combine_tensor[:, :, :self.num_ff_experts, :]
            
            # Apply partitioning constraints to feedforward routing
            ff_dispatch = partitioning.with_sharding_constraint(
                ff_dispatch, ('data', None, 'expert', None)  # (groups, group_size, num_ff_experts, capacity)
            )
            ff_combine = partitioning.with_sharding_constraint(
                ff_combine, ('data', None, 'expert', None)  # (groups, group_size, num_ff_experts, capacity)
            )
            
            # Route tokens to experts via einsum
            # [G, S, E, C] and [G, S, d] => [G, S, E, d]
            expert_inputs = jnp.einsum('GSEC,GSd->GSEd', ff_dispatch, x_grouped)
            
            # Apply partitioning constraint to expert inputs
            expert_inputs = partitioning.with_sharding_constraint(
                expert_inputs, ('data', None, 'expert', None)  # (groups, group_size, num_ff_experts, d_model)
            )
            
            # Process all feedforward experts in parallel
            expert_outputs = self.feedforward_experts(expert_inputs)
            
            # Apply partitioning constraint to expert outputs
            expert_outputs = partitioning.with_sharding_constraint(
                expert_outputs, ('data', None, 'expert', None)  # (groups, group_size, num_ff_experts, d_model)
            )
            
            # Combine expert outputs back: [G, S, E, d] and [G, S, E, C] => [G, S, d]
            ff_output = jnp.einsum('GSEd,GSEC->GSd', expert_outputs, ff_combine)
            
            # Apply partitioning constraint to combined output
            ff_output = partitioning.with_sharding_constraint(
                ff_output, ('data', None, None)  # (groups, group_size, d_model)
            )
            
            output = output + ff_output
        
        # Process constant experts
        if self.num_constant_experts > 0:
            start_idx = self.num_ff_experts
            end_idx = start_idx + self.num_constant_experts
            const_combine = combine_tensor[:, :, start_idx:end_idx, :]
            
            # Apply partitioning constraint to constant routing
            const_combine = partitioning.with_sharding_constraint(
                const_combine, ('data', None, 'expert', None)  # (groups, group_size, num_const_experts, capacity)
            )
            
            # Get all constant expert outputs in parallel
            const_outputs = self.constant_experts(x_grouped)
            
            # Apply partitioning constraint to constant outputs
            const_outputs = partitioning.with_sharding_constraint(
                const_outputs, ('data', None, 'expert', None)  # (groups, group_size, num_const_experts, d_model)
            )
            
            # Combine constant expert outputs
            const_output = jnp.einsum('GSEd,GSEC->GSd', const_outputs, const_combine)
            
            # Apply partitioning constraint to combined constant output
            const_output = partitioning.with_sharding_constraint(
                const_output, ('data', None, None)  # (groups, group_size, d_model)
            )
            
            output = output + const_output
        
        # Process noise experts
        if self.num_noise_experts > 0:
            start_idx = self.num_ff_experts + self.num_constant_experts
            end_idx = start_idx + self.num_noise_experts
            noise_combine = combine_tensor[:, :, start_idx:end_idx, :]
            
            # Apply partitioning constraint to noise routing
            noise_combine = partitioning.with_sharding_constraint(
                noise_combine, ('data', None, 'expert', None)  # (groups, group_size, num_noise_experts, capacity)
            )
            
            # Get all noise expert outputs in parallel
            noise_outputs = self.noise_experts(x_grouped)
            
            # Apply partitioning constraint to noise outputs
            noise_outputs = partitioning.with_sharding_constraint(
                noise_outputs, ('data', None, 'expert', None)  # (groups, group_size, num_noise_experts, d_model)
            )
            
            # Combine noise expert outputs
            noise_output = jnp.einsum('GSEd,GSEC->GSd', noise_outputs, noise_combine)
            
            # Apply partitioning constraint to combined noise output
            noise_output = partitioning.with_sharding_constraint(
                noise_output, ('data', None, None)  # (groups, group_size, d_model)
            )
            
            output = output + noise_output
        
        # Degroup the output back to original shape
        output = self._degroup_outputs(output, batch_size, seq_len)
        
        return output, router_loss
    
    def _eval_routing(self, x):
        # Apply partitioning constraint to input
        x = partitioning.with_sharding_constraint(
            x, ('data', None, None)  # (batch, seq, d_model)
        )
        
        # Get shared expert output (same as in training)
        shared_output = self._process_shared_experts(x)
        
        # Get top_k expert indices and weights from router
        expert_gate, expert_index = self.router._eval_routing(x)
        
        # Apply partitioning constraints to router outputs
        expert_gate = partitioning.with_sharding_constraint(
            expert_gate, ('data', None, 'expert')  # (batch, seq, top_k)
        )
        expert_index = partitioning.with_sharding_constraint(
            expert_index, ('data', None, 'expert')  # (batch, seq, top_k)
        )
        
        router_loss = 0.0  # No loss during evaluation
        
        output = shared_output
        
        # Process feedforward experts directly using indices
        if self.num_ff_experts > 0:
            # Filter indices and weights for feedforward experts only
            ff_mask = expert_index < self.num_ff_experts
            
            # Apply partitioning constraint to mask
            ff_mask = partitioning.with_sharding_constraint(
                ff_mask, ('data', None, 'expert')  # (batch, seq, top_k)
            )
            
            # Normalize weights for feedforward experts - multiply by mask to zero out non-FF experts
            ff_weights = expert_gate * ff_mask
            
            # Apply partitioning constraint to weights
            ff_weights = partitioning.with_sharding_constraint(
                ff_weights, ('data', None, 'expert')  # (batch, seq, top_k)
            )
            
            ff_weights_sum = jnp.sum(ff_weights, axis=-1, keepdims=True)
            # Add small epsilon to avoid division by zero
            ff_weights = ff_weights / (ff_weights_sum + 1e-9)
            
            # Apply partitioning constraint to normalized weights
            ff_weights = partitioning.with_sharding_constraint(
                ff_weights, ('data', None, 'expert')  # (batch, seq, top_k)
            )
            
            # Keep only feedforward expert indices, use zeros for masked positions
            ff_indices = jnp.where(ff_mask, expert_index, 0)
            
            # Apply partitioning constraint to indices
            ff_indices = partitioning.with_sharding_constraint(
                ff_indices, ('data', None, 'expert')  # (batch, seq, top_k)
            )
            
            # Process using direct parameter indexing
            ff_output = self.feedforward_experts.process_by_indices(
                x, ff_indices, ff_weights
            )
            
            # Apply partitioning constraint to output
            ff_output = partitioning.with_sharding_constraint(
                ff_output, ('data', None, None)  # (batch, seq, d_model)
            )
            
            output = output + ff_output
        
        # Process constant experts if needed
        if self.num_constant_experts > 0:
            start_idx = self.num_ff_experts
            end_idx = start_idx + self.num_constant_experts
            
            # Filter indices for constant experts
            const_mask = (expert_index >= start_idx) & (expert_index < end_idx)
            
            # Apply partitioning constraint to mask
            const_mask = partitioning.with_sharding_constraint(
                const_mask, ('data', None, 'expert')  # (batch, seq, top_k)
            )
            
            # Normalize weights for constant experts - multiply by mask to zero out non-constant experts
            const_weights = expert_gate * const_mask
            
            # Apply partitioning constraint to weights
            const_weights = partitioning.with_sharding_constraint(
                const_weights, ('data', None, 'expert')  # (batch, seq, top_k)
            )
            
            const_weights_sum = jnp.sum(const_weights, axis=-1, keepdims=True)
            # Add small epsilon to avoid division by zero
            const_weights = const_weights / (const_weights_sum + 1e-9)
            
            # Apply partitioning constraint to normalized weights
            const_weights = partitioning.with_sharding_constraint(
                const_weights, ('data', None, 'expert')  # (batch, seq, top_k)
            )
            
            # Adjust indices to be relative to the constant experts section
            const_indices = jnp.where(const_mask, expert_index - start_idx, 0)
            
            # Apply partitioning constraint to indices
            const_indices = partitioning.with_sharding_constraint(
                const_indices, ('data', None, 'expert')  # (batch, seq, top_k)
            )
            
            # Get constant expert outputs and combine directly with one einsum
            const_experts = self.constant_experts.jump  # Shape: [num_const_experts, d_model]
            
            # Use einsum to directly select and combine expert outputs
            # We need to generate one-hot encoding of const_indices to select the right experts
            const_one_hot = jax.nn.one_hot(const_indices, self.num_constant_experts)  # [batch, seq, top_k, num_const_experts]
            
            # Apply partitioning constraint to one-hot
            const_one_hot = partitioning.with_sharding_constraint(
                const_one_hot, ('data', None, 'expert', 'expert')  # (batch, seq, top_k, num_const_experts)
            )
            
            # Multiply one-hot by weights and mask, then use einsum to select experts
            # [batch, seq, top_k, num_const_experts] * [batch, seq, top_k] -> [batch, seq, top_k, num_const_experts]
            selection_weights = const_one_hot * const_weights[:, :, :, None] * const_mask[:, :, :, None]
            
            # Apply partitioning constraint to selection weights
            selection_weights = partitioning.with_sharding_constraint(
                selection_weights, ('data', None, 'expert', 'expert')  # (batch, seq, top_k, num_const_experts)
            )
            
            # Use einsum to apply selection weights to experts
            # [batch, seq, top_k, num_const_experts] @ [num_const_experts, d_model] -> [batch, seq, d_model]
            const_output = jnp.einsum('bske,ed->bsd', selection_weights, const_experts)
            
            # Apply partitioning constraint to output
            const_output = partitioning.with_sharding_constraint(
                const_output, ('data', None, None)  # (batch, seq, d_model)
            )
            
            output = output + const_output
        
        # Process noise experts if needed
        if self.num_noise_experts > 0 and self.training:  # Only use noise in training mode
            start_idx = self.num_ff_experts + self.num_constant_experts
            end_idx = start_idx + self.num_noise_experts
            
            # Filter indices for noise experts
            noise_mask = (expert_index >= start_idx) & (expert_index < end_idx)
            
            # Apply partitioning constraint to mask
            noise_mask = partitioning.with_sharding_constraint(
                noise_mask, ('data', None, 'expert')  # (batch, seq, top_k)
            )
            
            # Normalize weights for noise experts - multiply by mask to zero out non-noise experts
            noise_weights = expert_gate * noise_mask
            
            # Apply partitioning constraint to weights
            noise_weights = partitioning.with_sharding_constraint(
                noise_weights, ('data', None, 'expert')  # (batch, seq, top_k)
            )
            
            noise_weights_sum = jnp.sum(noise_weights, axis=-1, keepdims=True)
            # Add small epsilon to avoid division by zero
            noise_weights = noise_weights / (noise_weights_sum + 1e-9)
            
            # Apply partitioning constraint to normalized weights
            noise_weights = partitioning.with_sharding_constraint(
                noise_weights, ('data', None, 'expert')  # (batch, seq, top_k)
            )
            
            # For evaluation, noise experts can be skipped or applied with reduced magnitude
            # We'll apply a small fixed noise value instead of random noise
            noise_scale = 0.001  # Very small scale for eval
            noise_output = jnp.einsum('bsk->bs', noise_weights) * noise_scale
            
            # Apply partitioning constraint to summed weights
            noise_output = partitioning.with_sharding_constraint(
                noise_output, ('data', None)  # (batch, seq)
            )
            
            noise_output = noise_output[:, :, None]  # Add feature dimension
            
            # Apply partitioning constraint to expanded output
            noise_output = partitioning.with_sharding_constraint(
                noise_output, ('data', None, None)  # (batch, seq, 1)
            )
            
            output = output + noise_output
        
        return output, router_loss

    def __call__(self, x):
        # Apply partitioning constraint to input
        x = partitioning.with_sharding_constraint(
            x, ('data', None, None)  # (batch, seq, d_model)
        )
        
        if self.training or x.shape[0] * x.shape[1] > 18:
            return self._train_routing(x)
        else:
            return self._eval_routing(x)

class Block(nn.Module):
    """Transformer block with attention and MoE feed-forward layers."""
    num_heads: int
    d_model: int
    hidden_size: int
    max_seq_length: int
    attention_latent_dim: int
    num_experts: int = 8
    num_shared_experts: int = 1
    num_constant_experts: int = 0
    num_noise_experts: int = 0
    top_k: int = 2
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    training: bool = False
    layer_idx: int = 0
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_normal()

    def setup(self):
        impl = nn.remat if self.use_gradient_checkpointing else lambda x: x
        self.attention = impl(MultiHeadAttention)(
            num_heads=self.num_heads,
            d_model=self.d_model,
            latent_dim=self.attention_latent_dim,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            training=self.training,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            kernel_init=self.kernel_init
        )
        self.feedforward = impl(ExpertsFeedForward)(
            d_model=self.d_model,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            num_shared_experts=self.num_shared_experts,
            num_constant_experts=self.num_constant_experts,
            num_noise_experts=self.num_noise_experts,
            top_k=self.top_k,
            dtype=self.dtype,
            training=self.training,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            kernel_init=self.kernel_init
        )
        
        # Partition RMSNorm parameters along None dimension
        self.attention_norm = nn.RMSNorm(
            dtype=self.dtype,
            scale_init=nn.with_partitioning(nn.initializers.ones, (None,))
        )
        self.feedforward_norm = nn.RMSNorm(
            dtype=self.dtype,
            scale_init=nn.with_partitioning(nn.initializers.ones, (None,))
        )

    def __call__(self, x, attn_mask=None):
        # Apply partitioning constraint to input
        x = partitioning.with_sharding_constraint(
            x, ('data', None, None)  # (batch, seq, d_model)
        )
        
        # Apply attention norm
        attn_norm_out = self.attention_norm(x)
        
        # Apply partitioning constraint to normalized output
        attn_norm_out = partitioning.with_sharding_constraint(
            attn_norm_out, ('data', None, None)  # (batch, seq, d_model)
        )
        
        # Apply attention
        attn_out = self.attention(attn_norm_out, attn_mask)
        
        # Apply partitioning constraint to attention output
        attn_out = partitioning.with_sharding_constraint(
            attn_out, ('data', None, None)  # (batch, seq, d_model)
        )
        
        # Add residual connection
        residual1 = x + attn_out
        
        # Apply partitioning constraint to first residual
        residual1 = partitioning.with_sharding_constraint(
            residual1, ('data', None, None)  # (batch, seq, d_model)
        )
        
        # Apply feedforward norm
        ff_norm_out = self.feedforward_norm(residual1)
        
        # Apply partitioning constraint to normalized output
        ff_norm_out = partitioning.with_sharding_constraint(
            ff_norm_out, ('data', None, None)  # (batch, seq, d_model)
        )
        
        # Apply feedforward
        ff_out, router_loss = self.feedforward(ff_norm_out)
        
        # Apply partitioning constraint to feedforward output
        ff_out = partitioning.with_sharding_constraint(
            ff_out, ('data', None, None)  # (batch, seq, d_model)
        )
        
        # Add second residual connection
        final_output = residual1 + ff_out
        
        # Apply partitioning constraint to final output
        final_output = partitioning.with_sharding_constraint(
            final_output, ('data', None, None)  # (batch, seq, d_model)
        )
        
        return (final_output, router_loss)

class Transformer(nn.Module):
    """Transformer model with MoE layers."""
    num_blocks: int
    num_heads: int
    d_model: int
    hidden_size: int
    max_seq_length: int
    attention_latent_dim: int
    vocab_size: int
    num_experts: int = 8
    num_shared_experts: int = 1
    num_constant_experts: int = 0
    num_noise_experts: int = 0
    top_k: int = 2
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    training: bool = False
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_normal()

    def setup(self):
        # Partition embedding parameters
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.with_partitioning(nn.initializers.normal(0.02), ('data', None)),
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
                top_k=self.top_k,
                dtype=self.dtype,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                training=self.training,
                layer_idx=i,
                kernel_init=self.kernel_init
            ) for i in range(self.num_blocks)
        ]
        
        # Partition final norm parameters
        self.final_norm = nn.RMSNorm(
            dtype=self.dtype,
            scale_init=nn.with_partitioning(nn.initializers.ones, (None,))
        )
        
        # Partition LM head parameters
        self.lm_head = nn.Dense(
            features=self.vocab_size,
            kernel_init=nn.with_partitioning(self.kernel_init, ('model', 'data')),
            bias_init=nn.with_partitioning(nn.initializers.zeros, ('data',)),
            dtype=self.dtype
        )

    def __call__(self, input_ids, attn_mask=None):
        # Apply partitioning constraint to input
        input_ids = partitioning.with_sharding_constraint(
            input_ids, ('data', None)  # (batch, seq)
        )
        
        # Apply embedding
        x = self.embedding(input_ids)
        
        # Apply partitioning constraint to embedded input
        x = partitioning.with_sharding_constraint(
            x, ('data', None, None)  # (batch, seq, d_model)
        )
        
        total_router_loss = 0.0

        # Process through transformer blocks
        for block in self.blocks:
            x, router_loss = block(x, attn_mask)
            total_router_loss += router_loss
            
            # Apply partitioning constraint after each block
            x = partitioning.with_sharding_constraint(
                x, ('data', None, None)  # (batch, seq, d_model)
            )
            
        total_router_loss = total_router_loss / self.num_blocks

        # Apply final normalization
        x = self.final_norm(x)
        
        # Apply partitioning constraint to normalized output
        x = partitioning.with_sharding_constraint(
            x, ('data', None, None)  # (batch, seq, d_model)
        )
        
        # Apply LM head to get logits
        logits = self.lm_head(x)
        
        # Apply partitioning constraint to logits
        logits = partitioning.with_sharding_constraint(
            logits, ('data', None, 'data')  # (batch, seq, vocab_size)
        )
        
        return logits, total_router_loss