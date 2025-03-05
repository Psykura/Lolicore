import jax
import jax.numpy as jnp
from flax import linen as nn

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
        self.q_proj = dense_impl(features=self.head_dim, dtype=self.dtype)
        self.k_proj = dense_impl(features=self.head_dim, dtype=self.dtype)
        self.v_proj = dense_impl(features=self.head_dim, dtype=self.dtype)
        self.out_proj = nn.Dense(features=self.d_model, dtype=self.dtype)

    def _compute_qkv(self, x):
        batch_size, seq_len, _ = x.shape

        # Project inputs to q, k, v
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.latent_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.latent_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.latent_dim)

        # apply rotary embeddings
        q, k = self.rotary.rotate_queries_and_keys(q, k, seq_len)

        # Rearrange for attention computation
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, heads, seq, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        return q, k, v

    def __call__(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Compute query, key, value
        q, k, v = self._compute_qkv(x)
        
        # Compute attention scores
        scores = jnp.einsum('bnqd,bnkd->bnqk', q, k) / jnp.sqrt(self.latent_dim)
        
        # Standard causal mask
        def apply_causal_mask(score_matrix):
            row_idx = jnp.arange(seq_len)[None, :]
            col_idx = jnp.arange(seq_len)[:, None]
            return jnp.where(row_idx <= col_idx, score_matrix, -1e9)
            
        scores = jax.vmap(jax.vmap(apply_causal_mask))(scores)
        
        # Apply softmax after causal masking
        attn_weights = nn.softmax(scores, axis=-1)
        
        # Apply input attention mask after softmax to prevent NaN issues
        if attn_mask is not None and attn_mask.ndim == 2 and attn_mask.shape[0] == batch_size:
            if attn_mask.shape[1] > seq_len:
                attn_mask = attn_mask[:, :seq_len]
            attn_weights = jnp.where(
                attn_mask[:, None, :, None] > 0,
                attn_weights,
                0.0
            )

        attended = jnp.einsum('bnqk,bnkd->bnqd', attn_weights, v)
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        
        output = self.out_proj(attended.reshape(batch_size, seq_len, self.head_dim))
        return output

class FeedForward(nn.Module):
    """Feed-forward network that processes multiple experts in parallel with separate parameters."""
    hidden_size: int
    d_model: int
    num_experts: int = 1  # Number of parallel experts
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    
    def setup(self):
        self.keys = self.param('keys', nn.initializers.normal(0.02), (self.num_experts, self.hidden_size, self.d_model))
        self.values = self.param('values', nn.initializers.normal(0.02), (self.num_experts, self.d_model, self.hidden_size))
        self.activation = nn.gelu

    def __call__(self, x):
        # x shape: [batch, seq, num_experts, d_model]
        
        # First projection: x @ keys
        # [batch, seq, num_experts, d_model] @ [num_experts, hidden_size, d_model] 
        # -> [batch, seq, num_experts, hidden_size]
        hidden = jnp.einsum('bsed,ehd->bseh', x, self.keys)
        
        # Apply activation
        hidden = self.activation(hidden)
        
        # Second projection: hidden @ values
        # [batch, seq, num_experts, hidden_size] @ [num_experts, d_model, hidden_size] 
        # -> [batch, seq, num_experts, d_model]
        output = jnp.einsum('bseh,edh->bsed', hidden, self.values)
        
        return output

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
    top_k: int = 4

    def setup(self):
        self.gate = nn.Dense(
            features=self.num_experts,
            use_bias=True,
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
    
    def __call__(self, x, expert_capacity: int):
        # Compute gating probabilities for each token: shape [G, S, E]
        gating_logits = self.gate(x)        
        gating_probs = jax.nn.softmax(gating_logits)

        # expert_gate: [G, S, top_k] with the top_k probabilities.
        # expert_index: [G, S, top_k] with the corresponding expert indices.
        expert_gate, expert_index = jax.lax.top_k(gating_probs, self.top_k)

        # Resulting shape: [G, S, top_k, E]
        expert_mask = jax.nn.one_hot(expert_index, num_classes=gating_probs.shape[2])

        # Shape: [G, S, E]
        combined_expert_mask = jnp.sum(expert_mask, axis=2)

        if self.training:
            router_z_loss = self.z_loss(gating_logits)
            router_balance_loss = self.load_balance_loss(gating_probs, combined_expert_mask)
            loss = router_balance_loss * self.balance_loss_coef + router_z_loss * self.z_loss_coef
        else:
            loss = 0.0

        # Shape: [G, S, top_k, E]
        position_in_expert = jnp.cumsum(expert_mask, axis=1) * expert_mask

        # Shape: [G, S, top_k, E]
        valid_assignment = jnp.less(position_in_expert, expert_capacity)

        # Shape: [G, S, top_k, E]
        expert_gate_valid = expert_gate[..., None] * valid_assignment.astype(expert_gate.dtype)

        # Shape: [G, S, top_k, E, expert_capacity].
        combine_tensor_per_assignment = (
            expert_gate_valid[..., None] *
            jax.nn.one_hot(position_in_expert, num_classes=expert_capacity)
        )

        # Shape: [G, S, E, expert_capacity]
        combine_tensor = jnp.sum(combine_tensor_per_assignment, axis=2)

        # Often the 0th capacity slot is unused (or reserved), so we slice it off.
        combine_tensor = combine_tensor[..., 1:]

        # Create a boolean mask indicating which positions are valid.
        dispatch_mask = combine_tensor.astype(bool)
        
        return combine_tensor, dispatch_mask, loss

class JumpModule(nn.Module):
    """Jump module that processes multiple experts in parallel."""
    d_model: int
    num_experts: int = 1  # Number of parallel experts
    jump_type: str = 'constant'  # 'constant', or 'noise'
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False

    def setup(self):
        if self.jump_type == 'constant':
            # Each expert gets its own trainable constant
            self.jump = self.param('jump', 
                nn.initializers.normal(0.02), 
                (self.num_experts, self.d_model), 
                dtype=self.dtype
            )

    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        if self.jump_type == 'constant':
            # Broadcast jump to batch and sequence dimensions
            return jnp.broadcast_to(
                self.jump[None, None, :, :],  # [1, 1, num_experts, d_model]
                (batch_size, seq_len, self.num_experts, self.d_model)
            )
        else:
            # Generate different noise for each expert
            return jax.random.normal(
                self.make_rng('noise'), 
                (batch_size, seq_len, self.num_experts, self.d_model), 
                dtype=self.dtype
            ) * 0.02

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
    top_k: int = 4
    dtype: jnp.dtype = jnp.bfloat16
    training: bool = False
    use_gradient_checkpointing: bool = False

    def setup(self):
        self.num_ff_experts = (self.num_experts - self.num_constant_experts - self.num_noise_experts)
        assert self.num_ff_experts >= 0, "Total special experts exceeds num_experts"
        
        self.router = Router(
            d_model=self.d_model,
            num_experts=self.num_experts,
            dtype=self.dtype,
            training=self.training,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            top_k=self.top_k
        )
        
        # Single instance for all feedforward experts
        if self.num_ff_experts > 0:
            self.feedforward_experts = FeedForward(
                hidden_size=self.hidden_size,
                d_model=self.d_model,
                num_experts=self.num_ff_experts,
                dtype=self.dtype,
                use_gradient_checkpointing=self.use_gradient_checkpointing
            )
        
        # Single instance for all constant experts
        if self.num_constant_experts > 0:
            self.constant_experts = JumpModule(
                d_model=self.d_model,
                num_experts=self.num_constant_experts,
                jump_type='constant',
                dtype=self.dtype
            )
        
        # Single instance for all noise experts
        if self.num_noise_experts > 0:
            self.noise_experts = JumpModule(
                d_model=self.d_model,
                num_experts=self.num_noise_experts,
                jump_type='noise',
                dtype=self.dtype
            )
        
        # Single instance for all shared experts
        if self.num_shared_experts > 0:
            self.shared_experts = FeedForward(
                hidden_size=self.hidden_size,
                d_model=self.d_model,
                num_experts=self.num_shared_experts,
                dtype=self.dtype,
                use_gradient_checkpointing=self.use_gradient_checkpointing
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
        
        # Process all shared experts in parallel and take mean
        shared_outputs = self.shared_experts(x[:, :, None, :])  # [batch, seq, num_shared_experts, d_model]
        return jnp.mean(shared_outputs, axis=2)  # [batch, seq, d_model]
    
    def _group_inputs(self, x):
        batch_size, seq_len, _ = x.shape
        group_size, num_groups, expert_capacity = self._compute_group_size(batch_size, seq_len)
        
        # Calculate total size after grouping
        total_size = num_groups * group_size
        original_size = batch_size * seq_len
        
        # Reshape and pad input to grouped form
        x_flat = x.reshape(-1, self.d_model)
        padding_needed = total_size - original_size
        x_padded = jnp.pad(
            x_flat,
            pad_width=((0, padding_needed), (0, 0)),
            mode='constant',
            constant_values=0
        )
        x_grouped = x_padded.reshape(num_groups, group_size, self.d_model)
        return x_grouped, expert_capacity
    
    def _degroup_outputs(self, x, batch_size, seq_len):
        output_flat = x.reshape(-1, self.d_model)[:batch_size * seq_len]
        output = output_flat.reshape(batch_size, seq_len, self.d_model)
        return output
    
    def _train_routing(self, x):
        batch_size, seq_len, _ = x.shape
        x_grouped, expert_capacity = self._group_inputs(x)
        combine_tensor, dispatch_mask, router_loss = self.router(x_grouped, expert_capacity)
        
        # Initialize output with shared experts processing
        output = self._process_shared_experts(x_grouped)
        
        # Process feedforward experts
        if self.num_ff_experts > 0:
            # Get the routing for feedforward experts
            ff_dispatch = dispatch_mask[:, :, :self.num_ff_experts, :]
            ff_combine = combine_tensor[:, :, :self.num_ff_experts, :]
            
            # Route tokens to experts via einsum
            # [G, S, E, C] and [G, S, d] => [G, S, E, d]
            expert_inputs = jnp.einsum('GSEC,GSd->GSEd', ff_dispatch, x_grouped)
            # Process all feedforward experts in parallel
            expert_outputs = self.feedforward_experts(expert_inputs)
            
            # Combine expert outputs back: [G, S, E, d] and [G, S, E, C] => [G, S, d]
            ff_output = jnp.einsum('GSEd,GSEC->GSd', expert_outputs, ff_combine)
            output = output + ff_output
        
        # Process constant experts
        if self.num_constant_experts > 0:
            start_idx = self.num_ff_experts
            end_idx = start_idx + self.num_constant_experts
            const_combine = combine_tensor[:, :, start_idx:end_idx, :]
            
            # Get all constant expert outputs in parallel
            const_outputs = self.constant_experts(x_grouped)
            
            # Combine constant expert outputs
            const_output = jnp.einsum('GSEd,GSEC->GSd', const_outputs, const_combine)
            output = output + const_output
        
        # Process noise experts
        if self.num_noise_experts > 0:
            start_idx = self.num_ff_experts + self.num_constant_experts
            end_idx = start_idx + self.num_noise_experts
            noise_combine = combine_tensor[:, :, start_idx:end_idx, :]
            
            # Get all noise expert outputs in parallel
            noise_outputs = self.noise_experts(x_grouped)
            
            # Combine noise expert outputs
            noise_output = jnp.einsum('GSEd,GSEC->GSd', noise_outputs, noise_combine)
            output = output + noise_output
        
        # Degroup the output back to original shape
        output = self._degroup_outputs(output, batch_size, seq_len)
        
        return output, router_loss

    def __call__(self, x):
        return self._train_routing(x)

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
    top_k: int = 4
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    training: bool = False
    layer_idx: int = 0

    def setup(self):
        impl = nn.remat if self.use_gradient_checkpointing else lambda x: x
        self.attention = impl(MultiHeadAttention)(
            num_heads=self.num_heads,
            d_model=self.d_model,
            latent_dim=self.attention_latent_dim,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            training=self.training,
            use_gradient_checkpointing=self.use_gradient_checkpointing
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
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        self.attention_norm = nn.RMSNorm(dtype=self.dtype)
        self.feedforward_norm = nn.RMSNorm(dtype=self.dtype)

    def __call__(self, x, attn_mask=None):
        x = x + self.attention(self.attention_norm(x), attn_mask)
        residual, router_loss = self.feedforward(self.feedforward_norm(x))
        return (x + residual, router_loss)

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
    top_k: int = 4
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    training: bool = False

    def setup(self):
        self.embedding = nn.Embed(
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
                top_k=self.top_k,
                dtype=self.dtype,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                training=self.training,
                layer_idx=i
            ) for i in range(self.num_blocks)
        ]
        self.final_norm = nn.RMSNorm(dtype=self.dtype)
        self.lm_head = nn.Dense(self.vocab_size, dtype=self.dtype)

    def __call__(self, input_ids, attn_mask=None):
        x = self.embedding(input_ids)
        total_router_loss = 0.0

        for block in self.blocks:
            x, router_loss = block(x, attn_mask)
            total_router_loss += router_loss
        total_router_loss = total_router_loss / self.num_blocks

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, total_router_loss