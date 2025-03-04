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
        positions = jnp.arange(self.max_seq_length)
        angles = positions[:, None] * inv_freq[None, :]
        self.cos = jnp.cos(angles)
        self.sin = jnp.sin(angles)

    def get_rotary_cache(self, seq_len):
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

    def __call__(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.latent_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.latent_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.latent_dim)

        q, k = self.rotary.rotate_queries_and_keys(q, k, seq_len)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        scores = jnp.einsum('bnqd,bnkd->bnqk', q, k) / jnp.sqrt(self.latent_dim)
        
        # Apply causal mask before softmax
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
        
        return self.out_proj(attended.reshape(batch_size, seq_len, self.head_dim))

class FeedForward(nn.Module):
    """Simple feed-forward network with GELU activation."""
    hidden_size: int
    d_model: int
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    
    def setup(self):
        dense_impl = nn.remat(nn.Dense) if self.use_gradient_checkpointing else nn.Dense
        self.keys = dense_impl(features=self.hidden_size, dtype=self.dtype)
        self.values = dense_impl(features=self.d_model, dtype=self.dtype)
        self.activation = nn.gelu

    def __call__(self, x):
        return self.values(self.activation(self.keys(x)))

class Router(nn.Module):
    """Router module for Mixture of Experts."""
    d_model: int
    num_experts: int
    z_loss_coef: float = 1e-3
    balance_loss_coef: float = 1e-2
    confidence_loss_coef: float = 1e-4
    entropy_loss_coef: float = 1e-4
    dtype: jnp.dtype = jnp.bfloat16
    training: bool = False
    use_gradient_checkpointing: bool = False

    def setup(self):
        gate_init = nn.initializers.normal(stddev=0.1)
        bias_init = nn.initializers.normal(stddev=0.01)
        self.gate = nn.Dense(
            features=self.num_experts,
            use_bias=True,
            kernel_init=gate_init,
            bias_init=bias_init,
            dtype=self.dtype
        )
        self.temperature = self.param(
            'temperature',
            nn.initializers.constant(1.0),
            (1,),
            self.dtype
        )
    
    def expert_choose_tokens(self, x, expert_capacity: int):
        num_groups, group_size, _ = x.shape
        total_tokens = num_groups * group_size
        
        router_logits = self.gate(x)
        
        safe_temp = jnp.maximum(self.temperature, 0.1)
        router_probs = jax.nn.softmax(router_logits / safe_temp, axis=-1)
        
        if self.training:
            expert_usage = jnp.sum(router_probs, axis=(0, 1)) / (total_tokens + 1e-5)
            
            entropy = -jnp.sum(router_probs * jnp.log(router_probs + 1e-5), axis=-1)
            mean_entropy = jnp.mean(entropy)
            target_entropy = 0.5 * jnp.log(self.num_experts)
            entropy_loss = jnp.abs(mean_entropy - target_entropy)
            
            target_usage = 1.0 / self.num_experts
            usage_loss = jnp.mean(jnp.square(expert_usage - target_usage))
            
            confidence = jnp.max(router_probs, axis=-1)
            confidence_loss = jnp.mean(1.0 - confidence)
            
            router_z_loss = jnp.mean(jnp.square(jax.nn.logsumexp(router_logits, axis=-1)))
            
            loss = (
                usage_loss * self.balance_loss_coef +
                confidence_loss * self.confidence_loss_coef +
                entropy_loss * self.entropy_loss_coef +
                router_z_loss * self.z_loss_coef
            )
        else:
            loss = 0.0
        
        flat_probs = router_probs.transpose(2, 0, 1).reshape(self.num_experts, -1)

        safe_expert_capacity = min(expert_capacity, flat_probs.shape[-1])
        
        scores, token_indices = jax.lax.top_k(flat_probs, k=safe_expert_capacity)
        
        group_indices = token_indices // group_size
        pos_indices = token_indices % group_size

        return jnp.stack([group_indices, pos_indices], axis=-1), scores, loss

    def token_choose_experts(self, x, experts_top_k: int):
        # x: [batch_size * seq_len, d_model]        
        router_logits = self.gate(x)
        
        # Get top-k logits and indices directly from unnormalized logits
        top_k_logits, top_k_indices = jax.lax.top_k(router_logits, k=experts_top_k)
        
        # Apply temperature scaling and softmax only on the top-k logits
        top_k_weights = jax.nn.softmax(top_k_logits / self.temperature, axis=-1)

        return top_k_indices, top_k_weights

class JumpModule(nn.Module):
    """Jump module that can return trainable constant, or random noise"""
    d_model: int
    jump_type: str = 'constant'  # 'constant', or 'noise'
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False

    def setup(self):
        if self.jump_type == 'constant':
            self.jump = self.param('jump', nn.initializers.normal(0.02), (self.d_model,), dtype=self.dtype)

    def __call__(self, x):
        if self.jump_type == 'constant':
            return jnp.broadcast_to(self.jump, x.shape)
        else:
            return jax.random.normal(self.make_rng('noise'), x.shape, dtype=self.dtype) * 0.02

class ExpertsFeedForward(nn.Module):
    """Mixture of Experts layer with efficient routing."""
    d_model: int
    hidden_size: int
    num_experts: int
    num_shared_experts: int
    num_constant_experts: int = 0
    num_noise_experts: int = 0
    expert_capacity_factor: float = 2.0
    min_expert_capacity: int = 8
    max_group_size: int = 4096
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
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        
        self.experts = self._create_experts()
        self.shared_experts = [
            FeedForward(
                hidden_size=self.hidden_size, 
                d_model=self.d_model, 
                dtype=self.dtype, 
                use_gradient_checkpointing=self.use_gradient_checkpointing
            ) for _ in range(self.num_shared_experts)
        ]

    def _create_experts(self):
        experts = []
        for _ in range(self.num_ff_experts):
            experts.append(FeedForward(
                hidden_size=self.hidden_size, 
                d_model=self.d_model, 
                dtype=self.dtype, 
                use_gradient_checkpointing=self.use_gradient_checkpointing
            ))
        
        for _ in range(self.num_constant_experts):
            experts.append(JumpModule(
                d_model=self.d_model, 
                jump_type='constant', 
                dtype=self.dtype
            ))
        
        for _ in range(self.num_noise_experts):
            experts.append(JumpModule(
                d_model=self.d_model, 
                jump_type='noise', 
                dtype=self.dtype
            ))
            
        return experts

    def _compute_group_size(self, batch_size, seq_len):
        """Compute group size and related parameters in a JIT-compatible way."""
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
        if not self.shared_experts:
            return jnp.zeros_like(x)
        return jnp.mean(jnp.stack([
            jax.vmap(expert)(x) for expert in self.shared_experts
        ]), axis=0) if len(self.shared_experts) > 1 else jax.vmap(self.shared_experts[0])(x)
    
    def _expert_choose_tokens(self, x):
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
        
        expert_indices, routing_weights, router_loss = self.router.expert_choose_tokens(x_grouped, expert_capacity)
        output = self._process_shared_experts(x_grouped)

        # routing experts
        experts_output = jnp.zeros_like(x_grouped)
        for expert_idx, expert in enumerate(self.experts):
            indices = expert_indices[expert_idx]
            weights = routing_weights[expert_idx]
            tokens = x_grouped[indices[:, 0], indices[:, 1]]
            processed = jax.vmap(expert)(tokens) * weights[:, None]
            experts_output = experts_output.at[indices[:, 0], indices[:, 1]].add(processed)
        output = output + experts_output
        
        # Remove padding by reshaping and slicing
        output_flat = output.reshape(-1, self.d_model)[:original_size]
        output = output_flat.reshape(batch_size, seq_len, self.d_model)
        
        return output, router_loss
    
    def _compute_token_top_k(self, num_tokens, max_top_k):
        """Compute top_k value using the same capacity logic as expert_choose_tokens"""
        # Use the same group size computation as expert_choose_tokens
        if num_tokens < 12:
            return round(self.expert_capacity_factor)
        
        _, num_groups, expert_capacity = self._compute_group_size(1, num_tokens)
        
        # Calculate total assignments possible across all groups
        total_assignments = self.num_experts * expert_capacity * num_groups
        
        # Calculate ideal top_k based on assignments per token
        # Each token should get approximately the same number of expert assignments
        # as in the expert_choose_tokens method
        ideal_top_k = total_assignments / num_tokens
        
        # Round up and apply practical limits
        top_k = min(
            max(1, int(ideal_top_k + 0.5)),  # Minimum of 2 experts
            min(max_top_k, self.num_experts)  # Maximum of max_top_k or num_experts
        )
        
        return top_k  # Returns a Python int

    def _token_choose_experts(self, x, max_top_k: int = 16):
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.d_model)
        num_tokens = x_flat.shape[0]
        
        # Dynamically determine top_k based on input size
        experts_top_k = self._compute_token_top_k(num_tokens, max_top_k)
        print(f"experts_top_k: {experts_top_k}")
        
        expert_indices, expert_weights = self.router.token_choose_experts(x_flat, experts_top_k=experts_top_k)
        
        def expert_fn(i):
            return lambda mdl, x: mdl.experts[i](x)
        
        # Create branch functions for each expert
        expert_branches = [expert_fn(i) for i in range(len(self.experts))]
        
        # Initialize all experts during setup
        if self.is_mutable_collection('params'):
            for branch in expert_branches:
                _ = branch(self, x_flat[:1])  # Use first token to init all experts
        
        # Process single token through its experts
        def process_token(token, token_experts, token_weights):
            # Get outputs from all top-k experts
            expert_outputs = []
            for k in range(experts_top_k):
                expert_out = nn.switch(token_experts[k], expert_branches, self, token[None])
                expert_outputs.append(expert_out * token_weights[k])
            return jnp.squeeze(sum(expert_outputs), axis=0)  # Remove the added batch dim
        
        # Vectorize the token processing
        vectorized_process = jax.vmap(process_token)
        
        # Process all tokens at once
        outputs = vectorized_process(
            x_flat,
            expert_indices,
            expert_weights
        )
        
        # Add shared experts output
        shared_output = self._process_shared_experts(x_flat)
        combined_output = outputs + shared_output
        
        # Reshape back to original dimensions
        output = combined_output.reshape(batch_size, seq_len, self.d_model)
        
        return output, 0.0

    def __call__(self, x, use_token_choose_experts: bool = False):
        if use_token_choose_experts and not self.training:
            return self._token_choose_experts(x)
        else:
            return self._expert_choose_tokens(x)

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
    dtype: jnp.dtype = jnp.bfloat16
    use_gradient_checkpointing: bool = False
    training: bool = False

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
            dtype=self.dtype,
            training=self.training,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        self.attention_norm = nn.RMSNorm(dtype=self.dtype)
        self.feedforward_norm = nn.RMSNorm(dtype=self.dtype)

    def __call__(self, x, attn_mask=None):
        x = x + self.attention(self.attention_norm(x), attn_mask)
        residual, router_loss = self.feedforward(self.feedforward_norm(x))
        return x + residual, router_loss

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
                dtype=self.dtype,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                training=self.training,
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

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, total_router_loss