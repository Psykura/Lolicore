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
        attn_weights = nn.softmax(scores, axis=-1)

        def apply_causal_mask(weights):
            row_idx = jnp.arange(seq_len)[None, :]
            col_idx = jnp.arange(seq_len)[:, None]
            return jnp.where(row_idx >= col_idx, weights, 0.0)
        
        attn_weights = jax.vmap(jax.vmap(apply_causal_mask))(attn_weights)
        
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
    confidence_loss_coef: float = 1e-2
    entropy_loss_coef: float = 5e-3
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

    def __call__(self, x, expert_capacity: int, use_mask_routing: bool = False):
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
            confidence_loss = jnp.mean(1.0 - jnp.max(router_probs, axis=-1))
            router_z_loss = jnp.mean(jnp.square(jax.nn.logsumexp(router_logits, axis=-1)))
            
            loss = (
                usage_loss * self.balance_loss_coef +
                confidence_loss * self.confidence_loss_coef +
                entropy_loss * self.entropy_loss_coef +
                router_z_loss * self.z_loss_coef
            )
            
            noise = jax.random.normal(
                self.make_rng('noise'),
                router_probs.shape,
                dtype=router_probs.dtype
            ) * 0.01
            router_probs = nn.softmax(jnp.log(router_probs + 1e-5) + noise)
        else:
            loss = 0.0
        
        flat_probs = router_probs.transpose(2, 0, 1).reshape(self.num_experts, -1)
        scores, token_indices = jax.lax.top_k(flat_probs, k=expert_capacity)
        
        group_indices = token_indices // group_size
        pos_indices = token_indices % group_size
        
        if use_mask_routing:
            expert_masks = jnp.zeros((self.num_experts, num_groups, group_size), dtype=jnp.bool_)
            weight_masks = jnp.zeros((self.num_experts, num_groups, group_size), dtype=self.dtype)
            
            batch_indices = jnp.broadcast_to(
                jnp.arange(self.num_experts)[:, None],
                (self.num_experts, expert_capacity)
            )
            
            scatter_indices = jnp.stack([
                batch_indices.reshape(-1),
                group_indices.reshape(-1),
                pos_indices.reshape(-1)
            ], axis=1)
            
            expert_masks = expert_masks.at[
                scatter_indices[:, 0],
                scatter_indices[:, 1],
                scatter_indices[:, 2]
            ].set(True)
            weight_masks = weight_masks.at[
                scatter_indices[:, 0],
                scatter_indices[:, 1],
                scatter_indices[:, 2]
            ].set(scores.reshape(-1))
            
            return expert_masks, weight_masks, loss
        else:
            return jnp.stack([group_indices, pos_indices], axis=-1), scores, loss

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
    expert_capacity_factor: float = 1.0
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
        num_tokens = batch_size * seq_len
        min_num_groups = max(1, (num_tokens + self.max_group_size - 1) // self.max_group_size)
        
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        if num_tokens % min_num_groups == 0:
            num_groups = min_num_groups
        else:
            divisor = num_tokens // gcd(num_tokens, min_num_groups)
            if divisor >= min_num_groups:
                num_groups = divisor
            else:
                num_groups = ((min_num_groups + divisor - 1) // divisor) * divisor
                if num_groups > 2 * min_num_groups:
                    num_groups = min_num_groups
        
        group_size = num_tokens // num_groups
        if group_size > self.max_group_size:
            group_size = self.max_group_size
            num_groups = (num_tokens + group_size - 1) // group_size
            group_size = num_tokens // num_groups
        
        return group_size, num_groups

    def _process_shared_experts(self, x):
        if not self.shared_experts:
            return jnp.zeros_like(x)
        return jnp.mean(jnp.stack([
            jax.vmap(expert)(x) for expert in self.shared_experts
        ]), axis=0) if len(self.shared_experts) > 1 else jax.vmap(self.shared_experts[0])(x)

    def _process_routed_experts_inference(self, x, expert_indices, routing_weights):
        output = jnp.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            indices = expert_indices[expert_idx]
            weights = routing_weights[expert_idx]
            tokens = x[indices[:, 0], indices[:, 1]]
            processed = jax.vmap(expert)(tokens) * weights[:, None]
            output = output.at[indices[:, 0], indices[:, 1]].add(processed)
        return output
    
    def _process_routed_experts_training(self, x, expert_masks, weight_masks):
        output = jnp.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            mask = expert_masks[expert_idx]
            weights = weight_masks[expert_idx]
            masked_input = x * mask[:, :, None]
            processed = jax.vmap(jax.vmap(expert))(masked_input)
            output += processed * weights[:, :, None]
        return output

    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        group_size, num_groups = self._compute_group_size(batch_size, seq_len)
        
        if batch_size * seq_len == num_groups * group_size:
            x_grouped = x.reshape(num_groups, group_size, self.d_model)
        else:
            padded_size = num_groups * group_size
            padding = padded_size - (batch_size * seq_len)
            x_flat = x.reshape(-1, self.d_model)
            x_padded = jnp.pad(x_flat, ((0, padding), (0, 0)))
            x_grouped = x_padded.reshape(num_groups, group_size, self.d_model)
        
        safe_num_experts = max(1, self.num_experts)
        capacity_from_factor = int(round(self.expert_capacity_factor * group_size / safe_num_experts))
        min_capacity = max(1, self.min_expert_capacity, group_size // 100)
        expert_capacity = max(capacity_from_factor, min_capacity)
        
        if self.training:
            expert_masks, weight_masks, router_loss = self.router(x_grouped, expert_capacity, use_mask_routing=True)
            output = self._process_shared_experts(x_grouped)
            output += self._process_routed_experts_training(x_grouped, expert_masks, weight_masks)
        else:
            expert_indices, routing_weights, router_loss = self.router(x_grouped, expert_capacity)
            output = self._process_shared_experts(x_grouped)
            output += self._process_routed_experts_inference(x_grouped, expert_indices, routing_weights)
        
        return output.reshape(batch_size, seq_len, self.d_model), router_loss

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
    num_layers: int
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
            ) for i in range(self.num_layers)
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