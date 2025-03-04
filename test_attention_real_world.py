import jax
import jax.numpy as jnp
import flax.linen as nn
from transformer import MultiHeadAttention, RotaryEmbedding, FeedForward, ExpertsFeedForward
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import time
import optax

jax.config.update("jax_debug_nans", True)

# Define a simple sequence classification task
class SequenceClassifier(nn.Module):
    """Simple sequence classifier using MultiHeadAttention and ExpertsFeedForward."""
    num_heads: int
    d_model: int
    latent_dim: int
    max_seq_length: int
    num_classes: int
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # First attention layer
        self.attention1 = MultiHeadAttention(
            num_heads=self.num_heads,
            d_model=self.d_model,
            latent_dim=self.latent_dim,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            training=True
        )
        self.norm1 = nn.LayerNorm()
        self.ff1 = FeedForward(
            d_model=self.d_model,
            hidden_size=4 * self.d_model,
            dtype=self.dtype
        )
        
        # Second attention layer
        self.attention2 = MultiHeadAttention(
            num_heads=self.num_heads,
            d_model=self.d_model,
            latent_dim=self.latent_dim,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            training=True
        )
        self.norm2 = nn.LayerNorm()
        self.ff2 = FeedForward(
            d_model=self.d_model,
            hidden_size=4 * self.d_model,
            dtype=self.dtype
        )
        
        # Classification head
        self.classifier = nn.Dense(features=self.num_classes)
        
    def __call__(self, x):
        # First attention block with residual
        attended1 = self.attention1(x)
        x = x + attended1
        x = self.norm1(x)
        ff1_out = self.ff1(x)
        x = x + ff1_out
        
        # Second attention block with residual
        attended2 = self.attention2(x)
        x = x + attended2
        x = self.norm2(x)
        ff2_out = self.ff2(x)
        x = x + ff2_out
        
        # Global average pooling
        x = jnp.mean(x, axis=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits

# Define a baseline model without attention for comparison
class BaselineClassifier(nn.Module):
    """Baseline sequence classifier using only ExpertsFeedForward."""
    d_model: int
    num_classes: int
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.ff1 = FeedForward(
            d_model=self.d_model,
            hidden_size=4 * self.d_model,
            dtype=self.dtype
        )
        self.norm1 = nn.LayerNorm()
        
        self.ff2 = FeedForward(
            d_model=self.d_model,
            hidden_size=4 * self.d_model,
            dtype=self.dtype
        )
        self.norm2 = nn.LayerNorm()
        
        self.ff3 = FeedForward(
            d_model=self.d_model,
            hidden_size=4 * self.d_model,
            dtype=self.dtype
        )
        self.norm3 = nn.LayerNorm()
        
        self.classifier = nn.Dense(features=self.num_classes)
        
    def __call__(self, x):
        # First feed-forward block
        ff_out = self.ff1(x)
        x = x + ff_out
        x = self.norm1(x)
        
        # Second feed-forward block
        ff_out = self.ff2(x)
        x = x + ff_out
        x = self.norm2(x)
        
        # Third feed-forward block
        ff_out = self.ff3(x)
        x = x + ff_out
        x = self.norm3(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits

def generate_synthetic_data(num_samples, seq_len, d_model, num_classes, key):
    """Generate synthetic data for sequence classification with meaningful but challenging patterns."""
    key1, key2, key3, key4 = jax.random.split(key, 4)
    
    # Generate base sequences with more noise
    x = jax.random.normal(key1, (num_samples, seq_len, d_model)) * 0.5
    
    # Add class-specific patterns
    y = jax.random.randint(key2, (num_samples,), 0, num_classes)
    
    for i in range(num_samples):
        class_idx = y[i]
        key4 = jax.random.fold_in(key3, i)
        
        # Add some randomness to pattern positions
        pos_noise = jax.random.randint(key4, (), -2, 3)
        
        if class_idx == 0:
            # Pattern 1: Signal at the beginning with some position variance
            start = max(0, pos_noise)
            end = min(seq_len//4 + pos_noise, seq_len)
            if end > start:
                x = x.at[i, start:end, :].set(
                    jax.random.normal(key4, (end - start, d_model)) * 1.5
                )
        elif class_idx == 1:
            # Pattern 2: Signal in the middle with position variance
            mid_start = max(0, seq_len//3 + pos_noise)
            mid_end = min(2 * seq_len//3 + pos_noise, seq_len)
            if mid_end > mid_start:
                x = x.at[i, mid_start:mid_end, :].set(
                    jax.random.normal(key4, (mid_end - mid_start, d_model)) * 1.5
                )
        elif class_idx == 2:
            # Pattern 3: Signal at the end with position variance
            start = max(0, seq_len - seq_len//4 + pos_noise)
            if start < seq_len:
                x = x.at[i, start:, :].set(
                    jax.random.normal(key4, (seq_len - start, d_model)) * 1.5
                )
        elif class_idx == 3:
            # Pattern 4: Alternating signal with random phase
            phase = jax.random.randint(key4, (), 0, 2)
            x = x.at[i, phase::2, :].set(
                jax.random.normal(key4, (len(range(phase, seq_len, 2)), d_model)) * 1.5
            )
        else:
            # Pattern 5: Random bursts of activity
            num_bursts = jax.random.randint(key4, (), 2, 5)
            for _ in range(num_bursts):
                burst_key = jax.random.fold_in(key4, _)
                burst_pos = jax.random.randint(burst_key, (), 0, seq_len-3)
                burst_len = jax.random.randint(burst_key, (), 2, 5)
                end_pos = min(burst_pos + burst_len, seq_len)
                x = x.at[i, burst_pos:end_pos, :].set(
                    jax.random.normal(burst_key, (end_pos - burst_pos, d_model)) * 1.5
                )
    
    return x, y

def generate_causal_data(num_samples, seq_len, d_model, num_classes, key):
    """Generate synthetic data where each position causally depends on previous positions.
    
    The sequences will have different causal patterns:
    1. Exponential decay influence from past
    2. Oscillating patterns with frequency determined by past
    3. Cumulative sum with thresholds
    4. State transitions based on past values
    5. Markov chain-like dependencies
    """
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Initialize sequences with small random noise
    x = jax.random.normal(key1, (num_samples, seq_len, d_model)) * 0.1
    
    # Generate class labels
    y = jax.random.randint(key2, (num_samples,), 0, num_classes)
    
    for i in range(num_samples):
        class_idx = y[i]
        key_i = jax.random.fold_in(key3, i)
        
        if class_idx == 0:
            # Pattern 1: Exponential decay influence
            # Each position is influenced by all previous positions with exponential decay
            for t in range(1, seq_len):
                past_influence = jnp.sum(x[i, :t] * jnp.exp(-0.5 * (t - jnp.arange(t)))[:, None], axis=0)
                x = x.at[i, t].set(past_influence / (t + 1) + jax.random.normal(key_i, (d_model,)) * 0.1)
                
        elif class_idx == 1:
            # Pattern 2: Oscillating frequency determined by past average
            for t in range(1, seq_len):
                if t > 1:
                    freq = jnp.abs(jnp.mean(x[i, t-2:t])) + 0.1
                    x = x.at[i, t].set(jnp.sin(freq * t) + jax.random.normal(key_i, (d_model,)) * 0.1)
                
        elif class_idx == 2:
            # Pattern 3: Cumulative sum with thresholds
            cumsum = jnp.zeros(d_model)
            for t in range(seq_len):
                cumsum += x[i, t]
                if jnp.any(jnp.abs(cumsum) > 1.0):
                    # Reset with influence from past
                    x = x.at[i, t].set(-0.5 * cumsum + jax.random.normal(key_i, (d_model,)) * 0.1)
                    cumsum = jnp.zeros(d_model)
                
        elif class_idx == 3:
            # Pattern 4: State transitions
            # States are determined by the sum of previous k positions
            k = 3  # Look-back window
            for t in range(k, seq_len):
                window_sum = jnp.sum(x[i, t-k:t], axis=0)
                state = jnp.where(window_sum > 0, 1.0, -1.0)
                x = x.at[i, t].set(state + jax.random.normal(key_i, (d_model,)) * 0.1)
                
        else:
            # Pattern 5: Markov chain-like with learned transitions
            transition_matrix = jax.random.normal(key_i, (d_model, d_model)) * 0.1
            for t in range(1, seq_len):
                next_state = jnp.dot(x[i, t-1], transition_matrix)
                x = x.at[i, t].set(jax.nn.tanh(next_state) + jax.random.normal(key_i, (d_model,)) * 0.1)
    
    # Normalize sequences
    x = (x - jnp.mean(x, axis=(0, 1), keepdims=True)) / (jnp.std(x, axis=(0, 1), keepdims=True) + 1e-5)
    
    return x, y

def compute_accuracy(logits, labels):
    """Compute accuracy from logits and labels."""
    predicted_class = jnp.argmax(logits, axis=1)
    return jnp.mean(predicted_class == labels)

def train_step(params, opt_state, x_batch, y_batch, model, optimizer):
    """Single training step."""
    def loss_fn(params):
        logits = model.apply(params, x_batch)  # Removed deterministic flag
        one_hot = jax.nn.one_hot(y_batch, model.num_classes)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        # Add L2 regularization
        l2_loss = 0.01 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return loss + l2_loss, logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    accuracy = compute_accuracy(logits, y_batch)
    
    return params, opt_state, loss, accuracy

def evaluate(params, x, y, model):
    """Evaluate the model."""
    logits = model.apply(params, x)  # Removed deterministic flag
    accuracy = compute_accuracy(logits, y)
    return accuracy

def test_real_world_effectiveness():
    """Test the effectiveness of MultiHeadAttention in a real-world sequence classification task."""
    print("Testing MultiHeadAttention in a real-world sequence classification task...")
    
    # Set random seed
    key = jax.random.PRNGKey(42)
    
    # Parameters
    num_samples = 2000
    seq_len = 32
    d_model = 64
    num_heads = 4
    latent_dim = 16
    max_seq_length = 64
    num_classes = 5
    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-4
    
    # Generate causal synthetic data
    key, data_key = jax.random.split(key)
    x, y = generate_causal_data(num_samples, seq_len, d_model, num_classes, data_key)
    
    # Shuffle before splitting into train and test
    key, shuffle_key = jax.random.split(key)
    perm = jax.random.permutation(shuffle_key, num_samples)
    x = x[perm]
    y = y[perm]
    
    # Split into train and test sets
    train_size = int(0.8 * num_samples)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Add noise to test set
    key, noise_key = jax.random.split(key)
    test_noise = jax.random.normal(noise_key, x_test.shape) * 0.2
    x_test = x_test + test_noise
    
    # Initialize models
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    attention_model = SequenceClassifier(
        num_heads=num_heads,
        d_model=d_model,
        latent_dim=latent_dim,
        max_seq_length=max_seq_length,
        num_classes=num_classes,
        dtype=jnp.float32
    )
    
    baseline_model = BaselineClassifier(
        d_model=d_model,
        num_classes=num_classes,
        dtype=jnp.float32
    )
    
    # Initialize parameters
    attention_params = attention_model.init(subkey1, x_train[:1])
    baseline_params = baseline_model.init(subkey2, x_train[:1])
    
    # Create optimizers
    attention_optimizer = optax.adam(learning_rate)
    baseline_optimizer = optax.adam(learning_rate)
    
    attention_opt_state = attention_optimizer.init(attention_params)
    baseline_opt_state = baseline_optimizer.init(baseline_params)
    
    # Training loop
    attention_train_losses = []
    attention_train_accs = []
    attention_test_accs = []
    
    baseline_train_losses = []
    baseline_train_accs = []
    baseline_test_accs = []
    
    num_batches = train_size // batch_size
    
    for epoch in range(num_epochs):
        # Shuffle training data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, train_size)
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm]
        
        # Train attention model
        attention_epoch_losses = []
        attention_epoch_accs = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            attention_params, attention_opt_state, loss, acc = train_step(
                attention_params, attention_opt_state, x_batch, y_batch, 
                attention_model, attention_optimizer
            )
            
            attention_epoch_losses.append(loss)
            attention_epoch_accs.append(acc)
        
        attention_train_losses.append(np.mean(attention_epoch_losses))
        attention_train_accs.append(np.mean(attention_epoch_accs))
        
        # Evaluate attention model
        attention_test_acc = evaluate(attention_params, x_test, y_test, attention_model)
        attention_test_accs.append(attention_test_acc)
        
        # Train baseline model
        baseline_epoch_losses = []
        baseline_epoch_accs = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            baseline_params, baseline_opt_state, loss, acc = train_step(
                baseline_params, baseline_opt_state, x_batch, y_batch, 
                baseline_model, baseline_optimizer
            )
            
            baseline_epoch_losses.append(loss)
            baseline_epoch_accs.append(acc)
        
        baseline_train_losses.append(np.mean(baseline_epoch_losses))
        baseline_train_accs.append(np.mean(baseline_epoch_accs))
        
        # Evaluate baseline model
        baseline_test_acc = evaluate(baseline_params, x_test, y_test, baseline_model)
        baseline_test_accs.append(baseline_test_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Attention Model - Train Loss: {attention_train_losses[-1]:.4f}, Train Acc: {attention_train_accs[-1]:.4f}, Test Acc: {attention_test_accs[-1]:.4f}")
        print(f"  Baseline Model  - Train Loss: {baseline_train_losses[-1]:.4f}, Train Acc: {baseline_train_accs[-1]:.4f}, Test Acc: {baseline_test_accs[-1]:.4f}")
    
    # Plot training curves
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, attention_train_losses, 'b-', label='Attention')
    plt.plot(epochs, baseline_train_losses, 'r--', label='Baseline')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, attention_train_accs, 'b-', label='Attention')
    plt.plot(epochs, baseline_train_accs, 'r--', label='Baseline')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, attention_test_accs, 'b-', label='Attention')
    plt.plot(epochs, baseline_test_accs, 'r--', label='Baseline')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('attention_vs_baseline.png')
    print(f"Comparison plot saved to 'attention_vs_baseline.png'")
    
    # Final results
    print("\nFinal Results:")
    print(f"  Attention Model - Test Accuracy: {attention_test_accs[-1]:.4f}")
    print(f"  Baseline Model  - Test Accuracy: {baseline_test_accs[-1]:.4f}")
    
    return attention_test_accs[-1], baseline_test_accs[-1]

if __name__ == "__main__":
    test_real_world_effectiveness() 