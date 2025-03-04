import jax.numpy as jnp
import flax.linen as nn
import jax
from jax import random
from transformer import ExpertsFeedForward, FeedForward
import time
import optax
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Use full dataset
    train_size = 60000
    test_size = 10000
    
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]
    x_test = x_test[:test_size]
    y_test = y_test[:test_size]
    
    # Normalize pixel values to [0, 1] range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape images to (batch_size, seq_len, features)
    # Treat the entire image as one sequence with 784 features
    x_train = x_train.reshape(-1, 784, 1)
    x_test = x_test.reshape(-1, 784, 1)
    
    # Convert to JAX arrays
    x_train = jnp.array(x_train)
    y_train = jnp.array(y_train)
    x_test = jnp.array(x_test)
    y_test = jnp.array(y_test)
    
    return x_train, y_train, x_test, y_test

def train_and_evaluate_model(
    key,
    x_train, y_train, x_val, y_val,
    num_experts=8,
    num_shared_experts=1,
    num_constant_experts=1,
    num_noise_experts=1,
    hidden_size=512,
    num_epochs=50,
    plot_history=True
):
    """Train and evaluate a model with the given configuration."""
    key, subkey = random.split(key)
    
    # Create model instance with compatible parameters
    model = ExpertsFeedForward(
        d_model=32,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts, 
        num_constant_experts=num_constant_experts, 
        num_noise_experts=num_noise_experts, 
        dtype=jnp.float32,
        use_gradient_checkpointing=True,
        training=True
    )
    
    # Create a classification head to map from d_model to 10 classes
    class ClassificationHead(nn.Module):
        @nn.compact
        def __call__(self, x):
            # Input shape: (batch_size, seq_len, d_model)
            # First transform each position independently
            batch_size, seq_len, _ = x.shape
            x = x.reshape(-1, x.shape[-1])  # Flatten to (batch_size * seq_len, d_model)
            x = nn.Dense(features=32)(x)
            x = nn.gelu(x)
            x = x.reshape(batch_size, seq_len, 32)  # Restore sequence dimension
            
            # Global average pooling over sequence dimension
            x = jnp.mean(x, axis=1)  # Shape: (batch_size, 32)
            
            # Project to number of classes
            x = nn.Dense(features=10)(x)
            return x
    
    classifier = ClassificationHead()
    
    # Initialize model parameters with a small batch
    init_batch = jnp.zeros((2, 784, 1))
    variables = model.init({'noise': key, 'dropout': key, 'params': key}, init_batch)
    params = variables['params']
    
    # Initialize classifier parameters with model output shape
    model_output = model.apply({'params': params}, init_batch, rngs={'noise': key})[0]
    classifier_variables = classifier.init(subkey, model_output)
    classifier_params = classifier_variables['params']
    
    # Combine parameters
    combined_params = {'model': params, 'classifier': classifier_params}
    
    # Define cross entropy loss function
    def loss_fn(params, x, y):
        model_params = params['model']
        classifier_params = params['classifier']
        
        # Apply the MoE model
        output, router_loss = model.apply({'params': model_params}, x, rngs={'noise': key})
        
        # Apply the classifier
        logits = classifier.apply({'params': classifier_params}, output)
        
        # Calculate cross entropy loss
        labels = jax.nn.one_hot(y, num_classes=10)
        ce_loss = optax.softmax_cross_entropy(logits, labels).mean()
        
        # Reduce router loss weight
        total_loss = ce_loss + 0.001 * router_loss
        
        return total_loss, (ce_loss, router_loss, logits)
    
    # Create a gradient function
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    
    # Create an optimizer with a learning rate schedule
    warmup_steps = 1000  # Increased warmup steps
    decay_steps = 10000  # Increased decay steps
    
    def lr_schedule(step):
        warmup_factor = jnp.minimum(step / warmup_steps, 1.0)
        decay_factor = jnp.power(0.1, step / decay_steps)
        return 2e-3 * warmup_factor * decay_factor  # Increased initial learning rate
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
        optax.adam(learning_rate=lr_schedule)
    )
    
    opt_state = optimizer.init(combined_params)
    
    # JIT the training step
    @jax.jit
    def train_step(params, opt_state, x, y, step):
        (loss, aux), grads = grad_fn(params, x, y)
        ce_loss, router_loss, logits = aux
        updates, opt_state = optimizer.update(grads, opt_state, step)
        params = optax.apply_updates(params, updates)
        
        # Calculate accuracy
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == y)
        
        return params, opt_state, loss, ce_loss, router_loss, accuracy
    
    # JIT the evaluation step
    @jax.jit
    def eval_step(params, x, y):
        (loss, aux), _ = grad_fn(params, x, y)
        ce_loss, router_loss, logits = aux
        
        # Calculate accuracy
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == y)
        
        return loss, ce_loss, accuracy
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    
    batch_size = 64  # Reduced from 128/256
    
    # Create a simple data loader to shuffle the data
    def data_loader(x, y, batch_size, key):
        data_size = x.shape[0]
        num_batches = data_size // batch_size
        
        # Only process complete batches
        indices = random.permutation(key, jnp.arange(num_batches * batch_size))
        indices = indices.reshape(-1, batch_size)
        
        for batch_idx in indices:
            yield x[batch_idx], y[batch_idx]
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    step = 0
    
    for epoch in range(num_epochs):
        # Shuffle the data
        key, subkey = random.split(key)
        
        # Training loop
        epoch_losses = []
        epoch_accuracies = []
        
        for x_batch, y_batch in data_loader(x_train, y_train, batch_size, subkey):
            params, opt_state, train_loss, train_ce, train_router_loss, train_accuracy = train_step(
                combined_params, opt_state, x_batch, y_batch, step
            )
            combined_params = params
            epoch_losses.append(float(train_ce))
            epoch_accuracies.append(float(train_accuracy))
            step += 1
        
        # Evaluation step
        val_loss, val_ce, val_accuracy = eval_step(combined_params, x_val, y_val)
        
        # Store metrics
        train_losses.append(np.mean(epoch_losses))
        val_losses.append(float(val_ce))
        train_accuracies.append(np.mean(epoch_accuracies))
        val_accuracies.append(float(val_accuracy))
        
        # Print metrics every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")
            print(f"  Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")
    
    # Plot training history if requested
    if plot_history:
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracies
        plt.subplot(2, 1, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'mnist_moe_history_e{num_experts}_s{num_shared_experts}_c{num_constant_experts}_n{num_noise_experts}.png')
        plt.close()
    
    # Return the final metrics and model
    return {
        'final_loss': val_losses[-1],
        'final_accuracy': val_accuracies[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'model': model,
        'params': combined_params
    }

# Example usage
def main():
    # Create a key for random number generation
    key = random.key(0)
    
    # Load reduced MNIST data
    x_train, y_train, x_test, y_test = load_mnist_data()
    
    print("\nTesting model effectiveness with different configurations")
    print(f"Training data shape: {x_train.shape}")
    
    # Test the default configuration
    print("\n1. Testing default configuration:")
    results_default = train_and_evaluate_model(
        key, x_train, y_train, x_test, y_test,
        num_experts=8,
        num_shared_experts=1,
        num_constant_experts=1,
        num_noise_experts=1,
        num_epochs=200
    )
    
    # Test with more shared experts
    print("\n2. Testing with more shared experts:")
    results_more_shared = train_and_evaluate_model(
        key, x_train, y_train, x_test, y_test,
        num_experts=8,
        num_shared_experts=2,
        num_constant_experts=2,
        num_noise_experts=2,
        num_epochs=200
    )
    
    # Test with no special experts
    print("\n3. Testing with no special experts:")
    results_no_special = train_and_evaluate_model(
        key, x_train, y_train, x_test, y_test,
        num_experts=8,
        num_shared_experts=1,
        num_constant_experts=0,
        num_noise_experts=0,
        num_epochs=200
    )
    
    # Compare results
    print("\nComparison of Different Configurations:")
    print(f"1. Default Configuration: Loss = {results_default['final_loss']:.4f}, Accuracy = {results_default['final_accuracy']:.4f}")
    print(f"2. More Shared Experts: Loss = {results_more_shared['final_loss']:.4f}, Accuracy = {results_more_shared['final_accuracy']:.4f}")
    print(f"3. No Special Experts: Loss = {results_no_special['final_loss']:.4f}, Accuracy = {results_no_special['final_accuracy']:.4f}")
    
    # Now test inference speed
    print("\nTesting inference speed")
    
    # Create model instance for inference
    inference_model = ExpertsFeedForward(
        d_model=28,
        hidden_size=1024,  # Reduced
        num_experts=8,    # Reduced
        num_shared_experts=1,
        num_constant_experts=1,  # Reduced
        num_noise_experts=1,     # Reduced
        dtype=jnp.float32,
        use_gradient_checkpointing=False,
        training=False
    )
    
    # Initialize model parameters
    key, subkey = random.split(key)
    inference_variables = inference_model.init({'noise': key, 'dropout': key, 'params': key}, x_test)
    inference_params = inference_variables['params']
    
    @jax.jit
    def apply_fn(params, inputs):
        output, loss = inference_model.apply({'params': params}, inputs, rngs={'noise': key})
        return output, loss
    
    output, _ = apply_fn(inference_params, x_test)
    # Ensure the first run completes before timing
    output = jax.block_until_ready(output)
    print(f"Output shape: {output.shape}")
    
    # Time multiple runs
    num_runs = 10
    start_time = time.time()
    for i in range(num_runs):
        output, _ = apply_fn(inference_params, x_test)
        # Block on the last run to ensure all computation is complete
        if i == num_runs - 1:
            output = jax.block_until_ready(output)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"Average inference time per run: {avg_time*1000:.2f} ms")

if __name__ == "__main__":
    main()

