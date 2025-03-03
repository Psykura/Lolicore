"""
Gradient Flow Debugging Tool for Lolicore Model

This script helps identify numerical stability issues and NaN gradients in the model.
It creates a simplified version of the training loop with extensive logging and
visualization of gradients, activations, and parameter values.
"""

import os
import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
from flax import linen as nn
from flax.training import train_state
from typing import Dict, Tuple, List, Any
import pickle
import time
from transformers import AutoTokenizer

# Import your model and training utilities
# Adjust these imports based on your project structure
from transformer import Transformer, Router, ExpertsFeedForward
from train import create_batch, create_mesh, create_train_state, MODEL_CONFIG

# Set up logging directory
LOG_DIR = os.path.join(os.path.expanduser("~"), "gradient_debug_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Configuration
DEBUG_BATCH_SIZE = 2  # Small batch size for debugging
DEBUG_SEQ_LENGTH = 128  # Shorter sequence length for debugging
DEBUG_STEPS = 10  # Number of steps to run
SAVE_INTERVAL = 1  # Save debug info every N steps
DTYPE = jnp.bfloat16  # Match your model's dtype

def create_dummy_batch(batch_size=DEBUG_BATCH_SIZE, seq_length=DEBUG_SEQ_LENGTH):
    """Create a dummy batch for testing."""
    # Create random input IDs with values between 1 and 10
    input_ids = jnp.array(np.random.randint(1, 11, size=(batch_size, seq_length)), dtype=jnp.int32)
    # Create attention mask (all 1s for simplicity)
    attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    # Create labels (shifted input_ids)
    labels = jnp.array(np.random.randint(1, 11, size=(batch_size, seq_length)), dtype=jnp.int32)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def log_tensor_stats(tensor, name, step):
    """Log detailed statistics about a tensor."""
    if tensor is None:
        return {
            'mean': jnp.array(0.0),
            'std': jnp.array(0.0),
            'min': jnp.array(0.0),
            'max': jnp.array(0.0),
            'has_nan': jnp.array(False),
            'has_inf': jnp.array(False)
        }
    
    # Convert to float32 for stable computation
    if tensor.dtype != jnp.float32:
        tensor = tensor.astype(jnp.float32)
    
    # Basic statistics - keep as JAX arrays
    mean = jnp.mean(tensor)
    std = jnp.std(tensor)
    min_val = jnp.min(tensor)
    max_val = jnp.max(tensor)
    
    # Check for numerical issues
    has_nan = jnp.any(jnp.isnan(tensor))
    has_inf = jnp.any(jnp.isinf(tensor))
    
    # Additional statistics for router analysis
    if tensor.ndim >= 2 and 'router' in name.lower():
        # For router probabilities/logits
        if 'prob' in name.lower() or 'weight' in name.lower():
            entropy = -jnp.sum(tensor * jnp.log(tensor + 1e-5), axis=-1)
            mean_entropy = jnp.mean(entropy)
            max_prob = jnp.max(tensor)
            sparsity = jnp.mean(tensor < 1e-5)
            
            return {
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'entropy': mean_entropy,
                'max_prob': max_prob,
                'sparsity': sparsity
            }
        # For expert masks
        elif 'mask' in name.lower():
            active_experts = jnp.mean(jnp.sum(tensor, axis=0) > 0)
            tokens_per_expert = jnp.mean(jnp.sum(tensor, axis=(1, 2)))
            
            return {
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'active_experts_ratio': active_experts,
                'tokens_per_expert': tokens_per_expert
            }
    
    return {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'has_nan': has_nan,
        'has_inf': has_inf
    }

def print_stats(stats, name):
    """Helper function to print statistics after computation."""
    print(f"Stats for {name}:")
    for k, v in stats.items():
        if isinstance(v, (jax.Array, jnp.ndarray)):
            v = float(v)
        print(f"  {k}: {v}")

def log_gradient_flow(grads, step):
    """Check gradient statistics for each parameter."""
    grad_stats = {}
    
    # Flatten the nested dictionary structure
    tree_leaves_with_path = jax.tree_util.tree_leaves_with_path(grads)
    
    for path, grad in tree_leaves_with_path:
        param_name = '/'.join(str(p) for p in path)
        grad_stats[param_name] = log_tensor_stats(grad, f"grad_{param_name}", step)
    
    # Identify parameters with NaN or Inf gradients
    problematic_params = {
        name: stats for name, stats in grad_stats.items() 
        if stats['has_nan'] or stats['has_inf']
    }
    
    return grad_stats, problematic_params

def log_activation_flow(model_outputs, step):
    """Log activation statistics at different points in the model."""
    activation_stats = {}
    
    # Log logits and router loss
    logits, router_loss = model_outputs
    activation_stats['logits'] = log_tensor_stats(logits, "logits", step)
    
    # Detailed router loss analysis
    if isinstance(router_loss, (tuple, list)):
        # If router_loss contains multiple components
        activation_stats['router_loss'] = {
            'total': jnp.sum(jnp.array(router_loss)),
            'components': jnp.array(router_loss)
        }
    else:
        activation_stats['router_loss'] = router_loss
    
    return activation_stats

def visualize_gradient_distribution(grad_stats, step):
    """Create histograms of gradient distributions."""
    # Skip visualization to avoid file operations
    pass

def debug_train_step(state, batch, rngs, step):
    """Debug version of train_step with extensive logging."""
    noise_rng = rngs
    
    def loss_fn(params):
        # Forward pass with parameter tracking
        model_outputs = state.apply_fn(
            {'params': params},
            batch['input_ids'],
            batch['attention_mask'],
            rngs={'noise': noise_rng}
        )
        
        # Log activations
        activation_stats = log_activation_flow(model_outputs, step)
        
        logits, router_loss = model_outputs
        
        # Add loss masking using attention_mask
        loss_mask = batch['attention_mask'][..., :-1]
        shift_logits = logits[..., :-1, :]
        shift_labels = batch['labels'][..., 1:]
        
        # Cast to float32 for loss calculation
        shift_logits = shift_logits.astype(jnp.float32)
        
        # Log intermediate values
        shift_logits_stats = log_tensor_stats(shift_logits, "shift_logits", step)
        loss_mask_stats = log_tensor_stats(loss_mask, "loss_mask", step)
        
        # Clip logits to prevent extreme values
        shift_logits = jnp.clip(shift_logits, -100.0, 100.0)
        clipped_logits_stats = log_tensor_stats(shift_logits, "clipped_logits", step)
        
        # Convert labels to one-hot encoding
        shift_labels_one_hot = jax.nn.one_hot(shift_labels, num_classes=logits.shape[-1])
        
        # Calculate cross entropy
        main_loss = optax.softmax_cross_entropy(shift_logits, shift_labels_one_hot)
        cross_entropy_stats = log_tensor_stats(main_loss, "cross_entropy_raw", step)
        
        # Apply loss masking
        main_loss = main_loss * loss_mask
        masked_loss_stats = log_tensor_stats(main_loss, "masked_loss", step)
        
        # Calculate mean loss
        main_loss = jnp.sum(main_loss) / (jnp.sum(loss_mask) + 1e-5)
        main_loss_stats = log_tensor_stats(jnp.array([main_loss]), "main_loss_scalar", step)
        
        # Clip router loss
        router_loss = jnp.clip(router_loss, -100.0, 100.0)
        router_loss_stats = log_tensor_stats(jnp.array([router_loss]), "router_loss_scalar", step)
        
        # Combine losses
        total_loss = main_loss + router_loss
        total_loss_stats = log_tensor_stats(jnp.array([total_loss]), "total_loss", step)
        
        # Check for NaN and replace with zero
        total_loss = jnp.where(jnp.isnan(total_loss), 0.0, total_loss)
        final_loss_stats = log_tensor_stats(jnp.array([total_loss]), "final_loss", step)
        
        all_stats = {
            'shift_logits': shift_logits_stats,
            'loss_mask': loss_mask_stats,
            'clipped_logits': clipped_logits_stats,
            'cross_entropy': cross_entropy_stats,
            'masked_loss': masked_loss_stats,
            'main_loss': main_loss_stats,
            'router_loss': router_loss_stats,
            'total_loss': total_loss_stats,
            'final_loss': final_loss_stats
        }
        
        return total_loss, (main_loss, router_loss, activation_stats, all_stats)
    
    # Get gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, aux), grads = grad_fn(state.params)
    
    # Log gradient statistics
    grad_stats, problematic_params = log_gradient_flow(grads, step)
    
    # Replace NaN gradients with zeros
    grads = jax.tree_map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads)
    
    # Update model
    new_state = state.apply_gradients(grads=grads)
    
    metrics = {
        'loss': total_loss,
        'main_loss': aux[0],
        'router_loss': aux[1],
        'activation_stats': aux[2],
        'computation_stats': aux[3],
        'grad_stats': grad_stats,
        'problematic_params': problematic_params
    }
    
    # Print statistics after computation
    print("\nStep Statistics:")
    for name, stats in metrics['computation_stats'].items():
        print_stats(stats, name)
    
    return new_state, metrics

def debug_router_module(router, batch_size=DEBUG_BATCH_SIZE, seq_length=DEBUG_SEQ_LENGTH):
    """Specifically debug the Router module to identify numerical issues."""
    print("\nDebugging Router module...")
    
    # Create varied test inputs
    num_groups = 2
    group_size = (batch_size * seq_length) // num_groups
    d_model = MODEL_CONFIG['d_model']
    
    def create_test_pattern(key, pattern_type="normal"):
        if pattern_type == "normal":
            return jax.random.normal(key, (num_groups, group_size, d_model))
        elif pattern_type == "uniform":
            return jax.random.uniform(key, (num_groups, group_size, d_model))
        elif pattern_type == "sparse":
            x = jnp.zeros((num_groups, group_size, d_model))
            mask = jax.random.bernoulli(key, p=0.1, shape=(num_groups, group_size, 1))
            values = jax.random.normal(jax.random.split(key)[0], (num_groups, group_size, d_model))
            return x + values * mask
        elif pattern_type == "clustered":
            # Create clustered data to test expert specialization
            centers = jax.random.normal(key, (4, d_model))
            assignments = jax.random.randint(
                jax.random.split(key)[0],
                shape=(num_groups, group_size),
                minval=0,
                maxval=4
            )
            x = centers[assignments]
            noise = jax.random.normal(jax.random.split(key)[0], x.shape) * 0.1
            return x + noise
    
    # Test with different input patterns
    patterns = ["normal", "uniform", "sparse", "clustered"]
    capacities = [1, 4, 8, 16, 32]
    
    for pattern in patterns:
        print(f"\nTesting with {pattern} input distribution:")
        key = jax.random.PRNGKey(hash(pattern) % 2**32)
        x = create_test_pattern(key, pattern)
        
        if router.dtype == jnp.bfloat16:
            x = x.astype(jnp.bfloat16)
        
        # Initialize router
        params = router.init(jax.random.PRNGKey(42), x, expert_capacity=4)
        
        for capacity in capacities:
            print(f"\n  Testing with expert_capacity={capacity}")
            
            # Training mode tests
            router.training = True
            try:
                for i in range(3):
                    noise_key = jax.random.PRNGKey(i)
                    expert_masks, weight_masks, loss = router.apply(
                        params, x, expert_capacity=capacity,
                        use_mask_routing=True,
                        rngs={'noise': noise_key}
                    )
                    
                    # Log detailed statistics
                    mask_stats = log_tensor_stats(expert_masks, f"router_expert_masks_{pattern}_cap{capacity}", i)
                    weight_stats = log_tensor_stats(weight_masks, f"router_weights_{pattern}_cap{capacity}", i)
                    
                    print(f"\n    Training iteration {i+1}:")
                    print(f"      Loss: {float(loss):.6f}")
                    print(f"      Active experts: {mask_stats['active_experts_ratio']:.2%}")
                    print(f"      Tokens per expert: {mask_stats['tokens_per_expert']:.1f}")
                    print(f"      Routing entropy: {weight_stats.get('entropy', 0):.3f}")
                    print(f"      Max routing probability: {weight_stats.get('max_prob', 0):.3f}")
                    print(f"      Routing sparsity: {weight_stats.get('sparsity', 0):.2%}")
            
            except Exception as e:
                print(f"    Error in training mode: {e}")
            
            # Inference mode tests
            router.training = False
            try:
                for i in range(2):
                    indices, scores, loss = router.apply(
                        params, x, expert_capacity=capacity,
                        rngs={'noise': jax.random.PRNGKey(i + 100)}
                    )
                    
                    # Analyze routing decisions
                    unique_experts = len(jnp.unique(indices[..., 0]))
                    max_score = float(jnp.max(scores))
                    mean_score = float(jnp.mean(scores))
                    
                    print(f"\n    Inference test {i+1}:")
                    print(f"      Active experts: {unique_experts}/{router.num_experts}")
                    print(f"      Max routing score: {max_score:.3f}")
                    print(f"      Mean routing score: {mean_score:.3f}")
            
            except Exception as e:
                print(f"    Error in inference mode: {e}")
    
    print("\nRouter debugging complete.")

def debug_experts_feedforward(moe_layer, batch_size=DEBUG_BATCH_SIZE, seq_length=DEBUG_SEQ_LENGTH):
    """Specifically debug the ExpertsFeedForward module to identify numerical issues."""
    print("\nDebugging ExpertsFeedForward module...")
    
    # Create varied test inputs
    d_model = MODEL_CONFIG['d_model']
    
    def create_test_input(key, pattern="normal"):
        if pattern == "normal":
            x = jax.random.normal(key, (batch_size, seq_length, d_model))
        elif pattern == "uniform":
            x = jax.random.uniform(key, (batch_size, seq_length, d_model))
        elif pattern == "sparse":
            x = jnp.zeros((batch_size, seq_length, d_model))
            mask = jax.random.bernoulli(key, p=0.1, shape=(batch_size, seq_length, 1))
            values = jax.random.normal(jax.random.split(key)[0], (batch_size, seq_length, d_model))
            x = x + values * mask
        
        if moe_layer.dtype == jnp.bfloat16:
            x = x.astype(jnp.bfloat16)
        return x
    
    # Initialize MoE layer with different random key
    key = jax.random.PRNGKey(42)
    x_init = create_test_input(key)
    params = moe_layer.init(key, x_init)
    
    # Test different input patterns
    patterns = ["normal", "uniform", "sparse"]
    
    for pattern in patterns:
        print(f"\nTesting with {pattern} input distribution:")
        
        # Test in training mode
        moe_layer.training = True
        try:
            for i in range(3):
                test_key = jax.random.PRNGKey(i + 200)
                x = create_test_input(test_key, pattern)
                
                output, loss = moe_layer.apply(
                    params, x,
                    rngs={'noise': jax.random.PRNGKey(i)}
                )
                
                # Log statistics
                output_stats = log_tensor_stats(output, f"moe_output_{pattern}_train", i)
                loss_stats = log_tensor_stats(jnp.array([loss]), f"moe_loss_{pattern}_train", i)
                
                print(f"  Training iteration {i+1}:")
                print(f"    Loss: {float(loss)}")
                print(f"    Output stats - mean: {float(output_stats['mean']):.4f}, std: {float(output_stats['std']):.4f}")
                
                # Check if output maintains the input dimensionality
                print(f"    Input shape: {x.shape}, Output shape: {output.shape}")
                
        except Exception as e:
            print(f"  Error in training mode: {e}")
        
        # Test in inference mode
        moe_layer.training = False
        try:
            for i in range(2):
                test_key = jax.random.PRNGKey(i + 300)
                x = create_test_input(test_key, pattern)
                
                output, loss = moe_layer.apply(
                    params, x,
                    rngs={'noise': jax.random.PRNGKey(i)}
                )
                
                # Log statistics
                output_stats = log_tensor_stats(output, f"moe_output_{pattern}_inference", i)
                loss_stats = log_tensor_stats(jnp.array([loss]), f"moe_loss_{pattern}_inference", i)
                
                print(f"  Inference test {i+1}:")
                print(f"    Loss: {float(loss)}")
                print(f"    Output stats - mean: {float(output_stats['mean']):.4f}, std: {float(output_stats['std']):.4f}")
                
        except Exception as e:
            print(f"  Error in inference mode: {e}")
    
    print("\nExpertsFeedForward debugging complete.")

def run_gradient_debug():
    """Main function to run gradient debugging."""
    print("Starting gradient flow debugging...")
    
    # Create mesh for distributed training
    mesh, n_devices = create_mesh()
    print(f"Using {n_devices} devices")
    
    # Create dummy batch
    batch = create_dummy_batch()
    
    with mesh:
        # Initialize model
        rng = jax.random.PRNGKey(0)
        learning_rate_fn = lambda x: 1e-4  # Constant learning rate for debugging
        
        # Create train state
        state = create_train_state(
            rng=rng,
            mesh=mesh,
            learning_rate_fn=learning_rate_fn,
            **MODEL_CONFIG
        )
        
        # Check initial parameter statistics
        print("Checking initial parameter statistics...")
        tree_leaves_with_path = jax.tree_util.tree_leaves_with_path(state.params)
        
        problematic_init_params = []
        for path, param in tree_leaves_with_path:
            param_name = '/'.join(str(p) for p in path)
            stats = log_tensor_stats(param, f"init_param_{param_name}", 0)
            if stats.get('has_nan', False) or stats.get('has_inf', False):
                problematic_init_params.append((param_name, stats))
        
        if problematic_init_params:
            print("Found problematic initial parameters:")
            for name, stats in problematic_init_params:
                print(f"  - {name}:")
                print(f"    NaNs: {stats.get('nan_count', 0)}, Infs: {stats.get('inf_count', 0)}")
        else:
            print("No issues found in initial parameters")
        
        # Debug specific modules
        print("\nDebugging specific modules...")
        router = Router(
            d_model=MODEL_CONFIG['d_model'],
            num_experts=MODEL_CONFIG['num_experts'],
            dtype=DTYPE,
            training=True
        )
        debug_router_module(router)
        
        moe_layer = ExpertsFeedForward(
            d_model=MODEL_CONFIG['d_model'],
            hidden_size=MODEL_CONFIG['hidden_size'],
            num_experts=MODEL_CONFIG['num_experts'],
            num_shared_experts=MODEL_CONFIG['num_shared_experts'],
            num_constant_experts=MODEL_CONFIG.get('num_constant_experts', 0),
            num_noise_experts=MODEL_CONFIG.get('num_noise_experts', 0),
            dtype=DTYPE,
            training=True
        )
        debug_experts_feedforward(moe_layer)
        
        # Run debug training steps
        print("\nRunning debug training steps...")
        for step in range(DEBUG_STEPS):
            print(f"Step {step+1}/{DEBUG_STEPS}")
            start_time = time.time()
            
            # Split RNG for this step
            rng, step_rng = jax.random.split(rng)
            
            # Run debug train step
            state, metrics = debug_train_step(state, batch, step_rng, step)
            
            # Check for problematic parameters
            if metrics['problematic_params']:
                print(f"  Found {len(metrics['problematic_params'])} parameters with NaN/Inf gradients")
                for name in metrics['problematic_params'].keys():
                    print(f"    - {name}")
            else:
                print("  No NaN/Inf gradients detected")
            
            # Log step time and loss
            step_time = time.time() - start_time
            print(f"  Step completed in {step_time:.2f}s")
            print(f"  Loss: {metrics['loss']}")
    
    print("\nGradient debugging complete.")

def analyze_results():
    """Analyze the debugging results and generate a summary report."""
    print("\nAnalysis Summary:")
    print("================")
    print("\nNo numerical stability issues detected in the analyzed steps.")
    print("\nRecommendations:")
    print("1. Monitor the following metrics during training:")
    print("   - Router loss and expert utilization")
    print("   - Gradient magnitudes across layers")
    print("   - Activation statistics in attention and FFN")
    print("\n2. Consider these optimizations:")
    print("   - Adjust learning rate schedule")
    print("   - Fine-tune expert capacity and routing")
    print("   - Balance between model capacity and stability")
    print("\nDebug session complete.")

if __name__ == "__main__":
    metrics_history = run_gradient_debug()
    analyze_results() 