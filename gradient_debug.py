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
    # Create random input IDs
    input_ids = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    # Create attention mask (all 1s for simplicity)
    attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    # Create labels (shifted input_ids)
    labels = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def log_tensor_stats(tensor, name, step):
    """Log statistics about a tensor to help identify numerical issues."""
    # Convert to numpy for easier handling
    if hasattr(tensor, 'device_buffer'):
        tensor = np.array(tensor)
    
    # Basic statistics
    stats = {
        'min': float(np.min(tensor)),
        'max': float(np.max(tensor)),
        'mean': float(np.mean(tensor)),
        'std': float(np.std(tensor)),
        'has_nan': bool(np.isnan(np.sum(tensor))),
        'has_inf': bool(np.isinf(np.sum(tensor))),
    }
    
    # Count NaNs and Infs
    if stats['has_nan'] or stats['has_inf']:
        stats['nan_count'] = int(np.sum(np.isnan(tensor)))
        stats['inf_count'] = int(np.sum(np.isinf(tensor)))
        stats['total_elements'] = tensor.size
        stats['nan_percentage'] = stats['nan_count'] / stats['total_elements'] * 100
        stats['inf_percentage'] = stats['inf_count'] / stats['total_elements'] * 100
    
    # Save statistics
    filename = f"step_{step}_{name}_stats.pkl"
    with open(os.path.join(LOG_DIR, filename), 'wb') as f:
        pickle.dump(stats, f)
    
    # If tensor is small enough, save the full tensor
    if tensor.size < 10000:
        full_filename = f"step_{step}_{name}_full.npy"
        np.save(os.path.join(LOG_DIR, full_filename), tensor)
    
    return stats

def log_gradient_flow(grads, step):
    """Log gradient statistics for each parameter."""
    grad_stats = {}
    
    # Flatten the nested dictionary structure
    flat_paths, flat_grads = jax.tree_util.tree_leaves_with_path(grads)
    
    for path, grad in zip(flat_paths, flat_grads):
        param_name = '/'.join(str(p) for p in path)
        grad_stats[param_name] = log_tensor_stats(grad, f"grad_{param_name}", step)
    
    # Identify parameters with NaN or Inf gradients
    problematic_params = {
        name: stats for name, stats in grad_stats.items() 
        if stats.get('has_nan', False) or stats.get('has_inf', False)
    }
    
    # Save summary of problematic parameters
    if problematic_params:
        with open(os.path.join(LOG_DIR, f"step_{step}_problematic_grads.txt"), 'w') as f:
            for name, stats in problematic_params.items():
                f.write(f"Parameter: {name}\n")
                for k, v in stats.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
    
    return grad_stats, problematic_params

def log_activation_flow(model_outputs, step):
    """Log activation statistics at different points in the model."""
    activation_stats = {}
    
    # Log logits
    logits, router_loss = model_outputs
    activation_stats['logits'] = log_tensor_stats(logits, "logits", step)
    activation_stats['router_loss'] = log_tensor_stats(router_loss, "router_loss", step)
    
    return activation_stats

def visualize_gradient_distribution(grad_stats, step):
    """Create histograms of gradient distributions."""
    plt.figure(figsize=(12, 8))
    
    # Collect all gradient means and stds
    means = []
    stds = []
    names = []
    
    for name, stats in grad_stats.items():
        if not (stats.get('has_nan', False) or stats.get('has_inf', False)):
            means.append(stats['mean'])
            stds.append(stats['std'])
            names.append(name)
    
    # Plot histogram of means
    plt.subplot(2, 1, 1)
    plt.hist(means, bins=50)
    plt.title(f'Gradient Means Distribution (Step {step})')
    plt.xlabel('Mean Value')
    plt.ylabel('Count')
    
    # Plot histogram of standard deviations
    plt.subplot(2, 1, 2)
    plt.hist(stds, bins=50)
    plt.title(f'Gradient Standard Deviations Distribution (Step {step})')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f"step_{step}_gradient_distribution.png"))
    plt.close()

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
        log_tensor_stats(shift_logits, "shift_logits", step)
        log_tensor_stats(loss_mask, "loss_mask", step)
        
        # Clip logits to prevent extreme values
        shift_logits = jnp.clip(shift_logits, -100.0, 100.0)
        log_tensor_stats(shift_logits, "clipped_logits", step)
        
        # Convert labels to one-hot encoding
        shift_labels_one_hot = jax.nn.one_hot(shift_labels, num_classes=logits.shape[-1])
        
        # Calculate cross entropy
        main_loss = optax.softmax_cross_entropy(shift_logits, shift_labels_one_hot)
        log_tensor_stats(main_loss, "cross_entropy_raw", step)
        
        # Apply loss masking
        main_loss = main_loss * loss_mask
        log_tensor_stats(main_loss, "masked_loss", step)
        
        # Calculate mean loss
        main_loss = jnp.sum(main_loss) / (jnp.sum(loss_mask) + 1e-5)
        log_tensor_stats(jnp.array([main_loss]), "main_loss_scalar", step)
        
        # Clip router loss
        router_loss = jnp.clip(router_loss, -100.0, 100.0)
        log_tensor_stats(jnp.array([router_loss]), "router_loss_scalar", step)
        
        # Combine losses
        total_loss = main_loss + router_loss
        log_tensor_stats(jnp.array([total_loss]), "total_loss", step)
        
        # Check for NaN and replace with zero
        total_loss = jnp.where(jnp.isnan(total_loss), 0.0, total_loss)
        log_tensor_stats(jnp.array([total_loss]), "final_loss", step)
        
        return total_loss, (main_loss, router_loss, activation_stats)
    
    # Get gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, aux), grads = grad_fn(state.params)
    
    # Log gradient statistics
    grad_stats, problematic_params = log_gradient_flow(grads, step)
    
    # Visualize gradient distribution
    visualize_gradient_distribution(grad_stats, step)
    
    # Replace NaN gradients with zeros
    grads = jax.tree_map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads)
    
    # Update model
    new_state = state.apply_gradients(grads=grads)
    
    metrics = {
        'loss': total_loss,
        'main_loss': aux[0],
        'router_loss': aux[1],
        'activation_stats': aux[2],
        'grad_stats': grad_stats,
        'problematic_params': problematic_params
    }
    
    return new_state, metrics

def debug_router_module(router, batch_size=DEBUG_BATCH_SIZE, seq_length=DEBUG_SEQ_LENGTH):
    """Specifically debug the Router module to identify numerical issues."""
    print("Debugging Router module...")
    
    # Create a dummy input for the router
    num_groups = 2
    group_size = (batch_size * seq_length) // num_groups
    d_model = MODEL_CONFIG['d_model']
    
    # Create random input
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (num_groups, group_size, d_model))
    
    # Convert to bfloat16 if needed
    if router.dtype == jnp.bfloat16:
        x = x.astype(jnp.bfloat16)
    
    # Initialize router
    params = router.init(jax.random.PRNGKey(1), x, expert_capacity=4)
    
    # Test with different expert capacities
    capacities = [1, 4, 8, 16, 32]
    
    for capacity in capacities:
        print(f"Testing with expert_capacity={capacity}")
        
        # Run router in training mode
        router.training = True
        try:
            expert_masks, weight_masks, loss = router.apply(
                params, x, expert_capacity=capacity, use_mask_routing=True
            )
            
            # Log statistics
            log_tensor_stats(expert_masks, f"router_expert_masks_cap{capacity}", 0)
            log_tensor_stats(weight_masks, f"router_weight_masks_cap{capacity}", 0)
            log_tensor_stats(jnp.array([loss]), f"router_loss_cap{capacity}", 0)
            
            print(f"  Training mode successful. Loss: {loss}")
        except Exception as e:
            print(f"  Error in training mode: {e}")
        
        # Run router in inference mode
        router.training = False
        try:
            indices, scores, loss = router.apply(
                params, x, expert_capacity=capacity
            )
            
            # Log statistics
            log_tensor_stats(indices, f"router_indices_cap{capacity}", 0)
            log_tensor_stats(scores, f"router_scores_cap{capacity}", 0)
            log_tensor_stats(jnp.array([loss]), f"router_loss_inference_cap{capacity}", 0)
            
            print(f"  Inference mode successful.")
        except Exception as e:
            print(f"  Error in inference mode: {e}")
    
    print("Router debugging complete.")

def debug_experts_feedforward(moe_layer, batch_size=DEBUG_BATCH_SIZE, seq_length=DEBUG_SEQ_LENGTH):
    """Specifically debug the ExpertsFeedForward module to identify numerical issues."""
    print("Debugging ExpertsFeedForward module...")
    
    # Create a dummy input
    d_model = MODEL_CONFIG['d_model']
    
    # Create random input
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, seq_length, d_model))
    
    # Convert to bfloat16 if needed
    if moe_layer.dtype == jnp.bfloat16:
        x = x.astype(jnp.bfloat16)
    
    # Initialize MoE layer
    params = moe_layer.init(jax.random.PRNGKey(1), x)
    
    # Test in training mode
    moe_layer.training = True
    try:
        output, loss = moe_layer.apply(
            params, x, rngs={'noise': jax.random.PRNGKey(2)}
        )
        
        # Log statistics
        log_tensor_stats(output, "moe_output_training", 0)
        log_tensor_stats(jnp.array([loss]), "moe_loss_training", 0)
        
        print(f"  Training mode successful. Loss: {loss}")
    except Exception as e:
        print(f"  Error in training mode: {e}")
    
    # Test in inference mode
    moe_layer.training = False
    try:
        output, loss = moe_layer.apply(
            params, x, rngs={'noise': jax.random.PRNGKey(3)}
        )
        
        # Log statistics
        log_tensor_stats(output, "moe_output_inference", 0)
        log_tensor_stats(jnp.array([loss]), "moe_loss_inference", 0)
        
        print(f"  Inference mode successful.")
    except Exception as e:
        print(f"  Error in inference mode: {e}")
    
    print("ExpertsFeedForward debugging complete.")

def run_gradient_debug():
    """Main function to run gradient debugging."""
    print(f"Starting gradient flow debugging. Logs will be saved to {LOG_DIR}")
    
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
        
        # Log initial parameter statistics
        print("Logging initial parameter statistics...")
        flat_params = jax.tree_util.tree_leaves(state.params)
        flat_paths = jax.tree_util.tree_paths(state.params)
        
        for path, param in zip(flat_paths, flat_params):
            param_name = '/'.join(str(p) for p in path)
            log_tensor_stats(param, f"init_param_{param_name}", 0)
        
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
            
            # Log step time
            step_time = time.time() - start_time
            print(f"  Step completed in {step_time:.2f}s")
            print(f"  Loss: {metrics['loss']}")
            
            # Save summary for this step
            with open(os.path.join(LOG_DIR, f"step_{step}_summary.txt"), 'w') as f:
                f.write(f"Step {step} Summary\n")
                f.write(f"Total Loss: {metrics['loss']}\n")
                f.write(f"Main Loss: {metrics['main_loss']}\n")
                f.write(f"Router Loss: {metrics['router_loss']}\n")
                f.write(f"Step Time: {step_time:.2f}s\n")
                f.write(f"NaN/Inf Gradients: {len(metrics['problematic_params'])}\n")
    
    print("\nGradient debugging complete. Check the logs for detailed information.")
    print(f"Log directory: {LOG_DIR}")

def analyze_results():
    """Analyze the debugging results and generate a summary report."""
    print("Analyzing debugging results...")
    
    # Find all step summary files
    summary_files = [f for f in os.listdir(LOG_DIR) if f.endswith('_summary.txt')]
    step_numbers = [int(f.split('_')[1]) for f in summary_files]
    max_step = max(step_numbers)
    
    # Collect problematic parameters across all steps
    all_problematic_params = set()
    for step in range(max_step + 1):
        prob_file = os.path.join(LOG_DIR, f"step_{step}_problematic_grads.txt")
        if os.path.exists(prob_file):
            with open(prob_file, 'r') as f:
                content = f.read()
                param_names = [line.split(': ')[1] for line in content.split('\n') 
                              if line.startswith('Parameter: ')]
                all_problematic_params.update(param_names)
    
    # Generate summary report
    report_path = os.path.join(LOG_DIR, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("Gradient Flow Analysis Report\n")
        f.write("===========================\n\n")
        
        f.write(f"Total Steps Analyzed: {max_step + 1}\n")
        f.write(f"Total Problematic Parameters: {len(all_problematic_params)}\n\n")
        
        if all_problematic_params:
            f.write("Problematic Parameters:\n")
            for param in sorted(all_problematic_params):
                f.write(f"  - {param}\n")
            f.write("\n")
        
        f.write("Step-by-Step Analysis:\n")
        for step in range(max_step + 1):
            summary_file = os.path.join(LOG_DIR, f"step_{step}_summary.txt")
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as sf:
                    f.write(f"\nStep {step}:\n")
                    f.write(sf.read())
                    f.write("\n")
        
        f.write("\nRecommendations:\n")
        if all_problematic_params:
            f.write("1. Check the following components for numerical stability issues:\n")
            
            # Group parameters by component
            router_params = [p for p in all_problematic_params if 'router' in p.lower()]
            attention_params = [p for p in all_problematic_params if 'attention' in p.lower()]
            ffn_params = [p for p in all_problematic_params if 'feedforward' in p.lower() or 'ffn' in p.lower()]
            
            if router_params:
                f.write("   - Router module (likely source of NaNs)\n")
            if attention_params:
                f.write("   - Attention mechanism\n")
            if ffn_params:
                f.write("   - Feedforward networks\n")
                
            f.write("\n2. Consider the following fixes:\n")
            f.write("   - Add gradient clipping with smaller threshold\n")
            f.write("   - Use float32 instead of bfloat16 for critical operations\n")
            f.write("   - Add more robust NaN/Inf checking throughout the model\n")
            f.write("   - Reduce learning rate or use more stable optimizer\n")
        else:
            f.write("No numerical stability issues detected in the analyzed steps.\n")
    
    print(f"Analysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    run_gradient_debug()
    analyze_results() 