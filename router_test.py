"""
Router Module Test Script

This script specifically tests the Router module in isolation to identify
numerical stability issues and the exact conditions that cause NaN values.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import linen as nn
import pickle
import time

# Import the Router module
from transformer import Router

# Set up logging directory
LOG_DIR = os.path.join(os.path.expanduser("~"), "router_test_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Test configurations
TEST_CONFIGS = [
    # Format: (num_groups, group_size, d_model, num_experts, expert_capacity, dtype)
    (2, 64, 512, 8, 16, jnp.bfloat16),  # Standard config
    (2, 64, 512, 8, 16, jnp.float32),   # Higher precision
    (2, 64, 512, 8, 8, jnp.bfloat16),   # Lower capacity
    (2, 64, 512, 8, 32, jnp.bfloat16),  # Higher capacity
    (2, 64, 512, 16, 16, jnp.bfloat16), # More experts
    (2, 32, 512, 8, 16, jnp.bfloat16),  # Smaller group size
    (2, 128, 512, 8, 16, jnp.bfloat16), # Larger group size
    (4, 64, 512, 8, 16, jnp.bfloat16),  # More groups
]

# Input distributions to test
INPUT_DISTRIBUTIONS = [
    ("normal", lambda key, shape: jax.random.normal(key, shape)),
    ("uniform", lambda key, shape: jax.random.uniform(key, shape, minval=-1, maxval=1)),
    ("small_values", lambda key, shape: jax.random.normal(key, shape) * 0.01),
    ("large_values", lambda key, shape: jax.random.normal(key, shape) * 100),
]

def log_tensor_stats(tensor, name, config_idx, dist_name):
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
    filename = f"config_{config_idx}_{dist_name}_{name}_stats.pkl"
    with open(os.path.join(LOG_DIR, filename), 'wb') as f:
        pickle.dump(stats, f)
    
    # If tensor is small enough, save the full tensor
    if tensor.size < 10000:
        full_filename = f"config_{config_idx}_{dist_name}_{name}_full.npy"
        np.save(os.path.join(LOG_DIR, full_filename), tensor)
    
    return stats

def test_router_forward(config_idx, num_groups, group_size, d_model, num_experts, expert_capacity, dtype, dist_name, input_fn):
    """Test the forward pass of the Router module with specific configuration."""
    print(f"Testing config {config_idx} with {dist_name} distribution:")
    print(f"  num_groups={num_groups}, group_size={group_size}, d_model={d_model}")
    print(f"  num_experts={num_experts}, expert_capacity={expert_capacity}, dtype={dtype}")
    
    # Create router
    router = Router(
        d_model=d_model,
        num_experts=num_experts,
        dtype=dtype,
        training=True
    )
    
    # Create input
    key = jax.random.PRNGKey(config_idx)
    x = input_fn(key, (num_groups, group_size, d_model))
    if dtype == jnp.bfloat16:
        x = x.astype(jnp.bfloat16)
    
    # Log input statistics
    log_tensor_stats(x, "input", config_idx, dist_name)
    
    # Initialize router
    init_key = jax.random.PRNGKey(config_idx + 100)
    params = router.init(init_key, x, expert_capacity=expert_capacity)
    
    # Test training mode
    router.training = True
    try:
        # Forward pass
        expert_masks, weight_masks, loss = router.apply(
            params, x, expert_capacity=expert_capacity, use_mask_routing=True
        )
        
        # Log outputs
        log_tensor_stats(expert_masks, "expert_masks", config_idx, dist_name)
        log_tensor_stats(weight_masks, "weight_masks", config_idx, dist_name)
        log_tensor_stats(jnp.array([loss]), "loss", config_idx, dist_name)
        
        print(f"  Training mode successful. Loss: {loss}")
        return True, None
    except Exception as e:
        print(f"  Error in training mode: {e}")
        return False, str(e)

def test_router_with_modified_gate(config_idx, num_groups, group_size, d_model, num_experts, expert_capacity, dtype):
    """Test the router with different gate initializations to identify stability issues."""
    print(f"Testing router with modified gate (config {config_idx}):")
    
    # Create router
    router = Router(
        d_model=d_model,
        num_experts=num_experts,
        dtype=dtype,
        training=True
    )
    
    # Create normal input
    key = jax.random.PRNGKey(config_idx)
    x = jax.random.normal(key, (num_groups, group_size, d_model))
    if dtype == jnp.bfloat16:
        x = x.astype(jnp.bfloat16)
    
    # Initialize router
    init_key = jax.random.PRNGKey(config_idx + 100)
    params = router.init(init_key, x, expert_capacity=expert_capacity)
    
    # Test with different gate weight scales
    scales = [0.001, 0.01, 0.1, 1.0, 10.0]
    results = {}
    
    for scale in scales:
        print(f"  Testing with gate weight scale: {scale}")
        
        # Scale the gate weights
        scaled_params = params.copy()
        gate_weights = params['params']['gate']['kernel']
        scaled_params['params']['gate']['kernel'] = gate_weights * scale
        
        # Test forward pass
        router.training = True
        try:
            expert_masks, weight_masks, loss = router.apply(
                scaled_params, x, expert_capacity=expert_capacity, use_mask_routing=True
            )
            
            # Log outputs
            log_tensor_stats(expert_masks, f"expert_masks_scale{scale}", config_idx, "modified_gate")
            log_tensor_stats(weight_masks, f"weight_masks_scale{scale}", config_idx, "modified_gate")
            log_tensor_stats(jnp.array([loss]), f"loss_scale{scale}", config_idx, "modified_gate")
            
            has_nan = np.isnan(np.array(loss))
            results[scale] = (True, has_nan, loss)
            print(f"    Success. Loss: {loss}, Has NaN: {has_nan}")
        except Exception as e:
            results[scale] = (False, None, str(e))
            print(f"    Error: {e}")
    
    return results

def test_router_gradient(config_idx, num_groups, group_size, d_model, num_experts, expert_capacity, dtype):
    """Test the gradient computation of the Router module."""
    print(f"Testing router gradients (config {config_idx}):")
    
    # Create router
    router = Router(
        d_model=d_model,
        num_experts=num_experts,
        dtype=dtype,
        training=True
    )
    
    # Create input
    key = jax.random.PRNGKey(config_idx)
    x = jax.random.normal(key, (num_groups, group_size, d_model))
    if dtype == jnp.bfloat16:
        x = x.astype(jnp.bfloat16)
    
    # Initialize router
    init_key = jax.random.PRNGKey(config_idx + 100)
    params = router.init(init_key, x, expert_capacity=expert_capacity)
    
    # Define loss function
    def loss_fn(params):
        expert_masks, weight_masks, router_loss = router.apply(
            params, x, expert_capacity=expert_capacity, use_mask_routing=True
        )
        return router_loss
    
    # Compute gradients
    try:
        grads = jax.grad(loss_fn)(params)
        
        # Check for NaNs in gradients
        flat_grads = jax.tree_util.tree_leaves(grads)
        has_nan = any(np.isnan(np.sum(np.array(g))) for g in flat_grads)
        has_inf = any(np.isinf(np.sum(np.array(g))) for g in flat_grads)
        
        # Log gradient statistics
        for path, grad in zip(jax.tree_util.tree_paths(grads), jax.tree_util.tree_leaves(grads)):
            param_name = '/'.join(str(p) for p in path)
            log_tensor_stats(grad, f"grad_{param_name}", config_idx, "gradient_test")
        
        print(f"  Gradient computation successful. Has NaN: {has_nan}, Has Inf: {has_inf}")
        return True, has_nan, has_inf
    except Exception as e:
        print(f"  Error computing gradients: {e}")
        return False, None, None

def test_router_with_float32_intermediates(config_idx, num_groups, group_size, d_model, num_experts, expert_capacity):
    """Test the router with float32 for intermediate calculations."""
    print(f"Testing router with float32 intermediates (config {config_idx}):")
    
    # Create a modified router class with float32 intermediates
    class RouterFloat32(nn.Module):
        """Router with float32 intermediates for numerical stability."""
        d_model: int
        num_experts: int
        z_loss_coef: float = 1e-3
        balance_loss_coef: float = 4e-2
        dtype: jnp.dtype = jnp.bfloat16
        training: bool = False
        
        def setup(self):
            gate_init = nn.initializers.normal(stddev=0.02)
            self.gate = nn.Dense(
                features=self.num_experts,
                use_bias=False,
                kernel_init=gate_init,
                dtype=self.dtype
            )
        
        def __call__(self, x, expert_capacity: int, use_mask_routing: bool = False):
            num_groups, group_size, _ = x.shape
            total_tokens = num_groups * group_size
            
            # Cast to float32 for all intermediate calculations
            x_f32 = x.astype(jnp.float32)
            
            # Compute routing probabilities
            router_logits = self.gate(x_f32)  # Use float32
            
            # Clip logits to prevent overflow
            router_logits = jnp.clip(router_logits, -50.0, 50.0)
            
            # Use float32 for softmax
            router_probs = jax.nn.softmax(router_logits, axis=-1)
            
            if self.training:
                # Calculate load balancing loss with float32
                expert_usage = jnp.sum(router_probs, axis=(0, 1)) / (total_tokens + 1e-5)
                
                # Clip expert_usage to prevent extreme values
                expert_usage = jnp.clip(expert_usage, 1e-5, 1.0)
                
                balance_loss = (self.num_experts * jnp.sum(expert_usage ** 2) - 1.0) * self.balance_loss_coef
                
                # Calculate router z-loss
                router_z = jax.nn.logsumexp(router_logits, axis=-1)
                
                # Clip z values
                router_z = jnp.clip(router_z, -50.0, 50.0)
                
                router_z_loss = jnp.mean(jnp.square(router_z)) * self.z_loss_coef
                
                # Combined loss
                loss = balance_loss + router_z_loss
            else:
                loss = 0.0
            
            # Reshape for expert-wise token selection
            flat_probs = router_probs.transpose(2, 0, 1).reshape(self.num_experts, -1)
            
            # Pad with less extreme negative values
            padding_width = max(0, expert_capacity - flat_probs.shape[1])
            flat_probs = jnp.pad(
                flat_probs,
                ((0, 0), (0, padding_width)),
                mode='constant',
                constant_values=-1e4
            )
            
            # Select top tokens for each expert
            scores, token_indices = jax.lax.top_k(flat_probs, k=expert_capacity)
            
            # Zero out scores for padded values
            scores = jnp.where(scores > -1e3, scores, 0.0)
            
            # Convert flat indices to coordinates
            group_indices = token_indices // group_size
            pos_indices = token_indices % group_size
            
            # Cast back to original dtype for outputs
            if use_mask_routing:
                # Create masks
                expert_masks = jnp.zeros((self.num_experts, num_groups, group_size), dtype=jnp.bool_)
                weight_masks = jnp.zeros((self.num_experts, num_groups, group_size), dtype=jnp.float32)
                
                batch_indices = jnp.arange(self.num_experts)[:, None]
                batch_indices = jnp.broadcast_to(batch_indices, (self.num_experts, expert_capacity))
                
                flat_batch_indices = batch_indices.reshape(-1)
                flat_group_indices = group_indices.reshape(-1)
                flat_pos_indices = pos_indices.reshape(-1)
                flat_scores = scores.reshape(-1)
                
                scatter_indices = jnp.stack([flat_batch_indices, flat_group_indices, flat_pos_indices], axis=1)
                
                expert_masks = expert_masks.at[scatter_indices[:, 0], scatter_indices[:, 1], scatter_indices[:, 2]].set(True)
                weight_masks = weight_masks.at[scatter_indices[:, 0], scatter_indices[:, 1], scatter_indices[:, 2]].set(flat_scores)
                
                # Cast weight_masks back to original dtype if needed
                if self.dtype != jnp.float32:
                    weight_masks = weight_masks.astype(self.dtype)
                
                return expert_masks, weight_masks, loss
            else:
                indices = jnp.stack([group_indices, pos_indices], axis=-1)
                return indices, scores.astype(self.dtype) if self.dtype != jnp.float32 else scores, loss
    
    # Create router with float32 intermediates
    router = RouterFloat32(
        d_model=d_model,
        num_experts=num_experts,
        dtype=jnp.bfloat16,  # Output dtype
        training=True
    )
    
    # Create input
    key = jax.random.PRNGKey(config_idx)
    x = jax.random.normal(key, (num_groups, group_size, d_model))
    x = x.astype(jnp.bfloat16)
    
    # Initialize router
    init_key = jax.random.PRNGKey(config_idx + 100)
    params = router.init(init_key, x, expert_capacity=expert_capacity)
    
    # Test forward pass
    try:
        expert_masks, weight_masks, loss = router.apply(
            params, x, expert_capacity=expert_capacity, use_mask_routing=True
        )
        
        # Log outputs
        log_tensor_stats(expert_masks, "expert_masks_float32", config_idx, "float32_intermediates")
        log_tensor_stats(weight_masks, "weight_masks_float32", config_idx, "float32_intermediates")
        log_tensor_stats(jnp.array([loss]), "loss_float32", config_idx, "float32_intermediates")
        
        has_nan = np.isnan(np.array(loss))
        print(f"  Success with float32 intermediates. Loss: {loss}, Has NaN: {has_nan}")
        return True, has_nan, loss
    except Exception as e:
        print(f"  Error with float32 intermediates: {e}")
        return False, None, None

def run_router_tests():
    """Run all router tests and generate a comprehensive report."""
    print(f"Starting router tests. Logs will be saved to {LOG_DIR}")
    
    # Store all test results
    results = {
        "forward_pass": {},
        "modified_gate": {},
        "gradients": {},
        "float32_intermediates": {}
    }
    
    # Test forward pass with different input distributions
    for config_idx, config in enumerate(TEST_CONFIGS):
        results["forward_pass"][config_idx] = {}
        
        for dist_name, input_fn in INPUT_DISTRIBUTIONS:
            success, error = test_router_forward(
                config_idx, *config, dist_name, input_fn
            )
            results["forward_pass"][config_idx][dist_name] = {
                "success": success,
                "error": error
            }
        
        # Test with modified gate weights
        results["modified_gate"][config_idx] = test_router_with_modified_gate(
            config_idx, *config
        )
        
        # Test gradient computation
        success, has_nan, has_inf = test_router_gradient(
            config_idx, *config
        )
        results["gradients"][config_idx] = {
            "success": success,
            "has_nan": has_nan,
            "has_inf": has_inf
        }
        
        # Test with float32 intermediates
        success, has_nan, loss = test_router_with_float32_intermediates(
            config_idx, *config[:-1]  # Exclude dtype
        )
        results["float32_intermediates"][config_idx] = {
            "success": success,
            "has_nan": has_nan,
            "loss": loss
        }
    
    # Save all results
    with open(os.path.join(LOG_DIR, "all_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    # Generate summary report
    generate_report(results)
    
    print(f"Router tests complete. Check {LOG_DIR} for detailed results.")

def generate_report(results):
    """Generate a comprehensive report of all test results."""
    report_path = os.path.join(LOG_DIR, "router_test_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("Router Module Test Report\n")
        f.write("=======================\n\n")
        
        # Summarize configurations
        f.write("Test Configurations:\n")
        for config_idx, config in enumerate(TEST_CONFIGS):
            num_groups, group_size, d_model, num_experts, expert_capacity, dtype = config
            f.write(f"Config {config_idx}: ")
            f.write(f"num_groups={num_groups}, group_size={group_size}, d_model={d_model}, ")
            f.write(f"num_experts={num_experts}, expert_capacity={expert_capacity}, dtype={dtype}\n")
        f.write("\n")
        
        # Forward pass results
        f.write("Forward Pass Results:\n")
        for config_idx in results["forward_pass"]:
            f.write(f"  Config {config_idx}:\n")
            for dist_name in results["forward_pass"][config_idx]:
                result = results["forward_pass"][config_idx][dist_name]
                status = "Success" if result["success"] else f"Failed: {result['error']}"
                f.write(f"    {dist_name}: {status}\n")
        f.write("\n")
        
        # Modified gate results
        f.write("Modified Gate Results:\n")
        for config_idx in results["modified_gate"]:
            f.write(f"  Config {config_idx}:\n")
            for scale, (success, has_nan, result) in results["modified_gate"][config_idx].items():
                status = "Success" if success else f"Failed: {result}"
                if success:
                    f.write(f"    Scale {scale}: {status}, Loss: {result}, Has NaN: {has_nan}\n")
                else:
                    f.write(f"    Scale {scale}: {status}\n")
        f.write("\n")
        
        # Gradient results
        f.write("Gradient Computation Results:\n")
        for config_idx in results["gradients"]:
            result = results["gradients"][config_idx]
            status = "Success" if result["success"] else "Failed"
            if result["success"]:
                f.write(f"  Config {config_idx}: {status}, Has NaN: {result['has_nan']}, Has Inf: {result['has_inf']}\n")
            else:
                f.write(f"  Config {config_idx}: {status}\n")
        f.write("\n")
        
        # Float32 intermediates results
        f.write("Float32 Intermediates Results:\n")
        for config_idx in results["float32_intermediates"]:
            result = results["float32_intermediates"][config_idx]
            status = "Success" if result["success"] else "Failed"
            if result["success"]:
                f.write(f"  Config {config_idx}: {status}, Loss: {result['loss']}, Has NaN: {result['has_nan']}\n")
            else:
                f.write(f"  Config {config_idx}: {status}\n")
        f.write("\n")
        
        # Analysis and recommendations
        f.write("Analysis and Recommendations:\n")
        
        # Check if float32 helps
        float32_configs = [i for i, config in enumerate(TEST_CONFIGS) if config[5] == jnp.float32]
        float32_success = all(results["forward_pass"][i]["normal"]["success"] for i in float32_configs)
        float32_intermediates_success = all(results["float32_intermediates"][i]["success"] for i in results["float32_intermediates"])
        
        if not float32_success and float32_intermediates_success:
            f.write("1. Using float32 for intermediate calculations significantly improves stability.\n")
            f.write("   Recommendation: Modify the Router module to use float32 internally.\n\n")
        
        # Check if certain configurations are more stable
        stable_configs = []
        for config_idx in results["forward_pass"]:
            if all(results["forward_pass"][config_idx][dist]["success"] for dist in results["forward_pass"][config_idx]):
                stable_configs.append(config_idx)
        
        if stable_configs:
            f.write("2. The following configurations are more stable:\n")
            for config_idx in stable_configs:
                num_groups, group_size, d_model, num_experts, expert_capacity, dtype = TEST_CONFIGS[config_idx]
                f.write(f"   - Config {config_idx}: expert_capacity={expert_capacity}, dtype={dtype}\n")
            f.write("\n")
        
        # Check if certain input distributions cause more problems
        problematic_dists = {}
        for dist_name in INPUT_DISTRIBUTIONS[0][0]:  # Get all distribution names
            failures = sum(1 for config_idx in results["forward_pass"] 
                         if dist_name in results["forward_pass"][config_idx] 
                         and not results["forward_pass"][config_idx][dist_name]["success"])
            if failures > 0:
                problematic_dists[dist_name] = failures
        
        if problematic_dists:
            f.write("3. The following input distributions cause more stability issues:\n")
            for dist_name, count in sorted(problematic_dists.items(), key=lambda x: x[1], reverse=True):
                f.write(f"   - {dist_name}: {count} failures\n")
            f.write("\n")
        
        # Check if gate weight scaling affects stability
        gate_scale_issues = {}
        for config_idx in results["modified_gate"]:
            for scale, (success, has_nan, _) in results["modified_gate"][config_idx].items():
                if not success or has_nan:
                    gate_scale_issues[scale] = gate_scale_issues.get(scale, 0) + 1
        
        if gate_scale_issues:
            f.write("4. The following gate weight scales cause stability issues:\n")
            for scale, count in sorted(gate_scale_issues.items(), key=lambda x: x[1], reverse=True):
                f.write(f"   - Scale {scale}: {count} issues\n")
            f.write("   Recommendation: Initialize gate weights with smaller values or add weight normalization.\n\n")
        
        # Final recommendations
        f.write("Final Recommendations:\n")
        f.write("1. Use float32 for all intermediate calculations in the Router module.\n")
        f.write("2. Add more aggressive clipping to router logits and expert usage calculations.\n")
        f.write("3. Use smaller initialization for gate weights to prevent extreme values.\n")
        f.write("4. Consider adding layer normalization before the router to stabilize inputs.\n")
        f.write("5. Implement more robust NaN checking and handling throughout the module.\n")

if __name__ == "__main__":
    run_router_tests() 