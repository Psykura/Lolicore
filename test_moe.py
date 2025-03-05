import jax
import jax.numpy as jnp
from flax import linen as nn
from transformer import ExpertsFeedForward
import numpy as np
import time

def run_benchmark(batch_size, seq_len, d_model, hidden_size, num_experts, num_shared_experts, num_runs=5):
    """Run benchmark comparing training vs evaluation modes with specific dimensions."""
    # Initialize random key
    key = jax.random.PRNGKey(0)
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")
    print(f"{'='*70}")
    
    # Test both training and evaluation modes
    timing_stats = {}
    
    for training_mode in [True, False]:
        mode_name = "Training" if training_mode else "Evaluation"
        print(f"\n{'-'*20} Testing {mode_name} Mode {'-'*20}")
        timing_stats[mode_name] = []
        
        # Create model instance with appropriate training flag
        model = ExpertsFeedForward(
            d_model=d_model,
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            training=training_mode
        )
        
        # Create dummy input
        x = jax.random.normal(key, (batch_size, seq_len, d_model))
        
        # Initialize parameters (do this separately for each mode to avoid sharing state)
        variables = model.init(key, x)
        
        # Define apply function
        def apply_fn(x):
            output, router_loss = model.apply(
                variables, 
                x, 
            )
            return output, router_loss
        
        # JIT compile the forward pass
        jitted_apply = jax.jit(apply_fn)
        
        # Warmup run to compile
        print(f"Compiling {mode_name} mode...")
        warmup_out = jitted_apply(x)
        jax.block_until_ready(warmup_out)
        print(f"Compilation complete for {mode_name} mode")
        
        # Test with different inputs
        print(f"\nRunning performance tests for {mode_name} mode...")
        for i in range(num_runs):
            key_i = jax.random.PRNGKey(i)
            x_i = jax.random.normal(key_i, (batch_size, seq_len, d_model))
            
            # Time the execution
            start_time = time.perf_counter()
            output_i, router_loss_i = jitted_apply(x_i)
            # Block until computation is complete
            jax.block_until_ready((output_i, router_loss_i))
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            timing_stats[mode_name].append(execution_time)
            
            # Print progress for every run
            print(f"Run {i+1}/{num_runs}: {execution_time:.2f} ms")
            
            # Basic checks
            assert x_i.shape == output_i.shape, f"Shape mismatch: input {x_i.shape} != output {output_i.shape}"
            assert not jnp.allclose(output_i, 0), "Output is all zeros"
            assert not jnp.any(jnp.isnan(output_i)), "Output contains NaN values"
        
        # Calculate stats
        timing_array = np.array(timing_stats[mode_name])
        avg_time = np.mean(timing_array)
        std_time = np.std(timing_array)
        min_time = np.min(timing_array)
        max_time = np.max(timing_array)
        
        print(f"\n{mode_name} tests passed! ✓")
        print(f"Average execution time: {avg_time:.2f} ms (±{std_time:.2f} ms)")
        print(f"Min: {min_time:.2f} ms, Max: {max_time:.2f} ms")
    
    # Compare methods
    train_avg = np.mean(timing_stats["Training"])
    eval_avg = np.mean(timing_stats["Evaluation"])
    speedup = (train_avg / eval_avg) if eval_avg > 0 else 0
    
    print("\n" + "="*50)
    print("Performance Comparison:")
    print("="*50)
    for mode in timing_stats:
        avg = np.mean(timing_stats[mode])
        std = np.std(timing_stats[mode])
        print(f"{mode:12s}: {avg:.2f} ms (±{std:.2f} ms)")
    
    print("\nSpeedup from Training to Evaluation: {:.2f}x".format(speedup))
    percent_change = (speedup-1)*100
    if percent_change > 0:
        print("Evaluation is {:.1f}% faster than Training".format(percent_change))
    else:
        print("Training is {:.1f}% faster than Evaluation".format(-percent_change))
    
    return timing_stats

def test_experts_feed_forward():
    # Common parameters
    d_model = 128
    hidden_size = 256
    num_experts = 8
    num_shared_experts = 1
    
    # More granular sequence length tests to find crossover point
    # Focus on batch_size=1 with varying sequence lengths
    crossover_scenarios = [
        {"batch_size": 1, "seq_len": 16, "name": "Batch=1, SeqLen=16"},
        {"batch_size": 1, "seq_len": 24, "name": "Batch=1, SeqLen=24"},
        {"batch_size": 1, "seq_len": 32, "name": "Batch=1, SeqLen=32"},
        {"batch_size": 1, "seq_len": 40, "name": "Batch=1, SeqLen=40"},
        {"batch_size": 1, "seq_len": 48, "name": "Batch=1, SeqLen=48"},
        {"batch_size": 1, "seq_len": 56, "name": "Batch=1, SeqLen=56"},
        {"batch_size": 1, "seq_len": 64, "name": "Batch=1, SeqLen=64"},
    ]
    
    # Results storage
    all_results = {}
    
    # Run crossover benchmarks
    print(f"\n\n{'#'*80}")
    print(f"# FINDING CROSSOVER POINT BETWEEN TRAINING AND EVALUATION EFFICIENCY")
    print(f"{'#'*80}")
    
    for scenario in crossover_scenarios:
        results = run_benchmark(
            batch_size=scenario["batch_size"],
            seq_len=scenario["seq_len"],
            d_model=d_model,
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            num_runs=3  # Fewer runs to save time
        )
        
        all_results[scenario["name"]] = {
            "training": np.mean(results["Training"]),
            "evaluation": np.mean(results["Evaluation"]),
            "speedup": np.mean(results["Training"]) / np.mean(results["Evaluation"]),
            "seq_len": scenario["seq_len"]
        }
    
    # Final summary
    print("\n\n" + "="*80)
    print("CROSSOVER POINT ANALYSIS")
    print("="*80)
    print(f"{'Scenario':<20} {'SeqLen':<10} {'Training (ms)':<15} {'Evaluation (ms)':<15} {'Speedup':<10} {'Winner':<10}")
    print("-"*80)
    
    for name, data in all_results.items():
        winner = "Eval" if data["speedup"] > 1 else "Training"
        print(f"{name:<20} {data['seq_len']:<10} {data['training']:<15.2f} {data['evaluation']:<15.2f} {data['speedup']:<10.2f} {winner:<10}")
    
    # Identify crossover point through analysis
    seq_lengths = [data["seq_len"] for _, data in all_results.items()]
    speedups = [data["speedup"] for _, data in all_results.items()]
    
    # Find where speedup crosses 1.0
    crossover_seq_len = None
    for i in range(len(speedups) - 1):
        if (speedups[i] >= 1.0 and speedups[i+1] < 1.0) or (speedups[i] < 1.0 and speedups[i+1] >= 1.0):
            # Linear interpolation to estimate more precise crossover point
            seq_len1, seq_len2 = seq_lengths[i], seq_lengths[i+1]
            speedup1, speedup2 = speedups[i], speedups[i+1]
            
            # Interpolate to find where speedup = 1.0
            if speedup1 != speedup2:  # Avoid division by zero
                crossover_seq_len = seq_len1 + (1.0 - speedup1) * (seq_len2 - seq_len1) / (speedup2 - speedup1)
            else:
                crossover_seq_len = (seq_len1 + seq_len2) / 2
    
    if crossover_seq_len is not None:
        print("\nEstimated crossover sequence length: {:.1f} tokens".format(crossover_seq_len))
        print("At this sequence length, training and evaluation modes have equivalent performance.")
        print("For shorter sequences, evaluation mode is faster.")
        print("For longer sequences, training mode is faster.")
    else:
        print("\nNo crossover point detected in the tested range.")

if __name__ == "__main__":
    test_experts_feed_forward() 