import jax
import jax.numpy as jnp
import time
from functools import partial
import numpy as np
import signal
import sys

def print_device_info():
    """Print information about the available devices."""
    print(f"JAX devices: {jax.devices()}")
    print(f"Default device: {jax.default_backend()}")

def initialize_tpu():
    """Initialize TPU and return True if successful."""
    try:
        # Try to initialize TPU
        jax.devices('tpu')
        print("TPU initialized successfully!")
        return True
    except RuntimeError as e:
        print(f"TPU initialization failed: {e}")
        print("Running on available device instead.")
        return False

@partial(jax.jit, backend='tpu')
def matrix_multiply(a, b):
    """Perform matrix multiplication on TPU."""
    return jnp.matmul(a, b)

def stress_test(matrix_size=8192, num_iterations=100, warmup=10):
    """
    Run a stress test on TPU with matrix multiplication.
    
    Args:
        matrix_size: Size of the square matrices to multiply
        num_iterations: Number of multiplication iterations
        warmup: Number of warmup iterations (not counted in timing)
    """
    print(f"Generating random matrices of size {matrix_size}x{matrix_size}...")
    
    # Use a fixed seed for reproducibility
    key = jax.random.PRNGKey(42)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Generate large random matrices
    a = jax.random.normal(subkey1, (matrix_size, matrix_size))
    b = jax.random.normal(subkey2, (matrix_size, matrix_size))
    
    print("Starting warmup iterations...")
    # Warmup to compile and load the computation to TPU
    for i in range(warmup):
        result = matrix_multiply(a, b)
        # Force execution of the computation
        result.block_until_ready()
    
    print(f"Running {num_iterations} iterations of matrix multiplication...")
    
    # Measure performance
    start_time = time.time()
    
    for i in range(num_iterations):
        # Generate new random keys for each iteration
        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        # Update matrices with new random values to prevent optimization
        a = a * jax.random.normal(subkey1, (1, 1))
        b = b * jax.random.normal(subkey2, (1, 1))
        
        result = matrix_multiply(a, b)
        # Force execution of the computation
        result.block_until_ready()
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_iterations} iterations")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    ops = 2 * (matrix_size ** 3) * num_iterations  # Approximate FLOPs for matrix multiplication
    tflops = ops / (elapsed_time * 10**12)
    
    print(f"\nTest completed in {elapsed_time:.2f} seconds")
    print(f"Average time per iteration: {elapsed_time / num_iterations:.4f} seconds")
    print(f"Estimated performance: {tflops:.2f} TFLOPS")
    
    # Return a small sample of the result to verify correctness
    return result[0, 0]

def continuous_stress_test(matrix_size=8192):
    """
    Run a continuous stress test until interrupted with Ctrl+C.
    
    Args:
        matrix_size: Size of the square matrices to multiply
    """
    print(f"\n=== Starting Continuous TPU Stress Test (Press Ctrl+C to stop) ===\n")
    
    # Register signal handler for clean exit
    def signal_handler(sig, frame):
        print("\n\n=== Stress Test Interrupted by User ===")
        print(f"Completed {iteration_count} total iterations")
        print(f"Total runtime: {time.time() - global_start_time:.2f} seconds")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Use a fixed seed for initial matrices
    key = jax.random.PRNGKey(42)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Generate large random matrices
    print(f"Generating random matrices of size {matrix_size}x{matrix_size}...")
    a = jax.random.normal(subkey1, (matrix_size, matrix_size))
    b = jax.random.normal(subkey2, (matrix_size, matrix_size))
    
    # Warmup to compile and load the computation to TPU
    print("Starting warmup iterations...")
    for i in range(10):
        result = matrix_multiply(a, b)
        result.block_until_ready()
    
    print("Beginning continuous stress test. Press Ctrl+C to stop.")
    
    # Track global stats
    global_start_time = time.time()
    iteration_count = 0
    last_report_time = global_start_time
    iterations_since_report = 0
    
    try:
        while True:
            # Generate new random keys
            key, subkey1, subkey2 = jax.random.split(key, 3)
            
            # Update matrices with new random values
            a = a * jax.random.normal(subkey1, (1, 1))
            b = b * jax.random.normal(subkey2, (1, 1))
            
            # Perform matrix multiplication
            result = matrix_multiply(a, b)
            result.block_until_ready()
            
            iteration_count += 1
            iterations_since_report += 1
            
            # Report progress every 10 seconds
            current_time = time.time()
            if current_time - last_report_time >= 10:
                elapsed = current_time - last_report_time
                ops = 2 * (matrix_size ** 3) * iterations_since_report
                tflops = ops / (elapsed * 10**12)
                
                print(f"Status: {iteration_count} iterations completed | " 
                      f"Running for {current_time - global_start_time:.1f} seconds | "
                      f"Current performance: {tflops:.2f} TFLOPS")
                
                last_report_time = current_time
                iterations_since_report = 0
                
    except Exception as e:
        print(f"\nError during stress test: {e}")
        print(f"Completed {iteration_count} iterations before error")
        print(f"Total runtime: {time.time() - global_start_time:.2f} seconds")

if __name__ == "__main__":
    print_device_info()
    tpu_available = initialize_tpu()
    
    # Adjust matrix size based on available memory
    # For TPUs, we can use larger matrices
    matrix_size = 8192 if tpu_available else 4096
    
    # Run continuous stress test until interrupted
    continuous_stress_test(matrix_size=matrix_size)