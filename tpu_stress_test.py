import jax
import jax.numpy as jnp
import time
from functools import partial
import numpy as np

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

if __name__ == "__main__":
    print_device_info()
    tpu_available = initialize_tpu()
    
    # Adjust matrix size based on available memory
    # For TPUs, we can use larger matrices
    matrix_size = 8192 if tpu_available else 4096
    
    print("\n=== Starting TPU Stress Test ===\n")
    
    try:
        result_sample = stress_test(
            matrix_size=matrix_size,
            num_iterations=100,
            warmup=10
        )
        print(f"Sample result value: {result_sample}")
        print("\n=== Stress Test Completed Successfully ===")
    except Exception as e:
        print(f"Error during stress test: {e}")
        print("\n=== Stress Test Failed ===") 