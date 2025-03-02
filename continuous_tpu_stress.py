import os
import time
import signal
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
import psutil
import threading
import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Continuous TPU Stress Test')
parser.add_argument('--run_time', type=int, default=0, help='Total run time in minutes (0 for indefinite)')
parser.add_argument('--matrix_size', type=int, default=8192, help='Size of matrices to multiply')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for matrix operations')
parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type (float32, bfloat16)')
parser.add_argument('--profile_interval', type=int, default=0, help='Interval in minutes between profiles (0 to disable)')
parser.add_argument('--profile_dir', type=str, default='./profiles', help='Directory to save profiles')
parser.add_argument('--stats_interval', type=int, default=5, help='Interval in minutes to log performance stats')
args = parser.parse_args()

# Set up data type
if args.dtype == 'bfloat16':
    DTYPE = jnp.bfloat16
elif args.dtype == 'float32':
    DTYPE = jnp.float32
else:
    raise ValueError(f"Unsupported dtype: {args.dtype}")

# Global variables
running = True
iteration_count = 0
start_time = time.time()

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global running
    print("\nShutdown signal received. Completing current iteration and exiting...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Create a function that performs intensive matrix operations
@jax.jit(static_argnums=(1, 2))
def matrix_multiply_stress(key, size, batch_size):
    # Generate random matrices
    key1, key2 = random.split(key)
    matrix_a = random.normal(key1, (batch_size, size, size), dtype=DTYPE)
    matrix_b = random.normal(key2, (batch_size, size, size), dtype=DTYPE)
    
    # Perform matrix multiplication (very intensive operation)
    result = jnp.matmul(matrix_a, matrix_b)
    
    # Add some more operations to increase computational intensity
    result = jnp.sin(result) + jnp.cos(result)
    result = jnp.exp(jnp.tanh(result * 0.01))
    
    return result, key2

# Create a function that performs intensive JAX operations (replacing the Flax model)
@jax.jit(static_argnums=(1, 2))
def jax_intensive_operations(key, size, batch_size):
    # Generate random matrices
    key1, key2 = random.split(key)
    x = random.normal(key1, (batch_size, size, size), dtype=DTYPE)
    
    # Series of intensive JAX operations
    # First layer equivalent
    w1 = random.normal(key2, (size, size), dtype=DTYPE)
    b1 = random.normal(random.split(key2)[0], (size,), dtype=DTYPE)
    y1 = jnp.matmul(x, w1) + b1
    y1 = jnp.maximum(0, y1)  # ReLU
    
    # Second layer equivalent
    key2, subkey = random.split(key2)
    w2 = random.normal(subkey, (size, size), dtype=DTYPE)
    b2 = random.normal(random.split(subkey)[0], (size,), dtype=DTYPE)
    y2 = jnp.matmul(y1, w2) + b2
    y2 = jnp.maximum(0, y2)  # ReLU
    
    # Third layer equivalent
    key2, subkey = random.split(key2)
    w3 = random.normal(subkey, (size, size), dtype=DTYPE)
    b3 = random.normal(random.split(subkey)[0], (size,), dtype=DTYPE)
    y3 = jnp.matmul(y2, w3) + b3
    
    # Additional operations to stress the TPU
    y3 = jnp.sin(y3) + jnp.cos(y3)
    result = jnp.exp(jnp.tanh(y3 * 0.01))
    
    return result, key2

# Function to log system stats
def log_stats():
    global iteration_count, start_time
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Calculate iterations per second
    iterations_per_second = iteration_count / elapsed_time if elapsed_time > 0 else 0
    
    # Get CPU usage
    cpu_percent = psutil.cpu_percent()
    
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024 ** 3)
    memory_total_gb = memory.total / (1024 ** 3)
    
    # Log stats
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- Stats at {timestamp} ---")
    print(f"Running for: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Total iterations: {iteration_count}")
    print(f"Performance: {iterations_per_second:.2f} iterations/second")
    print(f"CPU usage: {cpu_percent:.1f}%")
    print(f"Memory usage: {memory_used_gb:.2f}GB / {memory_total_gb:.2f}GB ({memory.percent:.1f}%)")
    print("------------------------\n")

# Function to run profiling
def run_profiling():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_path = os.path.join(args.profile_dir, f"tpu_stress_{timestamp}")
    
    print(f"\nStarting profiling, saving to {profile_path}")
    jax.profiler.start_trace(profile_path)
    
    # Profile for 10 seconds
    time.sleep(10)
    
    jax.profiler.stop_trace()
    print(f"Profiling data saved to {profile_path}\n")

# Main stress test function
def run_continuous_stress_test():
    global running, iteration_count
    
    print(f"Starting continuous TPU stress test with {args.matrix_size}x{args.matrix_size} matrices")
    print(f"Using {args.dtype} precision")
    
    if args.run_time > 0:
        print(f"Will run for {args.run_time} minutes")
    else:
        print("Will run indefinitely until stopped with Ctrl+C")
    
    # Print JAX device information
    print(f"JAX devices: {jax.devices()}")
    print(f"Number of devices: {jax.device_count()}")
    
    # Create directories
    if args.profile_interval > 0:
        os.makedirs(args.profile_dir, exist_ok=True)
    
    # Initialize random key
    key = random.key(42)
    
    # Warm up TPU
    print("Warming up TPU...")
    for _ in range(5):
        _, key = matrix_multiply_stress(key, args.matrix_size, args.batch_size)
    
    print("Starting continuous stress test...")
    
    # Set up timers
    last_profile_time = time.time()
    last_stats_time = time.time()
    
    # Calculate end time if run_time is specified
    end_time = time.time() + (args.run_time * 60) if args.run_time > 0 else float('inf')
    
    # Main loop
    while running and time.time() < end_time:
        # Alternate between different JAX operations to stress the TPU
        if iteration_count % 2 == 0:
            _, key = matrix_multiply_stress(key, args.matrix_size, args.batch_size)
        else:
            _, key = jax_intensive_operations(key, args.matrix_size // 4, args.batch_size * 2)
        
        iteration_count += 1
        
        # Log stats periodically
        if args.stats_interval > 0 and time.time() - last_stats_time >= args.stats_interval * 60:
            log_stats()
            last_stats_time = time.time()
        
        # Run profiling periodically
        if args.profile_interval > 0 and time.time() - last_profile_time >= args.profile_interval * 60:
            run_profiling()
            last_profile_time = time.time()
        
        # Print a dot every 10 iterations to show progress without flooding the console
        if iteration_count % 10 == 0:
            print(".", end="", flush=True)
        
        # Print a newline every 500 dots for readability
        if iteration_count % 5000 == 0:
            elapsed = time.time() - start_time
            iterations_per_second = iteration_count / elapsed if elapsed > 0 else 0
            print(f" {iteration_count} iterations ({iterations_per_second:.2f}/s)")
    
    # Final stats
    log_stats()
    
    print("\nStress test completed.")
    print(f"Ran for {(time.time() - start_time) / 60:.2f} minutes")
    print(f"Completed {iteration_count} iterations")

if __name__ == "__main__":
    run_continuous_stress_test() 