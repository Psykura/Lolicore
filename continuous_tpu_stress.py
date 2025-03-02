import os
import time
import signal
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import linen as nn
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

# Create a simple MLP model that will stress the TPU
class StressModel(nn.Module):
    hidden_dim: int = 4096
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim, dtype=DTYPE)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim, dtype=DTYPE)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim, dtype=DTYPE)(x)
        x = nn.relu(x)
        x = nn.Dense(features=x.shape[-1], dtype=DTYPE)(x)
        return x

# Create a function that performs intensive matrix operations
@jax.jit
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

# Create a function that uses a neural network for stress testing
def create_and_apply_model(key, size, batch_size, hidden_dim=4096):
    # Initialize model
    model = StressModel(hidden_dim=hidden_dim)
    
    # Generate random input
    key, subkey = random.split(key)
    x = random.normal(subkey, (batch_size, size, size), dtype=DTYPE)
    
    # Initialize parameters
    key, subkey = random.split(key)
    params = model.init(subkey, x)
    
    # JIT the forward pass
    @jax.jit
    def forward(params, x):
        return model.apply(params, x)
    
    # Apply model
    result = forward(params, x)
    
    return result, key

# Function to create a device mesh for multi-device TPUs
def create_device_mesh():
    devices = jax.devices()
    n_devices = len(devices)
    
    if n_devices == 1:
        # Single device
        return Mesh(np.array(devices).reshape(1, 1), ('data', 'model'))
    elif n_devices == 8:
        # TPU v2/v3 with 8 cores
        return Mesh(np.array(devices).reshape(2, 4), ('data', 'model'))
    elif n_devices == 4:
        # TPU v2/v3 with 4 cores
        return Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))
    else:
        # Generic mesh
        mesh_shape = (1, n_devices)
        return Mesh(np.array(devices).reshape(*mesh_shape), ('data', 'model'))

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
    
    # Create device mesh for multi-device TPUs
    mesh = create_device_mesh()
    print(f"Created mesh with shape: {mesh.devices.shape}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Number of devices: {jax.device_count()}")
    
    # Create directories
    if args.profile_interval > 0:
        os.makedirs(args.profile_dir, exist_ok=True)
    
    # Initialize random key
    key = random.key(42)
    
    # Warm up TPU
    print("Warming up TPU...")
    with mesh:
        for _ in range(5):
            _, key = matrix_multiply_stress(key, args.matrix_size, args.batch_size)
    
    print("Starting continuous stress test...")
    
    # Set up timers
    last_profile_time = time.time()
    last_stats_time = time.time()
    
    # Calculate end time if run_time is specified
    end_time = time.time() + (args.run_time * 60) if args.run_time > 0 else float('inf')
    
    # Main loop
    with mesh:
        while running and time.time() < end_time:
            # Alternate between matrix multiplication and neural network operations
            if iteration_count % 2 == 0:
                _, key = matrix_multiply_stress(key, args.matrix_size, args.batch_size)
            else:
                _, key = create_and_apply_model(key, args.matrix_size // 4, args.batch_size * 2)
            
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