import os
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import linen as nn
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='TPU Stress Test')
parser.add_argument('--duration', type=int, default=60, help='Duration of the test in seconds')
parser.add_argument('--matrix_size', type=int, default=8192, help='Size of matrices to multiply')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for matrix operations')
parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type (float32, bfloat16)')
parser.add_argument('--profile', action='store_true', help='Enable JAX profiling')
parser.add_argument('--profile_dir', type=str, default='./profiles', help='Directory to save profiles')
args = parser.parse_args()

# Set up data type
if args.dtype == 'bfloat16':
    DTYPE = jnp.bfloat16
elif args.dtype == 'float32':
    DTYPE = jnp.float32
else:
    raise ValueError(f"Unsupported dtype: {args.dtype}")

print(f"JAX devices: {jax.devices()}")
print(f"Number of devices: {jax.device_count()}")
print(f"Process index: {jax.process_index()}")
print(f"Process count: {jax.process_count()}")
print(f"Local device count: {jax.local_device_count()}")

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

# Main stress test function
def run_stress_test():
    print(f"Starting TPU stress test with {args.matrix_size}x{args.matrix_size} matrices")
    print(f"Using {args.dtype} precision")
    
    # Create device mesh for multi-device TPUs
    mesh = create_device_mesh()
    print(f"Created mesh with shape: {mesh.devices.shape}")
    
    # Initialize random key
    key = random.key(42)
    
    # Warm up TPU
    print("Warming up TPU...")
    with mesh:
        for _ in range(5):
            _, key = matrix_multiply_stress(key, args.matrix_size, args.batch_size)
    
    # Start profiling if enabled
    if args.profile:
        os.makedirs(args.profile_dir, exist_ok=True)
        profile_path = os.path.join(args.profile_dir, f"tpu_stress_{int(time.time())}")
        print(f"Starting profiling, saving to {profile_path}")
        jax.profiler.start_trace(profile_path)
    
    # Run the stress test for the specified duration
    print(f"Running stress test for {args.duration} seconds...")
    start_time = time.time()
    iteration = 0
    
    with mesh:
        with tqdm(total=args.duration) as pbar:
            while time.time() - start_time < args.duration:
                # Alternate between matrix multiplication and neural network operations
                if iteration % 2 == 0:
                    _, key = matrix_multiply_stress(key, args.matrix_size, args.batch_size)
                else:
                    _, key = create_and_apply_model(key, args.matrix_size // 4, args.batch_size * 2)
                
                # Update progress bar
                elapsed = time.time() - start_time
                pbar.update(min(elapsed - pbar.n, args.duration - pbar.n))
                pbar.set_description(f"Iteration {iteration}")
                
                iteration += 1
    
    # Stop profiling if enabled
    if args.profile:
        jax.profiler.stop_trace()
        print(f"Profiling data saved to {profile_path}")
    
    print(f"Stress test completed. Ran {iteration} iterations in {args.duration} seconds.")
    print(f"Average: {iteration / args.duration:.2f} iterations per second")

if __name__ == "__main__":
    run_stress_test() 