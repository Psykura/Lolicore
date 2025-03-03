import os

import jax.tools
import jax.tools.build_utils
import jax.tools.colab_tpu
if os.path.exists('transformer.py'):
    from transformer import Transformer

import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from flax.training import train_state
from typing import Dict, Tuple
from functools import partial
from flax import jax_utils
from flax.training import checkpoints
from tqdm import tqdm
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import wandb
from jax.experimental.multihost_utils import sync_global_devices
import numpy as np
import collections
import itertools

# This script trains a Transformer model with MoE (Mixture of Experts) architecture
# It supports checkpoint saving and loading for resuming training from the last saved checkpoint

# Constants
CONTEXT_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-5
WARMUP_STEPS = 100
GRADIENT_CLIP_NORM = 0.5
BATCH_MESH_SIZE = 4
DTYPE = jnp.bfloat16  # Set default dtype to bfloat16
PARALLEL_PROCESSING = 16
TOKENIZED_DATASET_PATH = '/mnt/dataset/tokenized_dataset'

vocab_size = 50257
vocab_size = ((vocab_size + 127) // 128) * 128

# Model hyperparameters
MODEL_CONFIG = {
    'num_blocks': 6,
    'num_heads': 8,
    'd_model': 512,
    'hidden_size': 4096,
    'max_seq_length': CONTEXT_LENGTH,
    'vocab_size': vocab_size,  # GPT-2 vocab size
    'num_experts': 16,
    'num_shared_experts': 1,
    'use_gradient_checkpointing': True,
    'attention_latent_dim': 64,
    'num_constant_experts': 4,
    'num_noise_experts': 1,
}

# Dataset configuration
DATASET_CONFIG = {
    'path': "wikitext", #'HuggingFaceFW/fineweb',
    'name': "wikitext-103-v1", #'sample-10BT',
    'split': 'train',
}

def create_learning_rate_schedule(
    num_train_steps: int,
    warmup_steps: int,
    base_learning_rate: float
) -> optax.Schedule:
    """Creates learning rate schedule with warmup and cosine decay."""
    # Ensure warmup_steps is not larger than total steps
    warmup_steps = min(warmup_steps, num_train_steps)

    # Create warmup schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=warmup_steps
    )

    # Calculate remaining steps for cosine decay
    cosine_steps = max(num_train_steps - warmup_steps, 1)

    # Create cosine decay schedule
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_steps
    )

    # Combine schedules
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps]
    )

    return schedule_fn

def debug_param_sharding(params, param_shardings):
    """Print out parameter shapes and their sharding specifications for debugging."""
    print("\n=== Parameter Sharding Debug Information ===")
    print(f"{'Parameter Path':<60} {'Shape':<20} {'Size (MB)':<12} {'Sharding':<15}")
    print("-" * 110)
    
    # Flatten the parameter tree for easier iteration
    flat_params = jax.tree_util.tree_flatten_with_path(params)[0]
    flat_shardings = jax.tree_util.tree_flatten_with_path(param_shardings)[0]
    
    # Sort by parameter size (descending)
    param_info = []
    for (path, param), (_, sharding) in zip(flat_params, flat_shardings):
        path_str = '.'.join(str(p) for p in path)
        shape_str = str(param.shape)
        size_mb = param.size * param.dtype.itemsize / (1024 * 1024)
        sharding_str = str(sharding.spec)
        param_info.append((path_str, shape_str, size_mb, sharding_str))
    
    # Sort by size (largest first)
    param_info.sort(key=lambda x: x[2], reverse=True)
    
    # Print information
    total_size_mb = 0
    for path_str, shape_str, size_mb, sharding_str in param_info:
        print(f"{path_str[:57]+'...' if len(path_str) > 60 else path_str:<60} {shape_str:<20} {size_mb:<12.2f} {sharding_str:<15}")
        total_size_mb += size_mb
    
    print("-" * 110)
    print(f"Total parameter size: {total_size_mb:.2f} MB")
    print("=" * 110 + "\n")

def create_train_state(
    rng: jax.random.PRNGKey,
    mesh: Mesh,
    learning_rate_fn: optax.Schedule,
    **kwargs
) -> train_state.TrainState:
    """Creates initial `TrainState`."""
    model = Transformer(
        dtype=DTYPE,
        training=True,
        **kwargs
    )

    # Initialize model
    rng, params_rng, dropout_rng, noise_rng = jax.random.split(rng, 4)
    rngs = {
        'params': params_rng,
        'dropout': dropout_rng,
        'noise': noise_rng
    }

    # Initialize model on CPU explicitly
    print("Initializing model on CPU...")
    # Create dummy inputs on CPU
    cpu_dummy_input = jnp.ones((BATCH_SIZE, CONTEXT_LENGTH), dtype=jnp.int32)
    cpu_dummy_mask = jnp.ones((BATCH_SIZE, CONTEXT_LENGTH), dtype=jnp.int32)
    
    # Get CPU device
    cpu_device = jax.devices("cpu")[0]
    
    # Initialize on CPU using explicit device_put
    cpu_dummy_input = jax.device_put(cpu_dummy_input, cpu_device)
    cpu_dummy_mask = jax.device_put(cpu_dummy_mask, cpu_device)
    
    # Initialize with explicit CPU placement
    with jax.default_device(cpu_device):
        cpu_variables = model.init(
            rngs,
            cpu_dummy_input,
            cpu_dummy_mask,
        )
    print("CPU initialization complete")

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP_NORM),
        optax.adamw(
            learning_rate=learning_rate_fn,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.001,
            mu_dtype=jnp.float32,  # Use float32 for momentum
        )
    )

    with mesh:
        print("Transferring parameters to mesh...")
        # Create parameter shardings
        param_shardings = jax.tree.map_with_path(
            lambda path, p: NamedSharding(mesh, get_param_spec(p, path)),
            cpu_variables['params']
        )
        
        # Print parameters with inconsistent ndim and spec shape
        def check_param_spec_consistency(path, param, sharding):
            spec = sharding.spec
            # Count non-None dimensions in spec
            spec_ndim = sum(1 for axis in spec if axis is not None)
            if spec_ndim != param.ndim:
                print(f"Inconsistent parameter: {path}")
                print(f"  Shape: {param.shape} (ndim={param.ndim})")
                print(f"  Spec: {spec} (effective ndim={spec_ndim})")
            return sharding
            
        if jax.process_index() == 0:  # Only print on main process
            print("Parameters with inconsistent dimensions:")
            jax.tree.map_with_path(
                lambda path, p, s: check_param_spec_consistency(path, p, s),
                cpu_variables['params'],
                param_shardings
            )

        # Debug parameter sharding decisions
        if jax.process_index() == 0:  # Only print on main process
            debug_param_sharding(cpu_variables['params'], param_shardings)

        # Transfer CPU parameters to mesh with appropriate sharding
        sharded_params = jax.device_put(cpu_variables['params'], param_shardings)
        print("Parameter transfer complete")
        
        # Initialize state with sharded parameters
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=sharded_params,
            tx=optimizer
        )

    return state

def prepare_dataset(tokenizer):
    """Prepare dataset with tokenization and chunking."""
    if os.path.exists(TOKENIZED_DATASET_PATH):
        tokenized_dataset = load_from_disk(TOKENIZED_DATASET_PATH)
        print(f"Loaded tokenized dataset from disk with {len(tokenized_dataset)} examples")
        # Use numpy format for dataset   
        return tokenized_dataset, len(tokenized_dataset)

    dataset = load_dataset(**DATASET_CONFIG, num_proc=PARALLEL_PROCESSING)
    print(f"Raw dataset size: {len(dataset)}")

    def tokenize_and_chunk(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=False,
            add_special_tokens=True
        )

        chunks = []
        attention_masks = []

        for input_ids in tokenized['input_ids']:
            if len(input_ids) < 32 or not any(input_ids):
                continue

            # Split into chunks of exactly context_length
            for i in range(0, len(input_ids), CONTEXT_LENGTH):
                chunk = input_ids[i:i + CONTEXT_LENGTH]
                
                # Skip if too short
                if len(chunk) < 32:
                    continue
                    
                # Handle padding
                if len(chunk) < CONTEXT_LENGTH:
                    padding_length = CONTEXT_LENGTH - len(chunk)
                    attention_mask = [1] * len(chunk) + [0] * padding_length
                    chunk = chunk + [tokenizer.eos_token_id] * padding_length
                else:
                    # Truncate if longer than CONTEXT_LENGTH
                    chunk = chunk[:CONTEXT_LENGTH]
                    attention_mask = [1] * CONTEXT_LENGTH

                chunks.append(chunk)
                attention_masks.append(attention_mask)

        return {
            'input_ids': chunks,
            'attention_mask': attention_masks,
            'labels': chunks
        }

    # Use sequential processing without caching
    tokenized_dataset = dataset.map(
        tokenize_and_chunk,
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=PARALLEL_PROCESSING,
        cache_file_name=None  # Disable caching
    )

    print(f"Processed dataset size: {len(tokenized_dataset)}")
    tokenized_dataset.save_to_disk(TOKENIZED_DATASET_PATH)
    
    # Add these optimizations before returning:
    # Use numpy format instead of JAX
    return tokenized_dataset, len(tokenized_dataset)

def create_batch(mesh, inputs):
    """Create a sharded batch from dataset examples."""
    # Convert to JAX arrays and ensure attention_mask is properly shaped
    examples = {}
    for k, v in inputs.items():
        if k == 'attention_mask':
            # Ensure each attention mask is exactly CONTEXT_LENGTH
            masks = []
            for mask in v:
                if len(mask) > CONTEXT_LENGTH:
                    # Truncate if longer than context length
                    masks.append(mask[:CONTEXT_LENGTH])
                elif len(mask) < CONTEXT_LENGTH:
                    # Pad with zeros if shorter
                    padding = [0] * (CONTEXT_LENGTH - len(mask))
                    masks.append(list(mask) + padding)
                else:
                    masks.append(mask)
            examples[k] = jnp.array(masks)
        else:
            examples[k] = jnp.array(v)
    
    # Check if we're using a 3D mesh with expert dimension
    mesh_axes = mesh.axis_names
    has_expert_dim = 'expert' in mesh_axes
    
    # Handle different types of batch data structures
    if isinstance(examples, dict):
        # For dictionary of arrays (typical dataset format)
        sharded_examples = {}
        for key, value in examples.items():
            # Determine appropriate sharding based on array dimensionality
            if hasattr(value, 'ndim'):
                ndim = value.ndim
                # Create appropriate sharding spec based on tensor rank and mesh dimensions
                if has_expert_dim:
                    # For 3D mesh ('expert', 'model', 'batch')
                    if ndim == 3:
                        # For 3D tensors, can use full 3D sharding
                        spec = P('expert', 'model', 'batch')
                    elif ndim == 2:
                        # For 2D tensors, use only 2 dimensions of the mesh
                        # Shard on model and batch dimensions
                        spec = P('model', 'batch')
                    elif ndim == 1:
                        # For 1D tensors, use only 1 dimension or replicate
                        spec = P('model')
                    else:
                        # For scalars, replicate
                        spec = P(None)
                else:
                    # For 2D mesh ('model', 'batch')
                    if ndim >= 2:
                        # For 2D+ tensors, shard on batch dimension
                        spec = P('batch', None)
                    elif ndim == 1:
                        # For 1D tensors, replicate
                        spec = P(None)
                    else:
                        # For scalars, replicate
                        spec = P(None)
                
                # Apply sharding
                sharded_examples[key] = jax.device_put(value, NamedSharding(mesh, spec))
            else:
                # For non-array values, just pass through
                sharded_examples[key] = value
        
        return sharded_examples
    else:
        # For single array or other data structure
        if hasattr(examples, 'ndim'):
            ndim = examples.ndim
            
            # Create appropriate sharding spec based on tensor rank and mesh dimensions
            if has_expert_dim:
                # For 3D mesh ('expert', 'model', 'batch')
                if ndim == 3:
                    # For 3D tensors, can use full 3D sharding
                    spec = P('expert', 'model', 'batch')
                elif ndim == 2:
                    # For 2D tensors, use only 2 dimensions of the mesh
                    # Shard on model and batch dimensions
                    spec = P('model', 'batch')
                elif ndim == 1:
                    # For 1D tensors, use only 1 dimension or replicate
                    spec = P('model')
                else:
                    # For scalars, replicate
                    spec = P(None)
            else:
                # For 2D mesh ('model', 'batch')
                if ndim >= 2:
                    # For 2D+ tensors, shard on batch dimension
                    spec = P('batch', None)
                elif ndim == 1:
                    # For 1D tensors, replicate
                    spec = P(None)
                else:
                    # For scalars, replicate
                    spec = P(None)
            
            # Apply sharding
            return jax.device_put(examples, NamedSharding(mesh, spec))
        else:
            # For non-array values, just pass through
            return examples

@jax.jit
def train_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray], rngs: jax.random.PRNGKey):
    """Perform a single training step with model and data parallelism."""
    #rngs, noise_rng = jax.random.split(rngs)
    noise_rng = rngs

    def loss_fn(params):
        logits, router_loss = state.apply_fn(
            {'params': params},
            batch['input_ids'],
            batch['attention_mask'],
            rngs={'noise': noise_rng}
        )

        # Add loss masking using attention_mask
        loss_mask = batch['attention_mask'][..., :-1]  # Fix alignment
        shift_logits = logits[..., :-1, :]
        shift_labels = batch['labels'][..., 1:]
        # Cast to float32 for loss calculation
        shift_logits = shift_logits.astype(jnp.float32)
        # Convert labels to one-hot encoding for cross entropy
        shift_labels_one_hot = jax.nn.one_hot(shift_labels, num_classes=logits.shape[-1])
        # Calculate cross entropy and reduce to scalar by taking mean
        main_loss = optax.softmax_cross_entropy(
            shift_logits, 
            shift_labels_one_hot
        )
        main_loss = main_loss * loss_mask
        main_loss = jnp.sum(main_loss) / (jnp.sum(loss_mask) + 1e-5)  # Larger epsilon

        total_loss = main_loss + router_loss

        return total_loss, (main_loss, router_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, aux), grads = grad_fn(state.params)

    # Update model
    new_state = state.apply_gradients(grads=grads)

    metrics = {
        'loss': total_loss,
        'main_loss': aux[0],
        'router_loss': aux[1]
    }

    return new_state, metrics

def create_mesh():
    """Create an optimized device mesh for training MoE models.
    
    For MoE models, we use a 3D mesh with dimensions:
    - 'expert': to shard experts across devices
    - 'model': to shard model parameters
    - 'batch': to shard batch dimension
    
    This allows for more efficient expert parallelism in addition to
    model and data parallelism.
    """
    devices = jax.devices()
    n_devices = len(devices)
    
    # Determine optimal mesh shape based on number of devices
    # For MoE models, we want to prioritize expert parallelism
    
    # Get number of experts from model config
    num_experts = MODEL_CONFIG['num_experts']
    
    # Calculate factors of n_devices to find optimal mesh shape
    factors = []
    for i in range(1, int(n_devices**0.5) + 1):
        if n_devices % i == 0:
            factors.append((i, n_devices // i))
    
    # Choose mesh shape based on number of experts and available devices
    if n_devices >= 8:
        # For 8+ devices, use 3D mesh with expert dimension
        
        # Find best expert dimension size (closest to num_experts but not exceeding n_devices)
        expert_dim = 1
        for i in range(1, min(num_experts, n_devices) + 1):
            if n_devices % i == 0:
                expert_dim = i
        
        # Calculate remaining dimensions
        remaining_devices = n_devices // expert_dim
        
        # Find optimal split for model and batch dimensions
        model_dim = max(1, int(remaining_devices**0.5))
        while remaining_devices % model_dim != 0:
            model_dim -= 1
        
        batch_dim = remaining_devices // model_dim
        
        # Create 3D mesh
        mesh_shape = (expert_dim, model_dim, batch_dim)
        mesh = jax.make_mesh(mesh_shape, ('expert', 'model', 'batch'))
        
        print(f"Using 3D mesh with shape: expert={expert_dim}, model={model_dim}, batch={batch_dim}")
    else:
        # For fewer devices, fall back to 2D mesh
        model_dim = max(factors, key=lambda x: abs(x[0] - x[1]))[0]
        batch_dim = n_devices // model_dim
        
        mesh_shape = (model_dim, batch_dim)
        mesh = jax.make_mesh(mesh_shape, ('model', 'batch'))
        
        print(f"Using 2D mesh with shape: model={model_dim}, batch={batch_dim}")
    
    return mesh, n_devices

# Update get_param_spec to handle 3D mesh with expert dimension
def get_param_spec(param, path):
    """Get parameter sharding specification based on parameter path and shape."""
    path_str = str(path).lower()
    
    # Check if we're using a 3D mesh with expert dimension
    mesh_axes = jax.devices().mesh_axes if hasattr(jax.devices(), 'mesh_axes') else None
    has_expert_dim = mesh_axes is not None and 'expert' in mesh_axes
    
    # 1. Output projection and embedding layers - split across all dimensions
    if 'output_proj' in path_str and 'kernel' in path_str:
        return P('expert', 'model', 'batch') if has_expert_dim else P('model', 'batch')
    
    if 'token_embedding' in path_str and 'embedding' in path_str:
        return P('expert', 'model', None) if has_expert_dim else P('model', None)

    # 2. Attention projection layers - add kernel check
    if 'attention' in path_str:
        if has_expert_dim:
            if 'q_proj' in path_str or 'k_proj' in path_str or 'v_proj' in path_str:
                if 'kernel' in path_str:  # Only shard kernel matrices
                    return P('expert', None, 'model')
            if 'out_proj' in path_str and 'kernel' in path_str:
                return P('expert', 'model', None)
        else:
            # Only apply to kernel parameters, not biases
            if 'out_proj' in path_str and 'kernel' in path_str:
                return P('model', 'batch')

    # 3. Router parameters
    if 'router' in path_str and 'gate' in path_str and 'kernel' in path_str:
        return P('expert', 'model', None) if has_expert_dim else P('model', None)

    # 4. Expert FFN layers
    if 'experts' in path_str and ('keys' in path_str or 'values' in path_str) and 'kernel' in path_str:
        if has_expert_dim:
            return P('expert', 'model', None)
        else:
            return P('model', 'batch')

    # 5. Layer normalization parameters - always replicate 1D params
    if 'norm' in path_str or 'layernorm' in path_str or 'rmsnorm' in path_str:
        return P(None)  # Changed from 3D None spec
    
    # 6. Size-based fallbacks - replicate small params
    if param.size < 10_000:
        return P(None)  # Changed from 3D None spec
    
    # 7. Handle 1D parameters safely
    if param.ndim <= 1:
        return P(None)  # Changed from sharding along 'model'
    
    # Default for remaining matrices
    return P('batch', 'model')

def prefetch(iterator, size):
    queue = collections.deque()

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(iterator, n):
            queue.append(data)

    enqueue(size)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)

def create_prefetch_batches(dataset, indices, samples_per_step, mesh, num_prefetch=4):
    """Creates an iterator that prefetches batches while training."""
    # Create batch indices
    batch_indices = [indices[i:i + samples_per_step] 
                    for i in range(0, len(indices), samples_per_step)]
    
    # Filter out incomplete batches
    batch_indices = [idx for idx in batch_indices if len(idx) == samples_per_step]
    
    def _prepare_batch(batch_idx):
        batch_examples = dataset[batch_idx]
        return create_batch(mesh, batch_examples)
    
    # Create prefetch iterator
    batch_iter = map(_prepare_batch, batch_indices)
    return prefetch(batch_iter, num_prefetch)

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)

    # Prepare datasets - now returns tokenized dataset without batching
    tokenized_dataset, dataset_size = prepare_dataset(tokenizer)

    # if jax_smi is installed, track memory usage
    try:
        from jax_smi import initialise_tracking
        initialise_tracking()
        print("JAX SMI initialized")
    except ImportError:
        print("JAX SMI not installed, skipping memory tracking")

    CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if jax.process_index() == 0:
        # Initialize wandb
        wandb.init(
            project="lolicore",
            config={
                "context_length": CONTEXT_LENGTH,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "warmup_steps": WARMUP_STEPS,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "dtype": str(DTYPE),
            }
        )

    print(f"Syncing start state for process {jax.process_index()}")
    sync_global_devices('train')

    # Initialize TPU devices and create mesh
    mesh, n_devices = create_mesh()
    print(f"Training on {n_devices} devices with 2D sharding")

    print(f"Syncing mesh for process {jax.process_index()}")
    sync_global_devices('mesh_created')

    # Calculate steps per epoch and total steps
    samples_per_step = BATCH_SIZE * BATCH_MESH_SIZE
    steps_per_epoch = len(tokenized_dataset) // samples_per_step
    total_steps = steps_per_epoch * NUM_EPOCHS

    print(f"Original dataset size: {dataset_size}")
    print(f"Tokenized dataset size: {len(tokenized_dataset)}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")

    with mesh:
        # Create learning rate schedule
        learning_rate_fn = create_learning_rate_schedule(
            num_train_steps=total_steps,
            warmup_steps=WARMUP_STEPS,
            base_learning_rate=LEARNING_RATE
        )
        # Initialize model and state
        rng = jax.random.key(0)
        rng, init_rng = jax.random.split(rng)

        # Check for existing checkpoints
        step = 0
        latest_checkpoint = checkpoints.latest_checkpoint(CHECKPOINT_DIR)

        # Initialize training state with mesh
        state = create_train_state(
            init_rng,
            mesh=mesh,
            **MODEL_CONFIG,
            learning_rate_fn=learning_rate_fn
        )

        print(f"Syncing training state for process {jax.process_index()}")
        sync_global_devices('training_state_created')

        # Load from checkpoint if available
        if latest_checkpoint:
            print(f"Found checkpoint at {latest_checkpoint}. Restoring...")
            try:
                # Extract step from checkpoint path
                checkpoint_step = int(os.path.basename(latest_checkpoint).split("_")[-1])
                step = checkpoint_step

                # Restore checkpoint
                state = checkpoints.restore_checkpoint(latest_checkpoint, target=state)
                print(f"Successfully restored checkpoint at step {step}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Training from scratch instead.")
                step = 0
        else:
            print("No checkpoint found. Training from scratch.")

        print(f"Syncing checkpoint loaded for process {jax.process_index()}")
        sync_global_devices('checkpoint_loaded')

        # Calculate and print model size
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
        print(f"Number of parameters: {param_count/1e9:.2f}B")

        # Log model size to wandb
        if jax.process_index() == 0:
            wandb.run.summary["model_parameters_B"] = param_count/1e9

        # Calculate starting epoch and batch index based on loaded step
        start_epoch = step // steps_per_epoch
        start_batch_idx = step % steps_per_epoch

        print(f"Syncing starting training for process {jax.process_index()}")
        sync_global_devices('starting_training')

        print(f"Starting training from step {step} (epoch {start_epoch}, batch {start_batch_idx})")

        for epoch in range(start_epoch, NUM_EPOCHS):
            # Create shuffled indices for this epoch
            shuffled_indices = np.random.RandomState(seed=epoch).permutation(len(tokenized_dataset))
            
            print(f"Syncing epoch {epoch} for process {jax.process_index()}")
            sync_global_devices(f'epoch_{epoch}')

            # Create prefetching iterator for batches
            batch_iterator = create_prefetch_batches(
                tokenized_dataset,
                shuffled_indices,
                samples_per_step,
                mesh,
            )

            # Create tqdm progress bar for each epoch
            progress_bar = tqdm(
                range(steps_per_epoch),
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                position=0
            )

            # Skip batches if we're resuming from middle of an epoch
            if epoch == start_epoch and start_batch_idx > 0:
                for _ in range(start_batch_idx):
                    next(batch_iterator)
            
            for batch_idx in range(start_batch_idx if epoch == start_epoch else 0, steps_per_epoch):
                try:
                    # Get next prefetched batch
                    batch = next(batch_iterator)
                    
                    # Train step with sharding
                    state, metrics = train_step(state, batch, rngs=rng)

                    # Update progress bar and logging (existing code)
                    progress_bar.update(1)

                    if batch_idx % 50 == 0:
                        metrics = {
                            'loss': float(metrics['loss']),
                            'main_loss': float(metrics['main_loss']),
                            'router_loss': float(metrics['router_loss'])
                        }
                        progress_bar.set_postfix({
                            'loss': f"{metrics['loss']:.4f}",
                            'main_loss': f"{metrics['main_loss']:.4f}",
                            'router_loss': f"{metrics['router_loss']:.4f}"
                        })

                        if jax.process_index() == 0:
                            wandb.log({
                                'train/loss': metrics['loss'],
                                'train/main_loss': metrics['main_loss'],
                                'train/router_loss': metrics['router_loss'],
                                'train/step': step,
                                'train/epoch': epoch + (batch_idx / steps_per_epoch)
                            })

                    # Checkpoint saving (existing code)
                    if (batch_idx % 5000 == 0 or step % 5000 == 0) and step != 0:
                        print(f"\nSaving checkpoint at step {step}...")
                        checkpoints.save_checkpoint_multiprocess(
                            ckpt_dir=CHECKPOINT_DIR,
                            target=state,
                            step=step,
                            overwrite=True
                        )
                        print(f"Checkpoint saved at {os.path.join(CHECKPOINT_DIR, f'checkpoint_{step}')}")

                    step += 1
                    
                except StopIteration:
                    print(f"Reached end of dataset in epoch {epoch}")
                    break

        # Close wandb run when training is complete
        if jax.process_index() == 0:
            wandb.finish()

if __name__ == "__main__":
    main()
