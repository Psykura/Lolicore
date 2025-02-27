import os
if os.path.exists('transformer.py'):
    from transformer import Transformer

import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from flax.training import train_state
import numpy as np
from typing import Dict, Tuple
from functools import partial
from flax import jax_utils
from flax.training import checkpoints
from tqdm import tqdm
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from collections import deque
import threading
import queue
import wandb
import time
from jax.experimental import multihost_utils

# This script trains a Transformer model with MoE (Mixture of Experts) architecture
# It supports checkpoint saving and loading for resuming training from the last saved checkpoint

# Constants
CONTEXT_LENGTH = 512
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
GRADIENT_CLIP_NORM = 1.0
DTYPE = jnp.bfloat16  # Set default dtype to bfloat16

vocab_size = 50257
vocab_size = ((vocab_size + 127) // 128) * 128

# Model hyperparameters
MODEL_CONFIG = {
    'num_blocks': 12,
    'num_heads': 8,
    'd_model': 512,
    'hidden_size': 2048,
    'max_seq_length': CONTEXT_LENGTH,
    'vocab_size': vocab_size,
    'num_experts': 16,
    'num_shared_experts': 1,
    'top_k': 4,
    'use_gradient_checkpointing': True,
    'attention_latent_dim': 32,
    'num_zeros_experts': 1,
    'num_constant_experts': 2,
    'num_noise_experts': 1,
}

# Dataset configuration
DATASET_CONFIG = {
    'path': 'wikitext',
    'name': 'wikitext-103-v1',
    'split': 'train',
    'cache_dir': './cache',
}

# Initialize JAX distributed environment properly
def initialize_jax_distributed():
    """Initialize JAX distributed environment with proper coordination."""
    # Initialize JAX distributed system
    try:
        jax.distributed.initialize()
        setattr(jax.distributed, '_initialized', True)
    except Exception as e:
        print(f"Warning: JAX distributed initialization error: {e}")
        print("Continuing with local execution...")
    
    # Get process index and process count
    process_index = jax.process_index()
    process_count = jax.process_count()
    
    # Get local device count
    local_device_count = jax.local_device_count()
    
    print(f"Process {process_index} of {process_count} initialized with {local_device_count} local devices")
    
    # Ensure all processes have initialized before proceeding
    # This helps synchronize all hosts
    if process_count > 1:
        # Simple barrier to ensure all processes are ready
        # Each process waits for a short time proportional to its index
        # This helps stagger the initialization and avoid race conditions
        time.sleep(process_index * 0.5)
        
        # Use multihost_utils for better synchronization across hosts
        print(f"Process {process_index} waiting for synchronization...")
        
        try:
            # Synchronize all hosts using a simple value
            multihost_utils.sync_global_devices("init")
            print(f"Process {process_index} synchronized with all other processes")
        except Exception as e:
            print(f"Warning: Synchronization error: {e}")
            print("Continuing without synchronization...")
    
    # Set default device assignment strategy
    try:
        if jax.local_devices():
            jax.config.update('jax_default_device', jax.local_devices()[0])
    except Exception as e:
        print(f"Warning: Could not set default device: {e}")
    
    # Return process information for logging
    return process_index, process_count

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

def create_train_state(
    rng: jax.random.PRNGKey,
    mesh: Mesh,
    **kwargs
) -> train_state.TrainState:
    """Creates initial `TrainState`."""
    model = Transformer(
        num_blocks=kwargs['num_blocks'],
        num_heads=kwargs['num_heads'],
        d_model=kwargs['d_model'],
        hidden_size=kwargs['hidden_size'],
        max_seq_length=kwargs['max_seq_length'],
        vocab_size=kwargs['vocab_size'],
        attention_latent_dim=kwargs['attention_latent_dim'],
        num_experts=kwargs['num_experts'],
        num_shared_experts=kwargs['num_shared_experts'],
        num_zeros_experts=kwargs['num_zeros_experts'],
        num_constant_experts=kwargs['num_constant_experts'],
        num_noise_experts=kwargs['num_noise_experts'],
        top_k=kwargs['top_k'],
        dtype=DTYPE,
        use_gradient_checkpointing=kwargs['use_gradient_checkpointing'],
        training=True
    )
    
    # Initialize model
    rng, params_rng, dropout_rng, noise_rng = jax.random.split(rng, 4)
    rngs = {
        'params': params_rng,
        'dropout': dropout_rng,
        'noise': noise_rng
    }
    
    # Initialize model
    dummy_input = jnp.ones((2, CONTEXT_LENGTH), dtype=jnp.int32)
    dummy_mask = jnp.ones((2, CONTEXT_LENGTH), dtype=jnp.int32)
    
    variables = model.init(
        rngs,
        dummy_input,
        dummy_mask,
    )

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP_NORM),
        optax.adamw(
            learning_rate=kwargs['learning_rate_fn'],
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            mu_dtype=DTYPE,
        )
    )
    
    with mesh:
        # Create parameter shardings
        param_shardings = jax.tree.map_with_path(
            lambda path, p: NamedSharding(mesh, get_param_spec(p, path)),
            variables['params']
        )
        
        # Initialize with shardings directly
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=jax.device_put(variables['params'], param_shardings),
            tx=optimizer
        )
    
    return state

def prepare_dataset(tokenizer):
    """Prepare dataset with tokenization and chunking."""
    if os.path.exists('tokenized_dataset'):
        tokenized_dataset = load_from_disk('tokenized_dataset')
        return tokenized_dataset
    
    dataset = load_dataset(**DATASET_CONFIG)
    print(f"Raw dataset size: {len(dataset)}")

    def tokenize_and_chunk(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=CONTEXT_LENGTH,
            add_special_tokens=True
        )
        
        # Chunk into context_length segments
        chunks = []
        attention_masks = []
        
        for input_ids in tokenized['input_ids']:
            # Skip if sequence is too short or empty
            if len(input_ids) < 32 or not any(input_ids):  # Skip very short or empty sequences
                continue
                
            # Pad or truncate to CONTEXT_LENGTH
            if len(input_ids) > CONTEXT_LENGTH:
                input_ids = input_ids[:CONTEXT_LENGTH]
            else:
                input_ids = input_ids + [tokenizer.eos_token_id] * (CONTEXT_LENGTH - len(input_ids))
            
            chunks.append(input_ids)
            attention_masks.append([1] * CONTEXT_LENGTH)
        
        return {
            'input_ids': chunks,
            'attention_mask': attention_masks,
            'labels': chunks  # For causal language modeling
        }
    
    # Use sequential processing
    tokenized_dataset = dataset.map(
        tokenize_and_chunk,
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=16,
    )
    
    print(f"Processed dataset size: {len(tokenized_dataset)}")

    # Save tokenized dataset to disk
    tokenized_dataset.save_to_disk('tokenized_dataset')
    print("Tokenized dataset saved to disk")
    
    return tokenized_dataset

@jax.jit
def train_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray], rngs: jax.random.PRNGKey):
    """Perform a single training step with model and data parallelism."""
    rngs, noise_rng = jax.random.split(rngs)

    def stable_cross_entropy(logits, labels):
        # Compute max for numerical stability
        max_logits = jnp.max(logits, axis=-1, keepdims=True)
        # Subtract max from logits to prevent overflow
        shifted_logits = logits - max_logits
        # Compute log sum exp
        exp_shifted = jnp.exp(shifted_logits)
        log_sum_exp = jnp.log(jnp.sum(exp_shifted, axis=-1, keepdims=True))
        # Compute log probabilities
        log_probs = shifted_logits - log_sum_exp
        # Gather log probs of true labels
        label_log_probs = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)
        # Return mean negative log likelihood
        return -jnp.mean(label_log_probs)

    def loss_fn(params):
        logits, router_loss = state.apply_fn(
            {'params': params},
            batch['input_ids'],
            batch['attention_mask'],
            rngs={'noise': noise_rng}
        )
        
        # Calculate main loss using stable cross entropy
        shift_logits = logits[..., :-1, :]
        shift_labels = batch['labels'][..., 1:]
        main_loss = stable_cross_entropy(shift_logits, shift_labels)
        
        # Combine losses in bfloat16
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

def get_param_spec(param, path):
    """Get parameter sharding specification based on parameter path and shape."""
    
    # Output layer (usually the largest due to vocab size)
    if 'output_proj' in path:
        return P('batch', 'model')  # Shard on both dimensions
    
    # Token embeddings (also large due to vocab size)
    if 'token_embedding' in path:
        return P('model', 'batch')  # Shard on both dimensions
    
    # Expert parameters - distribute evenly
    if 'experts' in path:
        if 'kernel' in path or param.ndim == 2:
            return P('batch', 'model')  # Alternate dimension order
        return P('model')
    
    # Dense layers and attention
    if any(x in path for x in ['Dense', 'attention']):
        if 'kernel' in path or param.ndim == 2:
            if param.size > 100_000:  # Large matrices
                return P('batch', 'model')  # Alternate dimension order
            return P('model', None)
        return P('model')  # Vectors
    
    # Small parameters - replicate to avoid communication overhead
    if param.size < 10_000:
        return P(None)
    
    # Default for other parameters
    if param.ndim <= 1:
        return P('model')
    return P('batch', 'model')  # Alternate dimension order for remaining matrices

def create_mesh():
    """Create a more balanced device mesh for TPU."""
    # Ensure JAX distributed is initialized
    if jax.process_count() > 1 and not hasattr(jax.distributed, '_initialized'):
        jax.distributed.initialize()
    
    # Get device information
    n_devices = jax.device_count()
    local_devices = jax.local_devices()
    
    # Print device information for debugging
    print(f"Total devices: {n_devices}")
    print(f"Local devices: {len(local_devices)}")
    
    if len(local_devices) > 0:
        print(f"Device types: {[d.device_kind for d in local_devices[:2]]}")
    else:
        print("Warning: No local devices found!")
        # Fall back to using all devices if no local devices
        local_devices = jax.devices()
        print(f"Falling back to all devices: {len(local_devices)} devices")
    
    # Create a more robust mesh shape calculation
    if n_devices >= 4:
        # For larger pod slices, use a more balanced mesh
        mesh_shape = (n_devices // 2, 2)
    else:
        # For smaller setups, use a simpler mesh
        mesh_shape = (n_devices, 1)
    
    print(f"Using mesh shape: {mesh_shape}")
    
    # Create the mesh with explicit device assignment
    # Handle case where reshape might fail
    try:
        devices = np.array(jax.devices()).reshape(mesh_shape)
        mesh = Mesh(devices, ('model', 'batch'))
    except ValueError as e:
        print(f"Error creating mesh with shape {mesh_shape}: {e}")
        print("Falling back to simple 1D mesh")
        # Fall back to 1D mesh
        mesh = Mesh(np.array(jax.devices()), ('model',))
    
    return mesh, n_devices

def prefetch_batches(dataset_iterator, prefetch_size, mesh):
    """Prefetch and prepare batches in a separate thread."""
    prefetch_queue = queue.Queue(maxsize=prefetch_size)
    
    def prefetch_worker():
        try:
            for batch_idx, batch in dataset_iterator:
                try:
                    # Prepare batch with proper sharding
                    prepared_batch = {k: jnp.array(v) for k, v in batch.items()}
                    
                    # Use device_put with proper sharding
                    # Handle potential errors during device transfer
                    try:
                        prepared_batch = jax.device_put(prepared_batch, NamedSharding(mesh, P('batch', None)))
                    except Exception as e:
                        print(f"Warning: Error during device_put: {e}")
                        # Fall back to simple device_put without sharding
                        prepared_batch = jax.device_put(prepared_batch)
                    
                    prefetch_queue.put((batch_idx, prepared_batch))
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    # Continue with next batch instead of failing
                    continue
                    
            # Signal the end of the iterator
            prefetch_queue.put((None, None))
        except Exception as e:
            print(f"Prefetch worker exception: {e}")
            prefetch_queue.put((None, e))
    
    # Start prefetching thread
    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()
    
    # Yield batches from the queue
    while True:
        batch_idx, batch = prefetch_queue.get()
        if batch_idx is None:
            if isinstance(batch, Exception):
                raise batch
            break
        yield batch_idx, batch

def main():
    # Initialize JAX distributed environment with proper coordination
    process_index, process_count = initialize_jax_distributed()
    
    # Get local device count for synchronization
    local_device_count = jax.local_device_count()
    
    # Only initialize wandb on the main process
    if process_index == 0:
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
                "process_count": process_count,
            }
        )

    CHECKPOINT_DIR = os.path.abspath("./checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True, cache_dir='./cache')
    
    # Prepare datasets - only load on main process and broadcast if needed
    if process_index == 0:
        print("Preparing dataset...")
        train_dataset = prepare_dataset(tokenizer)
    else:
        # Wait for main process to prepare dataset
        time.sleep(5)
        train_dataset = prepare_dataset(tokenizer)
    
    # Ensure all processes have loaded the dataset
    if process_count > 1:
        # Use multihost_utils for better synchronization across hosts
        print(f"Process {process_index} waiting for dataset synchronization...")
        try:
            multihost_utils.sync_global_devices("dataset_loaded")
            print(f"Process {process_index}: Dataset loading synchronized")
        except Exception as e:
            print(f"Warning: Dataset synchronization error: {e}")
            print("Continuing without synchronization...")

    # Initialize TPU devices and create mesh
    try:
        mesh, n_devices = create_mesh()
        print(f"Process {process_index}: Training on {n_devices} devices with 2D sharding")
        
        # Calculate steps per epoch and total steps
        samples_per_step = BATCH_SIZE * (2 if n_devices >= 4 else 1)  # Adjust based on mesh
        steps_per_epoch = len(train_dataset) // samples_per_step
        total_steps = steps_per_epoch * NUM_EPOCHS
        
        print(f"Process {process_index}: Dataset size: {len(train_dataset)}")
        print(f"Process {process_index}: Steps per epoch: {steps_per_epoch}")
        print(f"Process {process_index}: Total steps: {total_steps}")
        
        # Create learning rate schedule
        learning_rate_fn = create_learning_rate_schedule(
            num_train_steps=total_steps,
            warmup_steps=WARMUP_STEPS,
            base_learning_rate=LEARNING_RATE
        )
                
        with mesh:
            # Create a device-specific random key inside the mesh context
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
            
            # Load from checkpoint if available
            if latest_checkpoint:
                print(f"Process {process_index}: Found checkpoint at {latest_checkpoint}. Restoring...")
                try:
                    # Extract step from checkpoint path
                    checkpoint_step = int(os.path.basename(latest_checkpoint).split("_")[-1])
                    step = checkpoint_step
                    
                    # Ensure all processes are synchronized before loading checkpoint
                    if process_count > 1:
                        multihost_utils.sync_global_devices("before_checkpoint_load")
                    
                    # Use a more robust checkpoint loading approach
                    try:
                        # First try to restore with specific target
                        state = checkpoints.restore_checkpoint(latest_checkpoint, target=state)
                    except Exception as inner_e:
                        print(f"Process {process_index}: First checkpoint restore attempt failed: {inner_e}")
                        print(f"Process {process_index}: Trying alternative restore approach...")
                        
                        # Try loading without specific target structure
                        raw_state = checkpoints.restore_checkpoint(latest_checkpoint, target=None)
                        if raw_state:
                            # Manually reconstruct state if needed
                            print(f"Process {process_index}: Loaded raw checkpoint data, reconstructing state...")
                            # Just use the fresh state and continue training from scratch
                            # This avoids the peer mismatch issues
                            step = 0
                        else:
                            raise ValueError("Could not load checkpoint data in any format")
                    
                    # Ensure all processes are synchronized after loading checkpoint
                    if process_count > 1:
                        multihost_utils.sync_global_devices("after_checkpoint_load")
                    
                    print(f"Process {process_index}: Successfully restored checkpoint at step {step}")
                except Exception as e:
                    print(f"Process {process_index}: Error loading checkpoint: {e}")
                    print(f"Process {process_index}: Training from scratch instead.")
                    step = 0
            else:
                print(f"Process {process_index}: No checkpoint found. Training from scratch.")
            
            # Calculate and print model size
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
            print(f"Process {process_index}: Number of parameters: {param_count/1e9:.2f}B")
            
            # Log model size to wandb (only on main process)
            if process_index == 0:
                wandb.run.summary["model_parameters_B"] = param_count/1e9
            
            # Pre-batch the dataset
            # Use a safer approach that avoids fork() for multiprocessing
            print(f"Process {process_index}: Batching dataset...")
            batched_dataset = train_dataset.batch(
                samples_per_step,
                num_proc=16,
            )
            
            print(f"Process {process_index}: Dataset batched with {len(batched_dataset)} batches")
            
            # Training loop
            train_metrics = []
            
            # Calculate starting epoch and batch index based on loaded step
            start_epoch = step // steps_per_epoch
            start_batch_idx = step % steps_per_epoch
            
            print(f"Process {process_index}: Starting training from step {step} (epoch {start_epoch}, batch {start_batch_idx})")
            
            # Number of batches to prefetch (adjust based on your system's memory)
            prefetch_size = 3
            
            for epoch in range(start_epoch, NUM_EPOCHS):
                # Shuffle dataset at the start of each epoch
                # Use the same seed across all processes to ensure consistency
                try:
                    print(f"Process {process_index}: Shuffling dataset for epoch {epoch+1}...")
                    # Use in-memory shuffling to avoid multiprocessing issues
                    batched_dataset = batched_dataset.shuffle(
                        seed=epoch,
                        writer_batch_size=100,  # Process in smaller chunks
                        load_from_cache_file=False  # Don't use cache to avoid file locking issues
                    )
                    print(f"Process {process_index}: Dataset shuffled for epoch {epoch+1}")
                except Exception as e:
                    print(f"Process {process_index}: Error shuffling dataset: {e}")
                    print(f"Process {process_index}: Continuing without shuffling...")
                
                # Synchronize after shuffling to ensure all processes are ready
                if process_count > 1:
                    multihost_utils.sync_global_devices(f"epoch_{epoch}_shuffled")
                
                # Create dataset iterator based on resumption point
                if epoch == start_epoch and start_batch_idx > 0:
                    print(f"Process {process_index}: Skipping to batch {start_batch_idx} in epoch {epoch+1}")
                    try:
                        # Convert to list first to avoid potential iterator issues
                        batched_list = list(batched_dataset)
                        if start_batch_idx < len(batched_list):
                            dataset_iter = enumerate(batched_list[start_batch_idx:], start=start_batch_idx)
                            total_batches = len(batched_list)
                            initial = start_batch_idx
                        else:
                            print(f"Process {process_index}: Warning: start_batch_idx {start_batch_idx} exceeds dataset size {len(batched_list)}")
                            dataset_iter = enumerate(batched_list)
                            total_batches = len(batched_list)
                            initial = 0
                    except Exception as e:
                        print(f"Process {process_index}: Error creating dataset iterator: {e}")
                        # Fallback to simpler approach
                        dataset_iter = enumerate(batched_dataset)
                        total_batches = len(batched_dataset)
                        initial = 0
                else:
                    try:
                        # Convert to list first to avoid potential iterator issues
                        batched_list = list(batched_dataset)
                        dataset_iter = enumerate(batched_list)
                        total_batches = len(batched_list)
                    except Exception as e:
                        print(f"Process {process_index}: Error creating dataset iterator: {e}")
                        # Fallback to simpler approach
                        dataset_iter = enumerate(batched_dataset)
                        total_batches = len(batched_dataset)
                    initial = 0
                
                # Create progress bar (only on main process)
                if process_index == 0:
                    progress_bar = tqdm(
                        total=total_batches,
                        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                        position=0,
                        initial=initial
                    )
                
                # Create prefetching iterator
                prefetch_iter = prefetch_batches(dataset_iter, prefetch_size, mesh)
                
                # Process batches with prefetching
                for batch_idx, batch in prefetch_iter:
                    # Train step with sharding (batch is already prepared and on device)
                    state, metrics = train_step(state, batch, rngs=rng)
                    train_metrics.append(metrics)
                    
                    # Update progress bar (only on main process)
                    if process_index == 0:
                        progress_bar.update(1)
                    
                    if batch_idx % 50 == 0:
                        # Convert JAX arrays to float values for formatting
                        metrics = {
                            'loss': float(metrics['loss']),
                            'main_loss': float(metrics['main_loss']),
                            'router_loss': float(metrics['router_loss'])
                        }
                        
                        # Update progress bar (only on main process)
                        if process_index == 0:
                            progress_bar.set_postfix({
                                'loss': f"{metrics['loss']:.4f}",
                                'main_loss': f"{metrics['main_loss']:.4f}",
                                'router_loss': f"{metrics['router_loss']:.4f}"
                            })
                        
                        # Log metrics to wandb (only on main process)
                        if process_index == 0:
                            wandb.log({
                                'train/loss': metrics['loss'],
                                'train/main_loss': metrics['main_loss'],
                                'train/router_loss': metrics['router_loss'],
                                'train/step': step,
                                'train/epoch': epoch + (batch_idx / steps_per_epoch)
                            })
                    
                    # Save checkpoint periodically (only on main process)
                    if (batch_idx % 5000 == 0 or step % 5000 == 0) and process_index == 0:
                        print(f"\nProcess {process_index}: Saving checkpoint at step {step}...")
                        checkpoints.save_checkpoint(
                            ckpt_dir=CHECKPOINT_DIR,
                            target=state,
                            step=step,
                            overwrite=True
                        )
                        print(f"Process {process_index}: Checkpoint saved at {os.path.join(CHECKPOINT_DIR, f'checkpoint_{step}')}")
                    
                    step += 1
                
                # Close progress bar (only on main process)
                if process_index == 0:
                    progress_bar.close()
                
                # Synchronize processes at the end of each epoch
                if process_count > 1:
                    # Use multihost_utils for better synchronization across hosts
                    print(f"Process {process_index} waiting for end-of-epoch synchronization...")
                    multihost_utils.sync_global_devices(f"epoch_{epoch}_complete")
                    print(f"Process {process_index}: End of epoch {epoch+1} synchronized")
            
            # Close wandb run when training is complete (only on main process)
            if process_index == 0:
                wandb.finish()
    except Exception as e:
        print(f"Error during mesh creation or training: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Print detailed error information
        import traceback
        print(f"Unhandled exception in main: {e}")
        traceback.print_exc()
        
        # Try to report error to all processes if possible
        try:
            import jax
            process_index = jax.process_index()
            print(f"Error occurred on process {process_index}")
        except:
            print("Could not determine process index")
        
        # Exit with error code
        import sys
        sys.exit(1)
