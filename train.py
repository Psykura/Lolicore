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

# This script trains a Transformer model with MoE (Mixture of Experts) architecture
# It supports checkpoint saving and loading for resuming training from the last saved checkpoint

# Constants
CONTEXT_LENGTH = 2048
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
GRADIENT_CLIP_NORM = 1.0
BATCH_MESH_SIZE = 4
DTYPE = jnp.bfloat16  # Set default dtype to bfloat16
PARALLEL_PROCESSING = 16
TOKENIZED_DATASET_PATH = '/mnt/dataset/tokenized_dataset'

vocab_size = 50257
vocab_size = ((vocab_size + 127) // 128) * 128

# Model hyperparameters
MODEL_CONFIG = {
    'num_blocks': 12,
    'num_heads': 8,
    'd_model': 768,
    'hidden_size': 4096,
    'max_seq_length': CONTEXT_LENGTH,
    'vocab_size': vocab_size,  # GPT-2 vocab size
    'num_experts': 24,
    'num_shared_experts': 1,
    'use_gradient_checkpointing': True,
    'attention_latent_dim': 64,
    'num_constant_experts': 3,
    'num_noise_experts': 1,
}

# Dataset configuration
DATASET_CONFIG = {
    'path': 'HuggingFaceFW/fineweb',
    'name': 'sample-10BT',
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
            mu_dtype=DTYPE,
        )
    )

    with mesh:
        print("Transferring parameters to mesh...")
        # Create parameter shardings
        param_shardings = jax.tree.map_with_path(
            lambda path, p: NamedSharding(mesh, get_param_spec(p, path)),
            cpu_variables['params']
        )

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
        tokenized_dataset = tokenized_dataset.with_format("jax")
        #tokenized_dataset = tokenized_dataset.shuffle(seed=42)
        return tokenized_dataset, len(tokenized_dataset)

    dataset = load_dataset(**DATASET_CONFIG, num_proc=PARALLEL_PROCESSING)
    print(f"Raw dataset size: {len(dataset)}")

    def tokenize_and_chunk(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=False,
            add_special_tokens=True
        )

        # Chunk into context_length segments
        chunks = []
        attention_masks = []

        for input_ids in tokenized['input_ids']:
            # Skip if sequence is too short or empty
            if len(input_ids) < 32 or not any(input_ids):  # Skip very short or empty sequences
                continue

            # Split into chunks of context_length
            for i in range(0, len(input_ids) + CONTEXT_LENGTH, CONTEXT_LENGTH):
                chunk = input_ids[i:i + CONTEXT_LENGTH]
                
                # Only process chunks that are reasonably sized
                if len(chunk) < 32:
                    continue
                    
                # Pad the last chunk if needed
                if len(chunk) < CONTEXT_LENGTH:
                    # Calculate padding length
                    padding_length = CONTEXT_LENGTH - len(chunk)
                    # Create attention mask with 1s for real tokens and 0s for padding
                    attention_mask = [1] * len(chunk) + [0] * padding_length
                    # Pad the chunk with EOS tokens
                    chunk = chunk + [tokenizer.eos_token_id] * padding_length
                else:
                    # No padding needed, attention mask is all 1s
                    attention_mask = [1] * CONTEXT_LENGTH
                
                chunks.append(chunk)
                attention_masks.append(attention_mask)

        return {
            'input_ids': chunks,
            'attention_mask': attention_masks,
            'labels': chunks  # For causal language modeling
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
    tokenized_dataset = tokenized_dataset.with_format("jax")
    #tokenized_dataset = tokenized_dataset.shuffle(seed=42)
    return tokenized_dataset, len(tokenized_dataset)

def create_batch(mesh, examples):
    """Create a sharded batch from dataset examples."""
    # Convert dataset to dictionary of arrays
    batch = {k: examples[k] for k in examples.features.keys()}
    
    # Apply sharding to each array in the batch
    return jax.tree_map(
        lambda x: jax.device_put(x, NamedSharding(mesh, P('batch', None))), 
        batch
    )

@jax.jit
def train_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray], rngs: jax.random.PRNGKey):
    """Perform a single training step with model and data parallelism."""
    #rngs, noise_rng = jax.random.split(rngs)
    noise_rng = rngs

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
    """Improved parameter sharding specification with MoE-aware distribution."""
    path_str = '/'.join(map(str, path))

    # 1. Embedding layers - split vocabulary dimension
    if 'token_embedding' in path_str:
        return P('model', None)  # Split vocab across model axis

    # 2. Attention layers - optimized for tensor parallelism
    if 'attention' in path_str:
        if 'q_proj' in path_str or 'k_proj' in path_str or 'v_proj' in path_str:
            return P(None, 'model') if param.ndim == 2 else P('model')
        if 'out_proj' in path_str:
            return P('model', None)  # Split output dimension

    # 3. Expert layers - distribute experts across devices
    if 'experts' in path_str:
        expert_idx = int(path_str.split('/')[2])  # Extract expert index
        if 'kernel' in path_str and param.ndim == 2:
            # Split expert weights across model and batch dimensions
            return P('model', 'batch')
        # Replicate small expert parameters
        return P('model') if param.size > 1e5 else P(None)

    # 4. Output projection - split vocabulary dimension
    if 'output_proj' in path_str:
        return P('model', 'batch')  # Split both dimensions

    # 5. FFN layers - optimized for model parallelism
    if 'Dense' in path_str or 'values' in path_str or 'keys' in path_str:
        if param.ndim == 2:
            return P('model', None)  # Split input dimension
        return P('model')

    # 6. Small parameters - replicate to avoid communication overhead
    if param.size < 2**14:  # 16KB threshold
        return P(None)

    # Default sharding for large parameters
    return P('model')

def create_mesh():
    """Create a more balanced device mesh for TPU."""
    devices = jax.devices()
    n_devices = len(devices)

    mesh_shape = (n_devices // BATCH_MESH_SIZE, BATCH_MESH_SIZE) 

    mesh = jax.make_mesh(mesh_shape, ('model', 'batch'))
    return mesh, n_devices

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
            # Replace dynamic batch creation with pre-batched dataset
            shuffled_indices = jax.random.permutation(jax.random.key(epoch), len(tokenized_dataset))
            
            print(f"Syncing epoch {epoch} for process {jax.process_index()}")
            sync_global_devices(f'epoch_{epoch}')

            # Create tqdm progress bar for each epoch
            progress_bar = tqdm(
                range(steps_per_epoch),
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                position=0
            )

            # Skip batches if we're resuming from middle of an epoch
            start_idx = start_batch_idx if epoch == start_epoch else 0
            
            for batch_idx in range(start_idx, steps_per_epoch):
                # Dynamically create batch
                batch_start_idx = batch_idx * samples_per_step
                batch_end_idx = min((batch_idx + 1) * samples_per_step, len(tokenized_dataset))
                
                # Get indices for this batch
                batch_indices = shuffled_indices[batch_start_idx:batch_end_idx]
                
                # Skip if batch is too small (should not happen with proper steps calculation)
                if len(batch_indices) < samples_per_step:
                    continue
                
                # Get examples for this batch
                batch_examples = tokenized_dataset[batch_indices]
                
                # Create batch
                batch = create_batch(mesh, batch_examples)
                
                # Train step with sharding
                state, metrics = train_step(state, batch, rngs=rng)

                # Update progress bar
                progress_bar.update(1)

                if batch_idx % 50 == 0:
                    # Convert JAX arrays to float values for formatting
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
                        # Log metrics to wandb
                        wandb.log({
                            'train/loss': metrics['loss'],
                            'train/main_loss': metrics['main_loss'],
                            'train/router_loss': metrics['router_loss'],
                            'train/step': step,
                            'train/epoch': epoch + (batch_idx / steps_per_epoch)
                        })

                # Save checkpoint periodically
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

        # Close wandb run when training is complete
        if jax.process_index() == 0:
            wandb.finish()

if __name__ == "__main__":
    main()
