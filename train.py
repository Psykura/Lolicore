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
import wandb
from jax.experimental.multihost_utils import sync_global_devices

# This script trains a Transformer model with MoE (Mixture of Experts) architecture
# It supports checkpoint saving and loading for resuming training from the last saved checkpoint

# Constants
CONTEXT_LENGTH = 2048
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
GRADIENT_CLIP_NORM = 1.0
BATCH_MESH_SIZE = 2
DTYPE = jnp.bfloat16  # Set default dtype to bfloat16
PARALLEL_PROCESSING = 16

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
    'top_k': 4,
    'use_gradient_checkpointing': True,
    'attention_latent_dim': 64,
    'num_zeros_experts': 1,
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
    if os.path.exists('tokenized_dataset'):
        tokenized_dataset = load_from_disk('tokenized_dataset')
        return tokenized_dataset

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
                    # Pad the chunk with EOS tokens
                    chunk = chunk + [tokenizer.eos_token_id] * padding_length
                    # Create attention mask with 1s for real tokens and 0s for padding
                    attention_mask = [1] * len(chunk) + [0] * padding_length
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

    # Use sequential processing
    tokenized_dataset = dataset.map(
        tokenize_and_chunk,
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=PARALLEL_PROCESSING,
    )

    print(f"Processed dataset size: {len(tokenized_dataset)}")

    # Calculate batch size for dataset
    samples_per_step = BATCH_SIZE * BATCH_MESH_SIZE
    
    # Batch the dataset
    print(f"Batching dataset with {samples_per_step} samples per batch...")
    batched_dataset = tokenized_dataset.batch(samples_per_step, drop_last_batch=True, num_proc=PARALLEL_PROCESSING)
    
    # Save batched dataset to disk
    batched_dataset.save_to_disk('batched_dataset')
    print(f"Batched dataset saved to disk with {len(batched_dataset)} batches")
    
    return batched_dataset, len(tokenized_dataset)

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
    devices = jax.devices()
    n_devices = len(devices)

    mesh_shape = (n_devices // BATCH_MESH_SIZE, BATCH_MESH_SIZE) 

    mesh = jax.make_mesh(mesh_shape, ('model', 'batch'))
    return mesh, n_devices

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)

    # Prepare datasets - now returns both batched dataset and original dataset size
    batched_dataset, dataset_size = prepare_dataset(tokenizer)

    # if jax_smi is installed, track memory usage
    import sys
    if 'jax_smi' in sys.modules:
        from jax_smi import initialise_tracking
        initialise_tracking()

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
    steps_per_epoch = len(batched_dataset)  # Now this is just the number of batches
    total_steps = steps_per_epoch * NUM_EPOCHS

    print(f"Original dataset size: {dataset_size}")
    print(f"Number of batches: {len(batched_dataset)}")
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
            # Shuffle dataset at the start of each epoch
            batched_dataset.shuffle(seed=epoch)

            print(f"Syncing epoch {epoch} for process {jax.process_index()}")
            sync_global_devices(f'epoch_{epoch}')

            # Create tqdm progress bar for each epoch
            progress_bar = tqdm(
                enumerate(batched_dataset),
                total=steps_per_epoch,
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                position=0
            )

            # Skip batches if we're resuming from middle of an epoch
            if epoch == start_epoch and start_batch_idx > 0:
                print(f"Skipping to batch {start_batch_idx} in epoch {epoch+1}")
                # Process only the remaining batches in this epoch
                progress_bar = tqdm(
                    enumerate(list(batched_dataset)[start_batch_idx:], start=start_batch_idx),
                    total=steps_per_epoch,
                    desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                    position=0,
                    initial=start_batch_idx
                )

            for batch_idx, batch in progress_bar:
                # Prepare batch with proper sharding
                batch = {k: jnp.array(v) for k, v in batch.items()}
                batch = jax.device_put(batch, NamedSharding(mesh, P('batch', None)))

                # Train step with sharding
                state, metrics = train_step(state, batch, rngs=rng)

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
