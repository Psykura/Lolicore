import os
if os.path.exists('transformer.py'):
    from transformer import Transformer
import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from flax.training import train_state
from typing import Dict
from tqdm import tqdm
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import wandb
from jax.experimental.multihost_utils import sync_global_devices
import numpy as np
import collections
import itertools
import orbax.checkpoint as ocp
import random
from functools import partial

# This script trains a Transformer model with MoE (Mixture of Experts) architecture
# It supports checkpoint saving and loading for resuming training from the last saved checkpoint

# Constants
CONTEXT_LENGTH = 512
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
GRADIENT_CLIP_NORM = 1.0
DTYPE = jnp.bfloat16  # Set default dtype to bfloat16
PARALLEL_PROCESSING = 8
TOKENIZED_DATASET_PATH = '/mnt/data/tokenized_dataset'
EVAL_STEPS = 1000  # How often to evaluate on test set

vocab_size = 50257
vocab_size = ((vocab_size + 127) // 128) * 128

# Model hyperparameters
MODEL_CONFIG = {
    'num_blocks': 12,
    'num_heads': 8,
    'd_model': 512,
    'hidden_size': 2048,
    'max_seq_length': CONTEXT_LENGTH,
    'vocab_size': vocab_size,  # GPT-2 vocab size
    'num_experts': 20 + 2 + 2,
    'num_shared_experts': 2,
    'use_gradient_checkpointing': True,
    'attention_latent_dim': 64,
    'num_constant_experts': 2,
    'num_noise_experts': 0,
    'top_k': 2,
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
    warmup_steps = min(warmup_steps, num_train_steps)
    warmup_fn = optax.linear_schedule(
        init_value=base_learning_rate * 0.1,
        end_value=base_learning_rate,
        transition_steps=warmup_steps
    )

    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=max(num_train_steps - warmup_steps, 1),
        alpha=0.2
    )

    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps]
    )

def create_train_state(
    rng: jax.random.PRNGKey,
    mesh: Mesh,
    learning_rate_fn: optax.Schedule,
    **kwargs
) -> train_state.TrainState:
    """Creates initial TrainState with model initialization and optimizer setup."""
    model = Transformer(dtype=DTYPE, training=True, **kwargs)
    rng, params_rng, dropout_rng, noise_rng = jax.random.split(rng, 4)
    rngs = {'params': params_rng, 'dropout': dropout_rng, 'noise': noise_rng}

    optimizer = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP_NORM),
        optax.adamw(
            learning_rate=learning_rate_fn,
            weight_decay=0.005,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
        )
    )

    with mesh:
        print("Initializing model directly on mesh...")
        # Create dummy inputs with proper sharding
        dummy_input = jnp.ones((BATCH_SIZE, CONTEXT_LENGTH), dtype=jnp.int32)
        dummy_mask = jnp.ones((BATCH_SIZE, CONTEXT_LENGTH), dtype=jnp.int32)
        
        # Initialize model directly on the mesh
        variables = model.init(rngs, dummy_input, dummy_mask)
        
        print("Creating optimizer state...")
        opt_state = optimizer.init(variables['params'])
        
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optimizer,
            opt_state=opt_state,
        )
        
        print("Train state created successfully")
        
        # Print total parameter count
        param_count = sum(p.size for p in jax.tree.leaves(variables['params']))
        print(f"Total parameters: {param_count:,}")
        
        return state

def create_mesh():
    """Create an optimized device mesh for training MoE models."""
    devices = jax.devices()
    n_devices = len(devices)
    
    print(f"Found {n_devices} devices: {devices}")
    
    if n_devices == 0:
        raise ValueError("No JAX devices found. Please check your JAX installation.")
    
    # Calculate optimal dimensions for device mesh
    # For MoE models, we want to maximize expert parallelism first
    # Then model parallelism, and finally batch parallelism
    
    # Determine optimal expert dimension based on model config
    num_experts = MODEL_CONFIG['num_experts']
    num_constant_experts = MODEL_CONFIG['num_constant_experts']
    
    # We need to ensure the expert dimension evenly divides both num_experts and num_constant_experts
    # So we need to find the largest divisor of n_devices that is <= min(num_experts, num_constant_experts)
    min_expert_dim = min(num_experts, num_constant_experts)
    
    # Try to find factors of n_devices that work well for expert sharding
    possible_expert_dims = []
    for i in range(1, min(min_expert_dim + 1, n_devices + 1)):
        if n_devices % i == 0 and i <= min_expert_dim:
            possible_expert_dims.append(i)
    
    # Choose the largest expert dimension that divides n_devices and is <= min_expert_dim
    expert_dim = max(possible_expert_dims) if possible_expert_dims else 1
    
    # Now distribute remaining devices between model and batch dimensions
    remaining_dims = n_devices // expert_dim
    
    # Ensure remaining_dims is at least 1
    if remaining_dims < 1:
        expert_dim = 1
        remaining_dims = n_devices
    
    # Default to model_dim=1 for single device case
    model_dim = 1
    
    # Prefer model parallelism over batch parallelism for better efficiency
    # but ensure at least some batch parallelism
    if remaining_dims > 1:
        model_dim = 2
        while remaining_dims % (model_dim * 2) == 0 and model_dim * 2 <= 8:  # Limit model dim to 8
            model_dim *= 2
    
    batch_dim = max(1, remaining_dims // model_dim)
    
    # Verify our dimensions multiply to give the total number of devices
    actual_devices = expert_dim * model_dim * batch_dim
    if actual_devices != n_devices:
        print(f"Warning: Calculated dimensions ({expert_dim}×{model_dim}×{batch_dim}={actual_devices}) " 
              f"don't match total devices ({n_devices})")
        # Fall back to a simpler configuration
        if n_devices > 1:
            expert_dim = 1
            model_dim = 1
            batch_dim = n_devices
        else:
            expert_dim = model_dim = batch_dim = 1
    
    print(f"Using 3D mesh with shape: expert={expert_dim}, model={model_dim}, batch={batch_dim}")
    print(f"Total devices: {expert_dim * model_dim * batch_dim}")
    
    # Create mesh with logical axes that match our model's partitioning
    mesh = jax.make_mesh((expert_dim, model_dim, batch_dim), ('expert', 'model', 'data'))
    return mesh, n_devices

def calculate_metrics(logits, labels, mask):
    """Calculate loss, accuracy, and perplexity metrics."""
    shift_logits = logits[..., :-1, :].astype(jnp.float32)
    shift_labels = labels[..., 1:]
    loss_mask = mask[..., :-1]
    
    # Calculate cross entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        shift_logits,
        shift_labels,
    )
    loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)
    
    # Calculate accuracy
    predictions = jnp.argmax(shift_logits, axis=-1)
    correct_predictions = (predictions == shift_labels) * loss_mask
    accuracy = correct_predictions.sum() / (loss_mask.sum() + 1e-9)
    
    # Calculate perplexity
    perplexity = jnp.exp(loss)
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }

@jax.jit
def train_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray], step: int, rngs: jax.random.PRNGKey):
    """Perform a single training step."""
    noise_rng = jax.random.fold_in(rngs, step)

    def loss_fn(params):
        logits, router_loss = state.apply_fn(
            {'params': params},
            batch['input_ids'],
            batch['attention_mask'],
            rngs={'noise': noise_rng}
        )
        
        metrics = calculate_metrics(logits, batch['labels'], batch['attention_mask'])
        total_loss = metrics['loss'] + router_loss
        
        return total_loss, (metrics, router_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (metrics, router_loss)), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)

    metrics['router_loss'] = router_loss
    metrics['total_loss'] = total_loss
    
    return new_state, metrics

@jax.jit
def test_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]):
    """Perform a single test step."""
    logits, router_loss = state.apply_fn(
        {'params': state.params},
        batch['input_ids'],
        batch['attention_mask'],
        rngs={'noise': jax.random.PRNGKey(0)}  # Fixed noise for evaluation
    )
    
    metrics = calculate_metrics(logits, batch['labels'], batch['attention_mask'])
    metrics['router_loss'] = router_loss
    metrics['total_loss'] = metrics['loss'] + router_loss
    
    return metrics

def prepare_dataset(tokenizer):
    """Prepare dataset with tokenization and chunking, and create a test set."""
    if os.path.exists(TOKENIZED_DATASET_PATH):
        tokenized_dataset = load_from_disk(TOKENIZED_DATASET_PATH)
        print(f"Loaded tokenized dataset from disk with {len(tokenized_dataset)} examples")
        
        # Create test set
        test_indices = random.Random(42).sample(range(len(tokenized_dataset)), BATCH_SIZE)
        test_dataset = tokenized_dataset.select(test_indices)
        train_indices = [i for i in range(len(tokenized_dataset)) if i not in test_indices]
        train_dataset = tokenized_dataset.select(train_indices)
        
        return train_dataset, test_dataset, len(train_dataset)

    dataset = load_dataset(**DATASET_CONFIG, num_proc=PARALLEL_PROCESSING)
    print(f"Raw dataset size: {len(dataset)}")

    def tokenize_and_chunk(examples):
        tokenized = tokenizer(examples['text'], truncation=False, add_special_tokens=True)
        chunks = []
        attention_masks = []

        for input_ids in tokenized['input_ids']:
            if len(input_ids) < 32 or not any(input_ids):
                continue

            for i in range(0, len(input_ids), CONTEXT_LENGTH):
                chunk = input_ids[i:i + CONTEXT_LENGTH]
                if len(chunk) < 32:
                    continue

                if len(chunk) < CONTEXT_LENGTH:
                    padding_length = CONTEXT_LENGTH - len(chunk)
                    attention_mask = [1] * len(chunk) + [0] * padding_length
                    chunk = chunk + [tokenizer.eos_token_id] * padding_length
                else:
                    chunk = chunk[:CONTEXT_LENGTH]
                    attention_mask = [1] * CONTEXT_LENGTH

                chunks.append(chunk)
                attention_masks.append(attention_mask)

        return {
            'input_ids': chunks,
            'attention_mask': attention_masks,
            'labels': chunks
        }

    tokenized_dataset = dataset.map(
        tokenize_and_chunk,
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=PARALLEL_PROCESSING,
        cache_file_name=None
    )

    print(f"Processed dataset size: {len(tokenized_dataset)}")
    tokenized_dataset.save_to_disk(TOKENIZED_DATASET_PATH)
    
    # Create test set
    test_indices = random.Random(42).sample(range(len(tokenized_dataset)), BATCH_SIZE)
    test_dataset = tokenized_dataset.select(test_indices)
    train_indices = [i for i in range(len(tokenized_dataset)) if i not in test_indices]
    train_dataset = tokenized_dataset.select(train_indices)
    
    return train_dataset, test_dataset, len(train_dataset)

def create_batch(inputs):
    """Create a batch from dataset examples."""
    examples = {}
    for k, v in inputs.items():
        if k == 'attention_mask':
            masks = [
                mask[:CONTEXT_LENGTH] if len(mask) > CONTEXT_LENGTH
                else list(mask) + [0] * (CONTEXT_LENGTH - len(mask))
                for mask in v
            ]
            examples[k] = jnp.array(masks)
        else:
            examples[k] = jnp.array(v)
    
    return examples

def get_sharding_spec(ndim: int) -> P:
    """Helper function to determine sharding spec based on tensor rank."""
    if ndim == 3:
        # For 3D tensors (batch, seq_len, features)
        return P(None, None, 'batch')
    elif ndim == 2:
        # For 2D tensors (batch, seq_len)
        return P(None, 'batch')
    elif ndim == 1:
        # For 1D tensors
        return P(None)
    else:
        # Default fallback
        return P(None)

def save_cpu_only_checkpoint(state, async_checkpointer: ocp.AsyncCheckpointer, checkpoint_dir, name="best_model"):
    """
    Saves a CPU-only version of the model parameters as a checkpoint.
    
    Args:
        state: The current train state containing model parameters
        async_checkpoint_manager: The checkpoint manager for saving
        checkpoint_dir: Directory to save the checkpoint
        name: Name for the checkpoint (default: "best_model")
        optimize_for_inference: If True, reinitialize model with training=False
    
    Returns:
        bool: True if a new checkpoint was saved, False otherwise
    """
    # Static variable to track best loss across function calls
    
    # Get a CPU device
    cpu_device = jax.devices("cpu")[0]
    
    # First, get the parameters off their current devices (potentially sharded)
    # Then transfer to CPU
    with jax.default_device(cpu_device):
        # Get model config without training flag        
        # Properly copy parameters from devices to CPU
        # First get from devices, then put to CPU
        params_copy = jax.device_put(
            jax.device_get(state.params),
            cpu_device
        )
        
        checkpoint_name = name
        # Save only the parameters
        checkpoint_data = {"params": params_copy}
        
        # Save the parameters-only checkpoint 
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        async_checkpointer.save(checkpoint_path, checkpoint_data, force=True)
    
    print(f"\nParameters checkpoint saved at {os.path.join(checkpoint_dir, checkpoint_name)}")
    return True

def prefetch(iterator, size):
    """Creates a prefetch queue for the iterator."""
    queue = collections.deque()

    def enqueue(n):
        for data in itertools.islice(iterator, n):
            queue.append(data)

    enqueue(size)
    while queue:
        yield queue.popleft()
        enqueue(1)

def create_prefetch_batches(dataset, indices, samples_per_step, start_idx=0, num_prefetch=4):
    """Creates an iterator that prefetches batches while training."""
    batch_indices = [
        idx for idx in [
            indices[i:i + samples_per_step]
            for i in range(0, len(indices), samples_per_step)
        ]
        if len(idx) == samples_per_step
    ]

    # Start from the specified batch index
    batch_indices = batch_indices[start_idx:]

    return prefetch(
        map(lambda idx:(dataset[idx]), batch_indices),
        num_prefetch
    )

def evaluate_model(state, test_dataset):
    """Evaluate the model on the test set."""
    test_batch = create_batch({
        'input_ids': test_dataset['input_ids'],
        'attention_mask': test_dataset['attention_mask'],
        'labels': test_dataset['labels']
    })
    
    metrics = test_step(state, test_batch)
    return {k: float(v) for k, v in metrics.items()}

def main():
    tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
    train_dataset, test_dataset, dataset_size = prepare_dataset(tokenizer)

    print("Initializing JAX distributed...")
    jax.distributed.initialize()

    try:
        from jax_smi import initialise_tracking
        initialise_tracking()
        print("JAX SMI initialized")
    except ImportError:
        print("JAX SMI not installed, skipping memory tracking")

    CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if jax.process_index() == 0:
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
                "model_config": MODEL_CONFIG,
            }
        )

    print(f"Syncing start state for process {jax.process_index()}")
    sync_global_devices('train')

    mesh, n_devices = create_mesh()
    print(f"Training on {n_devices} devices with 2D sharding")

    print(f"Syncing mesh for process {jax.process_index()}")
    sync_global_devices('mesh_created')

    # Initialize async checkpoint manager
    async_checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler(), timeout_secs=50)
    async_checkpoint_manager = ocp.CheckpointManager(CHECKPOINT_DIR, async_checkpointer, options=ocp.CheckpointManagerOptions(max_to_keep=2))

    samples_per_step = BATCH_SIZE
    steps_per_epoch = len(train_dataset) // samples_per_step
    total_steps = steps_per_epoch * NUM_EPOCHS

    print(f"Original dataset size: {dataset_size}")
    print(f"Tokenized dataset size: {len(train_dataset)}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")

    with mesh:
        learning_rate_fn = create_learning_rate_schedule(
            num_train_steps=total_steps,
            warmup_steps=WARMUP_STEPS,
            base_learning_rate=LEARNING_RATE
        )

        rng = jax.random.key(0)
        rng, init_rng = jax.random.split(rng)

        step = 0
        state = create_train_state(
          init_rng,
          mesh=mesh,
          **MODEL_CONFIG,
          learning_rate_fn=learning_rate_fn
        )

        print(f"Syncing training state for process {jax.process_index()}")
        sync_global_devices('training_state_created')
        # Restore from checkpoint
        just_loaded = False
        latest_step = async_checkpoint_manager.latest_step()
        if latest_step is not None:
            print(f"Restoring from checkpoint at step {latest_step}")
            step = latest_step
            loaded_state = async_checkpoint_manager.restore(latest_step)
            state = state.replace(params=loaded_state['state']['params'])
            async_checkpoint_manager.wait_until_finished()
            just_loaded = True
        else:
            print("No checkpoint found, training from scratch")

        print(f"Syncing checkpoint loaded for process {jax.process_index()}")
        sync_global_devices('checkpoint_loaded')

        param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
        print(f"Number of parameters: {param_count/1e9:.2f}B")

        if jax.process_index() == 0:
            wandb.run.summary["model_parameters_B"] = param_count/1e9

        start_epoch = step // steps_per_epoch
        start_batch_idx = step % steps_per_epoch

        print(f"Syncing starting training for process {jax.process_index()}")
        sync_global_devices('starting_training')

        print(f"Starting training from step {step} (epoch {start_epoch}, batch {start_batch_idx})")

        # Track best metrics
        best_metrics = {
            'train_loss': float('inf'),
            'test_loss': float('inf'),
            'test_perplexity': float('inf')
        }
        plateau_count = 0
        early_stop_patience = 3

        for epoch in range(start_epoch, NUM_EPOCHS):
            shuffled_indices = np.random.RandomState(seed=epoch).permutation(len(train_dataset))

            print(f"Syncing epoch {epoch} for process {jax.process_index()}")
            sync_global_devices(f'epoch_{epoch}')

            batch_iterator = create_prefetch_batches(
                train_dataset,
                shuffled_indices,
                samples_per_step,
                start_idx=start_batch_idx if epoch == start_epoch else 0
            )

            progress_bar = tqdm(
                range(steps_per_epoch),
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                position=0,
                initial=start_batch_idx if epoch == start_epoch else 0
            )

            for batch_idx in range(start_batch_idx if epoch == start_epoch else 0, steps_per_epoch):
                try:
                    batch = next(batch_iterator)
                    state, metrics = train_step(state, batch, step, rngs=rng)
                    progress_bar.update(1)

                    if batch_idx % 50 == 0:
                        metrics = {k: float(v) for k, v in metrics.items()}
                        progress_bar.set_postfix({
                            k: f"{v:.4f}" for k, v in metrics.items()
                        })

                        if jax.process_index() == 0:
                            # Log learning rate and training metrics
                            current_lr = float(learning_rate_fn(step))
                            wandb.log({
                                f"train/{k}": v for k, v in metrics.items()
                            } | {
                                'train/step': step,
                                'train/epoch': epoch + (batch_idx / steps_per_epoch),
                                'train/learning_rate': current_lr
                            })

                    # Evaluate and save checkpoint
                    if step % EVAL_STEPS == 0 and step != 0:
                        if just_loaded:
                            just_loaded = False
                        else:
                            print(f"\nStep {step}: Evaluating model...")
                            test_metrics = evaluate_model(state, test_dataset)
                            
                            if jax.process_index() == 0:
                                wandb.log({
                                    'test/loss': test_metrics['loss'],
                                    'test/perplexity': test_metrics['perplexity'],
                                    'test/step': step,
                                    'test/epoch': epoch + (batch_idx / steps_per_epoch)
                                })
                                
                                print(f"Test Loss: {test_metrics['loss']:.4f}, Perplexity: {test_metrics['perplexity']:.4f}")
                            
                            # Save checkpoint and best model if improved
                            print(f"\nSaving checkpoint...")
                            async_checkpoint_manager.save(step, {"state": state})
                            
                            if test_metrics['perplexity'] < best_metrics['test_perplexity']:
                                best_metrics['test_perplexity'] = test_metrics['perplexity']
                                best_metrics['test_loss'] = test_metrics['loss']
                                save_cpu_only_checkpoint(
                                    state, 
                                    async_checkpointer, 
                                    CHECKPOINT_DIR,
                                    name="best_model"
                                )
                                print(f"New best perplexity: {test_metrics['perplexity']:.4f}")

                    step += 1

                except StopIteration:
                    print(f"Reached end of dataset in epoch {epoch}")
                    break
            
            # Evaluate on test set at end of epoch
            test_metrics = evaluate_model(state, test_dataset)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Test Loss: {test_metrics['loss']:.4f}, Perplexity: {test_metrics['perplexity']:.4f}")
            
            # Check for improvement
            if test_metrics['perplexity'] < best_metrics['test_perplexity']:
                improvement = best_metrics['test_perplexity'] - test_metrics['perplexity']
                best_metrics['test_perplexity'] = test_metrics['perplexity']
                best_metrics['test_loss'] = test_metrics['loss']
                plateau_count = 0
                print(f"Test perplexity improved by {improvement:.4f}. New best: {test_metrics['perplexity']:.4f}")
                
                # Save best model
                save_cpu_only_checkpoint(
                    state,
                    async_checkpointer,
                    CHECKPOINT_DIR,
                    name="best_model"
                )
            else:
                plateau_count += 1
                print(f"No improvement for {plateau_count} epochs. Best test perplexity: {best_metrics['test_perplexity']:.4f}")
                
                if plateau_count >= early_stop_patience:
                    print(f"Early stopping after {early_stop_patience} epochs without improvement")
                    break
            
            if jax.process_index() == 0:
                wandb.log({
                    'test/epoch_loss': test_metrics['loss'],
                    'test/epoch_perplexity': test_metrics['perplexity'],
                    'test/epoch_accuracy': test_metrics['accuracy'],
                    'train/epoch': epoch + 1
                })

        if jax.process_index() == 0:
            wandb.finish()
            jax.distributed.shutdown()

if __name__ == "__main__":
    main()
