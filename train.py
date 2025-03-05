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
# This script trains a Transformer model with MoE (Mixture of Experts) architecture
# It supports checkpoint saving and loading for resuming training from the last saved checkpoint

# Constants
CONTEXT_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
GRADIENT_CLIP_NORM = 0.5
BATCH_MESH_SIZE = 2
DTYPE = jnp.bfloat16  # Set default dtype to bfloat16
PARALLEL_PROCESSING = 16
TOKENIZED_DATASET_PATH = './tokenized_dataset'

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
    'top_k': 2,
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
    warmup_steps = min(warmup_steps, num_train_steps)
    warmup_fn = optax.linear_schedule(
        init_value=base_learning_rate * 0.1,
        end_value=base_learning_rate,
        transition_steps=warmup_steps
    )

    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=max(num_train_steps - warmup_steps, 1),
        alpha=0.1
    )

    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps]
    )

def debug_param_sharding(params, param_shardings):
    """Print out parameter shapes and their sharding specifications for debugging."""
    print("\nParameter Sharding Debug Information")
    print(f"{'Parameter Path':<60} {'Shape':<20} {'Size (MB)':<12} {'Sharding':<15}")
    print("-" * 107)

    flat_params = jax.tree_util.tree_flatten_with_path(params)[0]
    flat_shardings = jax.tree_util.tree_flatten_with_path(param_shardings)[0]

    param_info = []
    for (path, param), (_, sharding) in zip(flat_params, flat_shardings):
        path_str = '.'.join(str(p) for p in path)
        size_mb = param.size * param.dtype.itemsize / (1024 * 1024)
        param_info.append((
            path_str[:57] + '...' if len(path_str) > 60 else path_str,
            str(param.shape),
            size_mb,
            str(sharding.spec)
        ))

    total_size_mb = 0
    for path_str, shape_str, size_mb, sharding_str in sorted(param_info, key=lambda x: x[2], reverse=True):
        print(f"{path_str:<60} {shape_str:<20} {size_mb:<12.2f} {sharding_str:<15}")
        total_size_mb += size_mb

    print("-" * 107)
    print(f"Total parameter size: {total_size_mb:.2f} MB\n")

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

    print("Initializing model on CPU...")
    cpu_device = jax.devices("cpu")[0]
    cpu_dummy_input = jax.device_put(jnp.ones((BATCH_SIZE, CONTEXT_LENGTH), dtype=jnp.int32), cpu_device)
    cpu_dummy_mask = jax.device_put(jnp.ones((BATCH_SIZE, CONTEXT_LENGTH), dtype=jnp.int32), cpu_device)

    with jax.default_device(cpu_device):
        cpu_variables = model.init(rngs, cpu_dummy_input, cpu_dummy_mask)
    print("CPU initialization complete")

    optimizer = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP_NORM),
        optax.adamw(
            learning_rate=learning_rate_fn,
            weight_decay=0.01,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
        )
    )

    with mesh:
        print("Transferring parameters to mesh...")
        param_shardings = jax.tree.map_with_path(
            lambda path, p: NamedSharding(mesh, get_param_spec_validated(p, path)),
            cpu_variables['params']
        )

        if jax.process_index() == 0:
            print("Parameters with inconsistent dimensions:")
            jax.tree.map_with_path(
                lambda path, p, s: check_param_spec_consistency(path, p, s),
                cpu_variables['params'],
                param_shardings
            )
            debug_param_sharding(cpu_variables['params'], param_shardings)

        sharded_params = jax.device_put(cpu_variables['params'], param_shardings)
        print("Parameter transfer complete")

        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=sharded_params,
            tx=optimizer
        )

def check_param_spec_consistency(path, param, sharding):
    spec = sharding.spec
    spec_ndim = sum(1 for axis in spec if axis is not None)
    if spec_ndim != param.ndim:
        path = '/'.join(str(p) for p in path)
        print(f"Inconsistent parameter: {path}")
        print(f"  Shape: {param.shape} (ndim={param.ndim})")
        print(f"  Spec: {spec} (effective ndim={spec_ndim})")
    return sharding

def prepare_dataset(tokenizer):
    """Prepare dataset with tokenization and chunking."""
    if os.path.exists(TOKENIZED_DATASET_PATH):
        tokenized_dataset = load_from_disk(TOKENIZED_DATASET_PATH)
        print(f"Loaded tokenized dataset from disk with {len(tokenized_dataset)} examples")
        return tokenized_dataset, len(tokenized_dataset)

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
    return tokenized_dataset, len(tokenized_dataset)

def create_batch(mesh, inputs):
    """Create a sharded batch from dataset examples."""
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

    if isinstance(examples, dict):
        sharded_examples = {}
        for key, value in examples.items():
            if hasattr(value, 'ndim'):
                ndim = value.ndim
                spec = get_sharding_spec(ndim)
                sharded_examples[key] = jax.device_put(value, NamedSharding(mesh, spec))
            else:
                sharded_examples[key] = value
        return sharded_examples
    else:
        if hasattr(examples, 'ndim'):
            spec = get_sharding_spec(examples.ndim)
            return jax.device_put(examples, NamedSharding(mesh, spec))
        return examples

def get_sharding_spec(ndim: int) -> P:
    """Helper function to determine sharding spec based on tensor rank."""
    if ndim == 3:
        return P('expert', 'model', 'batch')
    elif ndim == 2:
        return P('expert', 'model')
    elif ndim == 1:
        return P(None)
    return P(None)

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

        loss_mask = batch['attention_mask'][..., :-1]
        shift_logits = logits[..., :-1, :].astype(jnp.float32)
        shift_labels = batch['labels'][..., 1:]

        main_loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits,
            shift_labels,
        )

        main_loss = (main_loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)
        total_loss = main_loss + router_loss

        return total_loss, (main_loss, router_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, aux), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)

    return new_state, {
        'loss': total_loss,
        'main_loss': aux[0],
        'router_loss': aux[1]
    }

def create_mesh():
    """Create an optimized device mesh for training MoE models."""
    devices = jax.devices()
    n_devices = len(devices)

    expert_dim = 4
    model_dim = 2
    batch_dim = n_devices // (expert_dim * model_dim)
    
    mesh = jax.make_mesh((expert_dim, model_dim, batch_dim), ('expert', 'model', 'batch'))

    print(f"Using 3D mesh with shape: expert={expert_dim}, model={model_dim}, batch={batch_dim}")

    return mesh, n_devices

def get_param_spec(param, path):
    """Get parameter sharding specification based on parameter path and shape."""
    path_str = str(path).lower()

    # Handle 1D parameters (biases, scales, etc.)
    if param.ndim == 1:
        # For larger 1D parameters (>10k elements), shard across model dimension
        if param.size > 10000:
            return P(None, 'model', None)
        # For smaller 1D parameters, replicate across all dimensions
        return P(None)

    # Handle expert-related parameters
    if 'experts' in path_str:
        if 'keys' in path_str or 'values' in path_str:
            if 'kernel' in path_str:
                return P('expert', 'model', None)
            elif 'bias' in path_str:
                return P('expert')
        elif 'jump' in path_str:
            return P('expert')

    # Router parameters
    if 'router' in path_str:
        if 'gate' in path_str:
            if 'kernel' in path_str:
                # Router gate kernels are 2D (input_dim, num_experts)
                if param.ndim == 2:
                    return P('expert', 'model')
                return P(None, 'model', None)
            elif 'bias' in path_str:
                return P(None)
        elif 'temperature' in path_str:
            return P(None)

    # Output projection and embedding layers
    if 'output_proj' in path_str or 'lm_head' in path_str or 'out_proj' in path_str:
        if 'kernel' in path_str:
            if param.ndim == 2:
                return P('expert', 'model')
            return P(None, 'model', None)
        elif 'bias' in path_str:
            return P(None)

    if 'embedding' in path_str:
        if param.ndim == 2:
            return P('expert', 'model')
        return P(None, 'model', None)

    # Attention layers
    if 'attention' in path_str:
        if 'q_proj' in path_str or 'k_proj' in path_str or 'v_proj' in path_str:
            if 'kernel' in path_str:
                if param.ndim == 2:
                    return P('expert', 'model')
                return P(None, 'model', None)
            elif 'bias' in path_str:
                return P(None)
        if 'out_proj' in path_str:
            if 'kernel' in path_str:
                if param.ndim == 2:
                    return P('expert', 'model')
                return P(None, 'model', None)
            elif 'bias' in path_str:
                return P(None)

    # Layer normalization parameters
    if 'norm' in path_str or 'layernorm' in path_str or 'rmsnorm' in path_str:
        return P(None)

    # Size-based fallbacks
    if param.size < 10000:
        return P(None)

    # Default for remaining 2D+ parameters
    if param.ndim == 2:
        return P('expert', 'model')
    elif param.ndim >= 3:
        return P(None, 'model', None)

    # Default fallback for any other parameters
    return P(None)

def validate_param_spec(param, spec):
    """Validate that the parameter sharding specification is compatible with the parameter shape."""
    if spec is None:
        return True

    # Count non-None axes in the spec
    non_none_axes = sum(1 for axis in spec if axis is not None)

    # Check if the number of non-None axes matches the parameter ndim
    if non_none_axes > param.ndim:
        return False

    # For 1D parameters, ensure we don't have a 3D spec
    if param.ndim == 1 and len(spec) > 1:
        return False

    # For 2D parameters, ensure we don't have a 3D spec
    if param.ndim == 2 and len(spec) > 2:
        return False

    return True

def get_param_spec_validated(param, path):
    """Get validated parameter sharding specification."""
    spec = get_param_spec(param, path)
    if not validate_param_spec(param, spec):
        path_str = '/'.join(str(p) for p in path)
        print(f"Warning: Invalid sharding spec for {path_str} with shape {param.shape}: {spec}")
        # Fall back to a safe sharding spec
        if param.ndim == 0:
            return P(None)
        elif param.ndim == 1:
            return P(None)
        elif param.ndim == 2:
            return P('expert', 'model')
        else:
            return P(None, 'model', 'batch')
    return spec

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

def create_prefetch_batches(dataset, indices, samples_per_step, mesh, start_idx=0, num_prefetch=4):
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
        map(lambda idx: create_batch(mesh, dataset[idx]), batch_indices),
        num_prefetch
    )

def main():
    tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
    tokenized_dataset, dataset_size = prepare_dataset(tokenizer)

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
    async_checkpoint_manager = ocp.CheckpointManager(CHECKPOINT_DIR, async_checkpointer, options=ocp.CheckpointManagerOptions(enable_async_checkpointing=True, max_to_keep=2))

    samples_per_step = BATCH_SIZE * BATCH_MESH_SIZE
    steps_per_epoch = len(tokenized_dataset) // samples_per_step
    total_steps = steps_per_epoch * NUM_EPOCHS

    print(f"Original dataset size: {dataset_size}")
    print(f"Tokenized dataset size: {len(tokenized_dataset)}")
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

        for epoch in range(start_epoch, NUM_EPOCHS):
            shuffled_indices = np.random.RandomState(seed=epoch).permutation(len(tokenized_dataset))

            print(f"Syncing epoch {epoch} for process {jax.process_index()}")
            sync_global_devices(f'epoch_{epoch}')

            batch_iterator = create_prefetch_batches(
                tokenized_dataset,
                shuffled_indices,
                samples_per_step,
                mesh,
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
                            wandb.log({
                                f"train/{k}": v for k, v in metrics.items()
                            } | {
                                'train/step': step,
                                'train/epoch': epoch + (batch_idx / steps_per_epoch)
                            })

                    if (batch_idx % 5000 == 0 or step % 5000 == 0) and step != 0:
                        if just_loaded:
                            just_loaded = False
                        else:
                            print(f"\nSaving checkpoint at step {step}...")
                            async_checkpoint_manager.save(step, {"state": state})
                            async_checkpoint_manager.wait_until_finished()
                            print(f"Checkpoint saved at {os.path.join(CHECKPOINT_DIR, f'checkpoint_{step}')}")

                    step += 1

                except StopIteration:
                    print(f"Reached end of dataset in epoch {epoch}")
                    break

        if jax.process_index() == 0:
            wandb.finish()

if __name__ == "__main__":
    main()
