import jax
import jax.numpy as jnp
from jax2tf import convert
import tensorflow as tf
import tf2onnx
from typing import Dict, Any
import os
from transformer import Transformer
from inference import (
    create_inference_state,
    MODEL_CONFIG,
    DTYPE,
    CONTEXT_LENGTH
)

def convert_params_to_tf(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JAX parameters to TensorFlow format."""
    def _convert(x):
        if isinstance(x, jnp.ndarray):
            return tf.constant(x)
        elif isinstance(x, dict):
            return {k: _convert(v) for k, v in x.items()}
        else:
            return x
    return _convert(params)

def create_tf_inference_model(jax_model: Transformer, params: Dict[str, Any]) -> tf.keras.Model:
    """Create a TensorFlow model from JAX model for inference."""
    
    # Convert JAX function to TensorFlow
    tf_fn = convert(
        lambda params, x, attention_mask: jax_model.apply(
            {'params': params}, x, attention_mask, rngs={'noise': jax.random.PRNGKey(0)}
        ),
        with_gradient=False  # Inference only
    )
    
    # Convert parameters
    tf_params = convert_params_to_tf(params)
    
    # Create TF model
    class TFTransformerWrapper(tf.keras.Model):
        def __init__(self, tf_fn, params):
            super().__init__()
            self.tf_fn = tf_fn
            self.params = params
        
        def call(self, inputs):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            # Only return logits, not router loss for inference
            logits, _ = self.tf_fn(self.params, input_ids, attention_mask)
            return logits
    
    return TFTransformerWrapper(tf_fn, tf_params)

def export_to_onnx(
    jax_model: Transformer,
    params: Dict[str, Any],
    save_path: str,
    input_shape: tuple = (1, CONTEXT_LENGTH)  # batch_size, seq_len
):
    """Export JAX model to ONNX format."""
    
    # Create TF model
    tf_model = create_tf_inference_model(jax_model, params)
    
    # Convert to ONNX
    input_signature = {
        'input_ids': tf.TensorSpec(input_shape, tf.int32, name='input_ids'),
        'attention_mask': tf.TensorSpec(input_shape, tf.int32, name='attention_mask')
    }
    
    model_proto, _ = tf2onnx.convert.from_keras(
        tf_model,
        input_signature=input_signature,
        opset=13,
        output_path=save_path
    )
    
    print(f"Model exported to: {save_path}")
    return model_proto

def main():
    # Load the JAX model state
    checkpoint_dir = "/root/checkpoints"  # Adjust this path as needed
    print("Loading model from checkpoint...")
    
    # Create model instance with inference configuration
    model = Transformer(
        dtype=DTYPE,
        training=False,
        **MODEL_CONFIG
    )
    
    # Load the model state
    state = create_inference_state(checkpoint_dir)
    
    # Export to ONNX
    save_path = "transformer_model.onnx"
    print(f"Exporting model to ONNX format: {save_path}")
    
    export_to_onnx(
        jax_model=model,
        params=state.params,
        save_path=save_path,
        input_shape=(1, CONTEXT_LENGTH)
    )
    
    # Verify the exported model
    import onnxruntime as ort
    import numpy as np
    
    print("\nVerifying exported model...")
    session = ort.InferenceSession(save_path)
    
    # Create dummy input for verification
    dummy_input = {
        'input_ids': np.zeros((1, CONTEXT_LENGTH), dtype=np.int32),
        'attention_mask': np.ones((1, CONTEXT_LENGTH), dtype=np.int32)
    }
    
    # Run inference
    outputs = session.run(None, dummy_input)
    print(f"Model verification successful! Output shape: {outputs[0].shape}")

if __name__ == "__main__":
    main()

# Example usage:
"""
# Initialize your JAX model and parameters
model = Transformer(...)
params = model.init(...)

# Export to ONNX
export_to_onnx(
    jax_model=model,
    params=params,
    save_path='transformer_model.onnx',
    input_shape=(1, 512)  # Adjust based on your needs
)
""" 