"""
export_onnx.py - Export Models to ONNX Format
Phase 5: Model Deployment - ONNX Export
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tf2onnx
import onnx
import numpy as np
from transformer_model import MultiHeadAttention, EncoderLayer, TransformerEncoder

def export_model_to_onnx(model_path, onnx_path, model_name, custom_objects=None):
    """
    Export TensorFlow model to ONNX format
    
    Args:
        model_path: Path to .h5 model file
        onnx_path: Output path for .onnx file
        model_name: Name of the model for logging
        custom_objects: Dictionary of custom layers/objects
    """
    print(f"\n{'='*60}")
    print(f"Exporting {model_name} to ONNX")
    print(f"{'='*60}")
    
    # Load model
    print(f"[*] Loading model from: {model_path}")
    if custom_objects:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    else:
        model = tf.keras.models.load_model(model_path)
    print(f"   [OK] Model loaded")
    
    # Get model info
    print(f"[*] Model Information:")
    print(f"   Input shape: {model.input_shape}")
    if isinstance(model.output_shape, list):
        print(f"   Output shapes: {model.output_shape}")
    else:
        print(f"   Output shape: {model.output_shape}")
    
    # Convert to ONNX
    print(f"[*] Converting to ONNX format...")
    input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='input')]
    
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13
    )
    
    # Save ONNX model
    print(f"[*] Saving ONNX model to: {onnx_path}")
    onnx.save(onnx_model, onnx_path)
    
    # Get file sizes
    h5_size = os.path.getsize(model_path) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    
    print(f"[OK] Export complete!")
    print(f"   Original (.h5): {h5_size:.2f} MB")
    print(f"   ONNX (.onnx): {onnx_size:.2f} MB")
    print(f"   Size change: {((onnx_size/h5_size - 1) * 100):+.1f}%")
    
    return onnx_size

def verify_onnx_model(onnx_path, test_input):
    """
    Verify ONNX model can be loaded and run inference
    
    Args:
        onnx_path: Path to ONNX model
        test_input: Sample input for testing
    """
    print(f"\n[*] Verifying ONNX model: {onnx_path}")
    
    try:
        import onnxruntime as ort
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"   Input name: {input_name}")
        print(f"   Output names: {output_names}")
        
        # Run inference
        outputs = session.run(output_names, {input_name: test_input})
        
        print(f"   [OK] Inference successful")
        print(f"   Output shapes: {[out.shape for out in outputs]}")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Verification failed: {e}")
        return False

def main():
    """Main export function"""
    print("\n" + "="*60)
    print("ONNX MODEL EXPORT")
    print("="*60)
    
    # Create output directory
    os.makedirs('models/onnx', exist_ok=True)
    
    # Prepare test input
    test_input = np.random.randn(1, 875, 1).astype(np.float32)
    
    total_sizes = {}
    
    # 1. Export Domain Adaptation Model
    try:
        print("\n[1/3] Domain Adaptation Model")
        size = export_model_to_onnx(
            model_path='models/resnet_rppg_adapted.h5',
            onnx_path='models/onnx/domain_adaptation.onnx',
            model_name='Domain Adaptation (ResNet)'
        )
        total_sizes['Domain Adaptation'] = size
        verify_onnx_model('models/onnx/domain_adaptation.onnx', test_input)
    except Exception as e:
        print(f"[WARNING] Failed to export Domain Adaptation: {e}")
    
    # 2. Export Multi-Task Learning Model
    try:
        print("\n[2/3] Multi-Task Learning Model")
        size = export_model_to_onnx(
            model_path='models/multi_task_bp_model.h5',
            onnx_path='models/onnx/multi_task.onnx',
            model_name='Multi-Task Learning'
        )
        total_sizes['Multi-Task Learning'] = size
        verify_onnx_model('models/onnx/multi_task.onnx', test_input)
    except Exception as e:
        print(f"[WARNING] Failed to export Multi-Task Learning: {e}")
    
    # 3. Export Transformer Model
    try:
        print("\n[3/3] Transformer Model")
        custom_objects = {
            'MultiHeadAttention': MultiHeadAttention,
            'EncoderLayer': EncoderLayer,
            'TransformerEncoder': TransformerEncoder
        }
        size = export_model_to_onnx(
            model_path='models/transformer_bp_model.h5',
            onnx_path='models/onnx/transformer.onnx',
            model_name='Transformer',
            custom_objects=custom_objects
        )
        total_sizes['Transformer'] = size
        verify_onnx_model('models/onnx/transformer.onnx', test_input)
    except Exception as e:
        print(f"[WARNING] Failed to export Transformer: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    for model_name, size in total_sizes.items():
        print(f"   {model_name}: {size:.2f} MB")
    print(f"\n   Total ONNX models: {len(total_sizes)}")
    print(f"   Combined size: {sum(total_sizes.values()):.2f} MB")
    print("\n" + "="*60)
    print("OK All models exported to ONNX format!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Quantize models: python quantize_models.py")
    print("  2. Test inference: python test_onnx_inference.py")
    print("  3. Deploy to edge: python deploy_edge.py")

if __name__ == '__main__':
    main()
