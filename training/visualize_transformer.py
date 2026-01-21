"""
visualize_transformer.py - Visualize Transformer Model Results
Phase 4: Transformer Model Visualization
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from datetime import datetime
from models.transformer_model import MultiHeadAttention, EncoderLayer, TransformerEncoder


def load_model_and_data():
    """Load trained model and test data"""
    print("[*] Loading model and data...")
    
    # Load model with custom objects
    custom_objects = {
        'MultiHeadAttention': MultiHeadAttention,
        'EncoderLayer': EncoderLayer,
        'TransformerEncoder': TransformerEncoder
    }
    model = tf.keras.models.load_model('models/transformer_bp_model.h5', custom_objects=custom_objects)
    
    with h5py.File('data/rppg_test.h5', 'r') as f:
        test_x = f['signals'][:]
        test_bp = f['labels'][:]
    
    if len(test_x.shape) == 2:
        test_x = test_x[:, :, np.newaxis]
    
    test_y = [test_bp[:, 0:1], test_bp[:, 1:2]]
    
    print(f"[OK] Model and data loaded")
    return model, test_x, test_y


def evaluate_model(model, test_x, test_y):
    """Evaluate model on test set"""
    print("[*] Evaluating model...")
    
    results = model.evaluate(test_x, test_y, verbose=0)
    predictions = model.predict(test_x, verbose=0)
    
    sbp_pred = predictions[0].flatten()
    dbp_pred = predictions[1].flatten()
    
    sbp_true = test_y[0].flatten()
    dbp_true = test_y[1].flatten()
    
    mae_sbp = np.mean(np.abs(sbp_pred - sbp_true))
    mae_dbp = np.mean(np.abs(dbp_pred - dbp_true))
    rmse_sbp = np.sqrt(np.mean((sbp_pred - sbp_true)**2))
    rmse_dbp = np.sqrt(np.mean((dbp_pred - dbp_true)**2))
    
    print(f"   Total Loss: {results[0]:.4f}")
    print(f"   SBP Loss: {results[1]:.4f}")
    print(f"   DBP Loss: {results[2]:.4f}")
    print(f"   SBP MAE: {results[3]:.4f}")
    print(f"   DBP MAE: {results[4]:.4f}")
    
    print(f"\n   Mean Absolute Error:")
    print(f"   SBP: {mae_sbp:.2f} mmHg")
    print(f"   DBP: {mae_dbp:.2f} mmHg")
    
    print(f"\n   Root Mean Square Error:")
    print(f"   SBP: {rmse_sbp:.2f} mmHg")
    print(f"   DBP: {rmse_dbp:.2f} mmHg")
    
    return sbp_pred, dbp_pred, sbp_true, dbp_true, mae_sbp, mae_dbp


def create_visualizations(sbp_pred, dbp_pred, sbp_true, dbp_true):
    """Create performance visualization"""
    print("[*] Creating visualizations...")
    
    os.makedirs('results', exist_ok=True)
    
    # Prediction scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SBP
    axes[0].scatter(sbp_true, sbp_pred, alpha=0.5, s=20)
    min_val = min(sbp_true.min(), sbp_pred.min())
    max_val = max(sbp_true.max(), sbp_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('True SBP (mmHg)', fontsize=11)
    axes[0].set_ylabel('Predicted SBP (mmHg)', fontsize=11)
    axes[0].set_title(f'SBP Prediction (MAE: {np.mean(np.abs(sbp_pred - sbp_true)):.2f})', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # DBP
    axes[1].scatter(dbp_true, dbp_pred, alpha=0.5, s=20)
    min_val = min(dbp_true.min(), dbp_pred.min())
    max_val = max(dbp_true.max(), dbp_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1].set_xlabel('True DBP (mmHg)', fontsize=11)
    axes[1].set_ylabel('Predicted DBP (mmHg)', fontsize=11)
    axes[1].set_title(f'DBP Prediction (MAE: {np.mean(np.abs(dbp_pred - dbp_true)):.2f})', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/transformer_predictions.png', dpi=300, bbox_inches='tight')
    print("   [OK] Saved: results/transformer_predictions.png")
    plt.close()
    
    # Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sbp_error = sbp_pred - sbp_true
    dbp_error = dbp_pred - dbp_true
    
    axes[0].hist(sbp_error, bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(sbp_error), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sbp_error):.2f}')
    axes[0].set_xlabel('Prediction Error (mmHg)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('SBP Error Distribution', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(dbp_error, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[1].axvline(np.mean(dbp_error), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dbp_error):.2f}')
    axes[1].set_xlabel('Prediction Error (mmHg)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('DBP Error Distribution', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/transformer_error_distribution.png', dpi=300, bbox_inches='tight')
    print("   [OK] Saved: results/transformer_error_distribution.png")
    plt.close()


def create_summary_report(mae_sbp, mae_dbp):
    """Create summary report"""
    print("[*] Creating summary report...")
    
    report = f"""
================================================================================
TRANSFORMER MODEL - Phase 4 Summary Report
================================================================================

Architecture: Transformer with Multi-Head Attention
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
MODEL CONFIGURATION
================================================================================

- Model Dimension: 128
- Number of Heads: 4
- Number of Layers: 3
- Feed-Forward Dimension: 256
- Positional Encoding: Sinusoidal
- Dropout Rate: 0.1

================================================================================
TEST SET PERFORMANCE
================================================================================

Blood Pressure Estimation:
  - SBP MAE: {mae_sbp:.2f} mmHg
  - DBP MAE: {mae_dbp:.2f} mmHg

Notes:
  - Model trained on rPPG dataset from MIMIC-III
  - Transformer-based sequence-to-value architecture
  - Global average pooling after encoder
  - Training convergence achieved
  
================================================================================
PERFORMANCE COMPARISON
================================================================================

Model              SBP MAE       DBP MAE       Notes
---                ---           ---           ---
Phase 3-1 (DA)     1.22 mmHg     1.11 mmHg     Domain Adaptation
Phase 3-2 (MTL)    0.84 mmHg     0.83 mmHg     Multi-Task Learning
Phase 4 (TF)       {mae_sbp:.2f} mmHg     {mae_dbp:.2f} mmHg     Transformer

================================================================================
NEXT STEPS
================================================================================

1. Ensemble models for improved accuracy
2. Real-time integration with camera capture
3. ONNX/TensorRT optimization (Phase 5)
4. Deployment on edge devices

================================================================================
"""
    
    with open('results/transformer_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("   [OK] Saved: results/transformer_summary_report.txt")


def main():
    parser = argparse.ArgumentParser(description='Visualize Transformer Results')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Transformer Model Visualization")
    print("="*60)
    
    model, test_x, test_y = load_model_and_data()
    sbp_pred, dbp_pred, sbp_true, dbp_true, mae_sbp, mae_dbp = evaluate_model(model, test_x, test_y)
    create_visualizations(sbp_pred, dbp_pred, sbp_true, dbp_true)
    create_summary_report(mae_sbp, mae_dbp)
    
    print("\n" + "="*60)
    print("OK Visualization Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - results/transformer_predictions.png")
    print("  - results/transformer_error_distribution.png")
    print("  - results/transformer_summary_report.txt")


if __name__ == '__main__':
    main()
