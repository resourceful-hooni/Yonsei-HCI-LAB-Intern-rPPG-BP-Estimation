"""
train_transformer.py - Transformer Model Training Script
Phase 4: Transformer-based BP Estimation
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import argparse

from models.transformer_model import create_transformer_model, compile_transformer_model


def load_transformer_data(data_dir='data'):
    """Load data for transformer training"""
    print("[*] Loading data...")
    
    with h5py.File(f'{data_dir}/rppg_train.h5', 'r') as f:
        train_x = f['signals'][:]
        train_bp = f['labels'][:]
    
    with h5py.File(f'{data_dir}/rppg_val.h5', 'r') as f:
        val_x = f['signals'][:]
        val_bp = f['labels'][:]
    
    with h5py.File(f'{data_dir}/rppg_test.h5', 'r') as f:
        test_x = f['signals'][:]
        test_bp = f['labels'][:]
    
    if len(train_x.shape) == 2:
        train_x = train_x[:, :, np.newaxis]
        val_x = val_x[:, :, np.newaxis]
        test_x = test_x[:, :, np.newaxis]
    
    print(f"   Train: {train_x.shape[0]} samples")
    print(f"   Val:   {val_x.shape[0]} samples")
    print(f"   Test:  {test_x.shape[0]} samples")
    
    train_y = [train_bp[:, 0:1], train_bp[:, 1:2]]
    val_y = [val_bp[:, 0:1], val_bp[:, 1:2]]
    test_y = [test_bp[:, 0:1], test_bp[:, 1:2]]
    
    print("[OK] Data loaded")
    return train_x, train_y, val_x, val_y, test_x, test_y


def train_transformer_model(model, train_x, train_y, val_x, val_y,
                           epochs=30, batch_size=32, output_dir='models'):
    """Train Transformer model"""
    print("[*] Training model...")
    print(f"   Epochs: {epochs}, Batch size: {batch_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    best_model_path = os.path.join(output_dir, 'transformer_bp_model.h5')
    
    callbacks = [
        ModelCheckpoint(
            best_model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("[OK] Training completed")
    print(f"   Best model saved at: {best_model_path}")
    
    return history, best_model_path


def evaluate_transformer_model(model, test_x, test_y):
    """Evaluate Transformer model"""
    print("[*] Evaluating on test set")
    
    results = model.evaluate(test_x, test_y, verbose=0)
    
    print(f"   Total Loss: {results[0]:.4f}")
    print(f"   SBP Loss: {results[1]:.4f}")
    print(f"   DBP Loss: {results[2]:.4f}")
    print(f"   SBP MAE: {results[3]:.4f}")
    print(f"   DBP MAE: {results[4]:.4f}")
    
    predictions = model.predict(test_x, verbose=0)
    
    sbp_pred = predictions[0].flatten()
    dbp_pred = predictions[1].flatten()
    
    sbp_true = test_y[0].flatten()
    dbp_true = test_y[1].flatten()
    
    mae_sbp = np.mean(np.abs(sbp_pred - sbp_true))
    mae_dbp = np.mean(np.abs(dbp_pred - dbp_true))
    
    print("\n   Mean Absolute Error (MAE):")
    print(f"   SBP: {mae_sbp:.2f} mmHg")
    print(f"   DBP: {mae_dbp:.2f} mmHg")
    
    return mae_sbp, mae_dbp


def main():
    parser = argparse.ArgumentParser(description='Transformer Model Training')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Transformer Model Training")
    print("="*60)
    
    train_x, train_y, val_x, val_y, test_x, test_y = load_transformer_data(args.data_dir)
    
    model = create_transformer_model(
        input_shape=(875, 1),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    model = compile_transformer_model(model, learning_rate=args.learning_rate)
    
    print("\n[*] Model summary:")
    model.summary()
    
    history, best_model_path = train_transformer_model(
        model, train_x, train_y, val_x, val_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    mae_sbp, mae_dbp = evaluate_transformer_model(model, test_x, test_y)
    
    print("\n" + "="*60)
    print("OK Transformer Training Completed!")
    print("="*60)
    print("\nNext steps:")
    print(f"  1. Visualize results: python visualize_transformer.py")
    print(f"  2. Commit to GitHub: git add -A && git commit -m 'Phase 4: Transformer'")


if __name__ == '__main__':
    main()
