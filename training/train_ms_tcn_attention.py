"""
train_ms_tcn_attention.py - Training Script for MS-TCN + Linear Attention Model
================================================================================
Author: Yonsei HCI LAB
Date: 2026-01-22

Features:
---------
1. Comprehensive training pipeline with callbacks
2. Mixup data augmentation for better generalization
3. Learning rate scheduling with warmup
4. Detailed logging and visualization
5. Early stopping with best model checkpointing
6. Multi-metric evaluation (MAE, RMSE, R²)

Usage:
------
python training/train_ms_tcn_attention.py --epochs 100 --batch-size 32
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, LearningRateScheduler, CSVLogger
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ms_tcn_attention_model import (
    create_ms_tcn_attention_model, compile_model, get_custom_objects, print_model_info
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_dir='data'):
    """Load training, validation, and test data"""
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    datasets = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(data_dir, f'rppg_{split}.h5')
        with h5py.File(path, 'r') as f:
            signals = f['signals'][:]
            labels = f['labels'][:]
        
        # Add channel dimension if needed
        if len(signals.shape) == 2:
            signals = signals[:, :, np.newaxis]
        
        datasets[split] = {
            'x': signals.astype(np.float32),
            'y_sbp': labels[:, 0:1].astype(np.float32),
            'y_dbp': labels[:, 1:2].astype(np.float32)
        }
        
        print(f"  {split.upper():5s}: {signals.shape[0]:5d} samples, "
              f"signal shape {signals.shape[1:]}")
    
    # Load normalization info
    info_path = os.path.join(data_dir, 'rppg_info.txt')
    if os.path.exists(info_path):
        # Parse label statistics from info file
        with open(info_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Default values if parsing fails
        label_mean = np.array([143.398, 65.682])
        label_scale = np.array([14.967, 11.297])
        
        print(f"  Label denormalization: mean={label_mean}, scale={label_scale}")
        datasets['label_mean'] = label_mean
        datasets['label_scale'] = label_scale
    
    print("="*60 + "\n")
    return datasets


# =============================================================================
# DATA AUGMENTATION - MIXUP
# =============================================================================

def mixup_data(x, y_sbp, y_dbp, alpha=0.2):
    """
    Apply mixup augmentation
    
    Mixup creates virtual training examples:
    x_new = lambda * x_i + (1 - lambda) * x_j
    y_new = lambda * y_i + (1 - lambda) * y_j
    
    where lambda ~ Beta(alpha, alpha)
    """
    if alpha <= 0:
        return x, y_sbp, y_dbp
    
    batch_size = len(x)
    
    # Sample mixup coefficient
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.maximum(lam, 1 - lam)  # Ensure lam >= 0.5
    lam = lam.reshape(-1, 1, 1)
    lam_y = lam.reshape(-1, 1)
    
    # Shuffle indices
    indices = np.random.permutation(batch_size)
    
    # Mix
    x_mixed = lam * x + (1 - lam) * x[indices]
    y_sbp_mixed = lam_y * y_sbp + (1 - lam_y) * y_sbp[indices]
    y_dbp_mixed = lam_y * y_dbp + (1 - lam_y) * y_dbp[indices]
    
    return x_mixed, y_sbp_mixed, y_dbp_mixed


class MixupGenerator(tf.keras.utils.Sequence):
    """Data generator with mixup augmentation"""
    
    def __init__(self, x, y_sbp, y_dbp, batch_size=32, shuffle=True, mixup_alpha=0.2):
        self.x = x
        self.y_sbp = y_sbp
        self.y_dbp = y_dbp
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mixup_alpha = mixup_alpha
        self.indices = np.arange(len(x))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        x_batch = self.x[batch_indices]
        y_sbp_batch = self.y_sbp[batch_indices]
        y_dbp_batch = self.y_dbp[batch_indices]
        
        # Apply mixup
        if self.mixup_alpha > 0:
            x_batch, y_sbp_batch, y_dbp_batch = mixup_data(
                x_batch, y_sbp_batch, y_dbp_batch, self.mixup_alpha
            )
        
        return x_batch, {'sbp_output': y_sbp_batch, 'dbp_output': y_dbp_batch}
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def warmup_cosine_decay(epoch, total_epochs, warmup_epochs, initial_lr, min_lr=1e-6):
    """
    Learning rate schedule with linear warmup and cosine decay
    
    - Warmup: Linear increase from min_lr to initial_lr
    - Decay: Cosine annealing to min_lr
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return min_lr + (initial_lr - min_lr) * (epoch / warmup_epochs)
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))


class WarmupCosineDecay(tf.keras.callbacks.Callback):
    """Callback for warmup + cosine decay learning rate schedule"""
    
    def __init__(self, total_epochs, warmup_epochs=5, initial_lr=0.001, min_lr=1e-6):
        super().__init__()
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.history = []
    
    def on_epoch_begin(self, epoch, logs=None):
        lr = warmup_cosine_decay(
            epoch, self.total_epochs, self.warmup_epochs,
            self.initial_lr, self.min_lr
        )
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.history.append(lr)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_model(model, x, y_sbp, y_dbp, label_mean, label_scale, prefix=''):
    """
    Comprehensive model evaluation
    
    Returns metrics in both normalized and original scales
    """
    # Predict
    pred_sbp, pred_dbp = model.predict(x, verbose=0, batch_size=64)
    pred_sbp = pred_sbp.flatten()
    pred_dbp = pred_dbp.flatten()
    y_sbp = y_sbp.flatten()
    y_dbp = y_dbp.flatten()
    
    # Metrics in normalized scale
    metrics_norm = {
        'sbp_mae': mean_absolute_error(y_sbp, pred_sbp),
        'sbp_rmse': np.sqrt(mean_squared_error(y_sbp, pred_sbp)),
        'sbp_r2': r2_score(y_sbp, pred_sbp),
        'dbp_mae': mean_absolute_error(y_dbp, pred_dbp),
        'dbp_rmse': np.sqrt(mean_squared_error(y_dbp, pred_dbp)),
        'dbp_r2': r2_score(y_dbp, pred_dbp),
    }
    
    # Convert to original scale (mmHg)
    pred_sbp_mmhg = pred_sbp * label_scale[0] + label_mean[0]
    pred_dbp_mmhg = pred_dbp * label_scale[1] + label_mean[1]
    y_sbp_mmhg = y_sbp * label_scale[0] + label_mean[0]
    y_dbp_mmhg = y_dbp * label_scale[1] + label_mean[1]
    
    # Metrics in mmHg
    metrics_mmhg = {
        'sbp_mae_mmhg': mean_absolute_error(y_sbp_mmhg, pred_sbp_mmhg),
        'sbp_rmse_mmhg': np.sqrt(mean_squared_error(y_sbp_mmhg, pred_sbp_mmhg)),
        'dbp_mae_mmhg': mean_absolute_error(y_dbp_mmhg, pred_dbp_mmhg),
        'dbp_rmse_mmhg': np.sqrt(mean_squared_error(y_dbp_mmhg, pred_dbp_mmhg)),
    }
    
    # Prediction statistics
    pred_stats = {
        'sbp_pred_mean': float(np.mean(pred_sbp)),
        'sbp_pred_std': float(np.std(pred_sbp)),
        'dbp_pred_mean': float(np.mean(pred_dbp)),
        'dbp_pred_std': float(np.std(pred_dbp)),
    }
    
    all_metrics = {**metrics_norm, **metrics_mmhg, **pred_stats}
    
    # Add prefix
    if prefix:
        all_metrics = {f'{prefix}_{k}': v for k, v in all_metrics.items()}
    
    return all_metrics, (pred_sbp_mmhg, pred_dbp_mmhg, y_sbp_mmhg, y_dbp_mmhg)


def print_evaluation(metrics, title='Evaluation Results'):
    """Pretty print evaluation metrics"""
    print(f"\n{title}")
    print("-" * 50)
    
    # Normalized metrics
    print("Normalized Scale:")
    for key in ['sbp_mae', 'sbp_rmse', 'sbp_r2', 'dbp_mae', 'dbp_rmse', 'dbp_r2']:
        full_key = [k for k in metrics.keys() if k.endswith(key)][0]
        print(f"  {key:12s}: {metrics[full_key]:.4f}")
    
    # mmHg metrics
    print("\nOriginal Scale (mmHg):")
    for key in ['sbp_mae_mmhg', 'sbp_rmse_mmhg', 'dbp_mae_mmhg', 'dbp_rmse_mmhg']:
        full_key = [k for k in metrics.keys() if k.endswith(key)][0]
        print(f"  {key:15s}: {metrics[full_key]:.2f} mmHg")
    
    # Prediction stats
    print("\nPrediction Statistics:")
    for key in ['sbp_pred_mean', 'sbp_pred_std', 'dbp_pred_mean', 'dbp_pred_std']:
        full_key = [k for k in metrics.keys() if k.endswith(key)][0]
        print(f"  {key:15s}: {metrics[full_key]:.4f}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_history(history, save_dir='results'):
    """Plot training curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss
    axes[0, 0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=12)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # SBP MAE
    axes[0, 1].plot(history.history['sbp_output_mae'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_sbp_output_mae'], label='Val', linewidth=2)
    axes[0, 1].set_title('SBP MAE (Normalized)', fontsize=12)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # DBP MAE
    axes[1, 0].plot(history.history['dbp_output_mae'], label='Train', linewidth=2)
    axes[1, 0].plot(history.history['val_dbp_output_mae'], label='Val', linewidth=2)
    axes[1, 0].set_title('DBP MAE (Normalized)', fontsize=12)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2, color='green')
        axes[1, 1].set_title('Learning Rate', fontsize=12)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ms_tcn_training_curves.png'), dpi=150)
    plt.close()
    print(f"  Training curves saved to {save_dir}/ms_tcn_training_curves.png")


def plot_predictions(predictions, save_dir='results'):
    """Plot prediction vs actual scatter plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    pred_sbp, pred_dbp, y_sbp, y_dbp = predictions
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # SBP
    axes[0].scatter(y_sbp, pred_sbp, alpha=0.5, s=10, c='blue')
    axes[0].plot([y_sbp.min(), y_sbp.max()], [y_sbp.min(), y_sbp.max()], 
                'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual SBP (mmHg)', fontsize=12)
    axes[0].set_ylabel('Predicted SBP (mmHg)', fontsize=12)
    axes[0].set_title('SBP: Predicted vs Actual', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # DBP
    axes[1].scatter(y_dbp, pred_dbp, alpha=0.5, s=10, c='green')
    axes[1].plot([y_dbp.min(), y_dbp.max()], [y_dbp.min(), y_dbp.max()], 
                'r--', linewidth=2, label='Perfect prediction')
    axes[1].set_xlabel('Actual DBP (mmHg)', fontsize=12)
    axes[1].set_ylabel('Predicted DBP (mmHg)', fontsize=12)
    axes[1].set_title('DBP: Predicted vs Actual', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ms_tcn_predictions.png'), dpi=150)
    plt.close()
    print(f"  Prediction plots saved to {save_dir}/ms_tcn_predictions.png")


def plot_bland_altman(predictions, save_dir='results'):
    """Bland-Altman plot for agreement analysis"""
    os.makedirs(save_dir, exist_ok=True)
    
    pred_sbp, pred_dbp, y_sbp, y_dbp = predictions
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, (pred, actual, name, ax) in enumerate([
        (pred_sbp, y_sbp, 'SBP', axes[0]),
        (pred_dbp, y_dbp, 'DBP', axes[1])
    ]):
        mean = (pred + actual) / 2
        diff = pred - actual
        
        md = np.mean(diff)
        sd = np.std(diff)
        
        ax.scatter(mean, diff, alpha=0.5, s=10)
        ax.axhline(md, color='red', linestyle='-', label=f'Mean: {md:.2f}')
        ax.axhline(md + 1.96*sd, color='gray', linestyle='--', 
                  label=f'+1.96 SD: {md + 1.96*sd:.2f}')
        ax.axhline(md - 1.96*sd, color='gray', linestyle='--',
                  label=f'-1.96 SD: {md - 1.96*sd:.2f}')
        ax.set_xlabel(f'Mean of Predicted and Actual {name} (mmHg)', fontsize=12)
        ax.set_ylabel(f'Predicted - Actual {name} (mmHg)', fontsize=12)
        ax.set_title(f'Bland-Altman Plot: {name}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ms_tcn_bland_altman.png'), dpi=150)
    plt.close()
    print(f"  Bland-Altman plots saved to {save_dir}/ms_tcn_bland_altman.png")


def plot_error_distribution(predictions, save_dir='results'):
    """Error distribution histogram"""
    os.makedirs(save_dir, exist_ok=True)
    
    pred_sbp, pred_dbp, y_sbp, y_dbp = predictions
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SBP error
    sbp_error = pred_sbp - y_sbp
    axes[0].hist(sbp_error, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0].axvline(np.mean(sbp_error), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sbp_error):.2f}')
    axes[0].axvline(np.mean(sbp_error) + np.std(sbp_error), color='orange', 
                   linestyle=':', label=f'Std: {np.std(sbp_error):.2f}')
    axes[0].axvline(np.mean(sbp_error) - np.std(sbp_error), color='orange', linestyle=':')
    axes[0].set_xlabel('SBP Error (mmHg)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('SBP Error Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # DBP error
    dbp_error = pred_dbp - y_dbp
    axes[1].hist(dbp_error, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(np.mean(dbp_error), color='red', linestyle='--',
                   label=f'Mean: {np.mean(dbp_error):.2f}')
    axes[1].axvline(np.mean(dbp_error) + np.std(dbp_error), color='orange',
                   linestyle=':', label=f'Std: {np.std(dbp_error):.2f}')
    axes[1].axvline(np.mean(dbp_error) - np.std(dbp_error), color='orange', linestyle=':')
    axes[1].set_xlabel('DBP Error (mmHg)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('DBP Error Distribution', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ms_tcn_error_distribution.png'), dpi=150)
    plt.close()
    print(f"  Error distribution saved to {save_dir}/ms_tcn_error_distribution.png")


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(args):
    """Main training function"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "="*80)
    print("MS-TCN + Linear Attention - Blood Pressure Estimation Training")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Arguments: {vars(args)}")
    
    # Load data
    data = load_data(args.data_dir)
    
    # Create model
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)
    
    model = create_ms_tcn_attention_model(input_shape=(875, 1))
    model = compile_model(model, learning_rate=args.lr)
    print_model_info(model)
    
    # Setup callbacks
    callbacks = []
    
    # 1. Model checkpoint (save weights only to avoid h5py issues)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, 'ms_tcn_attention_best_weights.h5')
    callbacks.append(ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,  # Save weights only to avoid h5py serialization issues
        verbose=1
    ))
    
    # 2. Early stopping
    callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        restore_best_weights=True,
        verbose=1
    ))
    
    # 3. Learning rate schedule
    if args.use_warmup:
        lr_callback = WarmupCosineDecay(
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            initial_lr=args.lr,
            min_lr=1e-6
        )
        callbacks.append(lr_callback)
    else:
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ))
    
    # 4. CSV Logger
    csv_path = os.path.join(args.output_dir, 'ms_tcn_training_log.csv')
    callbacks.append(CSVLogger(csv_path))
    
    # 5. TensorBoard (optional)
    if args.tensorboard:
        tb_dir = os.path.join(args.output_dir, 'tensorboard', timestamp)
        callbacks.append(TensorBoard(log_dir=tb_dir, histogram_freq=1))
    
    # Create data generators
    if args.mixup_alpha > 0:
        print(f"\nUsing Mixup augmentation (alpha={args.mixup_alpha})")
        train_gen = MixupGenerator(
            data['train']['x'], data['train']['y_sbp'], data['train']['y_dbp'],
            batch_size=args.batch_size, shuffle=True, mixup_alpha=args.mixup_alpha
        )
        use_generator = True
    else:
        use_generator = False
    
    # Training
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    
    if use_generator:
        history = model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=(
                data['val']['x'],
                {'sbp_output': data['val']['y_sbp'], 'dbp_output': data['val']['y_dbp']}
            ),
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            data['train']['x'],
            {'sbp_output': data['train']['y_sbp'], 'dbp_output': data['train']['y_dbp']},
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(
                data['val']['x'],
                {'sbp_output': data['val']['y_sbp'], 'dbp_output': data['val']['y_dbp']}
            ),
            callbacks=callbacks,
            verbose=1
        )
    
    # Load best weights
    model.load_weights(checkpoint_path)
    print(f"\nLoaded best weights from: {checkpoint_path}")
    
    # Save final model in SavedModel format (more reliable)
    final_path = os.path.join(args.output_dir, 'ms_tcn_attention_final')
    model.save(final_path, save_format='tf')
    print(f"Final model saved to: {final_path}")
    
    # Also save weights to data directory for easy access
    import shutil
    best_model_path = os.path.join('data', 'ms_tcn_attention_bp_weights.h5')
    shutil.copy(checkpoint_path, best_model_path)
    print(f"Best weights copied to: {best_model_path}")
    
    # Evaluation
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    
    label_mean = data.get('label_mean', np.array([143.398, 65.682]))
    label_scale = data.get('label_scale', np.array([14.967, 11.297]))
    
    # Evaluate on all splits
    all_metrics = {}
    for split in ['train', 'val', 'test']:
        metrics, predictions = evaluate_model(
            model, data[split]['x'], data[split]['y_sbp'], data[split]['y_dbp'],
            label_mean, label_scale, prefix=split
        )
        all_metrics.update(metrics)
        print_evaluation(metrics, f"{split.upper()} Results")
        
        # Save test predictions for visualization
        if split == 'test':
            test_predictions = predictions
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'ms_tcn_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in all_metrics.items()}, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Visualization
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    plot_training_history(history, args.output_dir)
    plot_predictions(test_predictions, args.output_dir)
    plot_bland_altman(test_predictions, args.output_dir)
    plot_error_distribution(test_predictions, args.output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest Test Performance:")
    print(f"  SBP MAE: {all_metrics['test_sbp_mae_mmhg']:.2f} mmHg")
    print(f"  DBP MAE: {all_metrics['test_dbp_mae_mmhg']:.2f} mmHg")
    print(f"  SBP RMSE: {all_metrics['test_sbp_rmse_mmhg']:.2f} mmHg")
    print(f"  DBP RMSE: {all_metrics['test_dbp_rmse_mmhg']:.2f} mmHg")
    print(f"  SBP R²: {all_metrics['test_sbp_r2']:.4f}")
    print(f"  DBP R²: {all_metrics['test_dbp_r2']:.4f}")
    print("\nOutput Files:")
    print(f"  Best Model: {checkpoint_path}")
    print(f"  Final Model: {final_path}")
    print(f"  Quick Access: {best_model_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Training Log: {csv_path}")
    print("="*80 + "\n")
    
    return model, history, all_metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train MS-TCN + Linear Attention model for BP estimation'
    )
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing data files')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory for output files')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    # Augmentation
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                       help='Mixup alpha (0 = disabled)')
    
    # Learning rate schedule
    parser.add_argument('--use-warmup', action='store_true', default=True,
                       help='Use warmup + cosine decay LR schedule')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Number of warmup epochs')
    
    # Logging
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable TensorBoard logging')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
