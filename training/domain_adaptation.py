"""
domain_adaptation.py - Domain Adaptation (PPG ??rPPG) ëª¨ë“ˆ

Phase 3-1: Domain Adaptation ?™ìŠµ

ëª©í‘œ: PPG ëª¨ë¸??rPPG ?°ì´?°ë¡œ fine-tuning
- Pre-trained ResNet ë¡œë“œ
- ë§ˆì?ë§??ˆì´??unfreeze
- rPPG ?°ì´?°ë¡œ fine-tuning
- ëª¨ë¸ ?€??
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from kapre import STFT, Magnitude, MagnitudeToDecibel
import argparse
import json


def load_pretrained_model(model_path):
    """
    Pre-trained PPG ëª¨ë¸ ë¡œë“œ
    
    Args:
        model_path: ëª¨ë¸ ê²½ë¡œ (e.g., data/resnet_ppg_nonmixed.h5)
    
    Returns:
        model: Keras ëª¨ë¸
    """
    print(f"\n?”„ Pre-trained ëª¨ë¸ ë¡œë“œ ì¤? {model_path}")
    
    dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel
    }
    
    model = ks.models.load_model(model_path, custom_objects=dependencies)
    print(f"   ??ëª¨ë¸ ë¡œë“œ ?„ë£Œ")
    print(f"   ?…ë ¥ ?•íƒœ: {model.input_shape}")
    print(f"   ì¶œë ¥ ?•íƒœ: {model.output_shape}")
    print(f"   ì´??Œë¼ë¯¸í„°: {model.count_params():,}")
    
    return model


def freeze_base_layers(model, num_unfreeze=3):
    """
    ê¸°ë³¸ ?ˆì´???™ê²° (?„ì´ ?™ìŠµ??
    
    Args:
        model: Keras ëª¨ë¸
        num_unfreeze: ë§ˆì?ë§‰ì—??unfreeze???ˆì´????
    
    Returns:
        model: ?˜ì •??ëª¨ë¸
    """
    print(f"\n?„ï¸ ê¸°ë³¸ ?ˆì´???™ê²° ì¤?(ë§ˆì?ë§?{num_unfreeze}ê°??œì™¸)")
    
    # ëª¨ë“  ?ˆì´???™ê²°
    for layer in model.layers[:-num_unfreeze]:
        layer.trainable = False
    
    # ë§ˆì?ë§?num_unfreeze ?ˆì´???œì„±??
    for layer in model.layers[-num_unfreeze:]:
        layer.trainable = True
    
    # ?™ê²° ?íƒœ ?•ì¸
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    frozen_count = sum([1 for layer in model.layers if not layer.trainable])
    
    print(f"   Trainable ?ˆì´?? {trainable_count}")
    print(f"   Frozen ?ˆì´?? {frozen_count}")
    print(f"   Trainable ?Œë¼ë¯¸í„°: {model.count_params():,}")
    
    return model


def load_rppg_data(data_dir='data'):
    """
    ë¶„í• ??rPPG ?°ì´??ë¡œë“œ
    
    Args:
        data_dir: ?°ì´???”ë ‰? ë¦¬
    
    Returns:
        train_x, train_y, val_x, val_y, test_x, test_y
    """
    print(f"\n?“‚ rPPG ?°ì´??ë¡œë“œ ì¤?)
    
    # Train
    with h5py.File(f'{data_dir}/rppg_train.h5', 'r') as f:
        train_x = f['signals'][:]
        train_y = f['labels'][:]
    
    # Val
    with h5py.File(f'{data_dir}/rppg_val.h5', 'r') as f:
        val_x = f['signals'][:]
        val_y = f['labels'][:]
    
    # Test
    with h5py.File(f'{data_dir}/rppg_test.h5', 'r') as f:
        test_x = f['signals'][:]
        test_y = f['labels'][:]
    
    print(f"   Train: {train_x.shape[0]} ?˜í”Œ")
    print(f"   Val:   {val_x.shape[0]} ?˜í”Œ")
    print(f"   Test:  {test_x.shape[0]} ?˜í”Œ")
    
    # ? í˜¸ ?•íƒœ ?•ì¸ ë°?ì¡°ì •
    print(f"   ? í˜¸ ?•íƒœ: {train_x.shape}")
    print(f"   ?ˆì´ë¸??•íƒœ: {train_y.shape}")
    
    # ? í˜¸ë¥?ëª¨ë¸ ?…ë ¥ ?•íƒœë¡?ë³€??(N, 875) ??(N, 875, 1)
    if len(train_x.shape) == 2:
        train_x = train_x[:, :, np.newaxis]
        val_x = val_x[:, :, np.newaxis]
        test_x = test_x[:, :, np.newaxis]
        print(f"   ë³€????? í˜¸ ?•íƒœ: {train_x.shape}")
    
    return train_x, train_y, val_x, val_y, test_x, test_y


def compile_model(model, learning_rate=0.001):
    """
    ëª¨ë¸ ì»´íŒŒ??
    
    Args:
        model: Keras ëª¨ë¸
        learning_rate: ?™ìŠµë¥?
    
    Returns:
        model: ì»´íŒŒ?¼ëœ ëª¨ë¸
    """
    print(f"\n?™ï¸ ëª¨ë¸ ì»´íŒŒ??(?™ìŠµë¥? {learning_rate})")
    
    optimizer = Adam(learning_rate=learning_rate)
    
    # ëª¨ë¸ ì¶œë ¥ ê°œìˆ˜???°ë¼ ?ì‹¤?¨ìˆ˜ ê²°ì •
    if isinstance(model.output, list):
        # ?¬ëŸ¬ ì¶œë ¥ (e.g., [SBP, DBP])
        loss = ['mse', 'mse']
    else:
        loss = 'mse'
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['mae']
    )
    
    print(f"   ??ì»´íŒŒ???„ë£Œ")
    
    return model


def train_domain_adaptation(model, train_x, train_y, val_x, val_y,
                           epochs=50, batch_size=32, output_dir='models'):
    """
    Domain adaptation ?™ìŠµ
    
    Args:
        model: ì»´íŒŒ?¼ëœ ëª¨ë¸
        train_x, train_y: ?™ìŠµ ?°ì´??
        val_x, val_y: ê²€ì¦??°ì´??
        epochs: ?í¬????
        batch_size: ë°°ì¹˜ ?¬ê¸°
        output_dir: ì¶œë ¥ ?”ë ‰? ë¦¬
    
    Returns:
        history: ?™ìŠµ ?´ë ¥
        best_model_path: ìµœê³  ?±ëŠ¥ ëª¨ë¸ ê²½ë¡œ
    """
    print(f"\n?“ Domain Adaptation ?™ìŠµ ì¤?..")
    print(f"   Epochs: {epochs}, Batch size: {batch_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    best_model_path = os.path.join(output_dir, 'resnet_rppg_adapted.h5')
    
    # ì½œë°± ?•ì˜
    callbacks = [
        # ìµœê³  ?±ëŠ¥ ëª¨ë¸ ?€??
        ModelCheckpoint(
            best_model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # ?™ìŠµë¥?ê°ì†Œ
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # ?™ìŠµ
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n   ???™ìŠµ ?„ë£Œ")
    print(f"   ìµœê³  ?±ëŠ¥ ëª¨ë¸: {best_model_path}")
    
    return history, best_model_path


def evaluate_model(model, test_x, test_y):
    """
    ëª¨ë¸ ?‰ê?
    
    Args:
        model: Keras ëª¨ë¸
        test_x, test_y: ?ŒìŠ¤???°ì´??
    """
    print(f"\n?“Š ?ŒìŠ¤?????‰ê?")
    
    # ?‰ê?
    results = model.evaluate(test_x, test_y, verbose=0)
    
    if isinstance(results, list):
        # ?¬ëŸ¬ ì¶œë ¥
        print(f"   Loss (SBP): {results[0]:.4f}")
        print(f"   Loss (DBP): {results[1]:.4f}")
        print(f"   MAE (SBP): {results[2]:.4f} mmHg")
        print(f"   MAE (DBP): {results[3]:.4f} mmHg")
    else:
        print(f"   Loss: {results[0]:.4f}")
        print(f"   MAE: {results[1]:.4f}")
    
    # ?ˆì¸¡
    predictions = model.predict(test_x, verbose=0)
    
    if isinstance(predictions, list):
        pred_sbp = predictions[0].flatten()
        pred_dbp = predictions[1].flatten()
        true_sbp = test_y[:, 0]
        true_dbp = test_y[:, 1]
        
        mae_sbp = np.mean(np.abs(pred_sbp - true_sbp))
        mae_dbp = np.mean(np.abs(pred_dbp - true_dbp))
        
        print(f"\n   ?‰ê·  ?ˆë? ?¤ì°¨ (MAE):")
        print(f"   SBP: {mae_sbp:.2f} mmHg")
        print(f"   DBP: {mae_dbp:.2f} mmHg")
        
        # ê°œì„ ??(ê¸°ì¡´ PPG ëª¨ë¸ vs ?ì‘ ëª¨ë¸)
        # ê¸°ì¡´ PPG ëª¨ë¸???ˆìƒ ?±ëŠ¥: SBP MAE ~28.9, DBP MAE ~15.2
        ppg_mae_sbp = 28.9
        ppg_mae_dbp = 15.2
        
        improvement_sbp = (ppg_mae_sbp - mae_sbp) / ppg_mae_sbp * 100
        improvement_dbp = (ppg_mae_dbp - mae_dbp) / ppg_mae_dbp * 100
        
        print(f"\n   PPG ?€ë¹?ê°œì„ ??")
        print(f"   SBP: {improvement_sbp:+.1f}% (ê¸°ì¡´: {ppg_mae_sbp:.2f} ??ê°œì„ : {mae_sbp:.2f})")
        print(f"   DBP: {improvement_dbp:+.1f}% (ê¸°ì¡´: {ppg_mae_dbp:.2f} ??ê°œì„ : {mae_dbp:.2f})")


def save_training_info(output_dir, history, best_model_path):
    """
    ?™ìŠµ ?•ë³´ ?€??
    
    Args:
        output_dir: ì¶œë ¥ ?”ë ‰? ë¦¬
        history: ?™ìŠµ ?´ë ¥
        best_model_path: ìµœê³  ?±ëŠ¥ ëª¨ë¸ ê²½ë¡œ
    """
    info_file = os.path.join(output_dir, 'training_info.json')
    
    info = {
        'model': 'ResNet (PPG ??rPPG Domain Adaptation)',
        'best_model': best_model_path,
        'epochs_trained': len(history.history['loss']),
        'best_epoch': np.argmin(history.history['val_loss']) + 1,
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_loss': float(np.min(history.history['val_loss'])),
        'history': {
            'train_loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
    }
    
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n   ???™ìŠµ ?•ë³´ ?€?? {info_file}")


def main():
    parser = argparse.ArgumentParser(description='Domain Adaptation ?™ìŠµ')
    parser.add_argument('--pretrained', type=str,
                       default='data/resnet_ppg_nonmixed.h5',
                       help='Pre-trained ëª¨ë¸ ê²½ë¡œ')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='?°ì´???”ë ‰? ë¦¬')
    parser.add_argument('--epochs', type=int, default=50,
                       help='?í¬????)
    parser.add_argument('--batch-size', type=int, default=32,
                       help='ë°°ì¹˜ ?¬ê¸°')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='?™ìŠµë¥?)
    parser.add_argument('--output-dir', type=str, default='models',
                       help='ì¶œë ¥ ?”ë ‰? ë¦¬')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Domain Adaptation: PPG ??rPPG")
    print("="*60)
    
    # 1. ëª¨ë¸ ë¡œë“œ
    model = load_pretrained_model(args.pretrained)
    
    # 2. ê¸°ë³¸ ?ˆì´???™ê²°
    model = freeze_base_layers(model, num_unfreeze=3)
    
    # 3. ?°ì´??ë¡œë“œ
    train_x, train_y, val_x, val_y, test_x, test_y = load_rppg_data(args.data_dir)
    
    # 4. ëª¨ë¸ ì»´íŒŒ??
    model = compile_model(model, learning_rate=args.learning_rate)
    
    # 5. ?™ìŠµ
    history, best_model_path = train_domain_adaptation(
        model, train_x, train_y, val_x, val_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    # 6. ?‰ê?
    evaluate_model(model, test_x, test_y)
    
    # 7. ?•ë³´ ?€??
    save_training_info(args.output_dir, history, best_model_path)
    
    print("\n" + "="*60)
    print("??Domain Adaptation ?™ìŠµ ?„ë£Œ!")
    print("="*60)
    print(f"\n?¤ìŒ ?¨ê³„:")
    print(f"  1. ëª¨ë¸ ?‰ê?: camera_rppg_advanced.py --model {best_model_path}")
    print(f"  2. GitHub commit: git add -A && git commit -m 'Phase 3-1: Domain Adaptation'")


if __name__ == '__main__':
    main()
