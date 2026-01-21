"""
?¬ì „ ?™ìŠµ??ëª¨ë¸???¬ìš©?˜ì—¬ ?ˆì••???ˆì¸¡?˜ëŠ” ?ŒìŠ¤???¤í¬ë¦½íŠ¸

?¬ìš©ë²?
    python test_model.py --model data/alexnet_ppg_nonmixed.h5 --data data/MIMIC-III_ppg_dataset.h5
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU ?„ì „ ë¹„í™œ?±í™”
import argparse
import numpy as np
import h5py
import tensorflow as tf
import tensorflow.keras as ks
from kapre import STFT, Magnitude, MagnitudeToDecibel
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def load_model(model_path):
    """
    ?¬ì „ ?™ìŠµ??ëª¨ë¸??ë¡œë“œ?©ë‹ˆ??
    
    Args:
        model_path: ëª¨ë¸ ?Œì¼ ê²½ë¡œ (.h5 ?Œì¼)
    
    Returns:
        ë¡œë“œ??Keras ëª¨ë¸
    """
    print(f"ëª¨ë¸ ë¡œë“œ ì¤? {model_path}")
    
    # Kapre ?ˆì´?´ë? ?„í•œ custom objects ?•ì˜
    dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel
    }
    
    model = ks.models.load_model(model_path, custom_objects=dependencies)
    print("ëª¨ë¸ ë¡œë“œ ?„ë£Œ!")
    print("\nëª¨ë¸ êµ¬ì¡°:")
    model.summary()
    
    return model


def load_test_data(data_path, n_samples=100):
    """
    ?ŒìŠ¤???°ì´?°ë? ë¡œë“œ?©ë‹ˆ??
    
    Args:
        data_path: HDF5 ?°ì´???Œì¼ ê²½ë¡œ
        n_samples: ?ŒìŠ¤?¸í•  ?˜í”Œ ??
    
    Returns:
        ppg_signals: PPG ? í˜¸ ë°°ì—´ (n_samples, 875)
        sbp_values: ?¤ì œ ?˜ì¶•ê¸??ˆì•• ê°?
        dbp_values: ?¤ì œ ?´ì™„ê¸??ˆì•• ê°?
    """
    print(f"\n?ŒìŠ¤???°ì´??ë¡œë“œ ì¤? {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        print(f"?¬ìš© ê°€?¥í•œ ?°ì´?°ì…‹ ?? {list(f.keys())}")
        
        # PPG ? í˜¸ ë¡œë“œ
        if 'ppg' in f.keys():
            ppg_data = f['ppg'][:]
        elif 'signal' in f.keys():
            ppg_data = f['signal'][:]
        else:
            raise ValueError(f"PPG ?°ì´?°ë? ì°¾ì„ ???†ìŠµ?ˆë‹¤. ?¬ìš© ê°€?¥í•œ ?? {list(f.keys())}")
        
        # ?ˆì•• ê°?ë¡œë“œ
        if 'sbp' in f.keys() and 'dbp' in f.keys():
            sbp_data = f['sbp'][:]
            dbp_data = f['dbp'][:]
        else:
            raise ValueError(f"SBP/DBP ?°ì´?°ë? ì°¾ì„ ???†ìŠµ?ˆë‹¤. ?¬ìš© ê°€?¥í•œ ?? {list(f.keys())}")
    
    # ?˜í”Œ ???œí•œ
    n_samples = min(n_samples, len(ppg_data))
    ppg_signals = ppg_data[:n_samples]
    sbp_values = sbp_data[:n_samples]
    dbp_values = dbp_data[:n_samples]
    
    print(f"ë¡œë“œ???°ì´???•íƒœ:")
    print(f"  PPG ? í˜¸: {ppg_signals.shape}")
    print(f"  SBP ê°? {sbp_values.shape}")
    print(f"  DBP ê°? {dbp_values.shape}")
    
    return ppg_signals, sbp_values, dbp_values


def preprocess_signals(signals):
    """
    ? í˜¸ë¥?ëª¨ë¸ ?…ë ¥ ?•íƒœë¡??„ì²˜ë¦¬í•©?ˆë‹¤.
    
    Args:
        signals: PPG ? í˜¸ ë°°ì—´
    
    Returns:
        ?„ì²˜ë¦¬ëœ ? í˜¸ (n_samples, 875, 1)
    """
    # ? í˜¸ ê¸¸ì´ê°€ 875ê°€ ?„ë‹Œ ê²½ìš° ì²˜ë¦¬
    if signals.shape[1] != 875:
        print(f"ê²½ê³ : ? í˜¸ ê¸¸ì´ê°€ {signals.shape[1]}?…ë‹ˆ?? 875ë¡?ë¦¬ì‚¬?´ì§•?©ë‹ˆ??")
        # ê°„ë‹¨??ë¦¬ìƒ˜?Œë§ ?ëŠ” ?¨ë”©
        if signals.shape[1] > 875:
            signals = signals[:, :875]
        else:
            pad_width = ((0, 0), (0, 875 - signals.shape[1]))
            signals = np.pad(signals, pad_width, mode='edge')
    
    # ì°¨ì› ì¶”ê? (samples, 875) -> (samples, 875, 1)
    if len(signals.shape) == 2:
        signals = np.expand_dims(signals, axis=-1)
    
    return signals


def predict_bp(model, ppg_signals):
    """
    ëª¨ë¸???¬ìš©?˜ì—¬ ?ˆì••???ˆì¸¡?©ë‹ˆ??
    
    Args:
        model: ?™ìŠµ??Keras ëª¨ë¸
        ppg_signals: PPG ? í˜¸ ë°°ì—´ (n_samples, 875, 1)
    
    Returns:
        predictions: ?ˆì¸¡???ˆì•• ê°?(n_samples, 2) - [SBP, DBP]
    """
    print("\n?ˆì•• ?ˆì¸¡ ì¤?..")
    raw_pred = model.predict(ppg_signals, batch_size=32, verbose=1)
    print("?ˆì¸¡ ?„ë£Œ!")
    
    # ?¨ì¼ ?ì„œ (n,2)??ê²½ìš° ê·¸ë?ë¡?ë°˜í™˜
    if hasattr(raw_pred, 'shape') and raw_pred.ndim >= 2 and raw_pred.shape[-1] == 2:
        return raw_pred
    # ??ì¶œë ¥ ë¦¬ìŠ¤??([sbp_batch, dbp_batch])??ê²½ìš° ê²°í•©
    elif isinstance(raw_pred, (list, tuple)) and len(raw_pred) == 2:
        sbp_batch, dbp_batch = raw_pred
        sbp_batch = np.squeeze(sbp_batch)
        dbp_batch = np.squeeze(dbp_batch)
        # (n,) ?•íƒœë¥?(n,1)ë¡?ë³€????concatenate
        sbp_batch = sbp_batch.reshape(-1, 1)
        dbp_batch = dbp_batch.reshape(-1, 1)
        combined = np.concatenate([sbp_batch, dbp_batch], axis=1)
        return combined
    else:
        raise ValueError(f"?ˆìƒ?˜ì? ëª»í•œ ëª¨ë¸ ì¶œë ¥ ?•íƒœ: type={type(raw_pred)}, shape={getattr(raw_pred, 'shape', None)}")


def evaluate_predictions(y_true_sbp, y_true_dbp, y_pred):
    """
    ?ˆì¸¡ ?±ëŠ¥???‰ê??©ë‹ˆ??
    
    Args:
        y_true_sbp: ?¤ì œ SBP ê°?
        y_true_dbp: ?¤ì œ DBP ê°?
        y_pred: ?ˆì¸¡ ê°?(n_samples, 2)
    
    Returns:
        results: ?‰ê? ê²°ê³¼ ?•ì…”?ˆë¦¬
    """
    y_pred_sbp = y_pred[:, 0]
    y_pred_dbp = y_pred[:, 1]
    
    # MAE ê³„ì‚°
    mae_sbp = mean_absolute_error(y_true_sbp, y_pred_sbp)
    mae_dbp = mean_absolute_error(y_true_dbp, y_pred_dbp)
    
    # RMSE ê³„ì‚°
    rmse_sbp = np.sqrt(mean_squared_error(y_true_sbp, y_pred_sbp))
    rmse_dbp = np.sqrt(mean_squared_error(y_true_dbp, y_pred_dbp))
    
    # ?œì??¸ì°¨ ê³„ì‚°
    std_sbp = np.std(y_true_sbp - y_pred_sbp)
    std_dbp = np.std(y_true_dbp - y_pred_dbp)
    
    results = {
        'MAE_SBP': mae_sbp,
        'MAE_DBP': mae_dbp,
        'RMSE_SBP': rmse_sbp,
        'RMSE_DBP': rmse_dbp,
        'STD_SBP': std_sbp,
        'STD_DBP': std_dbp
    }
    
    return results


def visualize_results(y_true_sbp, y_true_dbp, y_pred, save_path='test_results.png'):
    """
    ?ˆì¸¡ ê²°ê³¼ë¥??œê°?”í•©?ˆë‹¤.
    
    Args:
        y_true_sbp: ?¤ì œ SBP ê°?
        y_true_dbp: ?¤ì œ DBP ê°?
        y_pred: ?ˆì¸¡ ê°?(n_samples, 2)
        save_path: ê·¸ë¦¼ ?€??ê²½ë¡œ
    """
    y_pred_sbp = y_pred[:, 0]
    y_pred_dbp = y_pred[:, 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # SBP ?°ì ??
    axes[0, 0].scatter(y_true_sbp, y_pred_sbp, alpha=0.5)
    axes[0, 0].plot([y_true_sbp.min(), y_true_sbp.max()], 
                    [y_true_sbp.min(), y_true_sbp.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('?¤ì œ SBP (mmHg)', fontsize=12)
    axes[0, 0].set_ylabel('?ˆì¸¡ SBP (mmHg)', fontsize=12)
    axes[0, 0].set_title('?˜ì¶•ê¸??ˆì•• ?ˆì¸¡', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # DBP ?°ì ??
    axes[0, 1].scatter(y_true_dbp, y_pred_dbp, alpha=0.5)
    axes[0, 1].plot([y_true_dbp.min(), y_true_dbp.max()], 
                    [y_true_dbp.min(), y_true_dbp.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('?¤ì œ DBP (mmHg)', fontsize=12)
    axes[0, 1].set_ylabel('?ˆì¸¡ DBP (mmHg)', fontsize=12)
    axes[0, 1].set_title('?´ì™„ê¸??ˆì•• ?ˆì¸¡', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # SBP ?¤ì°¨ ë¶„í¬
    error_sbp = y_true_sbp - y_pred_sbp
    axes[1, 0].hist(error_sbp, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('?ˆì¸¡ ?¤ì°¨ (mmHg)', fontsize=12)
    axes[1, 0].set_ylabel('ë¹ˆë„', fontsize=12)
    axes[1, 0].set_title(f'SBP ?¤ì°¨ ë¶„í¬ (Mean: {error_sbp.mean():.2f}, STD: {error_sbp.std():.2f})', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # DBP ?¤ì°¨ ë¶„í¬
    error_dbp = y_true_dbp - y_pred_dbp
    axes[1, 1].hist(error_dbp, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('?ˆì¸¡ ?¤ì°¨ (mmHg)', fontsize=12)
    axes[1, 1].set_ylabel('ë¹ˆë„', fontsize=12)
    axes[1, 1].set_title(f'DBP ?¤ì°¨ ë¶„í¬ (Mean: {error_dbp.mean():.2f}, STD: {error_dbp.std():.2f})', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nê²°ê³¼ ê·¸ë˜???€?¥ë¨: {save_path}")
    plt.close()


def save_results(results, predictions, y_true_sbp, y_true_dbp, save_path='test_results.csv'):
    """
    ?ŒìŠ¤??ê²°ê³¼ë¥?CSV ?Œì¼ë¡??€?¥í•©?ˆë‹¤.
    
    Args:
        results: ?‰ê? ì§€???•ì…”?ˆë¦¬
        predictions: ?ˆì¸¡ ê°?
        y_true_sbp: ?¤ì œ SBP ê°?
        y_true_dbp: ?¤ì œ DBP ê°?
        save_path: ?€??ê²½ë¡œ
    """
    # ?„ì²´ ?ˆì¸¡ ê²°ê³¼ë¥??°ì´?°í”„?ˆì„?¼ë¡œ ?€??
    df = pd.DataFrame({
        'True_SBP': y_true_sbp,
        'Pred_SBP': predictions[:, 0],
        'Error_SBP': y_true_sbp - predictions[:, 0],
        'True_DBP': y_true_dbp,
        'Pred_DBP': predictions[:, 1],
        'Error_DBP': y_true_dbp - predictions[:, 1]
    })
    
    df.to_csv(save_path, index=False)
    print(f"?ì„¸ ê²°ê³¼ ?€?¥ë¨: {save_path}")
    
    # ?‰ê? ì§€?œë? ë³„ë„ ?Œì¼ë¡??€??
    metrics_path = save_path.replace('.csv', '_metrics.csv')
    metrics_df = pd.DataFrame([results])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"?‰ê? ì§€???€?¥ë¨: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='?¬ì „ ?™ìŠµ??ëª¨ë¸???¬ìš©?˜ì—¬ ?ˆì•• ?ˆì¸¡ ?ŒìŠ¤??)
    parser.add_argument('--model', type=str, required=True,
                        help='ëª¨ë¸ ?Œì¼ ê²½ë¡œ (.h5)')
    parser.add_argument('--data', type=str, required=True,
                        help='?ŒìŠ¤???°ì´???Œì¼ ê²½ë¡œ (.h5)')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='?ŒìŠ¤?¸í•  ?˜í”Œ ??(ê¸°ë³¸ê°? 100)')
    parser.add_argument('--output', type=str, default='test_results',
                        help='ê²°ê³¼ ?Œì¼ ?€???´ë¦„ (?•ì¥???œì™¸)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("?ˆì•• ?ˆì¸¡ ëª¨ë¸ ?ŒìŠ¤???œì‘")
    print("="*80)

    # GPU ë¹„í™œ?±í™”ë¡?CUDA ê´€??ê²½ê³  ?Œí”¼ (CPU ê°•ì œ ?¬ìš©)
    try:
        tf.config.experimental.set_visible_devices([], 'GPU')
    except Exception:
        pass
    
    # 1. ëª¨ë¸ ë¡œë“œ
    model = load_model(args.model)
    
    # 2. ?ŒìŠ¤???°ì´??ë¡œë“œ
    ppg_signals, sbp_values, dbp_values = load_test_data(args.data, args.n_samples)
    
    # 3. ? í˜¸ ?„ì²˜ë¦?
    ppg_signals = preprocess_signals(ppg_signals)
    
    # 4. ?ˆì•• ?ˆì¸¡
    predictions = predict_bp(model, ppg_signals)
    
    # 5. ?±ëŠ¥ ?‰ê?
    results = evaluate_predictions(sbp_values, dbp_values, predictions)
    
    print("\n" + "="*80)
    print("?ŒìŠ¤??ê²°ê³¼")
    print("="*80)
    print(f"?˜ì¶•ê¸??ˆì•• (SBP):")
    print(f"  MAE:  {results['MAE_SBP']:.2f} mmHg")
    print(f"  RMSE: {results['RMSE_SBP']:.2f} mmHg")
    print(f"  STD:  {results['STD_SBP']:.2f} mmHg")
    print(f"\n?´ì™„ê¸??ˆì•• (DBP):")
    print(f"  MAE:  {results['MAE_DBP']:.2f} mmHg")
    print(f"  RMSE: {results['RMSE_DBP']:.2f} mmHg")
    print(f"  STD:  {results['STD_DBP']:.2f} mmHg")
    print("="*80)
    
    # 6. ê²°ê³¼ ?œê°??
    visualize_results(sbp_values, dbp_values, predictions, 
                     save_path=f'{args.output}.png')
    
    # 7. ê²°ê³¼ ?€??
    save_results(results, predictions, sbp_values, dbp_values,
                save_path=f'{args.output}.csv')
    
    print("\n?ŒìŠ¤???„ë£Œ!")


if __name__ == '__main__':
    main()
