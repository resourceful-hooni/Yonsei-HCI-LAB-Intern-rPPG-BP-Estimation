"""
prepare_rppg_dataset.py - rPPG ?°ì´?°ì…‹ ?„ì²˜ë¦?ë°?ë¶„í• 

Phase 3-1: Domain Adaptation ì¤€ë¹?
ëª©í‘œ: PPG ëª¨ë¸??rPPG ?°ì´?°ë¡œ fine-tuning?˜ê¸° ?„í•œ ?°ì´??ì¤€ë¹?

?‘ì—…:
1. rPPG-BP-UKL_rppg_7s.h5 ë¡œë“œ
2. Train/Val/Test split (70/15/15)
3. ?•ê·œ??
4. ë¶„í•  ?°ì´???€??
"""

import os
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse


def load_rppg_dataset(data_path):
    """
    rPPG ?°ì´?°ì…‹ ë¡œë“œ
    
    Args:
        data_path: rPPG-BP-UKL_rppg_7s.h5 ê²½ë¡œ
    
    Returns:
        signals: (N, 875) - rPPG ? í˜¸
        labels: (N, 2) - [SBP, DBP]
    """
    print(f"\n?“‚ ?°ì´??ë¡œë“œ ì¤? {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # ?°ì´?°ì…‹ êµ¬ì¡° ?•ì¸
        print(f"   ?°ì´?°ì…‹ ?? {list(f.keys())}")
        
        # rPPG-BP-UKL êµ¬ì¡°: label (2, N), rppg (875, N), subject_idx (1, N)
        # ?¬ê¸°??N?€ ?˜í”Œ ??
        labels_raw = f['label'][:]  # (2, N) - [SBP, DBP]
        signals_raw = f['rppg'][:]  # (875, N) - ? í˜¸ ê¸¸ì´ Ã— ?˜í”Œ
        
        # ?„ì¹˜: (2, N) ??(N, 2), (875, N) ??(N, 875)
        labels = labels_raw.T  # (N, 2)
        signals = signals_raw.T  # (N, 875)
    
    print(f"   rPPG ? í˜¸ ?•íƒœ: {signals.shape} (?˜í”Œ ?? ? í˜¸ ê¸¸ì´)")
    print(f"   SBP/DBP ?ˆì´ë¸??•íƒœ: {labels.shape} (?˜í”Œ ?? 2)")
    
    return signals, labels


def validate_data(signals, labels):
    """
    ?°ì´??? íš¨??ê²€??
    
    Args:
        signals: (N, 875)
        labels: (N, 2)
    """
    print("\n???°ì´??? íš¨??ê²€??)
    
    # ê²°ì¸¡ì¹??•ì¸
    nan_signals = np.sum(np.isnan(signals))
    nan_labels = np.sum(np.isnan(labels))
    print(f"   ? í˜¸ NaN: {nan_signals}, ?ˆì´ë¸?NaN: {nan_labels}")
    
    if nan_signals > 0 or nan_labels > 0:
        # NaN ?œê±°
        valid_idx = ~(np.isnan(signals).any(axis=1) | np.isnan(labels).any(axis=1))
        signals = signals[valid_idx]
        labels = labels[valid_idx]
        print(f"   NaN ?œê±° ?? signals {signals.shape}, labels {labels.shape}")
    
    # ?ˆì•• ë²”ìœ„ ?•ì¸ (?´ìƒì¹??œê±°)
    sbp, dbp = labels[:, 0], labels[:, 1]
    print(f"   SBP ë²”ìœ„: {sbp.min():.1f} - {sbp.max():.1f} mmHg")
    print(f"   DBP ë²”ìœ„: {dbp.min():.1f} - {dbp.max():.1f} mmHg")
    
    # ë¹„ì •??ë²”ìœ„ ?œê±° (ë¬¼ë¦¬?ìœ¼ë¡?ë¶ˆê??¥í•œ ê°?
    valid_idx = (sbp >= 50) & (sbp <= 250) & (dbp >= 20) & (dbp <= 150) & (sbp > dbp)
    signals = signals[valid_idx]
    labels = labels[valid_idx]
    print(f"   ?´ìƒì¹??œê±° ?? signals {signals.shape}, labels {labels.shape}")
    
    return signals, labels


def normalize_data(signals, labels):
    """
    ?°ì´???•ê·œ??
    
    Args:
        signals: (N, 875)
        labels: (N, 2)
    
    Returns:
        signals_normalized: (N, 875) - ?•ê·œ?”ëœ ? í˜¸
        labels_normalized: (N, 2) - ?•ê·œ?”ëœ ?ˆì´ë¸?
        signal_scaler: StandardScaler (? í˜¸??
        label_scaler: StandardScaler (?ˆì´ë¸”ìš©)
    """
    print("\n?™ï¸ ?•ê·œ??ì¤?..")
    
    # ? í˜¸ ?•ê·œ??(ê°??˜í”Œë³?z-score)
    signal_scaler = StandardScaler()
    signals_normalized = signal_scaler.fit_transform(signals)
    print(f"   ? í˜¸ ?•ê·œ???„ë£Œ")
    print(f"   ?‰ê· : {signals_normalized.mean():.4f}, ?œì??¸ì°¨: {signals_normalized.std():.4f}")
    
    # ?ˆì´ë¸??•ê·œ??
    label_scaler = StandardScaler()
    labels_normalized = label_scaler.fit_transform(labels)
    print(f"   ?ˆì´ë¸??•ê·œ???„ë£Œ")
    print(f"   SBP - ?‰ê· : {labels_normalized[:, 0].mean():.4f}, ?œì??¸ì°¨: {labels_normalized[:, 0].std():.4f}")
    print(f"   DBP - ?‰ê· : {labels_normalized[:, 1].mean():.4f}, ?œì??¸ì°¨: {labels_normalized[:, 1].std():.4f}")
    
    return signals_normalized, labels_normalized, signal_scaler, label_scaler


def split_dataset(signals, labels, train_ratio=0.7, val_ratio=0.15):
    """
    ?°ì´?°ì…‹ ë¶„í• 
    
    Args:
        signals: (N, 875)
        labels: (N, 2)
        train_ratio: ?™ìŠµ ë¹„ìœ¨
        val_ratio: ê²€ì¦?ë¹„ìœ¨ (?˜ë¨¸ì§€???ŒìŠ¤??
    
    Returns:
        train_signals, val_signals, test_signals
        train_labels, val_labels, test_labels
    """
    print(f"\n?‚ï¸ ?°ì´?°ì…‹ ë¶„í•  (Train:{train_ratio*100}%, Val:{val_ratio*100}%, Test:{(1-train_ratio-val_ratio)*100}%)")
    
    # ì²?ë²ˆì§¸ ë¶„í• : train vs (val+test)
    train_signals, temp_signals, train_labels, temp_labels = train_test_split(
        signals, labels, 
        test_size=(1-train_ratio), 
        random_state=42
    )
    
    # ??ë²ˆì§¸ ë¶„í• : val vs test
    val_size = val_ratio / (1 - train_ratio)
    val_signals, test_signals, val_labels, test_labels = train_test_split(
        temp_signals, temp_labels,
        test_size=(1-val_size),
        random_state=42
    )
    
    print(f"   Train: {train_signals.shape[0]} ({train_signals.shape[0]/len(signals)*100:.1f}%)")
    print(f"   Val:   {val_signals.shape[0]} ({val_signals.shape[0]/len(signals)*100:.1f}%)")
    print(f"   Test:  {test_signals.shape[0]} ({test_signals.shape[0]/len(signals)*100:.1f}%)")
    
    return (train_signals, val_signals, test_signals,
            train_labels, val_labels, test_labels)


def save_split_dataset(output_dir, 
                       train_signals, val_signals, test_signals,
                       train_labels, val_labels, test_labels,
                       signal_scaler, label_scaler):
    """
    ë¶„í• ???°ì´?°ì…‹ ?€??
    
    Args:
        output_dir: ?€???”ë ‰? ë¦¬
        train/val/test_signals: ? í˜¸
        train/val/test_labels: ?ˆì´ë¸?
        signal_scaler: ? í˜¸ ?¤ì??¼ëŸ¬
        label_scaler: ?ˆì´ë¸??¤ì??¼ëŸ¬
    """
    print(f"\n?’¾ ?°ì´???€??ì¤? {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train ?°ì´?°ì…‹
    train_file = os.path.join(output_dir, 'rppg_train.h5')
    with h5py.File(train_file, 'w') as f:
        f.create_dataset('signals', data=train_signals)
        f.create_dataset('labels', data=train_labels)
        f.attrs['split'] = 'train'
        f.attrs['count'] = len(train_signals)
    print(f"   ??{train_file} ({len(train_signals)} ?˜í”Œ)")
    
    # Val ?°ì´?°ì…‹
    val_file = os.path.join(output_dir, 'rppg_val.h5')
    with h5py.File(val_file, 'w') as f:
        f.create_dataset('signals', data=val_signals)
        f.create_dataset('labels', data=val_labels)
        f.attrs['split'] = 'val'
        f.attrs['count'] = len(val_signals)
    print(f"   ??{val_file} ({len(val_signals)} ?˜í”Œ)")
    
    # Test ?°ì´?°ì…‹
    test_file = os.path.join(output_dir, 'rppg_test.h5')
    with h5py.File(test_file, 'w') as f:
        f.create_dataset('signals', data=test_signals)
        f.create_dataset('labels', data=test_labels)
        f.attrs['split'] = 'test'
        f.attrs['count'] = len(test_signals)
    print(f"   ??{test_file} ({len(test_signals)} ?˜í”Œ)")
    
    # ?¤ì??¼ëŸ¬ ?•ë³´ ?€??
    info_file = os.path.join(output_dir, 'rppg_info.txt')
    with open(info_file, 'w') as f:
        f.write("rPPG Dataset Information\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(train_signals) + len(val_signals) + len(test_signals)}\n")
        f.write(f"Train: {len(train_signals)} samples\n")
        f.write(f"Val: {len(val_signals)} samples\n")
        f.write(f"Test: {len(test_signals)} samples\n\n")
        f.write("Signal Statistics:\n")
        f.write(f"  Mean: {signal_scaler.mean_}\n")
        f.write(f"  Scale: {signal_scaler.scale_}\n\n")
        f.write("Label Statistics:\n")
        f.write(f"  Mean: {label_scaler.mean_}\n")
        f.write(f"  Scale: {label_scaler.scale_}\n")
    print(f"   ??{info_file}")


def main():
    parser = argparse.ArgumentParser(description='rPPG ?°ì´?°ì…‹ ?„ì²˜ë¦?)
    parser.add_argument('--input', type=str, 
                       default='data/rPPG-BP-UKL_rppg_7s.h5',
                       help='?…ë ¥ ?°ì´??ê²½ë¡œ')
    parser.add_argument('--output', type=str,
                       default='data',
                       help='ì¶œë ¥ ?”ë ‰? ë¦¬')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='?™ìŠµ ?°ì´??ë¹„ìœ¨')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='ê²€ì¦??°ì´??ë¹„ìœ¨')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("rPPG ?°ì´?°ì…‹ ?„ì²˜ë¦?ë°?ë¶„í• ")
    print("="*60)
    
    # 1. ?°ì´??ë¡œë“œ
    signals, labels = load_rppg_dataset(args.input)
    print(f"   ??ë¡œë“œ ?„ë£Œ: {len(signals)} ?˜í”Œ")
    
    # 2. ? íš¨??ê²€??
    signals, labels = validate_data(signals, labels)
    print(f"   ??? íš¨??ê²€???„ë£Œ: {len(signals)} ?˜í”Œ")
    
    # 3. ?•ê·œ??
    signals_norm, labels_norm, signal_scaler, label_scaler = normalize_data(signals, labels)
    
    # 4. ?°ì´?°ì…‹ ë¶„í• 
    (train_signals, val_signals, test_signals,
     train_labels, val_labels, test_labels) = split_dataset(
        signals_norm, labels_norm,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    # 5. ?€??
    save_split_dataset(
        args.output,
        train_signals, val_signals, test_signals,
        train_labels, val_labels, test_labels,
        signal_scaler, label_scaler
    )
    
    print("\n" + "="*60)
    print("???°ì´???„ì²˜ë¦??„ë£Œ!")
    print("="*60)
    print(f"\n?¤ìŒ ?¨ê³„: domain_adaptation.py ?™ìŠµ ?¤í¬ë¦½íŠ¸ ?¤í–‰")
    print(f"   python domain_adaptation.py")


if __name__ == '__main__':
    main()
