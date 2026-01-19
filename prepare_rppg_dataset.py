"""
prepare_rppg_dataset.py - rPPG 데이터셋 전처리 및 분할

Phase 3-1: Domain Adaptation 준비
목표: PPG 모델을 rPPG 데이터로 fine-tuning하기 위한 데이터 준비

작업:
1. rPPG-BP-UKL_rppg_7s.h5 로드
2. Train/Val/Test split (70/15/15)
3. 정규화
4. 분할 데이터 저장
"""

import os
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse


def load_rppg_dataset(data_path):
    """
    rPPG 데이터셋 로드
    
    Args:
        data_path: rPPG-BP-UKL_rppg_7s.h5 경로
    
    Returns:
        signals: (N, 875) - rPPG 신호
        labels: (N, 2) - [SBP, DBP]
    """
    print(f"\n📂 데이터 로드 중: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # 데이터셋 구조 확인
        print(f"   데이터셋 키: {list(f.keys())}")
        
        # rPPG-BP-UKL 구조: label (2, N), rppg (875, N), subject_idx (1, N)
        # 여기서 N은 샘플 수
        labels_raw = f['label'][:]  # (2, N) - [SBP, DBP]
        signals_raw = f['rppg'][:]  # (875, N) - 신호 길이 × 샘플
        
        # 전치: (2, N) → (N, 2), (875, N) → (N, 875)
        labels = labels_raw.T  # (N, 2)
        signals = signals_raw.T  # (N, 875)
    
    print(f"   rPPG 신호 형태: {signals.shape} (샘플 수, 신호 길이)")
    print(f"   SBP/DBP 레이블 형태: {labels.shape} (샘플 수, 2)")
    
    return signals, labels


def validate_data(signals, labels):
    """
    데이터 유효성 검사
    
    Args:
        signals: (N, 875)
        labels: (N, 2)
    """
    print("\n✓ 데이터 유효성 검사")
    
    # 결측치 확인
    nan_signals = np.sum(np.isnan(signals))
    nan_labels = np.sum(np.isnan(labels))
    print(f"   신호 NaN: {nan_signals}, 레이블 NaN: {nan_labels}")
    
    if nan_signals > 0 or nan_labels > 0:
        # NaN 제거
        valid_idx = ~(np.isnan(signals).any(axis=1) | np.isnan(labels).any(axis=1))
        signals = signals[valid_idx]
        labels = labels[valid_idx]
        print(f"   NaN 제거 후: signals {signals.shape}, labels {labels.shape}")
    
    # 혈압 범위 확인 (이상치 제거)
    sbp, dbp = labels[:, 0], labels[:, 1]
    print(f"   SBP 범위: {sbp.min():.1f} - {sbp.max():.1f} mmHg")
    print(f"   DBP 범위: {dbp.min():.1f} - {dbp.max():.1f} mmHg")
    
    # 비정상 범위 제거 (물리적으로 불가능한 값)
    valid_idx = (sbp >= 50) & (sbp <= 250) & (dbp >= 20) & (dbp <= 150) & (sbp > dbp)
    signals = signals[valid_idx]
    labels = labels[valid_idx]
    print(f"   이상치 제거 후: signals {signals.shape}, labels {labels.shape}")
    
    return signals, labels


def normalize_data(signals, labels):
    """
    데이터 정규화
    
    Args:
        signals: (N, 875)
        labels: (N, 2)
    
    Returns:
        signals_normalized: (N, 875) - 정규화된 신호
        labels_normalized: (N, 2) - 정규화된 레이블
        signal_scaler: StandardScaler (신호용)
        label_scaler: StandardScaler (레이블용)
    """
    print("\n⚙️ 정규화 중...")
    
    # 신호 정규화 (각 샘플별 z-score)
    signal_scaler = StandardScaler()
    signals_normalized = signal_scaler.fit_transform(signals)
    print(f"   신호 정규화 완료")
    print(f"   평균: {signals_normalized.mean():.4f}, 표준편차: {signals_normalized.std():.4f}")
    
    # 레이블 정규화
    label_scaler = StandardScaler()
    labels_normalized = label_scaler.fit_transform(labels)
    print(f"   레이블 정규화 완료")
    print(f"   SBP - 평균: {labels_normalized[:, 0].mean():.4f}, 표준편차: {labels_normalized[:, 0].std():.4f}")
    print(f"   DBP - 평균: {labels_normalized[:, 1].mean():.4f}, 표준편차: {labels_normalized[:, 1].std():.4f}")
    
    return signals_normalized, labels_normalized, signal_scaler, label_scaler


def split_dataset(signals, labels, train_ratio=0.7, val_ratio=0.15):
    """
    데이터셋 분할
    
    Args:
        signals: (N, 875)
        labels: (N, 2)
        train_ratio: 학습 비율
        val_ratio: 검증 비율 (나머지는 테스트)
    
    Returns:
        train_signals, val_signals, test_signals
        train_labels, val_labels, test_labels
    """
    print(f"\n✂️ 데이터셋 분할 (Train:{train_ratio*100}%, Val:{val_ratio*100}%, Test:{(1-train_ratio-val_ratio)*100}%)")
    
    # 첫 번째 분할: train vs (val+test)
    train_signals, temp_signals, train_labels, temp_labels = train_test_split(
        signals, labels, 
        test_size=(1-train_ratio), 
        random_state=42
    )
    
    # 두 번째 분할: val vs test
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
    분할된 데이터셋 저장
    
    Args:
        output_dir: 저장 디렉토리
        train/val/test_signals: 신호
        train/val/test_labels: 레이블
        signal_scaler: 신호 스케일러
        label_scaler: 레이블 스케일러
    """
    print(f"\n💾 데이터 저장 중: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train 데이터셋
    train_file = os.path.join(output_dir, 'rppg_train.h5')
    with h5py.File(train_file, 'w') as f:
        f.create_dataset('signals', data=train_signals)
        f.create_dataset('labels', data=train_labels)
        f.attrs['split'] = 'train'
        f.attrs['count'] = len(train_signals)
    print(f"   ✓ {train_file} ({len(train_signals)} 샘플)")
    
    # Val 데이터셋
    val_file = os.path.join(output_dir, 'rppg_val.h5')
    with h5py.File(val_file, 'w') as f:
        f.create_dataset('signals', data=val_signals)
        f.create_dataset('labels', data=val_labels)
        f.attrs['split'] = 'val'
        f.attrs['count'] = len(val_signals)
    print(f"   ✓ {val_file} ({len(val_signals)} 샘플)")
    
    # Test 데이터셋
    test_file = os.path.join(output_dir, 'rppg_test.h5')
    with h5py.File(test_file, 'w') as f:
        f.create_dataset('signals', data=test_signals)
        f.create_dataset('labels', data=test_labels)
        f.attrs['split'] = 'test'
        f.attrs['count'] = len(test_signals)
    print(f"   ✓ {test_file} ({len(test_signals)} 샘플)")
    
    # 스케일러 정보 저장
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
    print(f"   ✓ {info_file}")


def main():
    parser = argparse.ArgumentParser(description='rPPG 데이터셋 전처리')
    parser.add_argument('--input', type=str, 
                       default='data/rPPG-BP-UKL_rppg_7s.h5',
                       help='입력 데이터 경로')
    parser.add_argument('--output', type=str,
                       default='data',
                       help='출력 디렉토리')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='학습 데이터 비율')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='검증 데이터 비율')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("rPPG 데이터셋 전처리 및 분할")
    print("="*60)
    
    # 1. 데이터 로드
    signals, labels = load_rppg_dataset(args.input)
    print(f"   ✓ 로드 완료: {len(signals)} 샘플")
    
    # 2. 유효성 검사
    signals, labels = validate_data(signals, labels)
    print(f"   ✓ 유효성 검사 완료: {len(signals)} 샘플")
    
    # 3. 정규화
    signals_norm, labels_norm, signal_scaler, label_scaler = normalize_data(signals, labels)
    
    # 4. 데이터셋 분할
    (train_signals, val_signals, test_signals,
     train_labels, val_labels, test_labels) = split_dataset(
        signals_norm, labels_norm,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    # 5. 저장
    save_split_dataset(
        args.output,
        train_signals, val_signals, test_signals,
        train_labels, val_labels, test_labels,
        signal_scaler, label_scaler
    )
    
    print("\n" + "="*60)
    print("✅ 데이터 전처리 완료!")
    print("="*60)
    print(f"\n다음 단계: domain_adaptation.py 학습 스크립트 실행")
    print(f"   python domain_adaptation.py")


if __name__ == '__main__':
    main()
