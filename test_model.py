"""
사전 학습된 모델을 사용하여 혈압을 예측하는 테스트 스크립트

사용법:
    python test_model.py --model data/alexnet_ppg_nonmixed.h5 --data data/MIMIC-III_ppg_dataset.h5
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU 완전 비활성화
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
    사전 학습된 모델을 로드합니다.
    
    Args:
        model_path: 모델 파일 경로 (.h5 파일)
    
    Returns:
        로드된 Keras 모델
    """
    print(f"모델 로드 중: {model_path}")
    
    # Kapre 레이어를 위한 custom objects 정의
    dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel
    }
    
    model = ks.models.load_model(model_path, custom_objects=dependencies)
    print("모델 로드 완료!")
    print("\n모델 구조:")
    model.summary()
    
    return model


def load_test_data(data_path, n_samples=100):
    """
    테스트 데이터를 로드합니다.
    
    Args:
        data_path: HDF5 데이터 파일 경로
        n_samples: 테스트할 샘플 수
    
    Returns:
        ppg_signals: PPG 신호 배열 (n_samples, 875)
        sbp_values: 실제 수축기 혈압 값
        dbp_values: 실제 이완기 혈압 값
    """
    print(f"\n테스트 데이터 로드 중: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        print(f"사용 가능한 데이터셋 키: {list(f.keys())}")
        
        # PPG 신호 로드
        if 'ppg' in f.keys():
            ppg_data = f['ppg'][:]
        elif 'signal' in f.keys():
            ppg_data = f['signal'][:]
        else:
            raise ValueError(f"PPG 데이터를 찾을 수 없습니다. 사용 가능한 키: {list(f.keys())}")
        
        # 혈압 값 로드
        if 'sbp' in f.keys() and 'dbp' in f.keys():
            sbp_data = f['sbp'][:]
            dbp_data = f['dbp'][:]
        else:
            raise ValueError(f"SBP/DBP 데이터를 찾을 수 없습니다. 사용 가능한 키: {list(f.keys())}")
    
    # 샘플 수 제한
    n_samples = min(n_samples, len(ppg_data))
    ppg_signals = ppg_data[:n_samples]
    sbp_values = sbp_data[:n_samples]
    dbp_values = dbp_data[:n_samples]
    
    print(f"로드된 데이터 형태:")
    print(f"  PPG 신호: {ppg_signals.shape}")
    print(f"  SBP 값: {sbp_values.shape}")
    print(f"  DBP 값: {dbp_values.shape}")
    
    return ppg_signals, sbp_values, dbp_values


def preprocess_signals(signals):
    """
    신호를 모델 입력 형태로 전처리합니다.
    
    Args:
        signals: PPG 신호 배열
    
    Returns:
        전처리된 신호 (n_samples, 875, 1)
    """
    # 신호 길이가 875가 아닌 경우 처리
    if signals.shape[1] != 875:
        print(f"경고: 신호 길이가 {signals.shape[1]}입니다. 875로 리사이징합니다.")
        # 간단한 리샘플링 또는 패딩
        if signals.shape[1] > 875:
            signals = signals[:, :875]
        else:
            pad_width = ((0, 0), (0, 875 - signals.shape[1]))
            signals = np.pad(signals, pad_width, mode='edge')
    
    # 차원 추가 (samples, 875) -> (samples, 875, 1)
    if len(signals.shape) == 2:
        signals = np.expand_dims(signals, axis=-1)
    
    return signals


def predict_bp(model, ppg_signals):
    """
    모델을 사용하여 혈압을 예측합니다.
    
    Args:
        model: 학습된 Keras 모델
        ppg_signals: PPG 신호 배열 (n_samples, 875, 1)
    
    Returns:
        predictions: 예측된 혈압 값 (n_samples, 2) - [SBP, DBP]
    """
    print("\n혈압 예측 중...")
    raw_pred = model.predict(ppg_signals, batch_size=32, verbose=1)
    print("예측 완료!")
    
    # 단일 텐서 (n,2)인 경우 그대로 반환
    if hasattr(raw_pred, 'shape') and raw_pred.ndim >= 2 and raw_pred.shape[-1] == 2:
        return raw_pred
    # 두 출력 리스트 ([sbp_batch, dbp_batch])인 경우 결합
    elif isinstance(raw_pred, (list, tuple)) and len(raw_pred) == 2:
        sbp_batch, dbp_batch = raw_pred
        sbp_batch = np.squeeze(sbp_batch)
        dbp_batch = np.squeeze(dbp_batch)
        # (n,) 형태를 (n,1)로 변환 후 concatenate
        sbp_batch = sbp_batch.reshape(-1, 1)
        dbp_batch = dbp_batch.reshape(-1, 1)
        combined = np.concatenate([sbp_batch, dbp_batch], axis=1)
        return combined
    else:
        raise ValueError(f"예상하지 못한 모델 출력 형태: type={type(raw_pred)}, shape={getattr(raw_pred, 'shape', None)}")


def evaluate_predictions(y_true_sbp, y_true_dbp, y_pred):
    """
    예측 성능을 평가합니다.
    
    Args:
        y_true_sbp: 실제 SBP 값
        y_true_dbp: 실제 DBP 값
        y_pred: 예측 값 (n_samples, 2)
    
    Returns:
        results: 평가 결과 딕셔너리
    """
    y_pred_sbp = y_pred[:, 0]
    y_pred_dbp = y_pred[:, 1]
    
    # MAE 계산
    mae_sbp = mean_absolute_error(y_true_sbp, y_pred_sbp)
    mae_dbp = mean_absolute_error(y_true_dbp, y_pred_dbp)
    
    # RMSE 계산
    rmse_sbp = np.sqrt(mean_squared_error(y_true_sbp, y_pred_sbp))
    rmse_dbp = np.sqrt(mean_squared_error(y_true_dbp, y_pred_dbp))
    
    # 표준편차 계산
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
    예측 결과를 시각화합니다.
    
    Args:
        y_true_sbp: 실제 SBP 값
        y_true_dbp: 실제 DBP 값
        y_pred: 예측 값 (n_samples, 2)
        save_path: 그림 저장 경로
    """
    y_pred_sbp = y_pred[:, 0]
    y_pred_dbp = y_pred[:, 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # SBP 산점도
    axes[0, 0].scatter(y_true_sbp, y_pred_sbp, alpha=0.5)
    axes[0, 0].plot([y_true_sbp.min(), y_true_sbp.max()], 
                    [y_true_sbp.min(), y_true_sbp.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('실제 SBP (mmHg)', fontsize=12)
    axes[0, 0].set_ylabel('예측 SBP (mmHg)', fontsize=12)
    axes[0, 0].set_title('수축기 혈압 예측', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # DBP 산점도
    axes[0, 1].scatter(y_true_dbp, y_pred_dbp, alpha=0.5)
    axes[0, 1].plot([y_true_dbp.min(), y_true_dbp.max()], 
                    [y_true_dbp.min(), y_true_dbp.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('실제 DBP (mmHg)', fontsize=12)
    axes[0, 1].set_ylabel('예측 DBP (mmHg)', fontsize=12)
    axes[0, 1].set_title('이완기 혈압 예측', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # SBP 오차 분포
    error_sbp = y_true_sbp - y_pred_sbp
    axes[1, 0].hist(error_sbp, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('예측 오차 (mmHg)', fontsize=12)
    axes[1, 0].set_ylabel('빈도', fontsize=12)
    axes[1, 0].set_title(f'SBP 오차 분포 (Mean: {error_sbp.mean():.2f}, STD: {error_sbp.std():.2f})', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # DBP 오차 분포
    error_dbp = y_true_dbp - y_pred_dbp
    axes[1, 1].hist(error_dbp, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('예측 오차 (mmHg)', fontsize=12)
    axes[1, 1].set_ylabel('빈도', fontsize=12)
    axes[1, 1].set_title(f'DBP 오차 분포 (Mean: {error_dbp.mean():.2f}, STD: {error_dbp.std():.2f})', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n결과 그래프 저장됨: {save_path}")
    plt.close()


def save_results(results, predictions, y_true_sbp, y_true_dbp, save_path='test_results.csv'):
    """
    테스트 결과를 CSV 파일로 저장합니다.
    
    Args:
        results: 평가 지표 딕셔너리
        predictions: 예측 값
        y_true_sbp: 실제 SBP 값
        y_true_dbp: 실제 DBP 값
        save_path: 저장 경로
    """
    # 전체 예측 결과를 데이터프레임으로 저장
    df = pd.DataFrame({
        'True_SBP': y_true_sbp,
        'Pred_SBP': predictions[:, 0],
        'Error_SBP': y_true_sbp - predictions[:, 0],
        'True_DBP': y_true_dbp,
        'Pred_DBP': predictions[:, 1],
        'Error_DBP': y_true_dbp - predictions[:, 1]
    })
    
    df.to_csv(save_path, index=False)
    print(f"상세 결과 저장됨: {save_path}")
    
    # 평가 지표를 별도 파일로 저장
    metrics_path = save_path.replace('.csv', '_metrics.csv')
    metrics_df = pd.DataFrame([results])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"평가 지표 저장됨: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='사전 학습된 모델을 사용하여 혈압 예측 테스트')
    parser.add_argument('--model', type=str, required=True,
                        help='모델 파일 경로 (.h5)')
    parser.add_argument('--data', type=str, required=True,
                        help='테스트 데이터 파일 경로 (.h5)')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='테스트할 샘플 수 (기본값: 100)')
    parser.add_argument('--output', type=str, default='test_results',
                        help='결과 파일 저장 이름 (확장자 제외)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("혈압 예측 모델 테스트 시작")
    print("="*80)

    # GPU 비활성화로 CUDA 관련 경고 회피 (CPU 강제 사용)
    try:
        tf.config.experimental.set_visible_devices([], 'GPU')
    except Exception:
        pass
    
    # 1. 모델 로드
    model = load_model(args.model)
    
    # 2. 테스트 데이터 로드
    ppg_signals, sbp_values, dbp_values = load_test_data(args.data, args.n_samples)
    
    # 3. 신호 전처리
    ppg_signals = preprocess_signals(ppg_signals)
    
    # 4. 혈압 예측
    predictions = predict_bp(model, ppg_signals)
    
    # 5. 성능 평가
    results = evaluate_predictions(sbp_values, dbp_values, predictions)
    
    print("\n" + "="*80)
    print("테스트 결과")
    print("="*80)
    print(f"수축기 혈압 (SBP):")
    print(f"  MAE:  {results['MAE_SBP']:.2f} mmHg")
    print(f"  RMSE: {results['RMSE_SBP']:.2f} mmHg")
    print(f"  STD:  {results['STD_SBP']:.2f} mmHg")
    print(f"\n이완기 혈압 (DBP):")
    print(f"  MAE:  {results['MAE_DBP']:.2f} mmHg")
    print(f"  RMSE: {results['RMSE_DBP']:.2f} mmHg")
    print(f"  STD:  {results['STD_DBP']:.2f} mmHg")
    print("="*80)
    
    # 6. 결과 시각화
    visualize_results(sbp_values, dbp_values, predictions, 
                     save_path=f'{args.output}.png')
    
    # 7. 결과 저장
    save_results(results, predictions, sbp_values, dbp_values,
                save_path=f'{args.output}.csv')
    
    print("\n테스트 완료!")


if __name__ == '__main__':
    main()
