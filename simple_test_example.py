"""
간단한 모델 테스트 예제
- 가장 기본적인 사용법을 보여주는 스크립트입니다
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU 완전 비활성화
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from kapre import STFT, Magnitude, MagnitudeToDecibel


def simple_test():
    """
    사전 학습된 모델로 간단한 테스트를 수행합니다.
    """
    print("="*80)
    print("간단한 모델 테스트 예제")
    print("="*80)

    # GPU를 사용하지 않도록 설정 (CUDA 경고 회피 및 CPU 강제)
    try:
        tf.config.experimental.set_visible_devices([], 'GPU')
    except Exception:
        pass
    
    # 1. 모델 로드
    model_path = 'data/alexnet_ppg_nonmixed.h5'  # 원하는 모델로 변경 가능
    print(f"\n1단계: 모델 로드 - {model_path}")
    
    dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel
    }
    
    try:
        model = ks.models.load_model(model_path, custom_objects=dependencies)
        print("✓ 모델 로드 성공!")
    except Exception as e:
        print(f"✗ 모델 로드 실패: {e}")
        return
    
    # 2. 더미 데이터 생성 (실제로는 실제 PPG 신호를 사용해야 합니다)
    print("\n2단계: 테스트 데이터 준비")
    # 형태: (샘플 수, 875, 1) - 875는 7초 * 125Hz
    dummy_ppg_signal = np.random.randn(5, 875, 1).astype(np.float32)
    print(f"테스트 데이터 형태: {dummy_ppg_signal.shape}")
    
    # 3. 혈압 예측
    print("\n3단계: 혈압 예측")
    predictions = model.predict(dummy_ppg_signal, verbose=0)
    
    # 4. 결과 출력 (모델 출력 형태에 따라 처리)
    print("\n4단계: 예측 결과")
    print("="*80)
    
    # 경우 1: 단일 텐서 출력 (shape: (batch, 2))
    if hasattr(predictions, 'shape') and predictions.ndim >= 2 and predictions.shape[-1] == 2:
        for i in range(predictions.shape[0]):
            sbp = float(predictions[i, 0])
            dbp = float(predictions[i, 1])
            print(f"샘플 {i+1}: SBP = {sbp:.1f} mmHg, DBP = {dbp:.1f} mmHg")
    # 경우 2: 두 개의 출력 리스트 ([sbp_batch, dbp_batch])
    elif isinstance(predictions, (list, tuple)) and len(predictions) == 2:
        sbp_batch, dbp_batch = predictions
        # 각 배치의 차원 정리 (예: (batch, 1) -> (batch,))
        sbp_batch = np.squeeze(sbp_batch)
        dbp_batch = np.squeeze(dbp_batch)
        for i in range(len(sbp_batch)):
            sbp = float(sbp_batch[i])
            dbp = float(dbp_batch[i])
            print(f"샘플 {i+1}: SBP = {sbp:.1f} mmHg, DBP = {dbp:.1f} mmHg")
    else:
        print(f"예상하지 못한 출력 형태: {type(predictions)}")
        try:
            print(f"출력 요약: shape={getattr(predictions, 'shape', None)}; len={len(predictions) if hasattr(predictions, '__len__') else 'N/A'}")
        except Exception:
            pass
    print("="*80)
    
    print("\n참고: 위 예측 값은 랜덤 데이터로 테스트한 것이므로 의미가 없습니다.")
    print("실제 PPG 신호를 사용하려면 test_model.py 스크립트를 사용하세요.")


if __name__ == '__main__':
    simple_test()
