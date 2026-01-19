"""
domain_adaptation.py - Domain Adaptation (PPG → rPPG) 모듈

Phase 3-1: Domain Adaptation 학습

목표: PPG 모델을 rPPG 데이터로 fine-tuning
- Pre-trained ResNet 로드
- 마지막 레이어 unfreeze
- rPPG 데이터로 fine-tuning
- 모델 저장
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
    Pre-trained PPG 모델 로드
    
    Args:
        model_path: 모델 경로 (e.g., data/resnet_ppg_nonmixed.h5)
    
    Returns:
        model: Keras 모델
    """
    print(f"\n🔄 Pre-trained 모델 로드 중: {model_path}")
    
    dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel
    }
    
    model = ks.models.load_model(model_path, custom_objects=dependencies)
    print(f"   ✓ 모델 로드 완료")
    print(f"   입력 형태: {model.input_shape}")
    print(f"   출력 형태: {model.output_shape}")
    print(f"   총 파라미터: {model.count_params():,}")
    
    return model


def freeze_base_layers(model, num_unfreeze=3):
    """
    기본 레이어 동결 (전이 학습용)
    
    Args:
        model: Keras 모델
        num_unfreeze: 마지막에서 unfreeze할 레이어 수
    
    Returns:
        model: 수정된 모델
    """
    print(f"\n❄️ 기본 레이어 동결 중 (마지막 {num_unfreeze}개 제외)")
    
    # 모든 레이어 동결
    for layer in model.layers[:-num_unfreeze]:
        layer.trainable = False
    
    # 마지막 num_unfreeze 레이어 활성화
    for layer in model.layers[-num_unfreeze:]:
        layer.trainable = True
    
    # 동결 상태 확인
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    frozen_count = sum([1 for layer in model.layers if not layer.trainable])
    
    print(f"   Trainable 레이어: {trainable_count}")
    print(f"   Frozen 레이어: {frozen_count}")
    print(f"   Trainable 파라미터: {model.count_params():,}")
    
    return model


def load_rppg_data(data_dir='data'):
    """
    분할된 rPPG 데이터 로드
    
    Args:
        data_dir: 데이터 디렉토리
    
    Returns:
        train_x, train_y, val_x, val_y, test_x, test_y
    """
    print(f"\n📂 rPPG 데이터 로드 중")
    
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
    
    print(f"   Train: {train_x.shape[0]} 샘플")
    print(f"   Val:   {val_x.shape[0]} 샘플")
    print(f"   Test:  {test_x.shape[0]} 샘플")
    
    # 신호 형태 확인 및 조정
    print(f"   신호 형태: {train_x.shape}")
    print(f"   레이블 형태: {train_y.shape}")
    
    # 신호를 모델 입력 형태로 변환 (N, 875) → (N, 875, 1)
    if len(train_x.shape) == 2:
        train_x = train_x[:, :, np.newaxis]
        val_x = val_x[:, :, np.newaxis]
        test_x = test_x[:, :, np.newaxis]
        print(f"   변환 후 신호 형태: {train_x.shape}")
    
    return train_x, train_y, val_x, val_y, test_x, test_y


def compile_model(model, learning_rate=0.001):
    """
    모델 컴파일
    
    Args:
        model: Keras 모델
        learning_rate: 학습률
    
    Returns:
        model: 컴파일된 모델
    """
    print(f"\n⚙️ 모델 컴파일 (학습률: {learning_rate})")
    
    optimizer = Adam(learning_rate=learning_rate)
    
    # 모델 출력 개수에 따라 손실함수 결정
    if isinstance(model.output, list):
        # 여러 출력 (e.g., [SBP, DBP])
        loss = ['mse', 'mse']
    else:
        loss = 'mse'
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['mae']
    )
    
    print(f"   ✓ 컴파일 완료")
    
    return model


def train_domain_adaptation(model, train_x, train_y, val_x, val_y,
                           epochs=50, batch_size=32, output_dir='models'):
    """
    Domain adaptation 학습
    
    Args:
        model: 컴파일된 모델
        train_x, train_y: 학습 데이터
        val_x, val_y: 검증 데이터
        epochs: 에포크 수
        batch_size: 배치 크기
        output_dir: 출력 디렉토리
    
    Returns:
        history: 학습 이력
        best_model_path: 최고 성능 모델 경로
    """
    print(f"\n🎓 Domain Adaptation 학습 중...")
    print(f"   Epochs: {epochs}, Batch size: {batch_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    best_model_path = os.path.join(output_dir, 'resnet_rppg_adapted.h5')
    
    # 콜백 정의
    callbacks = [
        # 최고 성능 모델 저장
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
        # 학습률 감소
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 학습
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n   ✓ 학습 완료")
    print(f"   최고 성능 모델: {best_model_path}")
    
    return history, best_model_path


def evaluate_model(model, test_x, test_y):
    """
    모델 평가
    
    Args:
        model: Keras 모델
        test_x, test_y: 테스트 데이터
    """
    print(f"\n📊 테스트 셋 평가")
    
    # 평가
    results = model.evaluate(test_x, test_y, verbose=0)
    
    if isinstance(results, list):
        # 여러 출력
        print(f"   Loss (SBP): {results[0]:.4f}")
        print(f"   Loss (DBP): {results[1]:.4f}")
        print(f"   MAE (SBP): {results[2]:.4f} mmHg")
        print(f"   MAE (DBP): {results[3]:.4f} mmHg")
    else:
        print(f"   Loss: {results[0]:.4f}")
        print(f"   MAE: {results[1]:.4f}")
    
    # 예측
    predictions = model.predict(test_x, verbose=0)
    
    if isinstance(predictions, list):
        pred_sbp = predictions[0].flatten()
        pred_dbp = predictions[1].flatten()
        true_sbp = test_y[:, 0]
        true_dbp = test_y[:, 1]
        
        mae_sbp = np.mean(np.abs(pred_sbp - true_sbp))
        mae_dbp = np.mean(np.abs(pred_dbp - true_dbp))
        
        print(f"\n   평균 절대 오차 (MAE):")
        print(f"   SBP: {mae_sbp:.2f} mmHg")
        print(f"   DBP: {mae_dbp:.2f} mmHg")
        
        # 개선도 (기존 PPG 모델 vs 적응 모델)
        # 기존 PPG 모델의 예상 성능: SBP MAE ~28.9, DBP MAE ~15.2
        ppg_mae_sbp = 28.9
        ppg_mae_dbp = 15.2
        
        improvement_sbp = (ppg_mae_sbp - mae_sbp) / ppg_mae_sbp * 100
        improvement_dbp = (ppg_mae_dbp - mae_dbp) / ppg_mae_dbp * 100
        
        print(f"\n   PPG 대비 개선도:")
        print(f"   SBP: {improvement_sbp:+.1f}% (기존: {ppg_mae_sbp:.2f} → 개선: {mae_sbp:.2f})")
        print(f"   DBP: {improvement_dbp:+.1f}% (기존: {ppg_mae_dbp:.2f} → 개선: {mae_dbp:.2f})")


def save_training_info(output_dir, history, best_model_path):
    """
    학습 정보 저장
    
    Args:
        output_dir: 출력 디렉토리
        history: 학습 이력
        best_model_path: 최고 성능 모델 경로
    """
    info_file = os.path.join(output_dir, 'training_info.json')
    
    info = {
        'model': 'ResNet (PPG → rPPG Domain Adaptation)',
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
    
    print(f"\n   ✓ 학습 정보 저장: {info_file}")


def main():
    parser = argparse.ArgumentParser(description='Domain Adaptation 학습')
    parser.add_argument('--pretrained', type=str,
                       default='data/resnet_ppg_nonmixed.h5',
                       help='Pre-trained 모델 경로')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='데이터 디렉토리')
    parser.add_argument('--epochs', type=int, default=50,
                       help='에포크 수')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='학습률')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='출력 디렉토리')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Domain Adaptation: PPG → rPPG")
    print("="*60)
    
    # 1. 모델 로드
    model = load_pretrained_model(args.pretrained)
    
    # 2. 기본 레이어 동결
    model = freeze_base_layers(model, num_unfreeze=3)
    
    # 3. 데이터 로드
    train_x, train_y, val_x, val_y, test_x, test_y = load_rppg_data(args.data_dir)
    
    # 4. 모델 컴파일
    model = compile_model(model, learning_rate=args.learning_rate)
    
    # 5. 학습
    history, best_model_path = train_domain_adaptation(
        model, train_x, train_y, val_x, val_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    # 6. 평가
    evaluate_model(model, test_x, test_y)
    
    # 7. 정보 저장
    save_training_info(args.output_dir, history, best_model_path)
    
    print("\n" + "="*60)
    print("✅ Domain Adaptation 학습 완료!")
    print("="*60)
    print(f"\n다음 단계:")
    print(f"  1. 모델 평가: camera_rppg_advanced.py --model {best_model_path}")
    print(f"  2. GitHub commit: git add -A && git commit -m 'Phase 3-1: Domain Adaptation'")


if __name__ == '__main__':
    main()
