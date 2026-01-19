"""
multi_task_model.py - Multi-Task Learning 모델 정의

Phase 3-2: Multi-Task Learning (BP + HR + SpO2)

목표: 
- Shared backbone (ResNet)에서 3개의 작업을 동시에 학습
- BP (SBP, DBP) 예측
- HR (Heart Rate) 예측  
- SpO2 (Oxygen Saturation) 예측
"""

import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import layers, Model


def create_multi_task_model(input_shape=(875, 1), backbone_path=None):
    """
    Multi-Task Learning 모델 생성
    
    Args:
        input_shape: 입력 형태
        backbone_path: Pre-trained backbone 모델 경로 (선택사항)
    
    Returns:
        model: Multi-task Keras 모델
    """
    print(f"\n[*] Creating Multi-Task Learning model...")
    
    # 입력 레이어
    inputs = layers.Input(shape=input_shape, name='input_signal')
    
    # ===== Shared Backbone (ResNet from rPPG domain) =====
    if backbone_path:
        print(f"   [OK] Loading pre-trained backbone: {backbone_path}")
        from kapre import STFT, Magnitude, MagnitudeToDecibel
        
        dependencies = {
            'ReLU': ks.layers.ReLU,
            'STFT': STFT,
            'Magnitude': Magnitude,
            'MagnitudeToDecibel': MagnitudeToDecibel
        }
        
        # Pre-trained 모델 로드
        full_model = ks.models.load_model(backbone_path, custom_objects=dependencies)
        
        # 마지막 Dense 레이어 제거 (출력층 제거)
        backbone = Model(inputs=full_model.input, 
                        outputs=full_model.layers[-2].output)
        
        # backbone 동결
        backbone.trainable = False
        
        x = backbone(inputs)
    else:
        # 기본 ResNet 백본 생성
        print(f"   [OK] Creating ResNet backbone from scratch")
        
        # Conv 블록
        x = layers.Conv1D(64, 7, strides=2, padding='same', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(3, strides=3, padding='same', name='pool1')(x)
        
        # Residual 블록들
        x = residual_block(x, 64, name='block1')
        x = residual_block(x, 128, name='block2')
        x = residual_block(x, 256, name='block3')
        
        # 글로벌 평균 풀링
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
    
    # ===== Shared Dense Layers =====
    shared = layers.Dense(512, activation='relu', name='shared_dense1')(x)
    shared = layers.Dropout(0.3)(shared)
    shared = layers.Dense(256, activation='relu', name='shared_dense2')(shared)
    shared = layers.Dropout(0.3)(shared)
    
    # ===== Task 1: Blood Pressure (BP) =====
    bp_head = layers.Dense(128, activation='relu', name='bp_dense1')(shared)
    bp_head = layers.Dropout(0.2)(bp_head)
    bp_head = layers.Dense(64, activation='relu', name='bp_dense2')(bp_head)
    
    # SBP와 DBP 별도 출력
    sbp_output = layers.Dense(1, name='sbp_output')(bp_head)  # Systolic BP
    dbp_output = layers.Dense(1, name='dbp_output')(bp_head)  # Diastolic BP
    
    # ===== Task 2: Heart Rate (HR) =====
    hr_head = layers.Dense(128, activation='relu', name='hr_dense1')(shared)
    hr_head = layers.Dropout(0.2)(hr_head)
    hr_head = layers.Dense(64, activation='relu', name='hr_dense2')(hr_head)
    
    # HR 출력 (50-150 bpm 범위)
    hr_output = layers.Dense(1, activation='relu', name='hr_output')(hr_head)
    
    # ===== Task 3: SpO2 (Oxygen Saturation) =====
    spo2_head = layers.Dense(128, activation='relu', name='spo2_dense1')(shared)
    spo2_head = layers.Dropout(0.2)(spo2_head)
    spo2_head = layers.Dense(64, activation='relu', name='spo2_dense2')(spo2_head)
    
    # SpO2 출력 (95-100% 범위)
    spo2_output = layers.Dense(1, activation='relu', name='spo2_output')(spo2_head)
    
    # ===== 모델 정의 =====
    model = Model(
        inputs=inputs,
        outputs=[sbp_output, dbp_output, hr_output, spo2_output],
        name='MultiTask_BP_HR_SpO2'
    )
    
    print(f"   [OK] Model created")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {[output.shape for output in model.outputs]}")
    print(f"   Total params: {model.count_params():,}")
    
    return model


def residual_block(x, filters, name='block'):
    """
    ResNet 잔차 블록
    
    Args:
        x: 입력
        filters: 필터 수
        name: 블록 이름
    
    Returns:
        x: 출력
    """
    shortcut = x
    
    # 필터 수가 변하면 shortcut 조정
    if int(x.shape[-1]) != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same', name=f'{name}_shortcut')(x)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
    
    # 메인 경로
    x = layers.Conv1D(filters, 3, padding='same', name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv1D(filters, 3, padding='same', name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    
    # 합산
    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x


def compile_multi_task_model(model, learning_rate=0.001):
    """
    Multi-Task 모델 컴파일
    
    Args:
        model: Keras 모델
        learning_rate: 학습률
    
    Returns:
        model: 컴파일된 모델
    """
    print(f"\n[*] Compiling model (learning rate: {learning_rate})")
    
    optimizer = ks.optimizers.Adam(learning_rate=learning_rate)
    
    # 각 작업별 손실함수
    losses = {
        'sbp_output': 'mse',
        'dbp_output': 'mse',
        'hr_output': 'mse',
        'spo2_output': 'mse'
    }
    
    # 작업별 가중치 (BP를 중점)
    loss_weights = {
        'sbp_output': 1.0,
        'dbp_output': 1.0,
        'hr_output': 0.3,      # HR은 보조 작업
        'spo2_output': 0.3     # SpO2는 보조 작업
    }
    
    # 메트릭
    metrics = {
        'sbp_output': 'mae',
        'dbp_output': 'mae',
        'hr_output': 'mae',
        'spo2_output': 'mae'
    }
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    print(f"   Loss Weights:")
    print(f"     - BP (SBP, DBP): 1.0 each")
    print(f"     - HR: 0.3")
    print(f"     - SpO2: 0.3")
    print(f"   [OK] Compilation complete")
    
    return model


def print_model_summary(model):
    """모델 요약 출력"""
    print(f"\n[*] Model Summary:")
    model.summary()


if __name__ == '__main__':
    # 테스트용 모델 생성
    print("\n" + "="*70)
    print("MULTI-TASK LEARNING MODEL TEST")
    print("="*70)
    
    # 모델 생성
    model = create_multi_task_model(input_shape=(875, 1))
    
    # 모델 컴파일
    model = compile_multi_task_model(model)
    
    # 모델 요약
    print_model_summary(model)
    
    # 더미 데이터로 테스트
    import numpy as np
    
    print(f"\n[OK] Forward pass test with dummy data")
    dummy_input = np.random.randn(4, 875, 1).astype(np.float32)
    
    predictions = model.predict(dummy_input, verbose=0)
    
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   SBP 출력: {predictions[0].shape}")
    print(f"   DBP 출력: {predictions[1].shape}")
    print(f"   HR 출력: {predictions[2].shape}")
    print(f"   SpO2 출력: {predictions[3].shape}")
    
    print(f"\n✅ Multi-Task 모델 생성 및 테스트 완료!")
