# 모델 테스트 가이드

이 가이드는 사전 학습된 혈압 예측 모델을 테스트하는 방법을 단계별로 설명합니다.

## 목차
1. [환경 설정](#1-환경-설정)
2. [간단한 테스트](#2-간단한-테스트)
3. [실제 데이터로 테스트](#3-실제-데이터로-테스트)
4. [사용 가능한 모델](#4-사용-가능한-모델)
5. [결과 해석](#5-결과-해석)
6. [문제 해결](#6-문제-해결)

---

## 1. 환경 설정

⚠️ **중요: 이 프로젝트는 Python 3.8과 TensorFlow 2.4.1을 사용해야 합니다!**

### 1.1 Python 버전 확인
먼저 Python 3.8이 설치되어 있는지 확인:
```powershell
python --version
```

Python 3.8이 없다면 [Python 3.8 다운로드](https://www.python.org/downloads/release/python-380/)에서 설치하세요.

### 1.2 가상 환경 생성 (처음 한 번만)
이미 `env` 폴더가 있다면 이 단계는 건너뛰세요.

```powershell
# virtualenv 설치
pip install virtualenv

# Python 3.8로 가상 환경 생성
virtualenv --python=python3.8 env
```

### 1.3 가상 환경 활성화
Windows PowerShell에서:
```powershell
.\env\Scripts\Activate.ps1
```

Windows CMD에서:
```cmd
.\env\Scripts\activate.bat
```

활성화되면 프롬프트에 `(env)`가 표시됩니다.

### 1.4 Python 버전 재확인
가상 환경 내에서:
```powershell
python --version
```
**반드시 Python 3.8.x가 표시되어야 합니다!**

### 1.5 필요한 패키지 설치
```powershell
pip install -r requirements.txt
```

주요 패키지:
- tensorflow-gpu==2.4.1
- kapre==0.3.5
- h5py==2.10.0
- pandas==1.3.1
- matplotlib==3.4.2

---

## 2. 간단한 테스트

모델이 제대로 로드되는지 확인하는 간단한 테스트:

```powershell
python simple_test_example.py
```

**이 테스트는:**
- 모델을 로드합니다
- 더미 데이터로 예측을 수행합니다
- 모델이 정상 작동하는지 확인합니다

**주의:** 랜덤 데이터를 사용하므로 예측 값은 의미가 없습니다!

---

## 3. 실제 데이터로 테스트

### 3.1 기본 사용법

```powershell
python test_model.py --model data/alexnet_ppg_nonmixed.h5 --data data/MIMIC-III_ppg_dataset.h5
```

### 3.2 옵션 설명

| 옵션 | 설명 | 기본값 | 예시 |
|------|------|--------|------|
| `--model` | 테스트할 모델 파일 경로 | 필수 | `data/resnet_ppg_nonmixed.h5` |
| `--data` | 테스트 데이터 파일 경로 | 필수 | `data/MIMIC-III_ppg_dataset.h5` |
| `--n_samples` | 테스트할 샘플 수 | 100 | `500` |
| `--output` | 결과 파일 이름 | `test_results` | `alexnet_test` |

### 3.3 다양한 예시

**AlexNet 모델로 100개 샘플 테스트:**
```powershell
python test_model.py --model data/alexnet_ppg_nonmixed.h5 --data data/MIMIC-III_ppg_dataset.h5 --n_samples 100 --output alexnet_results
```

**ResNet 모델로 500개 샘플 테스트:**
```powershell
python test_model.py --model data/resnet_ppg_nonmixed.h5 --data data/MIMIC-III_ppg_dataset.h5 --n_samples 500 --output resnet_results
```

**LSTM 모델 테스트:**
```powershell
python test_model.py --model data/lstm_ppg_nonmixed.h5 --data data/MIMIC-III_ppg_dataset.h5 --n_samples 200 --output lstm_results
```

**Slapnicar 모델 테스트:**
```powershell
python test_model.py --model data/slapnicar_ppg_nonmixed.h5 --data data/MIMIC-III_ppg_dataset.h5 --n_samples 200 --output slapnicar_results
```

### 3.4 생성되는 결과 파일

테스트 후 다음 파일들이 생성됩니다:

1. **`{output}.csv`** - 각 샘플의 예측 결과
   - True_SBP: 실제 수축기 혈압
   - Pred_SBP: 예측된 수축기 혈압
   - Error_SBP: SBP 오차
   - True_DBP: 실제 이완기 혈압
   - Pred_DBP: 예측된 이완기 혈압
   - Error_DBP: DBP 오차

2. **`{output}_metrics.csv`** - 전체 성능 지표
   - MAE_SBP/DBP: 평균 절대 오차
   - RMSE_SBP/DBP: 제곱근 평균 제곱 오차
   - STD_SBP/DBP: 표준 편차

3. **`{output}.png`** - 시각화 그래프
   - 산점도 (실제 vs 예측)
   - 오차 분포 히스토그램

---

## 4. 사용 가능한 모델

현재 `data/` 폴더에 있는 모델들:

| 모델 파일 | 설명 | 논문 참조 |
|-----------|------|-----------|
| `alexnet_ppg_nonmixed.h5` | AlexNet 기반 1D CNN | Krizhevsky et al. |
| `resnet_ppg_nonmixed.h5` | ResNet 기반 1D CNN | He et al. |
| `lstm_ppg_nonmixed.h5` | LSTM 네트워크 | - |
| `slapnicar_ppg_nonmixed.h5` | Slapnicar et al. 아키텍처 | Slapnicar et al. 2019 |

### 모델 입력 요구사항

모든 모델은 다음과 같은 입력을 요구합니다:
- **형태:** `(batch_size, 875, 1)`
- **875 샘플** = 7초 × 125Hz 샘플링 레이트
- **정규화:** 필요에 따라 전처리됨

### 모델 출력

- **형태:** `(batch_size, 2)`
- **첫 번째 값:** 수축기 혈압 (SBP) in mmHg
- **두 번째 값:** 이완기 혈압 (DBP) in mmHg

---

## 5. 결과 해석

### 5.1 성능 지표 이해

**MAE (Mean Absolute Error, 평균 절대 오차):**
- 예측 값과 실제 값의 평균 차이
- 낮을수록 좋음
- 일반적으로 5-10 mmHg 정도면 좋은 성능

**RMSE (Root Mean Squared Error, 제곱근 평균 제곱 오차):**
- MAE보다 큰 오차에 더 민감
- 낮을수록 좋음

**STD (Standard Deviation, 표준 편차):**
- 오차의 분산 정도
- 낮을수록 일관성 있는 예측

### 5.2 좋은 결과의 기준

**우수한 성능:**
- MAE < 5 mmHg
- 산점도에서 점들이 대각선에 가까움
- 오차 분포가 0을 중심으로 정규분포

**보통 성능:**
- MAE 5-10 mmHg
- 산점도에서 약간의 분산
- 오차 분포가 약간 치우침

**개선 필요:**
- MAE > 10 mmHg
- 산점도에서 큰 분산
- 오차 분포가 크게 치우침

---

## 6. 문제 해결

### 6.1 모델 로드 오류

**오류 메시지:**
```
ValueError: Unknown layer: STFT
```

**해결 방법:**
```powershell
pip install kapre==0.3.5
```

### 6.2 데이터 로드 오류

**오류 메시지:**
```
ValueError: PPG 데이터를 찾을 수 없습니다
```

**해결 방법:**
1. HDF5 파일 구조 확인:
```python
import h5py
with h5py.File('data/MIMIC-III_ppg_dataset.h5', 'r') as f:
    print(list(f.keys()))
```

2. 스크립트의 키 이름을 실제 키에 맞게 수정

### 6.3 메모리 오류

**오류 메시지:**
```
MemoryError: Unable to allocate array
```

**해결 방법:**
- `--n_samples` 값을 줄입니다:
```powershell
python test_model.py --model data/alexnet_ppg_nonmixed.h5 --data data/MIMIC-III_ppg_dataset.h5 --n_samples 50
```

### 6.4 GPU 메모리 오류

**오류 메시지:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**해결 방법:**
스크립트에 다음 코드 추가 (test_model.py 상단):
```python
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### 6.5 TensorFlow 버전 호환성

**오류 메시지:**
```
AttributeError: module 'tensorflow' has no attribute 'xxx'
```

**해결 방법:**
정확한 TensorFlow 버전 설치 **(Python 3.8 필수!)**:
```powershell
# 기존 TensorFlow 제거
pip uninstall tensorflow tensorflow-gpu

# Python 3.8 환경에서 TensorFlow 2.4.1 설치
pip install tensorflow-gpu==2.4.1
```

**참고:** TensorFlow 2.4.1은 Python 3.6-3.8만 지원합니다. Python 버전이 맞지 않으면 설치가 안 됩니다!

---

## 7. 카메라로 실제 테스트하기

### 7.1 개요

이 프로젝트는 **rPPG (Remote Photoplethysmography)** 기반으로 작동합니다:
- **PPG**: 손가락 센서로 직접 측정
- **rPPG**: 카메라로 얼굴의 피부 색상 변화로부터 추출

카메라로 테스트하려면 웹캠이 필요합니다.

### 7.2 필요한 패키지

```powershell
pip install opencv-python
```

### 7.3 카메라 테스트 실행

```powershell
python camera_rppg_test.py
```

### 7.4 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model` | 사용할 모델 파일 | `data/alexnet_ppg_nonmixed.h5` |
| `--camera` | 카메라 ID | `0` (기본 카메라) |
| `--duration` | 신호 수집 시간 (초) | `7` |

### 7.5 사용 예시

```powershell
# 기본 설정으로 테스트
python camera_rppg_test.py

# ResNet 모델 사용
python camera_rppg_test.py --model data/resnet_ppg_nonmixed.h5

# LSTM 모델 사용
python camera_rppg_test.py --model data/lstm_ppg_nonmixed.h5
```

### 7.6 사용 방법

1. **프로그램 시작** - "카메라 기반 rPPG 혈압 예측" 창이 열림
2. **얼굴 위치** - 얼굴을 카메라에 정면으로 맞춤
3. **신호 수집** - 7초 동안 신호를 수집 (진행률 표시)
4. **결과 확인** - 혈압 예측 결과가 터미널에 표시
5. **반복** - 새로운 측정을 위해 계속 신호 수집 가능
6. **종료** - `Ctrl+C` 또는 'q' 키를 눌러 종료

### 7.7 중요 주의사항 ⚠️

**정확도 제한:**
- 이 스크립트는 **간단한 Green 채널 평균** 방식을 사용
- 실제 rPPG는 **Wang et al. 2017의 Plane-Orthogonal-to-Skin 알고리즘** 사용
- 조명 조건, 움직임, 카메라 품질에 매우 민감
- **의료 용도로는 사용할 수 없습니다!**

**최적의 결과를 위해:**
1. ✅ 밝고 일정한 조명
2. ✅ 카메라로부터 30-50cm 거리 유지
3. ✅ 얼굴을 정면으로 향함
4. ✅ 움직임 최소화
5. ✅ 안경 제거 (반사 방지)

### 7.8 카메라 문제 해결

**카메라가 인식되지 않음:**
```powershell
# 다른 카메라 ID 시도
python camera_rppg_test.py --camera 1
python camera_rppg_test.py --camera 2
```

**얼굴을 인식하지 못함:**
- 조명을 밝게
- 카메라를 더 가까이
- 카메라를 정면으로 향함
- Haar Cascade 한계 - MediaPipe 사용 권장

**느린 프로세싱:**
- 프레임 해상도 낮추기
- 프레임 레이트 조정
- 고성능 GPU 사용

### 7.9 더 정교한 rPPG 구현

현재 스크립트는 간단한 방법을 사용합니다. 더 정확한 rPPG 추출을 위해 다음을 고려하세요:

**1. Wang et al. 2017 알고리즘:**
```python
# 피부 영역에서 RGB 채널 추출
# Plane-Orthogonal-to-Skin 변환 적용
# 주파수 영역 분석
```

**2. 고급 얼굴 인식:**
```python
import mediapipe as mp
# MediaPipe Face Detection 사용
```

**3. 신호 처리:**
```python
from scipy.signal import butter, filtfilt
# 밴드패스 필터 (0.7-4 Hz) 적용
# 움직임 보정
```

---

## 추가 리소스

### Python 코드에서 직접 사용하기

```python
import numpy as np
import tensorflow.keras as ks
from kapre import STFT, Magnitude, MagnitudeToDecibel

# 모델 로드
dependencies = {
    'ReLU': ks.layers.ReLU,
    'STFT': STFT,
    'Magnitude': Magnitude,
    'MagnitudeToDecibel': MagnitudeToDecibel
}
model = ks.models.load_model('data/alexnet_ppg_nonmixed.h5', 
                             custom_objects=dependencies)

# PPG 신호 준비 (875 샘플, 7초 @ 125Hz)
ppg_signal = your_ppg_data  # shape: (1, 875, 1)

# 예측
prediction = model.predict(ppg_signal)
sbp = prediction[0, 0]  # 수축기 혈압
dbp = prediction[0, 1]  # 이완기 혈압

print(f"예측 혈압: {sbp:.1f}/{dbp:.1f} mmHg")
```

### rPPG 데이터로 테스트

rPPG 데이터가 있다면:
```powershell
python test_model.py --model data/alexnet_ppg_nonmixed.h5 --data data/rPPG-BP-UKL_rppg_7s.h5 --output rppg_test
```

---

## 질문이나 문제가 있나요?

1. README.md 파일의 논문 참조 확인
2. requirements.txt에 명시된 정확한 패키지 버전 사용
3. GitHub Issues에서 유사한 문제 검색

**행운을 빕니다! 🩺📊**
