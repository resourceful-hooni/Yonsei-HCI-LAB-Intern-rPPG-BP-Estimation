# 카메라 rPPG 혈압 예측 구현 상태 보고서

## 📋 요약

현재 구현된 카메라 기반 혈압 예측은 **간소화된 프로토타입**입니다. README에서 언급한 정교한 rPPG 알고리즘과는 **차이**가 있습니다.

---

## 🔍 README vs 실제 구현 비교

### README에서 설명한 방법

**논문 (Schrumpf et al. 2021)에서 사용한 rPPG 추출 방법:**
- **알고리즘**: Wang et al. 2017의 **Plane-Orthogonal-to-Skin (POS)** 알고리즘
- **데이터**: Leipzig University Hospital에서 표준 RGB 카메라로 촬영한 얼굴 영상
- **신호 처리**: 피부 영역에서 정교한 색상 공간 변환 및 주파수 분석
- **입력 요구사항**: 7초 × 125Hz = **875 샘플**

### 현재 구현 (`camera_rppg_test.py`)

**실제 구현 방법:**
- **알고리즘**: 단순 **Green 채널 평균값** 추출
  ```python
  green_channel = face_region[:, :, 1]  # BGR에서 Green 채널
  signal_value = np.mean(green_channel)  # 평균값만 사용
  ```
- **얼굴 감지**: OpenCV Haar Cascade (간단하지만 부정확할 수 있음)
- **신호 처리**: 
  - 정규화만 수행 (Z-score normalization)
  - 밴드패스 필터 **없음**
  - 주파수 분석 **없음**
- **리샘플링**: 카메라 FPS(~30Hz)에서 수집 후 선형 보간으로 875 샘플로 변환

**차이점 요약:**

| 항목 | README/논문 | 현재 구현 | 영향 |
|------|------------|----------|------|
| rPPG 추출 | POS 알고리즘 | Green 채널 평균 | ⚠️ 정확도 크게 저하 |
| 얼굴 감지 | 고급 (논문 미명시) | Haar Cascade | ⚠️ 부정확한 영역 감지 |
| 신호 필터링 | 밴드패스 (0.7-4Hz) | 없음 | ⚠️ 노이즈 제거 안됨 |
| 움직임 보정 | 있음 | 없음 | ⚠️ 움직임에 민감 |
| 조명 보정 | 있음 | 없음 | ⚠️ 조명 변화에 민감 |

---

## 🎯 현재 구현의 동작 방식

### 1. 카메라 초기화
```python
# DirectShow 또는 MSMF 백엔드로 카메라 열기
cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 30)  # 30 FPS 요청
```

### 2. 프레임별 처리 루프
```python
while True:
    ret, frame = cap.read()  # 프레임 읽기
    
    # ① 얼굴 감지 (Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # ② 가장 큰 얼굴 영역 선택
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_region = frame[y:y+h, x:x+w]
    
    # ③ Green 채널 평균 추출
    green_channel = face_region[:, :, 1]
    signal_value = np.mean(green_channel)
    
    # ④ 버퍼에 저장 (7초분 수집)
    signal_buffer.append(signal_value)
```

### 3. 신호 수집 완료 후 예측
```python
# 7초 수집 후 (210 프레임 @ 30 FPS)
signal = np.array(signal_buffer)

# 정규화
signal = (signal - mean) / std

# 875 샘플로 리샘플링 (선형 보간)
signal = np.interp(x_new, x, signal)

# 모델 예측
prediction = model.predict(signal.reshape(1, 875, 1))
sbp, dbp = prediction[0]
```

---

## 🟩 초록 네모창 2개 현상 설명

### 원인

**OpenCV의 Haar Cascade가 여러 얼굴을 감지:**

```python
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 초록 사각형 그리기
```

**가능한 이유:**
1. **중복 감지**: 하나의 얼굴을 여러 스케일에서 감지 → 겹치는 박스
2. **거짓 양성**: 배경이나 다른 객체를 얼굴로 오인식
3. **얼굴이 여러 개**: 실제로 프레임에 여러 사람
4. **파라미터 문제**: `detectMultiScale(gray, 1.1, 4)`에서 `minNeighbors=4`가 너무 낮음

**신호 추출에 사용되는 것:**
```python
# 가장 큰 얼굴만 사용
face_region = extract_face_region(frame)  # 내부적으로 max() 사용
```

**해결 방법:**
- `minNeighbors` 값을 높여 거짓 감지 감소: `detectMultiScale(gray, 1.1, 8)`
- 비최대 억제 (Non-Maximum Suppression) 적용
- MediaPipe Face Detection 같은 더 정확한 방법 사용

---

## ⚠️ 혈압 변동값이 일정하지 않은 이유

### 측정 결과 예시
```
수축기 혈압 (SBP): 2028.0 mmHg  ← 비정상적으로 높음
이완기 혈압 (DBP): 946.8 mmHg   ← 비정상적으로 높음
```

**정상 범위:** SBP 90-140 mmHg, DBP 60-90 mmHg

### 근본 원인

#### 1️⃣ **rPPG 신호 품질 문제**

**Green 채널 평균만 사용의 한계:**
- 혈류 변화뿐 아니라 조명, 움직임, 그림자 모두 포함
- 주파수 대역 필터링 없음 → 심박과 무관한 노이즈 포함
- Wang et al. POS 알고리즘은 RGB를 선형 변환해 혈류 성분만 분리

**예시:**
```python
# 현재 구현
signal = np.mean(face_region[:, :, 1])  # Green 채널 평균

# 논문의 POS 알고리즘 (간략)
C = RGB_normalized  # (3, N) 정규화된 RGB
S = [[0, 1, -1], [-2, 1, 1]] @ C  # 직교 투영
P = S[0] + (std(S[0])/std(S[1])) * S[1]  # Pulse 신호 추출
```

#### 2️⃣ **모델-데이터 불일치**

**학습 데이터:**
- MIMIC-III PPG 센서 데이터 (직접 측정, 고품질)
- 125Hz 샘플링, 정교한 전처리

**테스트 데이터:**
- 카메라 Green 채널 평균 (간접 추정, 저품질)
- 30Hz → 875 샘플로 보간 (정보 손실)

**결과:** 모델이 본 적 없는 형태의 신호 → 예측 불가능

#### 3️⃣ **환경 요인**

| 요인 | 영향 | 현재 처리 |
|------|------|----------|
| 조명 변화 | Green 채널 값 크게 변동 | 없음 |
| 움직임 | 얼굴 영역 변화 → 신호 왜곡 | 없음 |
| 카메라 품질 | 노이즈, 압축 아티팩트 | 없음 |
| 초점/거리 | 픽셀 값 변화 | 없음 |

#### 4️⃣ **리샘플링 오류**

```python
# 30 FPS × 7초 = 210 샘플 → 875 샘플로 선형 보간
signal = np.interp(x_new, x, signal)
```

- 주파수 정보 왜곡 가능
- Nyquist 주파수 고려 안됨
- 심박수(60-100 bpm → 1-1.67 Hz)는 30 FPS로 충분하지만, 신호 품질이 이미 나쁨

---

## ✅ 정확한 구현을 위한 요구사항

### README 기준 올바른 rPPG 구현

#### 1. POS 알고리즘 구현
```python
import numpy as np

def pos_algorithm(video_frames, roi):
    """
    Wang et al. 2017 Plane-Orthogonal-to-Skin 알고리즘
    """
    # RGB 채널 평균 추출
    C = np.zeros((3, len(video_frames)))
    for i, frame in enumerate(video_frames):
        roi_pixels = frame[roi]
        C[0, i] = np.mean(roi_pixels[:, :, 2])  # R
        C[1, i] = np.mean(roi_pixels[:, :, 1])  # G
        C[2, i] = np.mean(roi_pixels[:, :, 0])  # B
    
    # 정규화
    C_norm = C / np.mean(C, axis=1, keepdims=True)
    
    # 투영 행렬
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    
    # 펄스 신호 추출
    S = P @ C_norm
    h = S[0] + (np.std(S[0]) / np.std(S[1])) * S[1]
    
    return h
```

#### 2. 신호 필터링
```python
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.7, highcut=4.0, fs=30):
    """
    심박수 범위만 통과 (0.7-4 Hz = 42-240 bpm)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)
```

#### 3. 고급 얼굴 감지
```python
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)

results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# 더 정확한 얼굴 경계 박스 제공
```

#### 4. 올바른 샘플링
```python
# 125 Hz로 수집하거나, 후처리에서 적절한 리샘플링
from scipy.signal import resample

signal_resampled = resample(signal, 875)
```

---

## 📊 현재 구현 vs 이상적 구현

### 현재 구현 파이프라인
```
카메라 (30 FPS)
  ↓
Haar Cascade 얼굴 감지 (부정확)
  ↓
Green 채널 평균 (노이즈 많음)
  ↓
정규화만 (필터링 없음)
  ↓
선형 보간 → 875 샘플
  ↓
AlexNet 모델 예측
  ↓
비정상적 결과 (2028/946 mmHg)
```

### 논문의 파이프라인
```
카메라 (고품질)
  ↓
정교한 얼굴/피부 영역 추출
  ↓
POS 알고리즘 (RGB → 펄스 신호)
  ↓
밴드패스 필터 (0.7-4 Hz)
  ↓
움직임/조명 보정
  ↓
적절한 리샘플링 → 875 샘플
  ↓
학습된 모델 예측
  ↓
정상 범위 결과
```

---

## 🎓 결론 및 권장사항

### 현재 상태

✅ **작동하는 것:**
- 카메라 열기 및 프레임 읽기
- 얼굴 감지 및 표시
- 신호 수집 (7초)
- 모델 로드 및 예측 실행

❌ **작동하지 않는 것:**
- 정확한 rPPG 신호 추출
- 의미있는 혈압 예측
- 안정적인 얼굴 감지

### 교육/데모 목적

현재 구현은 다음 용도로 적합:
- ✅ TensorFlow 모델 사용법 학습
- ✅ OpenCV 카메라 처리 연습
- ✅ 전체 파이프라인 이해
- ❌ 실제 혈압 측정 (절대 안됨!)

### 의료/연구 목적

논문 수준 구현 필요:
1. POS 알고리즘 구현
2. 신호 처리 파이프라인 추가
3. 고품질 얼굴/피부 감지
4. 적절한 데이터셋으로 재학습 또는 도메인 적응

### 초록 네모 2개 해결

간단한 수정으로 개선:
```python
# detectMultiScale 파라미터 조정
faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1, 
    minNeighbors=8,  # 4 → 8로 증가 (거짓 감지 감소)
    minSize=(100, 100)  # 최소 크기 지정
)
```

---

## 📝 면책 조항

⚠️ **중요:**
- 이 구현은 **교육 및 연구 목적**입니다
- **의료 진단에 절대 사용하지 마세요**
- 혈압 측정이 필요하면 **인증된 의료 기기**를 사용하세요
- 논문의 결과를 재현하려면 **전체 방법론**을 정확히 따라야 합니다

---

**작성일:** 2026-01-19  
**구현 파일:** `camera_rppg_test.py`  
**모델:** AlexNet (PPG 학습, rPPG 미세조정 안됨)  
**Python:** 3.8.10  
**TensorFlow:** 2.4.1 (CPU only)
