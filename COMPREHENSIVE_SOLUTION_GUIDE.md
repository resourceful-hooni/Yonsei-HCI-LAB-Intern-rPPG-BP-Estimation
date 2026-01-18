# 🩺 rPPG 기반 혈압 예측 - 종합 문제 분석 및 해결 가이드

**작성일:** 2026-01-19  
**프로젝트:** Non-invasive Blood Pressure Estimation Using Deep Learning  
**논문:** Schrumpf et al. 2021 - "Assessment of Non-Invasive Blood Pressure Prediction from PPG and rPPG Signals Using Deep Learning"

---

## 📋 목차

1. [현재 문제점 전체 정리](#1-현재-문제점-전체-정리)
2. [단계별 해결 방법](#2-단계별-해결-방법)
3. [논문 기반 정확한 구현](#3-논문-기반-정확한-구현)
4. [2026년 최신 방법론](#4-2026년-최신-방법론)
5. [구현 로드맵](#5-구현-로드맵)
6. [성능 비교표](#6-성능-비교표)
7. [코드 예제](#7-코드-예제)
8. [참고 자료](#8-참고-자료)

---

## 1. 현재 문제점 전체 정리

### 1.1 문제점 심각도 분류

| 심각도 | 문제 | 현재 상태 | 영향 |
|--------|------|-----------|------|
| 🔴 **Critical** | POS 알고리즘 미구현 | Green 채널 평균만 사용 | 신호 품질 매우 낮음 |
| 🔴 **Critical** | 밴드패스 필터 없음 | 필터링 전무 | 노이즈 제거 안됨 |
| 🟠 **High** | Haar Cascade 부정확 | 다중 감지, 오탐지 | 잘못된 ROI |
| 🟠 **High** | 모델-데이터 불일치 | PPG 모델에 rPPG 입력 | 예측값 비정상 |
| 🟡 **Medium** | 리샘플링 방식 | 단순 선형 보간 | 주파수 정보 손실 |
| 🟡 **Medium** | 움직임/조명 보정 없음 | 보정 전무 | 환경 변화에 취약 |

### 1.2 상세 문제 분석

#### 🔴 Critical Issue 1: POS 알고리즘 미구현

**현재 구현:**
```python
# camera_rppg_test.py - 현재 방식
green_channel = face_region[:, :, 1]  # BGR에서 Green만
signal_value = np.mean(green_channel)  # 단순 평균
```

**문제점:**
- Green 채널은 혈류 외에도 조명, 그림자, 움직임 모두 포함
- 피부 반사 특성을 고려하지 않음
- SNR (Signal-to-Noise Ratio) 매우 낮음

**논문의 POS 알고리즘:**
```
RGB 정규화 → 직교 투영 → 펄스 신호 분리
```

#### 🔴 Critical Issue 2: 밴드패스 필터 없음

**현재 구현:**
```python
# 정규화만 수행
signal = (signal - np.mean(signal)) / np.std(signal)
```

**문제점:**
- 심박수 범위: 0.7-4 Hz (42-240 bpm)
- 이 범위 외의 노이즈가 모두 포함됨
- 호흡 (0.1-0.5 Hz), 고주파 노이즈 등 제거 안됨

#### 🟠 High Issue 3: Haar Cascade 부정확

**현재 구현:**
```python
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
```

**문제점:**
- `minNeighbors=4`가 너무 낮음 → 다중 감지
- 얼굴이 아닌 영역 오탐지
- 프레임마다 ROI 위치 변동

#### 🟠 High Issue 4: 모델-데이터 불일치

**학습 데이터 (MIMIC-III PPG):**
- 손가락 PPG 센서로 직접 측정
- 125 Hz 고품질 샘플링
- 혈류 변화 직접 반영

**테스트 데이터 (카메라 rPPG):**
- 얼굴 피부에서 간접 추출
- 30 Hz → 보간 → 125 Hz
- 조명/움직임 아티팩트 포함

**결과:**
```
예측값: SBP=2028 mmHg, DBP=946 mmHg (비정상)
정상범위: SBP=90-140 mmHg, DBP=60-90 mmHg
```

#### 🟡 Medium Issue 5: 리샘플링 방식

**현재 구현:**
```python
# 선형 보간
signal = np.interp(np.linspace(0, len(signal), 875), 
                   np.arange(len(signal)), signal)
```

**문제점:**
- Nyquist 주파수 고려 안됨
- 앨리어싱 발생 가능
- Anti-aliasing 필터 없음

#### 🟡 Medium Issue 6: 움직임/조명 보정 없음

**필요한 보정:**
- 머리 움직임 추적 및 보상
- 조명 변화 정규화
- 피부 영역 마스킹

---

## 2. 단계별 해결 방법

### 2.1 Phase 1: Quick Fix (1-2일)

#### Step 1: Haar Cascade 파라미터 조정
```python
# Before
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# After
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=8,      # 4 → 8 (거짓 감지 감소)
    minSize=(100, 100),  # 최소 크기 지정
    maxSize=(400, 400)   # 최대 크기 지정
)
```

#### Step 2: 기본 밴드패스 필터 추가
```python
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.7, highcut=4.0, fs=30, order=4):
    """심박수 범위 필터 (0.7-4 Hz = 42-240 bpm)"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)
```

#### Step 3: 가장 큰 얼굴만 사용 + 단일 박스
```python
def get_largest_face(faces):
    if len(faces) == 0:
        return None
    # 면적 기준 가장 큰 얼굴
    largest = max(faces, key=lambda f: f[2] * f[3])
    return largest
```

#### Step 4: ResNet 모델로 변경
```python
# AlexNet → ResNet (논문 기준 최고 성능)
parser.add_argument('--model', type=str, 
                    default='data/resnet_ppg_nonmixed.h5')
```

**예상 효과:** 예측값이 여전히 부정확하지만, 범위가 줄어듦

---

### 2.2 Phase 2: 논문 구현 (2-4주)

#### Step 1: POS 알고리즘 구현

```python
import numpy as np
from scipy.signal import butter, filtfilt

class POSAlgorithm:
    """
    Wang et al. 2017 - Plane-Orthogonal-to-Skin Algorithm
    논문: "Algorithmic Principles of Remote PPG"
    IEEE Trans. Biomed. Eng., vol. 64, no. 7, pp. 1479-1491, 2017
    """
    
    def __init__(self, fs=30, window_size=32):
        self.fs = fs
        self.window_size = window_size
    
    def extract_pulse(self, rgb_signals):
        """
        RGB 신호에서 펄스 신호 추출
        
        Args:
            rgb_signals: (N, 3) array - R, G, B 채널 시계열
            
        Returns:
            pulse: (N,) array - 추출된 펄스 신호
        """
        N = rgb_signals.shape[0]
        H = np.zeros(N)
        
        for t in range(self.window_size, N):
            # 윈도우 내 RGB 신호
            C = rgb_signals[t-self.window_size:t, :].T  # (3, window_size)
            
            # 시간 정규화 (평균으로 나눔)
            mean_C = np.mean(C, axis=1, keepdims=True)
            C_norm = C / (mean_C + 1e-8)
            
            # POS 투영 행렬
            # P = [[0, 1, -1], [-2, 1, 1]]
            S = np.array([
                C_norm[1, :] - C_norm[2, :],           # G - B
                -2*C_norm[0, :] + C_norm[1, :] + C_norm[2, :]  # -2R + G + B
            ])
            
            # 표준편차 비율로 결합
            std_S0 = np.std(S[0, :])
            std_S1 = np.std(S[1, :])
            
            if std_S1 > 1e-8:
                alpha = std_S0 / std_S1
            else:
                alpha = 0
            
            # 펄스 신호
            h = S[0, :] + alpha * S[1, :]
            
            # 윈도우 중심값 저장
            H[t] = h[-1] - np.mean(h)
        
        return H
    
    def process_video(self, frames, face_detector):
        """
        비디오 프레임에서 rPPG 신호 추출
        
        Args:
            frames: list of BGR frames
            face_detector: 얼굴 감지기
            
        Returns:
            pulse: 추출된 펄스 신호
        """
        rgb_signals = []
        
        for frame in frames:
            # 얼굴 감지
            face_roi = face_detector.detect(frame)
            if face_roi is None:
                continue
            
            # 피부 영역에서 RGB 평균 추출
            r_mean = np.mean(face_roi[:, :, 2])  # R
            g_mean = np.mean(face_roi[:, :, 1])  # G
            b_mean = np.mean(face_roi[:, :, 0])  # B
            
            rgb_signals.append([r_mean, g_mean, b_mean])
        
        rgb_signals = np.array(rgb_signals)
        
        # POS 알고리즘으로 펄스 추출
        pulse = self.extract_pulse(rgb_signals)
        
        # 밴드패스 필터링
        pulse = self.bandpass_filter(pulse)
        
        return pulse
    
    def bandpass_filter(self, signal, lowcut=0.7, highcut=4.0):
        """심박수 범위 필터링"""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal)
```

#### Step 2: MediaPipe 얼굴 감지 도입

```python
import mediapipe as mp
import cv2

class MediaPipeFaceDetector:
    def __init__(self, min_detection_confidence=0.7):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0: 2m 이내, 1: 5m 이내
            min_detection_confidence=min_detection_confidence
        )
    
    def detect(self, frame):
        """얼굴 영역 반환"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        
        if not results.detections:
            return None
        
        # 가장 신뢰도 높은 얼굴
        detection = max(results.detections, 
                       key=lambda d: d.score[0])
        
        bbox = detection.location_data.relative_bounding_box
        h, w = frame.shape[:2]
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # 경계 체크
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        return frame[y:y+height, x:x+width]
    
    def detect_with_landmarks(self, frame):
        """얼굴 + 랜드마크 반환 (피부 영역 추출용)"""
        # MediaPipe Face Mesh 사용 시 더 정밀한 피부 영역 추출 가능
        pass
```

#### Step 3: 적절한 리샘플링

```python
from scipy.signal import resample

def proper_resample(signal, original_fs, target_fs, target_length):
    """
    적절한 리샘플링 (Anti-aliasing 포함)
    
    Args:
        signal: 원본 신호
        original_fs: 원본 샘플링 레이트
        target_fs: 목표 샘플링 레이트
        target_length: 목표 길이 (샘플 수)
    """
    # Anti-aliasing 필터
    if target_fs < original_fs:
        nyq = 0.5 * original_fs
        cutoff = 0.5 * target_fs / nyq
        b, a = butter(8, cutoff, btype='low')
        signal = filtfilt(b, a, signal)
    
    # scipy resample (FFT 기반)
    resampled = resample(signal, target_length)
    
    return resampled
```

#### Step 4: Transfer Learning / Fine-tuning

```python
# retrain_rppg_personalization.py 활용
# PPG로 학습된 모델을 rPPG 데이터로 fine-tuning

python retrain_rppg_personalization.py \
    "experiment_name" \
    "data/rPPG-BP-UKL_rppg_7s.h5" \
    "results/" \
    "data/resnet_ppg_nonmixed.h5" \
    "checkpoints/"
```

---

### 2.3 Phase 3: 2026 최신 기술 (4-8주)

[섹션 4에서 상세 설명]

---

## 3. 논문 기반 정확한 구현

### 3.1 POS 알고리즘 수학적 배경

**Wang et al. 2017 논문의 핵심 원리:**

피부 반사 모델:
```
I(t) = I_s(t) + I_d(t)
     = specular reflection + diffuse reflection
```

피부색 변화:
```
C(t) = C_0 · (1 + p(t))
```
- `C(t)`: 시간 t에서의 피부색 (RGB)
- `C_0`: 기저 피부색
- `p(t)`: 혈류 변화에 의한 미세 변동

**POS 투영:**
```
S = P · C_n

여기서:
P = [0,  1, -1]    (첫 번째 축)
    [-2, 1,  1]    (두 번째 축)

C_n = C(t) / mean(C(t))  (시간 정규화)
```

**펄스 신호 추출:**
```
H = S_1 + (σ(S_1) / σ(S_2)) · S_2
```
- `σ()`: 표준편차
- 두 직교 성분의 가중 합

### 3.2 4개 모델 아키텍처 상세

#### 3.2.1 AlexNet (1D 변형)

```
입력: (875, 1) - 7초 × 125Hz

Layer 1: Conv1D(96, 11, stride=4) → ReLU → MaxPool(3, 2)
Layer 2: Conv1D(256, 5, padding='same') → ReLU → MaxPool(3, 2)
Layer 3: Conv1D(384, 3, padding='same') → ReLU
Layer 4: Conv1D(384, 3, padding='same') → ReLU
Layer 5: Conv1D(256, 3, padding='same') → ReLU → MaxPool(3, 2)
Layer 6: Flatten → Dense(4096) → ReLU → Dropout(0.5)
Layer 7: Dense(4096) → ReLU → Dropout(0.5)
Layer 8: Dense(1, 'SBP') + Dense(1, 'DBP')

파라미터 수: ~60M
```

#### 3.2.2 ResNet50 (1D 변형) - **최고 성능**

```
입력: (875, 1)

Stage 0: Conv1D(64, 7, stride=2) → BN → ReLU → MaxPool(3, 3)
Stage 1: ConvBlock(64,64,256) → IdentityBlock × 2
Stage 2: ConvBlock(128,128,512) → IdentityBlock × 3
Stage 3: ConvBlock(256,256,1024) → IdentityBlock × 5
Stage 4: ConvBlock(512,512,2048) → IdentityBlock × 2
출력: AvgPool → Flatten → Dense(1, 'SBP') + Dense(1, 'DBP')

파라미터 수: ~25M
```

#### 3.2.3 LSTM (Bidirectional)

```
입력: (875, 1)

Layer 1: Conv1D(64, 5, padding='causal') → ReLU
Layer 2: Bidirectional(LSTM(128, return_sequences=True))
Layer 3: Bidirectional(LSTM(128, return_sequences=True))
Layer 4: Bidirectional(LSTM(64, return_sequences=False))
Layer 5: Dense(512) → ReLU
Layer 6: Dense(256) → ReLU
Layer 7: Dense(128) → ReLU
Layer 8: Dense(1, 'SBP') + Dense(1, 'DBP')

파라미터 수: ~3M
```

#### 3.2.4 Slapnicar (Spectro-Temporal)

```
입력: (875, 1)

시간 도메인 분기:
├── PPG 원본 → SingleChannelResNet → GRU(65) → BN
├── PPG 1차 미분 → SingleChannelResNet → GRU(65) → BN  (선택적)
└── PPG 2차 미분 → SingleChannelResNet → GRU(65) → BN  (선택적)

주파수 도메인 분기:
├── STFT(128, hop=64) → Magnitude → Dense(32) → BN
├── (각 미분에 대해 동일)
└── 

병합:
Concatenate([시간, 주파수]) → Dense(32) → Dropout(0.25)
→ Dense(32) → Dropout(0.25)
→ Dense(1, 'SBP') + Dense(1, 'DBP')

파라미터 수: ~5M
```

### 3.3 논문의 신호 처리 파이프라인

```
[원본 데이터]
     ↓
[전처리]
├── Butterworth Bandpass Filter (0.5-8 Hz)
├── Z-score Normalization
└── 윈도우 분할 (7초, 50% 오버랩)
     ↓
[품질 검사]
├── SNR > -7 dB
├── SBP: 75-165 mmHg
├── DBP: 40-80 mmHg
└── HR: 50-140 bpm
     ↓
[데이터셋 분할]
├── Subject-based split (Non-mixed)
├── Train: 3750 subjects, 1M samples
├── Val: 625 subjects, 250K samples
└── Test: 625 subjects, 250K samples
     ↓
[학습]
├── Optimizer: Adam (lr=0.001)
├── Loss: MSE
├── Early Stopping (patience=10)
└── Checkpoint: Best validation loss
```

---

## 4. 2026년 최신 방법론

### 4.1 Vision Transformer (ViT) for rPPG

**PhysFormer (2022-2024 발전):**

```python
class PhysFormer(nn.Module):
    """
    Transformer 기반 End-to-End rPPG 추출
    비디오 → 직접 BP 예측
    """
    def __init__(self, img_size=128, patch_size=4, 
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        # Temporal Difference Convolution
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, (1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1)),
        )
        
        # Patch Embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, 
                mlp_ratio=4.0, drop=0.1
            ) for _ in range(depth)
        ])
        
        # Temporal Attention
        self.temporal_attn = TemporalAttention(embed_dim)
        
        # BP Prediction Head
        self.bp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # SBP, DBP
        )
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.stem(x)
        x = self.patch_embed(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.temporal_attn(x)
        bp = self.bp_head(x)
        
        return bp
```

### 4.2 Self-Supervised Learning

**Contrastive Learning for rPPG:**

```python
class ContrastiveRPPG(nn.Module):
    """
    레이블 없이 rPPG 특징 학습
    같은 사람의 다른 시간대 = positive pair
    다른 사람 = negative pair
    """
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.temperature = 0.07
    
    def contrastive_loss(self, z_i, z_j):
        """NT-Xent Loss"""
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim = sim / self.temperature
        
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size),
            torch.arange(batch_size)
        ]).to(z.device)
        
        loss = F.cross_entropy(sim, labels)
        return loss
```

### 4.3 Multi-Task Learning

**동시 예측: BP + HR + SpO2**

```python
class MultiTaskBPModel(nn.Module):
    """
    여러 생체신호 동시 예측으로 특징 공유
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(backbone.output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads
        self.bp_head = nn.Linear(512, 2)    # SBP, DBP
        self.hr_head = nn.Linear(512, 1)    # Heart Rate
        self.spo2_head = nn.Linear(512, 1)  # SpO2
        
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared(features)
        
        bp = self.bp_head(shared)
        hr = self.hr_head(shared)
        spo2 = self.spo2_head(shared)
        
        return {
            'bp': bp,
            'hr': hr,
            'spo2': spo2
        }
    
    def compute_loss(self, pred, target, weights={'bp': 1.0, 'hr': 0.3, 'spo2': 0.3}):
        loss = 0
        for task, weight in weights.items():
            loss += weight * F.mse_loss(pred[task], target[task])
        return loss
```

### 4.4 Domain Adaptation (PPG → rPPG)

```python
class DomainAdaptationBP(nn.Module):
    """
    PPG 도메인에서 학습 → rPPG 도메인으로 적응
    Adversarial Training 사용
    """
    def __init__(self, feature_extractor, bp_predictor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.bp_predictor = bp_predictor
        
        # Domain discriminator
        self.domain_classifier = nn.Sequential(
            GradientReversal(lambda_=1.0),  # Gradient Reversal Layer
            nn.Linear(feature_extractor.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_domain=False):
        features = self.feature_extractor(x)
        bp = self.bp_predictor(features)
        
        if return_domain:
            domain = self.domain_classifier(features)
            return bp, domain
        return bp
```

### 4.5 Diffusion Models for Signal Denoising

```python
class DiffusionDenoiser(nn.Module):
    """
    Diffusion Model로 rPPG 신호 노이즈 제거
    """
    def __init__(self, signal_dim=875, time_embed_dim=128):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.GELU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )
        
        self.unet = UNet1D(
            in_channels=1,
            out_channels=1,
            time_embed_dim=time_embed_dim
        )
    
    def forward(self, x_noisy, t):
        t_embed = self.time_embed(t)
        noise_pred = self.unet(x_noisy, t_embed)
        return noise_pred
    
    @torch.no_grad()
    def denoise(self, x_noisy, num_steps=50):
        """DDPM 역과정으로 노이즈 제거"""
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((x_noisy.shape[0],), t)
            noise_pred = self.forward(x_noisy, t_tensor)
            x_noisy = self.ddpm_step(x_noisy, noise_pred, t)
        return x_noisy
```

### 4.6 Real-time Optimization

```python
# ONNX 변환 및 최적화
import onnx
import onnxruntime as ort

def export_to_onnx(model, sample_input, output_path):
    """PyTorch → ONNX 변환"""
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        input_names=['input'],
        output_names=['sbp', 'dbp'],
        dynamic_axes={'input': {0: 'batch_size'}},
        opset_version=13
    )

def optimize_onnx(input_path, output_path):
    """ONNX 모델 최적화"""
    from onnxruntime.transformers import optimizer
    optimized = optimizer.optimize_model(
        input_path,
        model_type='bert',  # transformer 구조
        num_heads=12,
        hidden_size=768
    )
    optimized.save_model_to_file(output_path)

# TensorRT 변환 (NVIDIA GPU용)
# trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

---

## 5. 구현 로드맵

### 5.1 전체 일정 (9주)

```
Week 1-2: Phase 1 (Quick Fix)
├── Day 1-2: 환경 설정 및 기존 코드 분석
├── Day 3-4: Haar Cascade 파라미터 조정
├── Day 5-7: 밴드패스 필터 추가
├── Day 8-10: MediaPipe 기본 통합
└── Day 11-14: ResNet 모델 테스트 및 기본 평가

Week 3-5: Phase 2 (논문 구현)
├── Week 3: POS 알고리즘 완전 구현
│   ├── RGB 추출 모듈
│   ├── 정규화 및 투영
│   └── 펄스 신호 추출
├── Week 4: 신호 처리 파이프라인
│   ├── 적절한 리샘플링
│   ├── SNR 계산 및 품질 필터
│   └── 전처리 통합
└── Week 5: Transfer Learning
    ├── rPPG 데이터셋 준비
    ├── Fine-tuning 실행
    └── 성능 평가

Week 6-8: Phase 3 (최신 기술)
├── Week 6: Transformer 기반 모델
│   ├── PhysFormer 구조 구현
│   └── 학습 파이프라인 구축
├── Week 7: Self-Supervised Pre-training
│   ├── Contrastive Learning 구현
│   └── Pre-training 실행
└── Week 8: Multi-Task & Domain Adaptation
    ├── Multi-Task 헤드 추가
    └── Domain Adaptation 적용

Week 9: 최적화 및 배포
├── Day 1-3: ONNX/TensorRT 변환
├── Day 4-5: Real-time 성능 테스트
└── Day 6-7: 문서화 및 최종 정리
```

### 5.2 주요 마일스톤

| Week | 목표 | 예상 결과 |
|------|------|----------|
| 2 | Quick Fix 완료 | MAE 개선 (비정상 → 100+ mmHg) |
| 5 | 논문 구현 완료 | MAE ~16 mmHg (논문 수준) |
| 8 | 최신 기술 적용 | MAE ~10 mmHg |
| 9 | 최적화 완료 | Real-time (>30 FPS) |

---

## 6. 성능 비교표

### 6.1 방법별 예상 성능

| 구현 단계 | SBP MAE (mmHg) | DBP MAE (mmHg) | 추론 시간 | 구현 난이도 |
|-----------|---------------|---------------|-----------|-------------|
| 현재 (Green 평균) | ~2000 (비정상) | ~900 (비정상) | <10ms | ✅ 완료 |
| Quick Fix | ~100-200 | ~50-100 | <20ms | ⭐ 쉬움 |
| 논문 구현 (ResNet) | **16.4** | **8.5** | ~50ms | ⭐⭐⭐ 보통 |
| 논문 + Fine-tuning | 12-14 | 6-7 | ~50ms | ⭐⭐⭐ 보통 |
| PhysFormer | 10-12 | 5-6 | ~100ms | ⭐⭐⭐⭐ 어려움 |
| Multi-Task + DA | **8-10** | **4-5** | ~100ms | ⭐⭐⭐⭐⭐ 매우 어려움 |

### 6.2 논문의 실제 결과 (Non-mixed Dataset)

| 모델 | SBP MAE | DBP MAE | 비고 |
|------|---------|---------|------|
| Mean Regressor | 20.2 | 10.7 | 기준선 |
| AlexNet | 17.1 | 8.8 | |
| **ResNet** | **16.4** | **8.5** | 최고 성능 |
| LSTM | 17.6 | 9.0 | |
| Slapnicar | 18.3 | 9.4 | |

### 6.3 rPPG Fine-tuning 후 결과

| 조건 | SBP MAE | DBP MAE |
|------|---------|---------|
| PPG 모델 (Fine-tuning 전) | 28.9 | 15.2 |
| rPPG Fine-tuning 후 | 14.1 | 8.3 |
| + Personalization (first 20%) | **12.7** | **7.1** |

---

## 7. 코드 예제

### 7.1 완전한 POS 알고리즘 구현

```python
"""
pos_algorithm.py - Wang et al. 2017 POS 알고리즘 완전 구현
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq

class POSExtractor:
    """
    Plane-Orthogonal-to-Skin (POS) rPPG 추출기
    """
    
    def __init__(self, fs=30, window_size=1.6):
        """
        Args:
            fs: 샘플링 주파수 (Hz)
            window_size: 윈도우 크기 (초)
        """
        self.fs = fs
        self.window_samples = int(window_size * fs)
        
    def extract_rgb_signals(self, frames, face_detector):
        """
        비디오 프레임에서 RGB 시계열 추출
        
        Args:
            frames: BGR 프레임 리스트
            face_detector: 얼굴 감지 객체
            
        Returns:
            rgb: (N, 3) RGB 신호
        """
        rgb_signals = []
        
        for frame in frames:
            roi = face_detector.detect(frame)
            
            if roi is not None and roi.size > 0:
                # 피부 영역 마스킹 (선택사항)
                skin_mask = self._get_skin_mask(roi)
                
                if skin_mask.sum() > 100:  # 최소 픽셀 수
                    r = np.mean(roi[:,:,2][skin_mask])
                    g = np.mean(roi[:,:,1][skin_mask])
                    b = np.mean(roi[:,:,0][skin_mask])
                else:
                    r = np.mean(roi[:,:,2])
                    g = np.mean(roi[:,:,1])
                    b = np.mean(roi[:,:,0])
                    
                rgb_signals.append([r, g, b])
            else:
                # 이전 값 사용 또는 보간
                if rgb_signals:
                    rgb_signals.append(rgb_signals[-1])
                    
        return np.array(rgb_signals)
    
    def _get_skin_mask(self, roi):
        """HSV 기반 피부색 마스크"""
        import cv2
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 피부색 범위 (HSV)
        lower = np.array([0, 20, 70])
        upper = np.array([20, 255, 255])
        
        mask = cv2.inRange(hsv, lower, upper)
        return mask > 0
    
    def pos_algorithm(self, rgb):
        """
        POS 알고리즘으로 펄스 신호 추출
        
        Args:
            rgb: (N, 3) RGB 시계열
            
        Returns:
            pulse: (N,) 펄스 신호
        """
        N = rgb.shape[0]
        l = self.window_samples
        H = np.zeros(N)
        
        for t in range(l, N):
            # 윈도우 내 RGB
            C = rgb[t-l:t, :].T  # (3, l)
            
            # 시간 정규화
            mean_C = np.mean(C, axis=1, keepdims=True)
            C_n = C / (mean_C + 1e-10)
            
            # POS 투영
            S = np.array([
                C_n[1] - C_n[2],              # G - B
                C_n[1] + C_n[2] - 2*C_n[0]    # G + B - 2R
            ])
            
            # 표준편차 기반 결합
            std1 = np.std(S[0])
            std2 = np.std(S[1])
            
            alpha = std1 / (std2 + 1e-10)
            
            # 윈도우 내 펄스 신호
            h = S[0] + alpha * S[1]
            
            # 중첩-가산 방식
            H[t-l:t] += (h - np.mean(h))
        
        return H
    
    def bandpass_filter(self, signal, lowcut=0.7, highcut=4.0, order=4):
        """
        Butterworth 밴드패스 필터
        
        Args:
            signal: 입력 신호
            lowcut: 하한 주파수 (Hz)
            highcut: 상한 주파수 (Hz)
            
        Returns:
            filtered: 필터링된 신호
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # 경계 조건 확인
        if low <= 0:
            low = 0.01
        if high >= 1:
            high = 0.99
            
        b, a = butter(order, [low, high], btype='band')
        
        # 신호가 너무 짧으면 필터링 건너뜀
        if len(signal) < 3 * max(len(b), len(a)):
            return signal
            
        return filtfilt(b, a, signal)
    
    def estimate_heart_rate(self, pulse_signal):
        """
        펄스 신호에서 심박수 추정
        
        Returns:
            hr: 심박수 (bpm)
        """
        # FFT 기반 추정
        n = len(pulse_signal)
        freq = fftfreq(n, 1/self.fs)
        fft_vals = np.abs(fft(pulse_signal))
        
        # 0.7-4 Hz 범위
        valid_idx = (freq > 0.7) & (freq < 4.0)
        
        if not np.any(valid_idx):
            return 60  # 기본값
            
        peak_freq = freq[valid_idx][np.argmax(fft_vals[valid_idx])]
        hr = peak_freq * 60  # Hz → bpm
        
        return hr
    
    def extract(self, frames, face_detector):
        """
        전체 파이프라인 실행
        
        Args:
            frames: BGR 프레임 리스트
            face_detector: 얼굴 감지 객체
            
        Returns:
            pulse: 추출된 펄스 신호
            hr: 추정 심박수
        """
        # 1. RGB 추출
        rgb = self.extract_rgb_signals(frames, face_detector)
        
        if len(rgb) < self.window_samples * 2:
            raise ValueError("프레임 수 부족")
        
        # 2. POS 알고리즘
        pulse = self.pos_algorithm(rgb)
        
        # 3. 밴드패스 필터
        pulse = self.bandpass_filter(pulse)
        
        # 4. 정규화
        pulse = (pulse - np.mean(pulse)) / (np.std(pulse) + 1e-10)
        
        # 5. 심박수 추정
        hr = self.estimate_heart_rate(pulse)
        
        return pulse, hr


# 사용 예시
if __name__ == "__main__":
    import cv2
    
    # MediaPipe 얼굴 감지기
    class SimpleFaceDetector:
        def __init__(self):
            self.cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
        def detect(self, frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.1, 8, minSize=(100, 100))
            
            if len(faces) == 0:
                return None
                
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            return frame[y:y+h, x:x+w]
    
    # 카메라 캡처
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    detector = SimpleFaceDetector()
    extractor = POSExtractor(fs=fps)
    
    frames = []
    duration = 7  # 7초
    
    print(f"Capturing {duration} seconds of video...")
    
    while len(frames) < duration * fps:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            cv2.imshow('Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 신호 추출
    pulse, hr = extractor.extract(frames, detector)
    
    print(f"Estimated Heart Rate: {hr:.1f} bpm")
    print(f"Pulse signal length: {len(pulse)}")
```

### 7.2 개선된 카메라 테스트 스크립트

```python
"""
improved_camera_rppg.py - Quick Fix 적용 버전
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, resample
import tensorflow as tf
import tensorflow.keras as ks
from kapre import STFT, Magnitude, MagnitudeToDecibel


def bandpass_filter(signal, lowcut=0.7, highcut=4.0, fs=30, order=4):
    """밴드패스 필터"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def get_largest_face(faces):
    """가장 큰 얼굴 반환"""
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def load_model(model_path):
    """모델 로드"""
    dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel
    }
    return ks.models.load_model(model_path, custom_objects=dependencies)


def main():
    # 설정
    MODEL_PATH = 'data/resnet_ppg_nonmixed.h5'  # ResNet 사용
    CAMERA_ID = 0
    DURATION = 7  # 초
    TARGET_SAMPLES = 875
    
    print("="*60)
    print("개선된 rPPG 혈압 예측")
    print("="*60)
    
    # 모델 로드
    print("\n모델 로딩...")
    model = load_model(MODEL_PATH)
    print("✓ 모델 로드 완료")
    
    # 카메라 초기화
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"✓ 카메라 초기화 (FPS: {fps})")
    
    # 얼굴 감지기
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # 신호 버퍼
    signal_buffer = []
    target_frames = int(DURATION * fps)
    
    print(f"\n{DURATION}초 동안 신호 수집 시작...")
    print("얼굴을 카메라에 맞춰주세요. 'q'를 눌러 취소.")
    
    while len(signal_buffer) < target_frames:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 얼굴 감지 (개선된 파라미터)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=8,        # 4 → 8
            minSize=(100, 100),
            maxSize=(400, 400)
        )
        
        # 가장 큰 얼굴만 사용
        face = get_largest_face(faces)
        
        if face is not None:
            x, y, w, h = face
            roi = frame[y:y+h, x:x+w]
            
            # Green 채널 평균 (추후 POS로 교체)
            green_mean = np.mean(roi[:, :, 1])
            signal_buffer.append(green_mean)
            
            # 단일 박스만 표시
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 진행 상황
            progress = len(signal_buffer) / target_frames * 100
            cv2.putText(frame, f"Progress: {progress:.1f}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('rPPG Capture', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("취소됨")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✓ {len(signal_buffer)} 샘플 수집 완료")
    
    # 신호 처리
    signal = np.array(signal_buffer)
    
    # 1. 밴드패스 필터
    signal = bandpass_filter(signal, fs=fps)
    print("✓ 밴드패스 필터 적용 (0.7-4 Hz)")
    
    # 2. 정규화
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    print("✓ 정규화 완료")
    
    # 3. 리샘플링 (scipy resample 사용)
    signal = resample(signal, TARGET_SAMPLES)
    print(f"✓ 리샘플링 완료 ({len(signal_buffer)} → {TARGET_SAMPLES})")
    
    # 예측
    input_data = signal.reshape(1, TARGET_SAMPLES, 1)
    prediction = model.predict(input_data, verbose=0)
    
    # 결과 파싱
    if isinstance(prediction, list):
        sbp = float(prediction[0][0])
        dbp = float(prediction[1][0])
    else:
        sbp = float(prediction[0, 0])
        dbp = float(prediction[0, 1])
    
    print("\n" + "="*60)
    print("예측 결과")
    print("="*60)
    print(f"수축기 혈압 (SBP): {sbp:.1f} mmHg")
    print(f"이완기 혈압 (DBP): {dbp:.1f} mmHg")
    
    # 경고
    if sbp > 200 or sbp < 50 or dbp > 150 or dbp < 30:
        print("\n⚠️  경고: 예측값이 정상 범위를 벗어났습니다.")
        print("   이는 rPPG 신호 품질 문제일 수 있습니다.")
        print("   POS 알고리즘 구현 후 재시도하세요.")


if __name__ == "__main__":
    main()
```

---

## 8. 참고 자료

### 8.1 핵심 논문

1. **Schrumpf et al. 2021** - 본 프로젝트 기반 논문
   - "Assessment of Non-Invasive Blood Pressure Prediction from PPG and rPPG Signals Using Deep Learning"
   - Sensors 2021, 21(18), 6022

2. **Wang et al. 2017** - POS 알고리즘
   - "Algorithmic Principles of Remote PPG"
   - IEEE Trans. Biomed. Eng., vol. 64, no. 7, pp. 1479-1491

3. **Slapničar et al. 2019** - Spectro-temporal 모델
   - "Blood Pressure Estimation from Photoplethysmogram Using a Spectro-Temporal Deep Neural Network"
   - Sensors 2019, 19(15), 3420

### 8.2 최신 논문 (2023-2026)

1. **PhysFormer** (2022)
   - "PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer"
   - CVPR 2022

2. **EfficientPhys** (2023)
   - "EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement"
   - WACV 2023

3. **Contrast-Phys** (2023)
   - "Contrast-Phys: Self-Supervised Learning for Remote Physiological Measurement"
   - ICCV 2023

### 8.3 오픈소스 프로젝트

1. **pyVHR** - Python Video Heart Rate
   - https://github.com/phuselab/pyVHR
   - POS, CHROM, ICA 등 다양한 알고리즘 구현

2. **rPPG-Toolbox**
   - https://github.com/ubicomplab/rPPG-Toolbox
   - 벤치마크 및 평가 도구

3. **PhysNet**
   - End-to-end rPPG 추출 네트워크

### 8.4 데이터셋

1. **MIMIC-III** - PPG 데이터
   - https://physionet.org/content/mimiciii/

2. **UBFC-rPPG** - rPPG 벤치마크
   - https://sites.google.com/view/yaboromance/ubfc-rppg

3. **PURE** - rPPG 데이터셋
   - 다양한 움직임 조건 포함

---

## 📝 면책 조항

⚠️ **중요 공지:**
- 이 문서의 구현은 **교육 및 연구 목적**입니다
- **의료 진단에 절대 사용하지 마세요**
- 실제 혈압 측정이 필요하면 **인증된 의료 기기**를 사용하세요
- 예측 결과는 참고용이며, 의료적 결정의 근거로 사용할 수 없습니다

---

**문서 작성:** 2026-01-19  
**버전:** 1.0  
**다음 업데이트 예정:** Phase 1 구현 완료 후
