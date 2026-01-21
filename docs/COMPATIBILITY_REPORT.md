# 호환성 분석 보고서
# Generated: 2026-01-21

## 현재 문제점

### 1. TensorFlow 버전 중복 설치
- tensorflow==2.4.4
- tensorflow-gpu==2.4.1  
- tensorflow-intel==2.13.0

**문제**: 3개 버전이 동시에 설치되어 충돌

### 2. 의존성 버전 불일치

| 패키지 | 필요 버전 (TF 2.4.x) | 현재 버전 | 상태 |
|--------|---------------------|----------|------|
| numpy | ~1.19.2 | 1.23.5 | ❌ 불일치 |
| absl-py | ~0.10 | 2.3.1 | ❌ 불일치 |
| flatbuffers | ~1.12.0 | 25.12.19 | ❌ 불일치 |
| protobuf | 3.x | 3.20.3 | ✅ 호환 |

### 3. MediaPipe 호환성
- MediaPipe 0.10.5: Python 3.8+, NumPy 1.21+ 필요
- TensorFlow 2.4.x: NumPy ~1.19.2 필요
- **충돌**: NumPy 버전 요구사항 상충

## 해결 방안

### Option 1: TensorFlow 2.13 기반 (권장)
- 최신 기능 지원
- MediaPipe와 완벽 호환
- GPU 지원 개선

```
tensorflow==2.13.0
numpy==1.23.5
mediapipe==0.10.5
opencv-python==4.8.0.76
protobuf==3.20.3
```

### Option 2: TensorFlow 2.4 유지 + MediaPipe 제거
- 기존 모델 완벽 호환
- Haar Cascade만 사용
- GPU 지원 (tensorflow-gpu 2.4.1)

```
tensorflow-gpu==2.4.1
numpy==1.19.5
opencv-python==4.5.3.56
protobuf==3.19.6
absl-py==0.10.0
flatbuffers==1.12
```

### Option 3: 하이브리드 (중간)
- TensorFlow 2.10.x
- MediaPipe 호환 + 기존 모델 지원

```
tensorflow==2.10.1
numpy==1.22.4
mediapipe==0.10.5
opencv-python==4.7.0.72
```

## 권장 솔루션: Option 1 (TensorFlow 2.13)

### 이유:
1. ✅ MediaPipe와 완벽 호환
2. ✅ 최신 보안 패치
3. ✅ 더 나은 성능
4. ✅ NumPy 1.23.5 지원
5. ✅ 기존 .h5 모델 로드 가능

### 마이그레이션 위험도: 낮음
- Keras 모델 API 동일
- custom_objects 로드 호환
- 성능 개선 예상

## 실행 계획

### 1단계: 기존 패키지 정리
```powershell
pip uninstall tensorflow tensorflow-gpu tensorflow-intel -y
pip uninstall numpy absl-py flatbuffers -y
```

### 2단계: 호환 패키지 설치
```powershell
pip install tensorflow==2.13.0
pip install numpy==1.23.5
pip install mediapipe==0.10.5
pip install opencv-python==4.8.0.76
pip install protobuf==3.20.3
pip install absl-py==1.4.0
pip install flatbuffers==23.5.26
```

### 3단계: 기존 패키지 재설치
```powershell
pip install -r requirements.txt --no-deps
```

### 4단계: 검증
```powershell
python -c "import tensorflow as tf; print('TF:', tf.__version__)"
python -c "import mediapipe; print('MP:', mediapipe.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

## 예상 효과

- ✅ MediaPipe Face Detection 정상 작동
- ✅ TensorFlow 모델 로드 안정화
- ✅ NumPy 호환성 문제 해결
- ✅ 의존성 충돌 제거
- ⚠️ 기존 모델 재테스트 필요 (호환성 높음)
