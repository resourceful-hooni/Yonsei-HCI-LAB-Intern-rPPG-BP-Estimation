# Phase 3: 실행 계획 (구체적 작업 목록)

**작성일:** 2026-01-19  
**현재 상태:** Phase 2 완료 ✅ (모든 기능 구현 및 테스트 완료)  
**다음 단계:** Phase 3 - 성능 개선 (4주)

---

## 📊 현재 상황 정리

### ✅ 이미 구현된 것 (중복 제외)

| 구현 항목 | 파일 | 상태 | 성능 개선 |
|-----------|------|------|----------|
| POS 알고리즘 | `pos_algorithm.py` | ✅ 완료 | 신호 품질 ↑ |
| 신호 처리 파이프라인 | `camera_rppg_advanced.py` | ✅ 완료 | 노이즈 ↓ |
| 신호 품질 평가 | `signal_quality.py` | ✅ 완료 | 품질 필터링 |
| BP 안정화 | `bp_stability.py` | ✅ 완료 | 편차 ↓ |
| 실시간 UI | `camera_rppg_advanced.py` | ✅ 완료 | UX ↑ |

**현재 성능 (예상):**
- SBP MAE: ~20-25 mmHg (논문: 16.4)
- DBP MAE: ~10-12 mmHg (논문: 8.5)
- 이유: PPG 모델을 rPPG 데이터로 사용

---

## 🎯 Phase 3: 우선순위 정렬 (추천 순서)

### 1️⃣ **Domain Adaptation (PPG → rPPG)** ⭐⭐⭐ [1-2주]

**문제 정의:**
```
현재:   PPG 모델 (125Hz, 손가락) ← rPPG 입력 (30Hz, 얼굴) → 예측 부정확
해결:   rPPG 데이터로 모델 적응 학습
```

**구체적 작업:**
1. `prepare_rppg_dataset.py` - rPPG 데이터 전처리
   - 기존 `rPPG-BP-UKL_rppg_7s.h5` 활용
   - 데이터 정규화 및 validation split
   
2. `domain_adaptation.py` - 적응 학습 모듈
   - Pre-trained ResNet 로드
   - rPPG 데이터로 fine-tuning (레이어 5개만 unfreeze)
   - Adversarial loss 추가 (선택사항)
   
3. 학습 실행
   ```bash
   python train_domain_adaptation.py \
       --model resnet_ppg_nonmixed.h5 \
       --data rPPG-BP-UKL_rppg_7s.h5 \
       --epochs 50 \
       --output models/resnet_rppg_adapted.h5
   ```

**예상 효과:** SBP MAE 28.9 → 14.1 mmHg (논문 실적)

**필요한 데이터:** 이미 있음 (`rPPG-BP-UKL_rppg_7s.h5`)

---

### 2️⃣ **Multi-Task Learning (BP + HR + SpO2)** ⭐⭐⭐ [1주]

**문제 정의:**
```
현재:   BP만 예측 (단일 작업)
해결:   HR, SpO2도 함께 예측 (특징 공유)
```

**구체적 작업:**
1. `multi_task_model.py` - 멀티테스크 헤드
   ```python
   shared_features (ResNet backbone)
       ├── BP Head: Dense → SBP, DBP
       ├── HR Head: Dense → HR (bpm)
       └── SpO2 Head: Dense → SpO2 (%)
   ```
   
2. `train_multi_task.py` - 멀티테스크 학습
   - Loss: `loss_bp + 0.3*loss_hr + 0.3*loss_spo2`
   - HR: 이미 `estimate_heart_rate()`로 추출 중
   - SpO2: 신호 품질에서 계산 가능
   
3. `camera_rppg_advanced.py` 수정
   - HR, SpO2 동시 예측 표시
   - 신뢰도 점수 통합

**예상 효과:** 
- BP 안정성 ↑ (특징 공유)
- HR 정확도 ↑
- 사용자 정보 ↑

---

### 3️⃣ **Enhanced Face Recognition (3D Landmarks)** ⭐⭐ [2-3일] [간소화됨]

**문제 정의:**
```
현재:   2D 얼굴 박스 + 기본 HSV 피부 마스킹
기존:   ROI 추출, 안정화 이미 완료 ✅
개선:   3D Face Mesh로 정확한 피부 영역만 추출
```

**주의:** ROI 안정화는 `signal_quality.py`의 `ROIStabilizer`에서 이미 구현됨

**구체적 작업:**
1. `face_mesh_extractor.py` - 3D 얼굴 랜드마크 추출 (신규)
   - MediaPipe Face Mesh (478 포인트)
   - 피부 영역 자동 추출 (이마, 볼, 턱만)
   - 눈, 코, 입 자동 제외
   - 얼굴 각도 보정
   
2. `camera_rppg_advanced.py` 수정
   - `--enable-face-mesh` flag 추가
   - Face Mesh 기반 신호 추출 (선택사항)
   - 기존 ROI 방식과 병행 가능

**예상 효과:**
- 신호 품질 ↑ (정확한 피부 영역만 사용)
- 노이즈 감소 (눈/입 등 움직이는 부위 제외)

---

### 4️⃣ **Attention + Lightweight Transformer** ⭐ [2-3주]

**문제 정의:**
```
현재:   CNN (고정 커널 크기)
해결:   Transformer (동적 주의 메커니즘)
```

**구체적 작업:**
1. `transformer_bp_model.py` - 경량 Transformer
   ```python
   Temporal Attention: 신호의 중요한 시간대 집중
   Spatial Attention: 신호 채널별 가중치
   ```
   
2. `train_transformer.py` - 학습
   - ResNet보다 작은 모델 (효율성)
   - 전이 학습 활용

**예상 효과:**
- 성능 약간 개선 (8-10%)
- 해석 가능성 ↑

---

### 5️⃣ **ONNX/TensorRT Optimization** ⭐ [1주]

**문제 정의:**
```
현재:   TensorFlow (100ms+ 추론 시간)
해결:   ONNX/TensorRT (10-50ms)
```

**구체적 작업:**
1. `export_onnx.py` - ONNX 변환
   ```bash
   python -m tf2onnx.convert --savedmodel resnet_bp_model --output_file model.onnx
   ```
   
2. `optimize_tensorrt.py` - TensorRT 최적화
   ```bash
   trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
   ```
   
3. `inference_optimized.py` - 최적화 추론

**예상 효과:**
- 추론 시간: 100ms → 20ms ⚡
- 배포 효율 ↑

---

## 📋 Phase 3 상세 작업 계획

### Week 1-2: Domain Adaptation (필수)

#### Day 1: 데이터 준비
- **작업:** `prepare_rppg_dataset.py` 작성
  ```python
  1. rPPG-BP-UKL_rppg_7s.h5 로드
  2. Train/Val/Test split (70/15/15)
  3. 정규화 및 데이터 증강
  4. 저장: data/rppg_train.h5, rppg_val.h5, rppg_test.h5
  ```
- **산출물:** 전처리된 rPPG 데이터셋
- **테스트:** 데이터 로드 성공 + 통계 출력

#### Day 2-3: 모델 적응 학습
- **작업:** `domain_adaptation.py` + `train_domain_adaptation.py`
  ```python
  1. ResNet 모델 로드
  2. 마지막 3개 레이어만 fine-tuning
  3. rPPG 데이터로 50 epoch 학습
  4. Validation loss 최소값에서 저장
  ```
- **산출물:** `models/resnet_rppg_adapted.h5`
- **테스트:** 학습 곡선 (loss ↓), 예측값 범위 확인

#### Day 4: 통합 및 평가
- **작업:** `camera_rppg_advanced.py` 수정
  ```python
  --model models/resnet_rppg_adapted.h5
  ```
- **산출물:** 개선된 실시간 예측
- **테스트:** 카메라 테스트 (7초 × 3회)

**GitHub Commit:** `Phase 3-1: Domain Adaptation - rPPG fine-tuning`

---

### Week 2-3: Multi-Task Learning

#### Day 1: 모델 설계
- **작업:** `multi_task_model.py`
  ```python
  Backbone: ResNet (기존)
  ├── BP Head: → SBP, DBP
  ├── HR Head: → HR (50-150 bpm)
  └── SpO2 Head: → SpO2 (90-100%)
  ```
- **산출물:** 모델 정의
- **테스트:** 모델 요약 출력

#### Day 2: 학습 구현
- **작업:** `train_multi_task.py`
  ```python
  Loss = MSE_BP + 0.3*MSE_HR + 0.3*MSE_SpO2
  ```
- **산출물:** `models/multi_task_bp_model.h5`
- **테스트:** 학습 완료

#### Day 3: 통합
- **작업:** `camera_rppg_advanced.py` 수정
  ```python
  bp, hr, spo2 = model.predict(input)
  ```
- **산출물:** 멀티테스크 실시간 예측
- **테스트:** 카메라 테스트

**GitHub Commit:** `Phase 3-2: Multi-Task Learning - BP+HR+SpO2`

---

### Week 3: Enhanced Face Recognition (간소화됨)

#### Day 1: 3D Face Mesh 추출기 작성
- **작업:** `face_mesh_extractor.py`
  ```python
  Face Mesh (478 포인트) → 피부 영역 마스크 생성
  (기존 ROI 안정화는 그대로 사용)
  ```
- **산출물:** 3D 랜드마크 기반 ROI 마스크 생성 모듈
- **테스트:** 얼굴 랜드마크 시각화

#### Day 2: 통합
- **작업:** `camera_rppg_advanced.py` 수정
  ```python
  --enable-face-mesh flag 추가
  Face Mesh 기반 신호 추출 옵션
  ```
- **산출물:** 선택 가능한 3D mesh 기반 예측
- **테스트:** 카메라 테스트 (기존 vs 3D Mesh 비교)

**GitHub Commit:** `Phase 3-3: Enhanced Face Recognition - 3D Face Mesh landmarks`

---

### Week 4: Optimization

#### Option A: Transformer (권장)
- `transformer_bp_model.py`
- `train_transformer.py`
- 성능 + 해석성 ↑

#### Option B: ONNX/TensorRT
- `export_onnx.py`
- `inference_optimized.py`
- 속도 ↑

---

## 🚀 지금 바로 할 일 (다음 단계)

### ✅ 체크리스트

- [ ] 1. `prepare_rppg_dataset.py` 작성 (30분)
- [ ] 2. rPPG 데이터 전처리 실행 (10분)
- [ ] 3. `domain_adaptation.py` 작성 (1시간)
- [ ] 4. 모델 fine-tuning 시작 (30분)
- [ ] 5. 결과 평가 및 GitHub commit (30분)

**예상 소요 시간:** 3시간

---

## 📌 중요 노트

### COMPREHENSIVE_SOLUTION_GUIDE.md와의 차이

| 가이드 | 용도 | 내용 |
|--------|------|------|
| COMPREHENSIVE_SOLUTION_GUIDE.md | 📚 **이론** | 개념, 논문, 알고리즘 설명 |
| PHASE3_ACTION_PLAN.md (이 문서) | ⚙️ **실행** | 구체적 작업 목록, 코드 예제 |

**결론:** 
- 이론은 이미 가이드에 다 있음
- **이 ACTION_PLAN을 따라 실행하면 됨**
- 각 단계마다 GitHub commit

---

### 중복 검토 결과

**검토 문서:** DUPLICATE_CHECK.md 참고

| 항목 | 상태 | 설명 |
|------|------|------|
| Domain Adaptation | ✅ 신규 | 전혀 구현 안됨 |
| Multi-Task Learning | ✅ 신규 | 전혀 구현 안됨 |
| Enhanced Face Recognition | ⚠️ 간소화 | ROI/안정화는 기존, 3D Mesh만 신규 |
| Transformer | ✅ 신규 | 전혀 구현 안됨 |
| ONNX/TensorRT | ✅ 신규 | 전혀 구현 안됨 |

**80% 신규 기능, 20% 기존 기능과 통합**

---

## 🎯 최종 목표

```
현재:         Phase 2 완료 (SBP MAE ~20-25)
     ↓
1-2주:        Domain Adaptation (SBP MAE ~14)  ← 가장 중요
     ↓
2-3주:        Multi-Task Learning (HR+SpO2 추가)
     ↓
3주:          Enhanced Face Recognition (안정성)
     ↓
4주:          최적화 선택 (Transformer 또는 ONNX)
     ↓
최종:         SBP MAE ~10-12 mmHg (논문 수준) 🎉
```

---

**작성:** 2026-01-19  
**상태:** 준비 완료 → 즉시 실행 가능 ✅
