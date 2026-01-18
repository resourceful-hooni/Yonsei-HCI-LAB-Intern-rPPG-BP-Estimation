"""
test_phase2_step3.py - Phase 2-Step 3 테스트

POS 알고리즘과 MediaPipe 기본 기능 테스트
"""

import numpy as np
from pos_algorithm import POSExtractor
from mediapipe_face_detector import MediaPipeFaceDetector, HaarCascadeFaceDetector

print("="*60)
print("Phase 2-Step 3 테스트: POS + MediaPipe 모듈")
print("="*60)

# 1. POS 알고리즘 초기화
print("\n[1] POS 알고리즘 초기화...")
pos = POSExtractor(fs=30, window_size=1.6)
print(f"✓ POS 초기화 완료")
print(f"  - 샘플링 주파수: 30 Hz")
print(f"  - 윈도우 크기: {pos.window_samples} 샘플 (1.6초)")

# 2. 가짜 RGB 신호 생성 (테스트용)
print("\n[2] 테스트 RGB 신호 생성...")
N = 300  # 10초 분량 (30fps)
t = np.arange(N) / 30.0

# 심박수 약 75 bpm = 1.25 Hz
hr_freq = 1.25
pulse = np.sin(2 * np.pi * hr_freq * t)
noise = 0.1 * np.random.randn(N)

# RGB 신호 생성 (heart rate modulation)
rgb = np.zeros((N, 3))
rgb[:, 0] = 100 + 5 * pulse + noise  # R (적게 변함)
rgb[:, 1] = 120 + 10 * pulse + noise  # G (많이 변함)
rgb[:, 2] = 110 + 3 * pulse + noise  # B (중간)
print(f"✓ RGB 신호 생성 완료: {rgb.shape}")

# 3. POS 알고리즘 테스트
print("\n[3] POS 알고리즘 실행...")
try:
    pulse_signal = pos.pos_algorithm(rgb)
    print(f"✓ POS 알고리즘 완료")
    print(f"  - 출력 신호 길이: {len(pulse_signal)}")
    print(f"  - 신호 범위: [{pulse_signal.min():.4f}, {pulse_signal.max():.4f}]")
except Exception as e:
    print(f"❌ POS 알고리즘 실패: {e}")

# 4. 밴드패스 필터 테스트
print("\n[4] 밴드패스 필터 테스트...")
try:
    filtered_signal = pos.bandpass_filter(pulse_signal, lowcut=0.7, highcut=4.0)
    print(f"✓ 필터링 완료")
    print(f"  - 필터 범위: 0.7-4.0 Hz (42-240 bpm)")
    print(f"  - 필터링 후 신호 범위: [{filtered_signal.min():.4f}, {filtered_signal.max():.4f}]")
except Exception as e:
    print(f"❌ 필터링 실패: {e}")

# 5. 심박수 추정 테스트
print("\n[5] 심박수 추정...")
try:
    hr, freqs = pos.estimate_heart_rate(filtered_signal)
    print(f"✓ 심박수 추정 완료")
    print(f"  - 예상 심박수: 75 bpm")
    print(f"  - 추정 심박수: {hr:.1f} bpm")
    print(f"  - 오차: {abs(hr - 75):.1f} bpm")
except Exception as e:
    print(f"❌ 심박수 추정 실패: {e}")

# 6. MediaPipe 얼굴 감지기 초기화
print("\n[6] MediaPipe 얼굴 감지기 초기화...")
try:
    mediapipe_detector = MediaPipeFaceDetector(min_detection_confidence=0.7)
    print(f"✓ MediaPipe 감지기 초기화 완료")
except Exception as e:
    print(f"❌ MediaPipe 초기화 실패: {e}")

# 7. Haar Cascade 비교 (폴백)
print("\n[7] Haar Cascade 감지기 초기화...")
try:
    haar_detector = HaarCascadeFaceDetector(min_neighbors=8)
    print(f"✓ Haar Cascade 감지기 초기화 완료")
except Exception as e:
    print(f"❌ Haar Cascade 초기화 실패: {e}")

print("\n" + "="*60)
print("테스트 완료!")
print("="*60)
print("\n다음 단계: 실제 카메라로 camera_rppg_advanced.py 테스트")
