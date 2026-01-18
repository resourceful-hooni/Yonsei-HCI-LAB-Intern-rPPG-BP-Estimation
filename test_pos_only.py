"""
test_pos_only.py - POS 알고리즘만 테스트 (MediaPipe 제외)
"""

import numpy as np
from pos_algorithm import POSExtractor

print("="*60)
print("Phase 2-Step 3 테스트: POS 알고리즘")
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
print(f"  - R 범위: [{rgb[:, 0].min():.1f}, {rgb[:, 0].max():.1f}]")
print(f"  - G 범위: [{rgb[:, 1].min():.1f}, {rgb[:, 1].max():.1f}]")
print(f"  - B 범위: [{rgb[:, 2].min():.1f}, {rgb[:, 2].max():.1f}]")

# 3. POS 알고리즘 테스트
print("\n[3] POS 알고리즘 실행...")
try:
    pulse_signal = pos.pos_algorithm(rgb)
    print(f"✓ POS 알고리즘 완료")
    print(f"  - 출력 신호 길이: {len(pulse_signal)}")
    print(f"  - 신호 범위: [{pulse_signal.min():.4f}, {pulse_signal.max():.4f}]")
    print(f"  - 신호 표준편차: {np.std(pulse_signal):.4f}")
except Exception as e:
    print(f"❌ POS 알고리즘 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. 밴드패스 필터 테스트
print("\n[4] 밴드패스 필터 테스트...")
try:
    filtered_signal = pos.bandpass_filter(pulse_signal, lowcut=0.7, highcut=4.0)
    print(f"✓ 필터링 완료")
    print(f"  - 필터 범위: 0.7-4.0 Hz (42-240 bpm)")
    print(f"  - 필터링 후 신호 범위: [{filtered_signal.min():.4f}, {filtered_signal.max():.4f}]")
except Exception as e:
    print(f"❌ 필터링 실패: {e}")
    import traceback
    traceback.print_exc()

# 5. 심박수 추정 테스트
print("\n[5] 심박수 추정...")
try:
    hr, freqs = pos.estimate_heart_rate(filtered_signal)
    print(f"✓ 심박수 추정 완료")
    print(f"  - 예상 심박수: 75 bpm (1.25 Hz)")
    print(f"  - 추정 심박수: {hr:.1f} bpm ({hr/60:.3f} Hz)")
    print(f"  - 오차: {abs(hr - 75):.1f} bpm ({abs(hr-75)/75*100:.1f}%)")
except Exception as e:
    print(f"❌ 심박수 추정 실패: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✓ POS 알고리즘 테스트 완료!")
print("="*60)
print("\n다음 단계:")
print("1. MediaPipe 호환성 문제 해결")
print("2. camera_rppg_advanced.py 테스트")
