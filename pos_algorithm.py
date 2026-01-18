"""
pos_algorithm.py - Wang et al. 2017 POS (Plane-Orthogonal-to-Skin) 알고리즘 구현

논문: "Algorithmic Principles of Remote PPG"
저자: Wang et al. (2017)
출처: IEEE Transactions on Biomedical Engineering, vol. 64, no. 7, pp. 1479-1491, 2017

POS 알고리즘은 비디오에서 원격 광용적맥파 (rPPG) 신호를 추출하는 알고리즘입니다.
Green 채널만 사용하는 간단한 방법보다 훨씬 더 정확한 신호를 추출합니다.
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample
from typing import Optional, Tuple


class POSExtractor:
    """
    Plane-Orthogonal-to-Skin (POS) rPPG 추출기
    
    Wang et al. 2017의 POS 알고리즘을 구현합니다.
    
    핵심 원리:
    - 피부색의 시간 변화는 주로 혈류 변화로 인함
    - RGB 신호를 피부에 직교하는 평면에 투영
    - 두 개의 직교 성분을 결합하여 펄스 신호 추출
    """
    
    def __init__(self, fs: float = 30, window_size: float = 1.6):
        """
        Args:
            fs: 샘플링 주파수 (Hz), 기본값 30 (카메라 FPS)
            window_size: 슬라이딩 윈도우 크기 (초), 기본값 1.6초
                        원논문에서는 40프레임 권장 (30fps 기준 약 1.33초)
        """
        self.fs = fs
        self.window_samples = int(window_size * fs)
        
    def extract_rgb_signals(self, frames: list, get_roi_func) -> np.ndarray:
        """
        비디오 프레임에서 RGB 시계열 신호 추출
        
        Args:
            frames: BGR 형식의 프레임 리스트
            get_roi_func: 각 프레임에서 얼굴 ROI를 반환하는 함수
                         함수 시그니처: (frame) -> roi (또는 None)
        
        Returns:
            rgb: (N, 3) shape의 RGB 신호 배열
                 각 행은 [R, G, B] 평균값
        """
        rgb_signals = []
        last_valid = None
        
        for frame in frames:
            roi = get_roi_func(frame)
            
            if roi is not None and roi.size > 0:
                # BGR to RGB 변환
                # OpenCV는 BGR, 우리는 RGB 필요
                b = np.mean(roi[:, :, 0])
                g = np.mean(roi[:, :, 1])
                r = np.mean(roi[:, :, 2])
                
                rgb_signals.append([r, g, b])
                last_valid = [r, g, b]
            else:
                # ROI를 찾지 못한 경우 이전 값으로 보간
                if last_valid is not None:
                    rgb_signals.append(last_valid)
                else:
                    # 아직 유효한 값이 없으면 회색 (128, 128, 128)으로 시작
                    rgb_signals.append([128, 128, 128])
        
        return np.array(rgb_signals, dtype=np.float32)
    
    def pos_algorithm(self, rgb: np.ndarray) -> np.ndarray:
        """
        POS 알고리즘으로 펄스 신호 추출
        
        알고리즘:
        1. 시간 윈도우 내 RGB 신호 정규화
        2. 두 개의 직교 신호 성분 계산:
           - S1 = G - B
           - S2 = G + B - 2R
        3. 표준편차 비율로 가중 결합:
           - H = S1 + (std(S1) / std(S2)) * S2
        
        Args:
            rgb: (N, 3) RGB 시계열 신호
        
        Returns:
            pulse: (N,) 추출된 펄스 신호
        """
        N = rgb.shape[0]
        l = self.window_samples
        
        # 슬라이딩 윈도우 기반 펄스 신호
        H = np.zeros(N)
        
        for t in range(l, N):
            # 현재 윈도우 내 RGB 신호 (window_size x 3)
            C = rgb[t-l:t, :].T  # (3, window_size)
            
            # 1단계: 시간 정규화
            # 각 채널의 평균값으로 나눔 (0 나눗셈 방지)
            mean_C = np.mean(C, axis=1, keepdims=True)  # (3, 1)
            C_n = C / (mean_C + 1e-10)  # (3, window_size)
            
            # 2단계: 직교 신호 계산
            S = np.array([
                C_n[1, :] - C_n[2, :],              # S1 = G - B
                C_n[1, :] + C_n[2, :] - 2*C_n[0, :]  # S2 = G + B - 2R
            ])
            
            # 3단계: 표준편차 기반 가중 결합
            std1 = np.std(S[0, :])
            std2 = np.std(S[1, :])
            
            # 가중치 계산 (0 나눗셈 방지)
            if std2 > 1e-10:
                alpha = std1 / std2
            else:
                alpha = 0.0
            
            # 펄스 신호 계산 (0-mean)
            h = S[0, :] + alpha * S[1, :]
            h_mean_removed = h - np.mean(h)
            
            # 윈도우 중심 샘플 저장 (슬라이딩 윈도우 기반)
            # 마지막 샘플만 저장
            H[t] = h_mean_removed[-1]
        
        return H
    
    def bandpass_filter(self, signal: np.ndarray, 
                       lowcut: float = 0.7, 
                       highcut: float = 4.0, 
                       order: int = 4) -> np.ndarray:
        """
        Butterworth 밴드패스 필터
        
        심박수 범위 (0.7-4 Hz = 42-240 bpm)에 해당하는 신호만 통과
        
        Args:
            signal: 입력 신호
            lowcut: 하한 주파수 (Hz), 기본값 0.7
            highcut: 상한 주파수 (Hz), 기본값 4.0
            order: 필터 차수, 기본값 4
        
        Returns:
            filtered: 필터링된 신호
        """
        # Nyquist 주파수
        nyq = 0.5 * self.fs
        
        # 정규화된 주파수 계산
        low = lowcut / nyq
        high = highcut / nyq
        
        # 범위 확인 및 클리핑
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)
        
        # low가 high보다 크면 실패
        if low >= high:
            return signal
        
        try:
            # Butterworth 필터 설계
            b, a = butter(order, [low, high], btype='band')
            
            # 양방향 필터링 (filtfilt는 phase distortion을 제거함)
            filtered = filtfilt(b, a, signal)
            return filtered
        except Exception as e:
            print(f"필터링 오류: {e}")
            return signal
    
    def estimate_heart_rate(self, pulse_signal: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        펄스 신호에서 심박수와 주파수 스펙트럼 추정
        
        FFT를 사용하여 주파수 도메인에서 피크 주파수 검출
        
        Args:
            pulse_signal: 펄스 신호
        
        Returns:
            hr: 심박수 (bpm)
            freqs: 주파수 배열 (debugging용)
        """
        N = len(pulse_signal)
        
        # FFT 계산
        fft_vals = np.fft.fft(pulse_signal)
        freqs = np.fft.fftfreq(N, 1/self.fs)
        
        # 양수 주파수만 고려
        positive_idx = freqs > 0
        freqs_positive = freqs[positive_idx]
        fft_positive = np.abs(fft_vals[positive_idx])
        
        # 심박수 범위 (0.7-4 Hz = 42-240 bpm)
        valid_idx = (freqs_positive > 0.7) & (freqs_positive < 4.0)
        
        if not np.any(valid_idx):
            # 유효한 주파수 범위가 없으면 기본값 반환
            return 60.0, freqs_positive
        
        # 최대 피크의 주파수
        peak_freq = freqs_positive[valid_idx][np.argmax(fft_positive[valid_idx])]
        
        # Hz → bpm 변환
        hr = peak_freq * 60.0
        
        return hr, freqs_positive
    
    def extract(self, frames: list, get_roi_func) -> Tuple[np.ndarray, float]:
        """
        완전한 POS 파이프라인 실행
        
        Args:
            frames: BGR 프레임 리스트
            get_roi_func: ROI 추출 함수
        
        Returns:
            pulse: 정규화된 펄스 신호
            hr: 추정된 심박수 (bpm)
        """
        # 1단계: RGB 신호 추출
        if len(frames) < self.window_samples * 2:
            raise ValueError(f"프레임 수 부족: {len(frames)} < {self.window_samples * 2}")
        
        rgb = self.extract_rgb_signals(frames, get_roi_func)
        print(f"✓ RGB 신호 추출 완료: {rgb.shape}")
        
        # 2단계: POS 알고리즘 적용
        pulse = self.pos_algorithm(rgb)
        print(f"✓ POS 알고리즘 적용 완료: {pulse.shape}")
        
        # 3단계: 밴드패스 필터링
        pulse = self.bandpass_filter(pulse, lowcut=0.7, highcut=4.0)
        print(f"✓ 밴드패스 필터 적용 완료")
        
        # 4단계: 정규화
        mean_pulse = np.mean(pulse)
        std_pulse = np.std(pulse)
        if std_pulse > 1e-10:
            pulse = (pulse - mean_pulse) / std_pulse
        else:
            pulse = pulse - mean_pulse
        print(f"✓ 정규화 완료")
        
        # 5단계: 심박수 추정
        hr, _ = self.estimate_heart_rate(pulse)
        print(f"✓ 심박수 추정: {hr:.1f} bpm")
        
        return pulse, hr


# 테스트 코드
if __name__ == "__main__":
    print("POS 알고리즘 모듈 로드됨")
    print("사용 예시:")
    print("  from pos_algorithm import POSExtractor")
    print("  extractor = POSExtractor(fs=30, window_size=1.6)")
    print("  pulse, hr = extractor.extract(frames, roi_func)")
