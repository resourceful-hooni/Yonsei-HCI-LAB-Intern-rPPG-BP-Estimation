"""
signal_quality.py - rPPG 신호 품질 평가 및 개선

신호 품질 문제를 해결하기 위한 모듈:
1. SNR (Signal-to-Noise Ratio) 계산
2. 신호 품질 점수 계산
3. 조명 변화 감지 및 보정
4. 움직임 아티팩트 감지
5. 시간적 평활화
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional


class SignalQualityAssessor:
    """
    rPPG 신호 품질 평가 및 개선
    """
    
    def __init__(self, fs: float = 30):
        """
        Args:
            fs: 샘플링 주파수 (Hz)
        """
        self.fs = fs
        self.hr_range = (40, 180)  # BPM
        self.freq_range = (self.hr_range[0]/60, self.hr_range[1]/60)  # Hz
    
    def compute_snr(self, signal_data: np.ndarray, hr_freq: float) -> float:
        """
        SNR (Signal-to-Noise Ratio) 계산
        
        Args:
            signal_data: 신호 데이터
            hr_freq: 심박수 주파수 (Hz)
        
        Returns:
            SNR (dB)
        """
        # FFT
        N = len(signal_data)
        yf = fft(signal_data)
        xf = fftfreq(N, 1/self.fs)[:N//2]
        power = np.abs(yf[:N//2])**2
        
        # HR 주파수 주변의 파워 (신호)
        hr_band_width = 0.1  # ±0.1 Hz
        hr_mask = (xf >= hr_freq - hr_band_width) & (xf <= hr_freq + hr_band_width)
        signal_power = np.sum(power[hr_mask])
        
        # 전체 파워에서 신호 제외한 부분 (노이즈)
        noise_power = np.sum(power[~hr_mask])
        
        # SNR (dB)
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 0
        
        return snr
    
    def assess_quality(self, signal_data: np.ndarray) -> Tuple[float, dict]:
        """
        신호 품질 종합 평가
        
        Args:
            signal_data: 신호 데이터
        
        Returns:
            quality_score: 0-1 사이 품질 점수
            metrics: 세부 메트릭
        """
        metrics = {}
        
        # 1. 표준편차 (신호 변동성)
        std_val = np.std(signal_data)
        metrics['std'] = std_val
        
        # 2. 피크 품질
        peaks, properties = sp_signal.find_peaks(
            signal_data,
            height=np.mean(signal_data),
            distance=int(self.fs * 0.4)  # 최소 0.4초 간격
        )
        metrics['num_peaks'] = len(peaks)
        
        if len(peaks) > 1:
            # 피크 간 간격 일관성 (심박수 변동성)
            peak_intervals = np.diff(peaks)
            metrics['peak_regularity'] = 1 - (np.std(peak_intervals) / np.mean(peak_intervals))
        else:
            metrics['peak_regularity'] = 0
        
        # 3. 주파수 도메인 품질
        N = len(signal_data)
        yf = fft(signal_data)
        xf = fftfreq(N, 1/self.fs)[:N//2]
        power = np.abs(yf[:N//2])**2
        
        # HR 범위 내 파워 비율
        hr_mask = (xf >= self.freq_range[0]) & (xf <= self.freq_range[1])
        hr_power_ratio = np.sum(power[hr_mask]) / (np.sum(power) + 1e-10)
        metrics['hr_power_ratio'] = hr_power_ratio
        
        # 가장 강한 주파수 (심박수)
        dominant_freq_idx = np.argmax(power[hr_mask])
        dominant_freq = xf[hr_mask][dominant_freq_idx]
        metrics['dominant_hr'] = dominant_freq * 60  # BPM
        
        # SNR
        snr = self.compute_snr(signal_data, dominant_freq)
        metrics['snr'] = snr
        
        # 4. 품질 점수 계산 (0-1)
        score = 0
        
        # 표준편차 (0.1-1.0 정규화)
        if std_val > 0.1:
            score += 0.2
        
        # 피크 수 (최소 3개 이상)
        if metrics['num_peaks'] >= 3:
            score += 0.2
        
        # 피크 일관성 (0.7 이상)
        if metrics['peak_regularity'] > 0.7:
            score += 0.2
        
        # HR 파워 비율 (0.3 이상)
        if hr_power_ratio > 0.3:
            score += 0.2
        
        # SNR (0 dB 이상)
        if snr > 0:
            score += 0.2
        
        return score, metrics
    
    def detect_motion_artifacts(self, signal_data: np.ndarray, 
                                window_size: int = 30) -> np.ndarray:
        """
        움직임 아티팩트 감지
        
        Args:
            signal_data: 신호 데이터
            window_size: 윈도우 크기 (프레임 수)
        
        Returns:
            motion_mask: True = 정상, False = 아티팩트
        """
        N = len(signal_data)
        motion_mask = np.ones(N, dtype=bool)
        
        for i in range(window_size, N):
            window = signal_data[i-window_size:i]
            
            # 1. 급격한 변화 감지 (표준편차 기준)
            if i > 0:
                diff = abs(signal_data[i] - signal_data[i-1])
                window_std = np.std(window)
                
                # 3 표준편차 이상 변화 = 아티팩트
                if diff > 3 * window_std:
                    motion_mask[i] = False
            
            # 2. 윈도우 내 비정상 분산
            window_std = np.std(window)
            global_std = np.std(signal_data)
            
            # 전체 표준편차의 3배 이상 = 아티팩트
            if window_std > 3 * global_std:
                motion_mask[i-window_size:i] = False
        
        return motion_mask
    
    def temporal_smoothing(self, signal_data: np.ndarray, 
                          alpha: float = 0.3) -> np.ndarray:
        """
        지수 이동 평균으로 시간적 평활화
        
        Args:
            signal_data: 원본 신호
            alpha: 평활화 계수 (0-1, 작을수록 부드러움)
        
        Returns:
            smoothed: 평활화된 신호
        """
        smoothed = np.zeros_like(signal_data)
        smoothed[0] = signal_data[0]
        
        for i in range(1, len(signal_data)):
            smoothed[i] = alpha * signal_data[i] + (1 - alpha) * smoothed[i-1]
        
        return smoothed
    
    def adaptive_filtering(self, signal_data: np.ndarray,
                          hr_estimate: Optional[float] = None) -> np.ndarray:
        """
        적응형 밴드패스 필터링
        
        Args:
            signal_data: 원본 신호
            hr_estimate: 추정 심박수 (BPM), None이면 자동 추정
        
        Returns:
            filtered: 필터링된 신호
        """
        if hr_estimate is None:
            # FFT로 심박수 추정
            N = len(signal_data)
            yf = fft(signal_data)
            xf = fftfreq(N, 1/self.fs)[:N//2]
            power = np.abs(yf[:N//2])**2
            
            hr_mask = (xf >= self.freq_range[0]) & (xf <= self.freq_range[1])
            dominant_freq_idx = np.argmax(power[hr_mask])
            dominant_freq = xf[hr_mask][dominant_freq_idx]
            hr_estimate = dominant_freq * 60
        
        # 심박수 중심으로 적응형 밴드패스
        hr_freq = hr_estimate / 60  # Hz
        lowcut = max(0.5, hr_freq - 0.5)  # ±0.5 Hz
        highcut = min(4.0, hr_freq + 0.5)
        
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = sp_signal.butter(4, [low, high], btype='band')
        filtered = sp_signal.filtfilt(b, a, signal_data)
        
        return filtered
    
    def detrend_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """
        선형 트렌드 제거 (조명 변화 보정)
        
        Args:
            signal_data: 원본 신호
        
        Returns:
            detrended: 트렌드 제거된 신호
        """
        return sp_signal.detrend(signal_data, type='linear')
    
    def normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Z-score 정규화
        
        Args:
            signal_data: 원본 신호
        
        Returns:
            normalized: 정규화된 신호
        """
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        
        if std > 1e-10:
            return (signal_data - mean) / std
        else:
            return signal_data - mean


class ROIStabilizer:
    """
    얼굴 ROI 안정화 (떨림 방지)
    """
    
    def __init__(self, smoothing_factor: float = 0.3):
        """
        Args:
            smoothing_factor: 평활화 계수 (0-1, 클수록 이전 값에 가중치)
        """
        self.smoothing_factor = smoothing_factor
        self.prev_rect = None
        self.lost_frames = 0
        self.max_lost_frames = 5
    
    def stabilize(self, rect: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        얼굴 박스 좌표 안정화
        
        Args:
            rect: (x, y, w, h) 현재 얼굴 박스
        
        Returns:
            stabilized_rect: 안정화된 얼굴 박스
        """
        if rect is None:
            self.lost_frames += 1
            # 5프레임 이상 감지 안되면 초기화
            if self.lost_frames > self.max_lost_frames:
                self.prev_rect = None
            return self.prev_rect
        
        self.lost_frames = 0
        
        if self.prev_rect is None:
            self.prev_rect = rect
            return rect
        
        # 지수 이동 평균
        x, y, w, h = rect
        px, py, pw, ph = self.prev_rect
        
        alpha = 1 - self.smoothing_factor
        
        new_x = int(alpha * x + self.smoothing_factor * px)
        new_y = int(alpha * y + self.smoothing_factor * py)
        new_w = int(alpha * w + self.smoothing_factor * pw)
        new_h = int(alpha * h + self.smoothing_factor * ph)
        
        stabilized = (new_x, new_y, new_w, new_h)
        self.prev_rect = stabilized
        
        return stabilized
    
    def reset(self):
        """상태 초기화"""
        self.prev_rect = None


if __name__ == "__main__":
    # 테스트
    import matplotlib.pyplot as plt
    
    # 합성 신호 생성
    fs = 30
    t = np.arange(0, 10, 1/fs)
    hr = 75  # 75 BPM
    clean_signal = np.sin(2 * np.pi * (hr/60) * t)
    
    # 노이즈 추가
    noise = 0.3 * np.random.randn(len(t))
    noisy_signal = clean_signal + noise
    
    # 품질 평가
    assessor = SignalQualityAssessor(fs=fs)
    score, metrics = assessor.assess_quality(noisy_signal)
    
    print("=== 신호 품질 평가 ===")
    print(f"품질 점수: {score:.2f}")
    print(f"표준편차: {metrics['std']:.3f}")
    print(f"피크 수: {metrics['num_peaks']}")
    print(f"피크 일관성: {metrics['peak_regularity']:.3f}")
    print(f"HR 파워 비율: {metrics['hr_power_ratio']:.3f}")
    print(f"추정 심박수: {metrics['dominant_hr']:.1f} BPM")
    print(f"SNR: {metrics['snr']:.2f} dB")
    
    # 신호 개선
    detrended = assessor.detrend_signal(noisy_signal)
    filtered = assessor.adaptive_filtering(detrended)
    smoothed = assessor.temporal_smoothing(filtered)
    
    # 시각화
    plt.figure(figsize=(12, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(t, clean_signal, label='Clean')
    plt.plot(t, noisy_signal, label='Noisy', alpha=0.7)
    plt.legend()
    plt.title('Original Signals')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 2)
    plt.plot(t, detrended)
    plt.title('Detrended')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 3)
    plt.plot(t, filtered)
    plt.title('Adaptive Filtered')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 4)
    plt.plot(t, smoothed)
    plt.title('Smoothed')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig('signal_quality_test.png', dpi=150)
    print("\n✓ 테스트 완료: signal_quality_test.png 저장됨")
