"""
bp_stability.py - 혈압 예측값 안정화 모듈

예측값의 안정성을 높이기 위한 기법:
1. 칼만 필터 (Kalman Filter)
2. 이동 평균 (Moving Average)
3. 이상치 제거 (Outlier Rejection)
4. 신뢰도 기반 가중 평균
"""

import numpy as np
from typing import Tuple, Optional, List
from collections import deque


class KalmanFilter:
    """
    1D 칼만 필터 (혈압 예측값 평활화용)
    """
    
    def __init__(self, process_variance: float = 0.01, 
                 measurement_variance: float = 1.0):
        """
        Args:
            process_variance: 프로세스 노이즈 (작을수록 안정적)
            measurement_variance: 측정 노이즈 (클수록 새 측정값 신뢰 낮음)
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        self.estimate = None
        self.estimate_error = 1.0
    
    def update(self, measurement: float) -> float:
        """
        새 측정값으로 추정값 업데이트
        
        Args:
            measurement: 새 측정값
        
        Returns:
            filtered_value: 필터링된 값
        """
        if self.estimate is None:
            self.estimate = measurement
            return measurement
        
        # 예측 단계
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance
        
        # 업데이트 단계
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate
    
    def reset(self):
        """필터 초기화"""
        self.estimate = None
        self.estimate_error = 1.0


class BPStabilizer:
    """
    혈압 예측값 안정화
    """
    
    def __init__(self, window_size: int = 5, outlier_threshold: float = 2.5):
        """
        Args:
            window_size: 이동 평균 윈도우 크기
            outlier_threshold: 이상치 판단 기준 (표준편차 배수)
        """
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold
        
        # 칼만 필터
        self.sbp_kalman = KalmanFilter(process_variance=0.01, measurement_variance=5.0)
        self.dbp_kalman = KalmanFilter(process_variance=0.01, measurement_variance=3.0)
        
        # 이동 평균용 버퍼
        self.sbp_buffer = deque(maxlen=window_size)
        self.dbp_buffer = deque(maxlen=window_size)
        
        # 품질 기반 가중치 버퍼
        self.quality_buffer = deque(maxlen=window_size)
        
        # 통계
        self.sbp_stats = {'mean': 120, 'std': 15}
        self.dbp_stats = {'mean': 80, 'std': 10}
    
    def is_outlier(self, value: float, bp_type: str = 'sbp') -> bool:
        """
        이상치 여부 판단
        
        Args:
            value: 예측값
            bp_type: 'sbp' 또는 'dbp'
        
        Returns:
            True if outlier
        """
        stats = self.sbp_stats if bp_type == 'sbp' else self.dbp_stats
        
        # Z-score 계산
        z_score = abs(value - stats['mean']) / stats['std']
        
        return z_score > self.outlier_threshold
    
    def update_stats(self):
        """버퍼 기반으로 통계 업데이트"""
        if len(self.sbp_buffer) >= 3:
            self.sbp_stats['mean'] = np.mean(list(self.sbp_buffer))
            self.sbp_stats['std'] = np.std(list(self.sbp_buffer))
        
        if len(self.dbp_buffer) >= 3:
            self.dbp_stats['mean'] = np.mean(list(self.dbp_buffer))
            self.dbp_stats['std'] = np.std(list(self.dbp_buffer))
    
    def stabilize(self, sbp: float, dbp: float, 
                  quality_score: float = 1.0) -> Tuple[float, float, dict]:
        """
        혈압 예측값 안정화
        
        Args:
            sbp: 수축기 혈압
            dbp: 이완기 혈압
            quality_score: 신호 품질 점수 (0-1)
        
        Returns:
            stabilized_sbp, stabilized_dbp, info
        """
        info = {
            'sbp_outlier': False,
            'dbp_outlier': False,
            'sbp_raw': sbp,
            'dbp_raw': dbp,
            'method': 'kalman'
        }
        
        # 1. 이상치 검사
        sbp_outlier = self.is_outlier(sbp, 'sbp')
        dbp_outlier = self.is_outlier(dbp, 'dbp')
        
        info['sbp_outlier'] = sbp_outlier
        info['dbp_outlier'] = dbp_outlier
        
        # 이상치면 칼만 필터 예측값만 사용 (측정값 무시)
        if sbp_outlier:
            if self.sbp_kalman.estimate is not None:
                sbp = self.sbp_kalman.estimate
                info['sbp_corrected'] = True
        
        if dbp_outlier:
            if self.dbp_kalman.estimate is not None:
                dbp = self.dbp_kalman.estimate
                info['dbp_corrected'] = True
        
        # 2. 범위 체크 및 클리핑
        sbp = np.clip(sbp, 70, 200)
        dbp = np.clip(dbp, 40, 130)
        
        # 3. 칼만 필터 적용
        sbp_kalman = self.sbp_kalman.update(sbp)
        dbp_kalman = self.dbp_kalman.update(dbp)
        
        # 4. 버퍼에 추가
        self.sbp_buffer.append(sbp_kalman)
        self.dbp_buffer.append(dbp_kalman)
        self.quality_buffer.append(quality_score)
        
        # 5. 품질 기반 가중 평균
        if len(self.sbp_buffer) >= 3:
            weights = np.array(list(self.quality_buffer))
            weights = weights / (np.sum(weights) + 1e-10)
            
            sbp_weighted = np.average(list(self.sbp_buffer), weights=weights)
            dbp_weighted = np.average(list(self.dbp_buffer), weights=weights)
            
            info['method'] = 'weighted_average'
        else:
            sbp_weighted = sbp_kalman
            dbp_weighted = dbp_kalman
        
        # 6. 통계 업데이트
        self.update_stats()
        
        # 7. 생리학적 타당성 검사 (SBP > DBP)
        if sbp_weighted <= dbp_weighted:
            # DBP가 SBP보다 크면 안됨
            sbp_weighted = dbp_weighted + 20
            info['physiological_correction'] = True
        
        return sbp_weighted, dbp_weighted, info
    
    def get_confidence(self) -> float:
        """
        현재 예측값의 신뢰도 반환
        
        Returns:
            confidence: 0-1 사이 값
        """
        if len(self.sbp_buffer) < 3:
            return 0.5
        
        # 최근 예측값들의 일관성 기반
        sbp_std = np.std(list(self.sbp_buffer))
        dbp_std = np.std(list(self.dbp_buffer))
        
        # 표준편차가 작을수록 신뢰도 높음
        sbp_conf = 1 / (1 + sbp_std / 10)
        dbp_conf = 1 / (1 + dbp_std / 10)
        
        return (sbp_conf + dbp_conf) / 2
    
    def reset(self):
        """상태 초기화"""
        self.sbp_kalman.reset()
        self.dbp_kalman.reset()
        self.sbp_buffer.clear()
        self.dbp_buffer.clear()
        self.quality_buffer.clear()


class MultiReadingAverager:
    """
    다중 측정값 평균 (여러 번 측정 후 평균)
    """
    
    def __init__(self, required_readings: int = 3, max_std: float = 10.0):
        """
        Args:
            required_readings: 필요한 측정 횟수
            max_std: 허용 최대 표준편차 (mmHg)
        """
        self.required_readings = required_readings
        self.max_std = max_std
        
        self.sbp_readings = []
        self.dbp_readings = []
        self.quality_scores = []
    
    def add_reading(self, sbp: float, dbp: float, quality: float = 1.0):
        """측정값 추가"""
        self.sbp_readings.append(sbp)
        self.dbp_readings.append(dbp)
        self.quality_scores.append(quality)
    
    def is_ready(self) -> bool:
        """측정 완료 여부"""
        return len(self.sbp_readings) >= self.required_readings
    
    def get_average(self) -> Optional[Tuple[float, float, dict]]:
        """
        평균값 계산
        
        Returns:
            (sbp_avg, dbp_avg, info) or None
        """
        if not self.is_ready():
            return None
        
        sbp_array = np.array(self.sbp_readings)
        dbp_array = np.array(self.dbp_readings)
        quality_array = np.array(self.quality_scores)
        
        # 이상치 제거 (IQR 방법)
        sbp_cleaned, dbp_cleaned, quality_cleaned = self._remove_outliers(
            sbp_array, dbp_array, quality_array
        )
        
        if len(sbp_cleaned) == 0:
            return None
        
        # 품질 가중 평균
        weights = quality_cleaned / np.sum(quality_cleaned)
        
        sbp_avg = np.average(sbp_cleaned, weights=weights)
        dbp_avg = np.average(dbp_cleaned, weights=weights)
        
        sbp_std = np.std(sbp_cleaned)
        dbp_std = np.std(dbp_cleaned)
        
        info = {
            'sbp_std': sbp_std,
            'dbp_std': dbp_std,
            'num_readings': len(sbp_cleaned),
            'reliable': sbp_std < self.max_std and dbp_std < self.max_std
        }
        
        return sbp_avg, dbp_avg, info
    
    def _remove_outliers(self, sbp, dbp, quality):
        """IQR 방법으로 이상치 제거"""
        # SBP 이상치 제거
        q1_sbp, q3_sbp = np.percentile(sbp, [25, 75])
        iqr_sbp = q3_sbp - q1_sbp
        lower_sbp = q1_sbp - 1.5 * iqr_sbp
        upper_sbp = q3_sbp + 1.5 * iqr_sbp
        
        # DBP 이상치 제거
        q1_dbp, q3_dbp = np.percentile(dbp, [25, 75])
        iqr_dbp = q3_dbp - q1_dbp
        lower_dbp = q1_dbp - 1.5 * iqr_dbp
        upper_dbp = q3_dbp + 1.5 * iqr_dbp
        
        # 양쪽 조건 모두 만족하는 값만
        mask = (sbp >= lower_sbp) & (sbp <= upper_sbp) & \
               (dbp >= lower_dbp) & (dbp <= upper_dbp)
        
        return sbp[mask], dbp[mask], quality[mask]
    
    def reset(self):
        """초기화"""
        self.sbp_readings.clear()
        self.dbp_readings.clear()
        self.quality_scores.clear()


if __name__ == "__main__":
    # 테스트
    print("=== BP Stabilizer 테스트 ===\n")
    
    stabilizer = BPStabilizer(window_size=5)
    
    # 시뮬레이션: 노이즈가 있는 혈압 측정값
    true_sbp = 120
    true_dbp = 80
    
    measurements = []
    stabilized_values = []
    
    for i in range(20):
        # 노이즈 추가
        noise_sbp = np.random.randn() * 10
        noise_dbp = np.random.randn() * 7
        
        sbp_raw = true_sbp + noise_sbp
        dbp_raw = true_dbp + noise_dbp
        
        # 가끔 이상치 추가
        if i % 7 == 0:
            sbp_raw += 40
        
        # 품질 점수 (랜덤)
        quality = 0.5 + 0.5 * np.random.rand()
        
        # 안정화
        sbp_stable, dbp_stable, info = stabilizer.stabilize(sbp_raw, dbp_raw, quality)
        
        measurements.append((sbp_raw, dbp_raw))
        stabilized_values.append((sbp_stable, dbp_stable))
        
        print(f"측정 {i+1:2d}: SBP={sbp_raw:6.1f} → {sbp_stable:6.1f} mmHg, "
              f"DBP={dbp_raw:6.1f} → {dbp_stable:6.1f} mmHg "
              f"{'[이상치]' if info.get('sbp_outlier') else ''}")
    
    # 최종 통계
    final_sbp = [s[0] for s in stabilized_values[-5:]]
    final_dbp = [d[1] for d in stabilized_values[-5:]]
    
    print(f"\n최근 5개 평균: SBP={np.mean(final_sbp):.1f} mmHg, DBP={np.mean(final_dbp):.1f} mmHg")
    print(f"신뢰도: {stabilizer.get_confidence():.2f}")
    
    # 시각화
    try:
        import matplotlib.pyplot as plt
        
        meas_sbp = [m[0] for m in measurements]
        meas_dbp = [m[1] for m in measurements]
        stab_sbp = [s[0] for s in stabilized_values]
        stab_dbp = [s[1] for s in stabilized_values]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(meas_sbp, 'o-', alpha=0.5, label='Raw SBP')
        plt.plot(stab_sbp, 's-', label='Stabilized SBP')
        plt.axhline(true_sbp, color='r', linestyle='--', label='True SBP')
        plt.xlabel('Measurement')
        plt.ylabel('SBP (mmHg)')
        plt.legend()
        plt.title('Systolic Blood Pressure')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(meas_dbp, 'o-', alpha=0.5, label='Raw DBP')
        plt.plot(stab_dbp, 's-', label='Stabilized DBP')
        plt.axhline(true_dbp, color='r', linestyle='--', label='True DBP')
        plt.xlabel('Measurement')
        plt.ylabel('DBP (mmHg)')
        plt.legend()
        plt.title('Diastolic Blood Pressure')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bp_stability_test.png', dpi=150)
        print("\n✓ 테스트 완료: bp_stability_test.png 저장됨")
    except ImportError:
        print("\n✓ 테스트 완료 (matplotlib 없음)")
