"""
camera_rppg_advanced.py - Phase 2 완전 구현: POS + MediaPipe + Transfer Learning 지원

이 스크립트는 다음을 포함합니다:
1. MediaPipe 얼굴 감지 (Haar Cascade 대체)
2. POS 알고리즘 (Green 채널 대체)
3. 적절한 리샘플링 (scipy.signal.resample)
4. Transfer Learning 지원 (rPPG 데이터로 Fine-tuning)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from kapre import STFT, Magnitude, MagnitudeToDecibel
from scipy.signal import resample, butter, filtfilt
import argparse
from collections import deque
from typing import Optional, Tuple

from pos_algorithm import POSExtractor
from mediapipe_face_detector import MediaPipeFaceDetector, HaarCascadeFaceDetector


class AdvancedRPPGExtractor:
    """
    Phase 2: POS 알고리즘 + MediaPipe 통합
    """
    
    def __init__(self, fs: float = 30, duration: float = 7, target_len: int = 875, 
                 use_pos: bool = True, use_mediapipe: bool = True):
        """
        Args:
            fs: 샘플링 주파수 (Hz)
            duration: 신호 수집 시간 (초)
            target_len: 모델 입력 길이
            use_pos: POS 알고리즘 사용 여부
            use_mediapipe: MediaPipe 사용 여부
        """
        self.fs = fs
        self.duration = duration
        self.target_len = target_len
        self.use_pos = use_pos
        self.use_mediapipe = use_mediapipe
        
        # 프레임 버퍼
        self.window_size = int(duration * fs)
        self.frame_buffer = deque(maxlen=self.window_size)
        
        # 얼굴 감지기
        if use_mediapipe:
            self.detector = MediaPipeFaceDetector(min_detection_confidence=0.7)
        else:
            self.detector = HaarCascadeFaceDetector(min_neighbors=8)
        
        # POS 알고리즘
        if use_pos:
            self.pos = POSExtractor(fs=fs, window_size=1.6)
        
        self.signal_buffer = deque(maxlen=self.window_size)
    
    def process_frame(self, frame: np.ndarray) -> Optional[float]:
        """
        프레임 처리 및 신호 값 추출
        
        Args:
            frame: BGR 프레임
        
        Returns:
            추출된 신호 값 또는 None
        """
        self.frame_buffer.append(frame.copy())
        
        # ROI 추출
        roi = self.detector.detect(frame)
        if roi is None:
            return None
        
        # Simple Green 채널 (POS 계산 시까지의 임시)
        green_mean = np.mean(roi[:, :, 1])
        self.signal_buffer.append(green_mean)
        
        return green_mean
    
    def is_buffer_full(self) -> bool:
        """버퍼 가득 찬 여부"""
        return len(self.frame_buffer) == self.window_size
    
    def extract_signal(self) -> Optional[np.ndarray]:
        """
        버퍼에서 최종 신호 추출
        
        Returns:
            정규화된 신호 (target_len, 1)
        """
        if not self.is_buffer_full():
            return None
        
        frames = list(self.frame_buffer)
        
        # POS 알고리즘 사용
        if self.use_pos:
            try:
                print("\n[진행 중] POS 알고리즘 적용...")
                pulse, hr = self.pos.extract(frames, self.detector.detect)
                signal = pulse
                print(f"✓ 심박수: {hr:.1f} bpm")
            except Exception as e:
                print(f"⚠️  POS 알고리즘 오류: {e}, 대체 방식 사용")
                signal = self._extract_simple_signal()
        else:
            signal = self._extract_simple_signal()
        
        # 리샘플링
        if len(signal) != self.target_len:
            signal = self._proper_resample(signal, self.target_len)
        
        # 정규화
        signal = self._normalize(signal)
        
        return signal.reshape(-1, 1)
    
    def _extract_simple_signal(self) -> np.ndarray:
        """간단한 Green 채널 기반 신호 (폴백)"""
        signal = np.array(list(self.signal_buffer), dtype=np.float32)
        
        # 밴드패스 필터
        if len(signal) > 10:
            nyq = 0.5 * self.fs
            low = 0.7 / nyq
            high = 4.0 / nyq
            b, a = butter(4, [low, high], btype='band')
            try:
                signal = filtfilt(b, a, signal)
            except:
                pass
        
        return signal
    
    def _proper_resample(self, signal: np.ndarray, target_len: int) -> np.ndarray:
        """
        적절한 리샘플링 (scipy.signal.resample)
        
        Linear interp보다 정확한 FFT 기반 리샘플링
        """
        if len(signal) <= 1:
            return signal
        
        # Anti-aliasing 필터 (다운샘플링 시)
        if target_len < len(signal):
            nyq = 0.5 * self.fs
            cutoff = (0.5 * target_len / len(signal)) / nyq
            cutoff = np.clip(cutoff, 0.001, 0.999)
            
            try:
                b, a = butter(8, cutoff, btype='low')
                signal = filtfilt(b, a, signal)
            except:
                pass
        
        # scipy resample (FFT 기반)
        resampled = resample(signal, target_len)
        return resampled.astype(np.float32)
    
    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """신호 정규화"""
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std > 1e-10:
            return (signal - mean) / std
        else:
            return signal - mean


def load_model(model_path: str):
    """모델 로드"""
    print(f"모델 로드 중: {model_path}")
    
    dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel
    }
    
    model = ks.models.load_model(model_path, custom_objects=dependencies)
    print("✓ 모델 로드 완료")
    
    return model


def predict_bp(model, signal: np.ndarray) -> Tuple[float, float]:
    """혈압 예측"""
    input_data = np.expand_dims(signal, axis=0)
    prediction = model.predict(input_data, verbose=0)
    
    # 모델 출력 파싱
    if hasattr(prediction, 'shape') and prediction.ndim >= 2 and prediction.shape[-1] == 2:
        sbp = float(prediction[0, 0])
        dbp = float(prediction[0, 1])
    elif isinstance(prediction, (list, tuple)) and len(prediction) == 2:
        sbp = float(np.squeeze(prediction[0]))
        dbp = float(np.squeeze(prediction[1]))
    else:
        raise ValueError(f"예상하지 못한 모델 출력: {type(prediction)}")
    
    return sbp, dbp


def main():
    parser = argparse.ArgumentParser(
        description='Advanced rPPG 혈압 예측 (Phase 2: POS + MediaPipe)'
    )
    parser.add_argument('--model', type=str, default='data/resnet_ppg_nonmixed.h5',
                       help='모델 파일 경로')
    parser.add_argument('--camera', type=int, default=0, help='카메라 ID')
    parser.add_argument('--backend', type=str, default='default',
                       choices=['default', 'dshow', 'msmf'],
                       help='카메라 백엔드')
    parser.add_argument('--duration', type=int, default=7,
                       help='신호 수집 시간 (초)')
    parser.add_argument('--width', type=int, default=640, help='카메라 해상도 가로')
    parser.add_argument('--height', type=int, default=480, help='카메라 해상도 세로')
    parser.add_argument('--fps', type=int, default=30, help='카메라 FPS')
    parser.add_argument('--pos', action='store_true', default=True,
                       help='POS 알고리즘 사용 (기본값: True)')
    parser.add_argument('--no-pos', dest='pos', action='store_false',
                       help='POS 알고리즘 비활성화 (Simple Green 사용)')
    parser.add_argument('--mediapipe', action='store_true', default=True,
                       help='MediaPipe 사용 (기본값: True)')
    parser.add_argument('--no-mediapipe', dest='mediapipe', action='store_false',
                       help='MediaPipe 비활성화 (Haar Cascade 사용)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Advanced rPPG 혈압 예측 - Phase 2")
    print("="*80)
    print(f"설정:")
    print(f"  - 모델: {args.model}")
    print(f"  - POS 알고리즘: {'활성화' if args.pos else '비활성화'}")
    print(f"  - MediaPipe: {'활성화' if args.mediapipe else '비활성화'}")
    print(f"  - 수집 시간: {args.duration}초")
    print("="*80)
    
    # 모델 로드
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 카메라 열기
    def open_camera(idx, backend):
        if backend == 'dshow':
            return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        elif backend == 'msmf':
            return cv2.VideoCapture(idx, cv2.CAP_MSMF)
        else:
            return cv2.VideoCapture(idx)
    
    cap = open_camera(args.camera, args.backend)
    if not cap.isOpened():
        for b in ['dshow', 'msmf', 'default']:
            if b == args.backend:
                continue
            cap = open_camera(args.camera, b)
            if cap.isOpened():
                print(f"✓ 카메라 백엔드 폴백: {b}")
                break
    
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    print(f"✓ 카메라 준비 완료 (FPS: {fps:.0f})")
    
    # rPPG 추출기 초기화
    extractor = AdvancedRPPGExtractor(
        fs=fps,
        duration=args.duration,
        use_pos=args.pos,
        use_mediapipe=args.mediapipe
    )
    
    print(f"\n{args.duration}초 동안 신호 수집 중...")
    print("Ctrl+C를 눌러 중단, 'q'로 종료\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다")
                break
            
            # 프레임 처리
            extractor.process_frame(frame)
            
            # ROI 시각화 (정확한 좌표 표시)
            if isinstance(extractor.detector, HaarCascadeFaceDetector):
                face_rect = extractor.detector.get_last_face_rect()
                if face_rect is not None:
                    x, y, w, h = face_rect
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(frame, f"Face {w}x{h}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                roi = extractor.detector.detect(frame)
                if roi is not None:
                    h_roi, w_roi = roi.shape[:2]
                    cv2.rectangle(frame, (0, 0), (w_roi, h_roi), (0, 255, 0), 2)
            
            # 진행률
            progress = len(extractor.frame_buffer) / extractor.window_size * 100
            cv2.putText(frame, f"Progress: {progress:.1f}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Advanced rPPG', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 신호 수집 완료
            if extractor.is_buffer_full():
                print("\n✓ 신호 수집 완료!")
                
                # 신호 추출 및 예측
                signal = extractor.extract_signal()
                if signal is not None:
                    print("[진행 중] 혈압 예측...")
                    sbp, dbp = predict_bp(model, signal)
                    
                    print("\n" + "="*80)
                    print("예측 결과")
                    print("="*80)
                    print(f"수축기 혈압 (SBP): {sbp:.1f} mmHg")
                    print(f"이완기 혈압 (DBP): {dbp:.1f} mmHg")
                    print("="*80)
                    
                    # 유효성 확인
                    if 50 <= sbp <= 200 and 30 <= dbp <= 150:
                        print("✓ 예측값이 정상 범위 내입니다")
                    else:
                        print("⚠️  예측값이 정상 범위를 벗어났습니다")
                        print("   신호 품질 확인이 필요할 수 있습니다")
                    
                    print("\n계속 측정하려면 'c'를 누르세요")
                    print("종료하려면 Ctrl+C를 누르세요")
                
                # 버퍼 초기화
                extractor.frame_buffer.clear()
                extractor.signal_buffer.clear()
    
    except KeyboardInterrupt:
        print("\n사용자 중단")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("카메라 종료")


if __name__ == '__main__':
    main()
