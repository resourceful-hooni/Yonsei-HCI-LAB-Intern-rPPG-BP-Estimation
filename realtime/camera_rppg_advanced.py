"""
camera_rppg_advanced.py - Phase 2 개선: 신호 품질 향상 + POS + ROI 안정화

이 스크립트는 다음을 포함합니다:
1. MediaPipe 얼굴 감지 (Haar Cascade 대체)
2. POS 알고리즘 (Green 채널 대체)
3. 적절한 리샘플링 (scipy.signal.resample)
4. 신호 품질 평가 및 개선
5. ROI 안정화
6. 조명 변화 보정
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
import re
from collections import deque
from typing import Optional, Tuple

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ onnxruntime 미설치: ONNX 모델 사용 불가")

from .pos_algorithm import POSExtractor
from .mediapipe_face_detector import MediaPipeFaceDetector, HaarCascadeFaceDetector
from .signal_quality import SignalQualityAssessor, ROIStabilizer
from .bp_stability import BPStabilizer
from models.transformer_model import MultiHeadAttention, EncoderLayer, TransformerEncoder


class AdvancedRPPGExtractor:
    """
    Phase 2 개선: POS 알고리즘 + 신호 품질 향상 + ROI 안정화
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
        self.signal_mean = None
        self.signal_scale = None
        self.label_mean = None
        self.label_scale = None
        
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
        
        # 신호 품질 평가기 및 ROI 안정화
        self.quality_assessor = SignalQualityAssessor(fs=fs)
        self.roi_stabilizer = ROIStabilizer(smoothing_factor=0.7)
        
        # 품질 메트릭 저장
        self.last_quality_score = 0
        self.last_quality_metrics = {}
        
        # 학습 시 스케일러 통계 로드
        self.load_scaler_stats()
    
    def load_scaler_stats(self, info_path='data/rppg_info.txt'):
        """학습 시 StandardScaler 통계 로드"""
        try:
            with open(info_path, 'r') as f:
                text = f.read()
            signal_mean_match = re.search(r"Signal Statistics:\s*\n\s*Mean:\s*\[([^\]]+)\]", text, re.S)
            signal_scale_match = re.search(r"Signal Statistics:.*?Scale:\s*\[([^\]]+)\]", text, re.S)
            label_mean_match = re.search(r"Label Statistics:\s*\n\s*Mean:\s*\[([^\]]+)\]", text, re.S)
            label_scale_match = re.search(r"Label Statistics:.*?Scale:\s*\[([^\]]+)\]", text, re.S)

            if signal_mean_match and signal_scale_match:
                self.signal_mean = np.fromstring(signal_mean_match.group(1), sep=' ')
                self.signal_scale = np.fromstring(signal_scale_match.group(1), sep=' ')
                if len(self.signal_mean) != self.target_len or len(self.signal_scale) != self.target_len:
                    print("[경고] 신호 스케일러 길이 불일치; 우회로 z-score 사용")
                    self.signal_mean = None
                    self.signal_scale = None

            if label_mean_match and label_scale_match:
                self.label_mean = np.fromstring(label_mean_match.group(1), sep=' ')
                self.label_scale = np.fromstring(label_scale_match.group(1), sep=' ')
            print("[OK] 스케일러 통계 로드 완료")
        except FileNotFoundError:
            print(f"[경고] 스케일러 파일 없음 ({info_path}); 우회로 z-score 및 원본 출력 사용")
        except Exception as exc:
            print(f"[경고] 스케일러 로드 실패: {exc}; 우회로 z-score 및 원본 출력 사용")
    
    def process_frame(self, frame: np.ndarray) -> Optional[float]:
        """
        프레임 처리 및 신호 값 추출
        
        Args:
            frame: BGR 프레임
        
        Returns:
            추출된 신호 값 또는 None
        """
        self.frame_buffer.append(frame.copy())
        
        # ROI 추출 with stabilization
        if isinstance(self.detector, HaarCascadeFaceDetector):
            roi = self.detector.detect(frame)
            # ROI 좌표 안정화
            face_rect = self.detector.get_last_face_rect()
            if face_rect is not None:
                stabilized_rect = self.roi_stabilizer.stabilize(face_rect)
                if stabilized_rect is not None:
                    x, y, w, h = stabilized_rect
                    roi = frame[y:y+h, x:x+w]
                    # 안정화된 좌표를 detector에 저장
                    self.detector.last_face_rect = stabilized_rect
        else:
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
        버퍼에서 최종 신호 추출 (품질 개선 포함)
        
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
        
        # === 신호 품질 개선 단계 ===
        
        # 1. 트렌드 제거 (조명 변화 보정)
        signal = self.quality_assessor.detrend_signal(signal)
        print("✓ 조명 변화 보정 완료")
        
        # 2. 움직임 아티팩트 감지
        motion_mask = self.quality_assessor.detect_motion_artifacts(signal)
        artifact_ratio = 1 - np.mean(motion_mask)
        if artifact_ratio > 0.3:
            print(f"⚠️  움직임 아티팩트 감지됨: {artifact_ratio*100:.1f}%")
        
        # 3. 적응형 필터링
        signal = self.quality_assessor.adaptive_filtering(signal)
        print("✓ 적응형 필터링 완료")
        
        # 4. 시간적 평활화
        signal = self.quality_assessor.temporal_smoothing(signal, alpha=0.3)
        print("✓ 시간적 평활화 완료")
        
        # 5. 신호 품질 평가
        quality_score, metrics = self.quality_assessor.assess_quality(signal)
        self.last_quality_score = quality_score
        self.last_quality_metrics = metrics
        
        print(f"✓ 신호 품질 점수: {quality_score:.2f}/1.00")
        print(f"  - SNR: {metrics['snr']:.2f} dB")
        print(f"  - 피크 수: {metrics['num_peaks']}")
        print(f"  - 피크 일관성: {metrics['peak_regularity']:.2f}")
        
        # 품질이 너무 낮으면 경고
        if quality_score < 0.3:
            print("⚠️  신호 품질이 낮습니다. 조명을 확인하거나 움직임을 줄이세요.")
        
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
        """신호 정규화 (학습 통계 또는 우회로 z-score)"""
        if self.signal_mean is not None and self.signal_scale is not None:
            return (signal - self.signal_mean) / (self.signal_scale + 1e-8)
        else:
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
        'MagnitudeToDecibel': MagnitudeToDecibel,
        'MultiHeadAttention': MultiHeadAttention,
        'EncoderLayer': EncoderLayer,
        'TransformerEncoder': TransformerEncoder
    }
    
    model = ks.models.load_model(model_path, custom_objects=dependencies)
    print("✓ 모델 로드 완료")
    
    return model


def predict_bp(model, signal: np.ndarray, label_mean=None, label_scale=None) -> Tuple[float, float]:
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
    
    # 레이블 역변환 (정규화된 값 → mmHg)
    if label_mean is not None and label_scale is not None:
        if label_mean.shape[0] >= 2 and label_scale.shape[0] >= 2:
            sbp = sbp * label_scale[0] + label_mean[0]
            dbp = dbp * label_scale[1] + label_mean[1]
    
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
    
    # 혈압 안정화 초기화
    bp_stabilizer = BPStabilizer(window_size=5, outlier_threshold=2.5)
    
    print(f"\n{args.duration}초 동안 신호 수집 중...")
    print("Ctrl+C를 눌러 중단, 'q'로 종료\n")
    
    # 상태 정보 저장용
    last_sbp, last_dbp, last_hr = None, None, None
    last_confidence = 0
    frame_count = 0
    import time
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다")
                break
            
            frame_count += 1
            current_fps = frame_count / (time.time() - start_time)
            
            # 프레임 처리
            extractor.process_frame(frame)
            
            # 화면 정보 표시 영역 생성 (오른쪽 패널)
            h, w = frame.shape[:2]
            info_panel = np.zeros((h, 300, 3), dtype=np.uint8)
            
            # ROI 시각화 (정확한 좌표 표시)
            if isinstance(extractor.detector, HaarCascadeFaceDetector):
                face_rect = extractor.detector.get_last_face_rect()
                if face_rect is not None:
                    x, y, w_face, h_face = face_rect
                    cv2.rectangle(frame, (x, y), (x+w_face, y+h_face), (0, 255, 0), 3)
                    cv2.putText(frame, f"Face {w_face}x{h_face}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # MediaPipe도 내부적으로 Haar Cascade 사용, last_face_rect 확인
                if hasattr(extractor.detector, 'last_face_rect') and extractor.detector.last_face_rect is not None:
                    x, y, w_face, h_face = extractor.detector.last_face_rect
                    cv2.rectangle(frame, (x, y), (x+w_face, y+h_face), (0, 255, 0), 3)
                    cv2.putText(frame, f"Face {w_face}x{h_face}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 진행률
            progress = len(extractor.frame_buffer) / extractor.window_size * 100
            cv2.putText(frame, f"Progress: {progress:.1f}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # === 정보 패널 그리기 ===
            y_offset = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            color = (255, 255, 255)
            
            # 제목
            cv2.putText(info_panel, "rPPG BP Monitor", (10, y_offset), 
                       font, 0.8, (0, 255, 255), 2)
            y_offset += 40
            
            # FPS
            cv2.putText(info_panel, f"FPS: {current_fps:.1f}", (10, y_offset), 
                       font, font_scale, color, thickness)
            y_offset += 30
            
            # 프레임 수
            cv2.putText(info_panel, f"Frames: {frame_count}", (10, y_offset), 
                       font, font_scale, color, thickness)
            y_offset += 30
            
            # 진행률 바
            cv2.putText(info_panel, f"Buffer: {progress:.0f}%", (10, y_offset), 
                       font, font_scale, color, thickness)
            y_offset += 25
            cv2.rectangle(info_panel, (10, y_offset), (290, y_offset+20), (100, 100, 100), -1)
            cv2.rectangle(info_panel, (10, y_offset), (int(10 + 280 * progress/100), y_offset+20), 
                         (0, 255, 0), -1)
            y_offset += 40
            
            # 구분선
            cv2.line(info_panel, (10, y_offset), (290, y_offset), (100, 100, 100), 1)
            y_offset += 30
            
            # 혈압 정보
            if last_sbp is not None:
                cv2.putText(info_panel, "Blood Pressure:", (10, y_offset), 
                           font, 0.7, (0, 255, 255), 2)
                y_offset += 35
                
                sbp_color = (0, 255, 0) if 90 <= last_sbp <= 140 else (0, 165, 255)
                cv2.putText(info_panel, f"SBP: {last_sbp:.1f} mmHg", (10, y_offset), 
                           font, font_scale, sbp_color, thickness)
                y_offset += 30
                
                dbp_color = (0, 255, 0) if 60 <= last_dbp <= 90 else (0, 165, 255)
                cv2.putText(info_panel, f"DBP: {last_dbp:.1f} mmHg", (10, y_offset), 
                           font, font_scale, dbp_color, thickness)
                y_offset += 30
                
                # 신뢰도 표시
                conf_color = (0, 255, 0) if last_confidence >= 0.7 else (0, 165, 255) if last_confidence >= 0.4 else (0, 0, 255)
                cv2.putText(info_panel, f"Confidence: {last_confidence:.2f}", (10, y_offset), 
                           font, 0.5, conf_color, thickness)
                y_offset += 30
            else:
                cv2.putText(info_panel, "BP: Waiting...", (10, y_offset), 
                           font, font_scale, (128, 128, 128), thickness)
                y_offset += 40
            
            # 심박수
            if last_hr is not None:
                hr_color = (0, 255, 0) if 60 <= last_hr <= 100 else (0, 165, 255)
                cv2.putText(info_panel, f"Heart Rate: {last_hr:.1f} bpm", (10, y_offset), 
                           font, 0.7, hr_color, 2)
                y_offset += 40
            else:
                cv2.putText(info_panel, "HR: Waiting...", (10, y_offset), 
                           font, font_scale, (128, 128, 128), thickness)
                y_offset += 40
            
            # 신호 품질 정보
            if hasattr(extractor, 'last_quality_score') and extractor.last_quality_score > 0:
                cv2.line(info_panel, (10, y_offset), (290, y_offset), (100, 100, 100), 1)
                y_offset += 25
                
                cv2.putText(info_panel, "Signal Quality:", (10, y_offset), 
                           font, 0.7, (0, 255, 255), 2)
                y_offset += 30
                
                # 품질 점수
                quality_score = extractor.last_quality_score
                quality_color = (0, 255, 0) if quality_score >= 0.7 else (0, 165, 255) if quality_score >= 0.4 else (0, 0, 255)
                cv2.putText(info_panel, f"Score: {quality_score:.2f}/1.00", (10, y_offset), 
                           font, font_scale, quality_color, thickness)
                y_offset += 25
                
                # 품질 바
                cv2.rectangle(info_panel, (10, y_offset), (290, y_offset+15), (100, 100, 100), -1)
                cv2.rectangle(info_panel, (10, y_offset), (int(10 + 280 * quality_score), y_offset+15), 
                             quality_color, -1)
                y_offset += 25
                
                # SNR
                if 'snr' in extractor.last_quality_metrics:
                    snr = extractor.last_quality_metrics['snr']
                    cv2.putText(info_panel, f"SNR: {snr:.1f} dB", (10, y_offset), 
                               font, 0.5, color, thickness)
                    y_offset += 20
                
                # 피크 수
                if 'num_peaks' in extractor.last_quality_metrics:
                    num_peaks = extractor.last_quality_metrics['num_peaks']
                    cv2.putText(info_panel, f"Peaks: {num_peaks}", (10, y_offset), 
                               font, 0.5, color, thickness)
                    y_offset += 25
            
            y_offset += 10
            
            # 구분선
            cv2.line(info_panel, (10, y_offset), (290, y_offset), (100, 100, 100), 1)
            y_offset += 30
            
            # 신호 그래프
            if len(extractor.signal_buffer) > 1:
                cv2.putText(info_panel, "PPG Signal:", (10, y_offset), 
                           font, font_scale, color, thickness)
                y_offset += 25
                
                # 신호 정규화 및 그래프 그리기
                signal_data = np.array(list(extractor.signal_buffer))
                signal_norm = (signal_data - signal_data.min()) / (signal_data.max() - signal_data.min() + 1e-10)
                
                graph_height = 100
                graph_width = 280
                graph_x = 10
                graph_y = y_offset
                
                # 그래프 배경
                cv2.rectangle(info_panel, (graph_x, graph_y), 
                             (graph_x + graph_width, graph_y + graph_height), 
                             (50, 50, 50), -1)
                
                # 신호 그리기
                points = []
                for i, val in enumerate(signal_norm):
                    x = int(graph_x + (i / len(signal_norm)) * graph_width)
                    y = int(graph_y + graph_height - val * graph_height)
                    points.append((x, y))
                
                for i in range(len(points) - 1):
                    cv2.line(info_panel, points[i], points[i+1], (0, 255, 0), 2)
                
                y_offset += graph_height + 20
            
            # 프레임과 정보 패널 결합
            combined = np.hstack([frame, info_panel])
            
            cv2.imshow('Advanced rPPG', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 신호 수집 완료
            if extractor.is_buffer_full():
                print("\n✓ 신호 수집 완료!")
                
                # 신호 추출 및 예측
                signal = extractor.extract_signal()
                if signal is not None:
                    print("[진행 중] 혈압 예측...")
                    sbp_raw, dbp_raw = predict_bp(model, signal, extractor.label_mean, extractor.label_scale)
                    
                    # POS 알고리즘에서 심박수 추출
                    if hasattr(extractor, 'pos') and extractor.use_pos:
                        try:
                            frames = list(extractor.frame_buffer)
                            _, hr = extractor.pos.extract(frames, extractor.detector.detect)
                            last_hr = hr
                        except:
                            pass
                    
                    # === 혈압 안정화 적용 ===
                    quality_score = extractor.last_quality_score if hasattr(extractor, 'last_quality_score') else 0.5
                    sbp, dbp, stab_info = bp_stabilizer.stabilize(sbp_raw, dbp_raw, quality_score)
                    last_confidence = bp_stabilizer.get_confidence()
                    
                    print(f"[안정화] {sbp_raw:.1f} → {sbp:.1f} mmHg (SBP), {dbp_raw:.1f} → {dbp:.1f} mmHg (DBP)")
                    if stab_info.get('sbp_outlier') or stab_info.get('dbp_outlier'):
                        print("⚠️  이상치 감지됨 - 안정화 필터 적용")
                    print(f"✓ 신뢰도: {last_confidence:.2f}")
                    
                    last_sbp = sbp
                    last_dbp = dbp
                    
                    print("\n" + "="*80)
                    print("예측 결과")
                    print("="*80)
                    print(f"수축기 혈압 (SBP): {sbp:.1f} mmHg")
                    print(f"이완기 혈압 (DBP): {dbp:.1f} mmHg")
                    if last_hr:
                        print(f"심박수 (HR): {last_hr:.1f} bpm")
                    print(f"신뢰도: {last_confidence:.2f}/1.00")
                    print("="*80)
                    
                    # 유효성 확인
                    if 50 <= sbp <= 200 and 30 <= dbp <= 150:
                        print("✓ 예측값이 정상 범위 내입니다")
                    else:
                        print("⚠️  예측값이 정상 범위를 벗어났습니다")
                        print("   신호 품질 확인이 필요할 수 있습니다")
                    
                    # 신뢰도 기반 조언
                    if last_confidence < 0.5:
                        print("⚠️  낮은 신뢰도 - 여러 번 측정하여 평균값 사용 권장")
                    elif last_confidence >= 0.8:
                        print("✓ 높은 신뢰도 - 측정값이 안정적입니다")
                    
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
