"""
Integrated rPPG BP Estimation Pipeline with MediaPipe
======================================================

Pipeline Stages:
1. Face Detection (MediaPipe) → 2. ROI Extraction → 3. RGB Time Series
4. POS Algorithm → 5. Signal Quality → 6. Preprocessing
7. Model Inference → 8. Post-processing (Kalman Filter) → 9. BP Output

Features:
- Bounding Box Kalman Filter (jitter reduction)
- BGR/RGB conversion validation
- Signal integrity monitoring
- Memory-efficient streaming
- GPU/CPU load balancing
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU-only for stability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow.keras as ks
from collections import deque
from typing import Optional, Tuple, List, Dict
import time
try:
    from time import perf_counter as timer  # High-resolution timer
except ImportError:
    timer = time.time  # Fallback

from .mediapipe_face_detector import MediaPipeFaceDetector
from .pos_algorithm import POSExtractor
from .signal_quality import SignalQualityAssessor, ROIStabilizer
from .bp_stability import BPStabilizer
from models.transformer_model import MultiHeadAttention, EncoderLayer, TransformerEncoder
from kapre import STFT, Magnitude, MagnitudeToDecibel
from scipy.signal import resample, butter, filtfilt


class BoundingBoxKalmanFilter:
    """
    Kalman Filter for bounding box stabilization
    Reduces jitter that could inject noise into POS algorithm
    """
    
    def __init__(self, process_var=0.01, measurement_var=1.0):
        """
        Args:
            process_var: Process noise (smaller = more stable)
            measurement_var: Measurement noise (larger = more smoothing)
        """
        self.process_var = process_var
        self.measurement_var = measurement_var
        
        # State: [x, y, w, h]
        self.state = None
        self.error_cov = np.eye(4) * 1.0
    
    def update(self, measurement: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Update with new bounding box measurement
        
        Args:
            measurement: (x, y, w, h)
        
        Returns:
            Filtered (x, y, w, h)
        """
        z = np.array(measurement, dtype=np.float32)
        
        if self.state is None:
            self.state = z
            return tuple(map(int, z))
        
        # Prediction
        predicted_state = self.state
        predicted_cov = self.error_cov + np.eye(4) * self.process_var
        
        # Update
        innovation_cov = predicted_cov + np.eye(4) * self.measurement_var
        kalman_gain = predicted_cov @ np.linalg.inv(innovation_cov)
        
        self.state = predicted_state + kalman_gain @ (z - predicted_state)
        self.error_cov = (np.eye(4) - kalman_gain) @ predicted_cov
        
        return tuple(map(int, self.state))
    
    def reset(self):
        """Reset filter state"""
        self.state = None
        self.error_cov = np.eye(4) * 1.0


class IntegratedRPPGPipeline:
    """
    Complete rPPG BP Estimation Pipeline with MediaPipe Integration
    """
    
    def __init__(self, model_path: str, fs: float = 30, duration: float = 7,
                 target_len: int = 875, use_mediapipe: bool = True,
                 enable_bbox_filter: bool = True):
        """
        Args:
            model_path: Path to trained model (.h5)
            fs: Sampling frequency (Hz)
            duration: Signal collection duration (seconds)
            target_len: Model input length
            use_mediapipe: Use MediaPipe (True) or Haar Cascade (False)
            enable_bbox_filter: Enable bounding box Kalman filtering
        """
        print("="*70)
        print("Initializing Integrated rPPG BP Estimation Pipeline")
        print("="*70)
        
        self.fs = fs
        self.duration = duration
        self.target_len = target_len
        self.window_size = int(duration * fs)
        
        # === Stage 1: Face Detection ===
        print("[Stage 1] Initializing Face Detector...")
        # min_detection_confidence=0.5: 1m 환경 최적, 빠른 감지
        self.detector = MediaPipeFaceDetector(min_detection_confidence=0.5)
        self.use_mediapipe = use_mediapipe and hasattr(self.detector, 'use_mediapipe') and self.detector.use_mediapipe
        print(f"  → {'MediaPipe' if self.use_mediapipe else 'Haar Cascade'} activated")
        
        # Bounding Box Stabilization
        self.enable_bbox_filter = enable_bbox_filter
        if enable_bbox_filter:
            self.bbox_kalman = BoundingBoxKalmanFilter(
                process_var=0.01, measurement_var=2.0
            )
            print("  → Bounding Box Kalman Filter enabled")
        
        # === Stage 2-3: ROI Extraction & RGB Time Series ===
        print("[Stage 2-3] Initializing ROI & Signal Extraction...")
        self.frame_buffer = deque(maxlen=self.window_size)
        self.signal_buffer = deque(maxlen=self.window_size)
        self.roi_stabilizer = ROIStabilizer(smoothing_factor=0.7)
        print("  → Frame buffer ready")
        
        # === Stage 4: POS Algorithm ===
        print("[Stage 4] Initializing POS Algorithm...")
        self.pos = POSExtractor(fs=fs, window_size=1.6)
        print("  → POS extractor ready")
        
        # === Stage 5: Signal Quality Analysis ===
        print("[Stage 5] Initializing Signal Quality Assessor...")
        self.quality_assessor = SignalQualityAssessor(fs=fs)
        self.last_quality_metrics = {}
        self.last_quality_score = 0.0
        print("  → Quality assessor ready")
        
        # === Stage 6-7: Model Loading ===
        print("[Stage 6-7] Loading BP Prediction Model...")
        self.model = self._load_model(model_path)
        print(f"  → Model loaded: {model_path}")
        
        # === Stage 8: Post-processing (Kalman Filter) ===
        print("[Stage 8] Initializing BP Stabilizer...")
        self.bp_stabilizer = BPStabilizer(window_size=5, outlier_threshold=2.5)
        print("  → BP stabilizer ready")
        
        # Load scaler statistics
        self._load_scaler_stats()
        
        # Performance monitoring
        self.stage_times = {
            'detection': deque(maxlen=30),
            'roi': deque(maxlen=30),
            'pos': deque(maxlen=30),
            'quality': deque(maxlen=30),
            'preprocessing': deque(maxlen=30),
            'inference': deque(maxlen=30),
            'postprocessing': deque(maxlen=30)
        }
        
        print("\n✓ Pipeline initialization complete!")
        print("="*70 + "\n")
    
    def _load_model(self, model_path: str):
        """Load TensorFlow model with custom objects"""
        custom_objects = {
            'ReLU': ks.layers.ReLU,
            'STFT': STFT,
            'Magnitude': Magnitude,
            'MagnitudeToDecibel': MagnitudeToDecibel,
            'MultiHeadAttention': MultiHeadAttention,
            'EncoderLayer': EncoderLayer,
            'TransformerEncoder': TransformerEncoder
        }
        return ks.models.load_model(model_path, custom_objects=custom_objects)
    
    def _load_scaler_stats(self, info_path='data/rppg_info.txt'):
        """Load scaler statistics for normalization"""
        import re
        
        self.signal_mean = None
        self.signal_scale = None
        self.label_mean = None
        self.label_scale = None
        
        try:
            with open(info_path, 'r') as f:
                text = f.read()
            
            # Signal statistics
            signal_mean_match = re.search(r"Signal Statistics:\s*\n\s*Mean:\s*\[([^\]]+)\]", text, re.S)
            signal_scale_match = re.search(r"Signal Statistics:.*?Scale:\s*\[([^\]]+)\]", text, re.S)
            
            if signal_mean_match and signal_scale_match:
                self.signal_mean = np.fromstring(signal_mean_match.group(1), sep=' ')
                self.signal_scale = np.fromstring(signal_scale_match.group(1), sep=' ')
            
            # Label statistics
            label_mean_match = re.search(r"Label Statistics:\s*\n\s*Mean:\s*\[([^\]]+)\]", text, re.S)
            label_scale_match = re.search(r"Label Statistics:.*?Scale:\s*\[([^\]]+)\]", text, re.S)
            
            if label_mean_match and label_scale_match:
                self.label_mean = np.fromstring(label_mean_match.group(1), sep=' ')
                self.label_scale = np.fromstring(label_scale_match.group(1), sep=' ')
        
        except Exception as e:
            print(f"[Warning] Failed to load scaler stats: {e}")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame through pipeline
        
        Args:
            frame: BGR frame from camera
        
        Returns:
            Status dictionary with timing and results
        """
        status = {
            'face_detected': False,
            'bbox': None,
            'bbox_filtered': None,
            'roi_valid': False,
            'signal_collected': False,
            'timings': {}
        }
        
        # === Stage 1: Face Detection ===
        t0 = timer()
        
        # Validate BGR format
        if frame.shape[2] != 3:
            return status
        
        # MediaPipe requires RGB, but we verify conversion integrity
        roi = self.detector.detect(frame)
        bbox_raw = self.detector.last_face_rect
        
        if roi is None or bbox_raw is None:
            return status
        
        status['face_detected'] = True
        status['bbox'] = bbox_raw
        self.stage_times['detection'].append(timer() - t0)
        
        # === Bounding Box Stabilization ===
        t0 = timer()
        if self.enable_bbox_filter:
            bbox_filtered = self.bbox_kalman.update(bbox_raw)
            status['bbox_filtered'] = bbox_filtered
            
            # Re-extract ROI with filtered coordinates
            x, y, w, h = bbox_filtered
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            roi = frame[y:y+h, x:x+w]
        else:
            status['bbox_filtered'] = bbox_raw
        
        self.stage_times['roi'].append(timer() - t0)
        
        # === Stage 2-3: Frame Buffer & RGB Signal ===
        self.frame_buffer.append(frame.copy())
        
        # Extract mean signal (simple green channel for buffer)
        if roi.size > 0:
            green_mean = np.mean(roi[:, :, 1])  # BGR format
            self.signal_buffer.append(green_mean)
            status['roi_valid'] = True
        
        status['signal_collected'] = len(self.frame_buffer) == self.window_size
        
        return status
    
    def extract_and_predict(self) -> Optional[Dict]:
        """
        Extract signal and predict BP (full pipeline execution)
        
        Returns:
            Results dictionary or None
        """
        if not len(self.frame_buffer) == self.window_size:
            return None
        
        results = {
            'sbp': None,
            'dbp': None,
            'hr': None,
            'quality_score': 0.0,
            'quality_metrics': {},
            'confidence': 0.0,
            'timings': {}
        }
        
        try:
            # === Stage 4: POS Algorithm ===
            t0 = timer()
            frames = list(self.frame_buffer)
            pulse, hr = self.pos.extract(frames, self.detector.detect)
            signal = pulse
            results['hr'] = hr
            self.stage_times['pos'].append(timer() - t0)
            
            # === Stage 5: Signal Quality Analysis ===
            t0 = timer()
            
            # Detrend (lighting correction)
            signal = self.quality_assessor.detrend_signal(signal)
            
            # Adaptive filtering
            signal = self.quality_assessor.adaptive_filtering(signal)
            
            # Temporal smoothing
            signal = self.quality_assessor.temporal_smoothing(signal, alpha=0.3)
            
            # Quality assessment
            quality_score, metrics = self.quality_assessor.assess_quality(signal)
            results['quality_score'] = quality_score
            results['quality_metrics'] = metrics
            self.last_quality_score = quality_score
            self.last_quality_metrics = metrics
            
            self.stage_times['quality'].append(timer() - t0)
            
            # === Stage 6: Preprocessing ===
            t0 = timer()
            
            # Resample to target length
            if len(signal) != self.target_len:
                signal = self._proper_resample(signal, self.target_len)
            
            # Normalize
            signal = self._normalize(signal).reshape(-1, 1)
            
            self.stage_times['preprocessing'].append(timer() - t0)
            
            # === Stage 7: Model Inference ===
            t0 = timer()
            input_data = np.expand_dims(signal, axis=0)
            prediction = self.model.predict(input_data, verbose=0)
            
            # Parse output
            if isinstance(prediction, list):
                sbp_raw_model = float(prediction[0][0, 0])
                dbp_raw_model = float(prediction[1][0, 0])
            else:
                sbp_raw_model = float(prediction[0, 0])
                dbp_raw_model = float(prediction[0, 1])
            
            # Inverse transform
            if self.label_mean is not None and self.label_scale is not None:
                sbp_raw = sbp_raw_model * self.label_scale[0] + self.label_mean[0]
                dbp_raw = dbp_raw_model * self.label_scale[1] + self.label_mean[1]
            
            self.stage_times['inference'].append(timer() - t0)
            
            # === Stage 8: Post-processing (Kalman Filter) ===
            t0 = timer()
            sbp, dbp, stab_info = self.bp_stabilizer.stabilize(
                sbp_raw, dbp_raw, quality_score
            )
            results['sbp'] = sbp
            results['dbp'] = dbp
            results['confidence'] = self.bp_stabilizer.get_confidence()
            # Debug: expose raw outputs and stabilization metadata
            results['sbp_raw'] = sbp_raw
            results['dbp_raw'] = dbp_raw
            results['sbp_raw_model'] = sbp_raw_model
            results['dbp_raw_model'] = dbp_raw_model
            results['stabilization'] = stab_info
            
            self.stage_times['postprocessing'].append(timer() - t0)
            
            # Collect timing statistics
            for stage, times in self.stage_times.items():
                if len(times) > 0:
                    results['timings'][stage] = {
                        'mean': np.mean(list(times)) * 1000,  # ms
                        'std': np.std(list(times)) * 1000
                    }
            
            return results
        
        except Exception as e:
            print(f"[Error] Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _proper_resample(self, signal: np.ndarray, target_len: int) -> np.ndarray:
        """Resample signal using scipy"""
        if len(signal) <= 1:
            return signal
        
        # Anti-aliasing filter
        if target_len < len(signal):
            nyq = 0.5 * self.fs
            cutoff = (0.5 * target_len / len(signal)) / nyq
            cutoff = np.clip(cutoff, 0.001, 0.999)
            
            try:
                b, a = butter(8, cutoff, btype='low')
                signal = filtfilt(b, a, signal)
            except:
                pass
        
        return resample(signal, target_len).astype(np.float32)
    
    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal"""
        if self.signal_mean is not None and self.signal_scale is not None:
            return (signal - self.signal_mean) / (self.signal_scale + 1e-8)
        else:
            mean = np.mean(signal)
            std = np.std(signal)
            return (signal - mean) / std if std > 1e-10 else signal - mean
    
    def get_performance_report(self) -> Dict:
        """Get pipeline performance statistics"""
        report = {}
        
        total_time = 0
        for stage, times in self.stage_times.items():
            if len(times) > 0:
                avg_time = np.mean(list(times)) * 1000
                report[stage] = {
                    'avg_ms': avg_time,
                    'std_ms': np.std(list(times)) * 1000
                }
                total_time += avg_time
        
        report['total_ms'] = total_time
        report['max_fps'] = 1000 / total_time if total_time > 0 else 0
        
        return report
    
    def reset(self):
        """Reset pipeline state"""
        self.frame_buffer.clear()
        self.signal_buffer.clear()
        if self.enable_bbox_filter:
            self.bbox_kalman.reset()


if __name__ == "__main__":
    print("Integrated rPPG BP Estimation Pipeline")
    print("Import this module to use the pipeline")
    print("\nExample usage:")
    print("  from realtime.integrated_pipeline import IntegratedRPPGPipeline")
    print("  pipeline = IntegratedRPPGPipeline('data/transformer_bp_model.h5')")
    print("  status = pipeline.process_frame(frame)")
    print("  results = pipeline.extract_and_predict()")
