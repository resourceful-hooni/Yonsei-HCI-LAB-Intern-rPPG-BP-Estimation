"""
test_real_time_models.py - Real-time BP prediction with trained models
Tests Domain Adaptation, Multi-Task Learning, and Transformer models
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import time
import re
from scipy.signal import butter, filtfilt, detrend
from scipy.signal import find_peaks, welch

# Import custom modules
from realtime.pos_algorithm import POSExtractor
from models.transformer_model import MultiHeadAttention, EncoderLayer, TransformerEncoder

class RealtimeBPPredictor:
    def __init__(self, model_type='transformer'):
        """
        Initialize real-time BP predictor
        
        Args:
            model_type: 'domain', 'multitask', or 'transformer'
        """
        self.model_type = model_type
        self.model = None
        self.signal_buffer = []
        self.fps = 30
        self.target_length = 875  # 7 seconds at 125 Hz
        self.signal_duration = 7  # seconds
        self.signal_mean = None
        self.signal_scale = None
        self.label_mean = None
        self.label_scale = None
        
        # POS algorithm
        self.pos_extractor = POSExtractor(fs=self.fps)
        
        # Load model
        self.load_model()
        self.load_scaler_stats()
        
        # Face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Signal processing
        self.rgb_buffer = []
        self.max_buffer_size = int(self.fps * self.signal_duration)
        self.quality_threshold = 0.5  # minimum quality score to run inference
        self.min_signal_len = 120     # minimum samples to attempt quality/prediction
        
    def load_model(self):
        """Load trained model"""
        print(f"\n[*] Loading {self.model_type} model...")
        
        if self.model_type == 'domain':
            model_path = 'models/resnet_rppg_adapted.h5'
            self.model = keras.models.load_model(model_path)
            print(f"   [OK] Domain Adaptation model loaded (62.1 MB)")
            
        elif self.model_type == 'multitask':
            model_path = 'models/multi_task_bp_model.h5'
            self.model = keras.models.load_model(model_path)
            print(f"   [OK] Multi-Task Learning model loaded (9.7 MB)")
            
        elif self.model_type == 'transformer':
            model_path = 'models/transformer_bp_model.h5'
            custom_objects = {
                'MultiHeadAttention': MultiHeadAttention,
                'EncoderLayer': EncoderLayer,
                'TransformerEncoder': TransformerEncoder
            }
            self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
            print(f"   [OK] Transformer model loaded (7.7 MB)")
        
        print(f"   Model input shape: {self.model.input_shape}")
        print(f"   Model output shape: {self.model.output_shape}")
    
    def process_frame(self, frame):
        """Extract RGB signal from face ROI"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, frame, False
        
        # Use largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract ROI (forehead region)
        roi_y = y + int(h * 0.2)
        roi_h = int(h * 0.3)
        roi = frame[roi_y:roi_y+roi_h, x:x+w]
        
        if roi.size == 0:
            return None, frame, False
        
        # Calculate mean RGB
        rgb_mean = cv2.mean(roi)[:3]  # (B, G, R)
        rgb_signal = np.array([rgb_mean[2], rgb_mean[1], rgb_mean[0]])  # Convert to RGB
        
        return rgb_signal, frame, True
    
    def extract_rppg_signal(self):
        """Extract rPPG signal using POS algorithm"""
        if len(self.rgb_buffer) < 30:  # Need at least 1 second
            return None
        
        # Convert to numpy array (N, 3) - RGB format
        rgb_array = np.array(self.rgb_buffer)
        
        # Extract POS signal
        pos_signal = self.pos_extractor.pos_algorithm(rgb_array)
        
        return pos_signal

    def load_scaler_stats(self, info_path='data/rppg_info.txt'):
        """Load training-time StandardScaler stats for signals and labels."""
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
                if len(self.signal_mean) != self.target_length or len(self.signal_scale) != self.target_length:
                    print("[WARN] Signal scaler length mismatch; falling back to per-window z-score")
                    self.signal_mean = None
                    self.signal_scale = None

            if label_mean_match and label_scale_match:
                self.label_mean = np.fromstring(label_mean_match.group(1), sep=' ')
                self.label_scale = np.fromstring(label_scale_match.group(1), sep=' ')
            print("[OK] Loaded scaler statistics")
        except FileNotFoundError:
            print(f"[WARN] Scaler info file not found at {info_path}; using window z-score and raw outputs")
        except Exception as exc:
            print(f"[WARN] Failed to load scaler stats: {exc}; using window z-score and raw outputs")
    
    def preprocess_signal(self, signal):
        """Preprocess signal for model input"""
        # Detrend
        signal = detrend(signal)
        
        # Bandpass filter (0.7-4 Hz for HR: 42-240 bpm)
        nyquist = self.fps / 2
        low = 0.7 / nyquist
        high = 4.0 / nyquist
        b, a = butter(4, [low, high], btype='band')
        signal = filtfilt(b, a, signal)
        
        # Resample to 125 Hz (875 samples for 7 seconds)
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, self.target_length)
        f = interp1d(x_old, signal, kind='cubic')
        signal_resampled = f(x_new)

        # Normalize with training scaler if available, otherwise window z-score
        if self.signal_mean is not None and self.signal_scale is not None:
            signal_normalized = (signal_resampled - self.signal_mean) / (self.signal_scale + 1e-8)
        else:
            signal_mean = np.mean(signal_resampled)
            signal_std = np.std(signal_resampled)
            if signal_std > 0:
                signal_normalized = (signal_resampled - signal_mean) / signal_std
            else:
                signal_normalized = signal_resampled

        return signal_normalized
    
    def calculate_signal_quality(self, signal):
        """Calculate signal quality score"""
        # SNR
        signal_power = np.mean(signal**2)
        noise_power = np.var(np.diff(signal))
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 0
        
        # Peak detection
        peaks, _ = find_peaks(signal, distance=self.fps*0.5)
        peak_regularity = 0
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            peak_regularity = 1 - (np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-6))
        
        # Frequency analysis
        f, psd = welch(signal, fs=self.fps, nperseg=min(256, len(signal)))
        hr_band = (f >= 0.7) & (f <= 4.0)
        hr_power_ratio = np.sum(psd[hr_band]) / (np.sum(psd) + 1e-6)
        
        # Quality score (0-1)
        quality_score = (
            0.4 * min(snr_db / 20, 1.0) +
            0.3 * max(0, peak_regularity) +
            0.3 * hr_power_ratio
        )
        
        return quality_score, snr_db, len(peaks)
    
    def predict_bp(self, signal):
        """Predict blood pressure"""
        # Reshape for model input
        signal_input = signal.reshape(1, -1, 1)
        
        # Predict
        start_time = time.time()
        prediction = self.model.predict(signal_input, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Extract SBP and DBP with robust parsing
        sbp = None
        dbp = None
        if isinstance(prediction, (list, tuple)):
            if len(prediction) >= 2:
                sbp = float(np.squeeze(prediction[0]))
                dbp = float(np.squeeze(prediction[1]))
        elif hasattr(prediction, 'shape'):
            if prediction.ndim >= 2 and prediction.shape[-1] >= 2:
                sbp = float(prediction[0, 0])
                dbp = float(prediction[0, 1])
        
        if sbp is None or dbp is None:
            raise ValueError(f"Unexpected model output type/shape: type={type(prediction)}, shape={getattr(prediction, 'shape', None)}")
        
        # Inverse-transform labels back to mmHg if scaler is available
        if self.label_mean is not None and self.label_scale is not None:
            if self.label_mean.shape[0] >= 2 and self.label_scale.shape[0] >= 2:
                sbp = sbp * self.label_scale[0] + self.label_mean[0]
                dbp = dbp * self.label_scale[1] + self.label_mean[1]

        return sbp, dbp, inference_time
    
    def run(self, camera_idx=0, duration=None):
        """Run real-time BP prediction"""
        print(f"\n{'='*60}")
        print(f"Real-time BP Prediction - {self.model_type.upper()} Model")
        print(f"{'='*60}")
        print(f"[*] Opening camera {camera_idx}...")
        
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_idx}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.pos_extractor.fs = self.fps
        self.max_buffer_size = int(self.fps * self.signal_duration)
        
        print(f"[OK] Camera opened")
        print(f"\nInstructions:")
        print(f"  - Keep your face in the frame")
        print(f"  - Stay still for best results")
        print(f"  - Wait for {self.signal_duration} seconds to collect signal")
        print(f"  - Press 'q' to quit, 'r' to reset")
        print(f"\nCollecting data...")
        
        start_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            rgb_signal, display_frame, detected = self.process_frame(frame)
            
            if detected and rgb_signal is not None:
                self.rgb_buffer.append(rgb_signal)
                
                # Keep buffer size
                if len(self.rgb_buffer) > self.max_buffer_size:
                    self.rgb_buffer.pop(0)
            else:
                cv2.putText(display_frame, "No face detected", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display info
            elapsed = time.time() - start_time
            buffer_len = len(self.rgb_buffer)
            progress = min(buffer_len / self.max_buffer_size * 100, 100)
            
            cv2.putText(display_frame, f"Model: {self.model_type.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Buffer: {buffer_len}/{self.max_buffer_size} ({progress:.1f}%)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"FPS: {frame_count / (elapsed + 1e-6):.1f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Cam FPS (actual): {self.fps:.1f}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            
            # Try prediction if buffer is full
            if buffer_len >= self.max_buffer_size:
                # Extract rPPG signal
                rppg_signal = self.extract_rppg_signal()
                
                if rppg_signal is not None and len(rppg_signal) > self.min_signal_len:
                    # Calculate quality
                    quality, snr, peaks = self.calculate_signal_quality(rppg_signal)
                    cv2.putText(display_frame, f"Quality: {quality:.2f}", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"SNR: {snr:.1f} dB", 
                               (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if quality < self.quality_threshold:
                        cv2.putText(display_frame, "Low quality - skip", (10, 210),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # Preprocess
                        signal_processed = self.preprocess_signal(rppg_signal)
                        
                        # Predict
                        sbp, dbp, inf_time = self.predict_bp(signal_processed)
                        
                        # Display results
                        cv2.putText(display_frame, f"SBP: {sbp:.1f} mmHg", 
                                   (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.putText(display_frame, f"DBP: {dbp:.1f} mmHg", 
                                   (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.putText(display_frame, f"Inference: {inf_time:.1f} ms", 
                                   (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # Print to console
                        print(f"\n{'='*60}")
                        print(f"[PREDICTION] Quality: {quality:.2f}, SNR: {snr:.1f} dB, peaks: {peaks}")
                        print(f"  SBP: {sbp:.1f} mmHg")
                        print(f"  DBP: {dbp:.1f} mmHg")
                        print(f"  Inference time: {inf_time:.1f} ms")
                        print(f"{'='*60}")
            
            cv2.imshow('Real-time BP Prediction', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.rgb_buffer = []
                print("\n[*] Buffer reset. Collecting new data...")
                start_time = time.time()
                frame_count = 0
            
            frame_count += 1
            
            # Auto-stop if duration specified
            if duration and elapsed >= duration:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n[OK] Test complete")


def main():
    parser = argparse.ArgumentParser(description='Real-time BP prediction with trained models')
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['domain', 'multitask', 'transformer'],
                       help='Model to use (default: transformer)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Test duration in seconds (default: unlimited)')
    
    args = parser.parse_args()
    
    predictor = RealtimeBPPredictor(model_type=args.model)
    predictor.run(camera_idx=args.camera, duration=args.duration)


if __name__ == '__main__':
    main()
