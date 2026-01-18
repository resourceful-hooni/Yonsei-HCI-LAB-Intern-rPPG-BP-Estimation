"""
카메라를 사용하여 rPPG (remote Photoplethysmography) 신호를 추출하고 혈압을 예측하는 스크립트

필요한 패키지:
    - opencv-python
    - tensorflow-gpu==2.4.1
    - kapre
    - scipy (필터링용)

Phase 2 개선사항:
    - POS 알고리즘 통합 가능 (pos_algorithm.py)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU 완전 비활성화
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from kapre import STFT, Magnitude, MagnitudeToDecibel
from collections import deque
import argparse
from scipy.signal import butter, filtfilt


class RPPGExtractor:
    """
    카메라로부터 rPPG (원격 PPG) 신호를 추출하는 클래스
    
    방법: Green 채널의 평균 강도 변화 추출 (간단한 rPPG 방법)
    더 정교한 방법: Wang et al. 2017의 Plane-Orthogonal-to-Skin 알고리즘
    """
    
    def __init__(self, window_size=875, fps=30, target_len=875):
        """
        Args:
            window_size: PPG 신호 윈도우 크기 (샘플 수)
            fps: 카메라 프레임 레이트 (Hz)
        """
        self.window_size = window_size
        self.fps = fps
        self.target_len = target_len  # 모델 입력 길이 (875)
        self.duration = window_size / max(fps, 1)  # 초 단위 시간
        self.signal_buffer = deque(maxlen=window_size)
        
    def extract_face_region(self, frame):
        """
        프레임에서 얼굴 영역을 추출합니다 (개선된 파라미터)
        
        Phase 1-Step 1: Haar Cascade 파라미터 최적화
        - minNeighbors: 4 → 8 (거짓 감지 감소)
        - minSize/maxSize 지정 (불필요한 감지 제거)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 개선된 파라미터
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,      # 4 → 8 (거짓 감지 감소)
            minSize=(100, 100),  # 최소 크기 지정
            maxSize=(400, 400)   # 최대 크기 지정
        )
        
        if len(faces) == 0:
            return None
        
        # 가장 큰 얼굴 선택
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return frame[y:y+h, x:x+w]
    
    def extract_skin_color(self, face_region):
        """
        얼굴 영역에서 피부색 픽셀을 추출합니다 (간단한 방법)
        
        HSV 색 공간에서 피부색 범위를 정의
        """
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # 피부색 범위 정의 (HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = face_region[mask == 255]
        
        return skin_pixels
    
    def extract_green_channel_mean(self, face_region):
        """
        얼굴 영역의 Green 채널 평균값을 추출합니다
        
        간단한 rPPG 신호 추출 방법:
        - Green 채널이 Red, Blue 채널보다 피부의 혈류 변화에 더 민감함
        """
        if face_region is None or face_region.size == 0:
            return None
        
        # BGR 형식에서 Green은 인덱스 1
        green_channel = face_region[:, :, 1]
        mean_value = np.mean(green_channel)
        
        return mean_value
    
    def process_frame(self, frame):
        """
        프레임을 처리하여 rPPG 신호 값을 추출합니다
        
        Returns:
            추출된 신호 값 (또는 None)
        """
        # 1. 얼굴 영역 추출
        face_region = self.extract_face_region(frame)
        if face_region is None:
            return None
        
        # 2. Green 채널 평균값 추출 (간단한 rPPG 신호)
        signal_value = self.extract_green_channel_mean(face_region)
        
        if signal_value is not None:
            self.signal_buffer.append(signal_value)
        
        return signal_value
    
    def get_signal(self):
        """
        현재까지 추출된 신호를 반환합니다 (Phase 1-Step 2: 밴드패스 필터 적용)
        
        Phase 1-Step 2: Butterworth 밴드패스 필터 추가
        - 심박수 범위: 0.7-4 Hz (42-240 bpm)
        - Order: 4 (기본값)
        - 필터링: scipy.signal.filtfilt (양방향 필터링)
        
        Returns:
            신호 배열 (shape: (n_samples, 1))
        """
        if len(self.signal_buffer) == 0:
            return None
        
        signal = np.array(list(self.signal_buffer), dtype=np.float32)
        
        # Phase 1-Step 2: 밴드패스 필터 적용
        # 신호가 충분히 길어야 필터링 가능
        if len(signal) > 10:
            signal = self._bandpass_filter(signal, lowcut=0.7, highcut=4.0, order=4)
        
        # 정규화 (표준편차가 0인 경우 방지)
        std = np.std(signal)
        mean = np.mean(signal)
        if std > 1e-8:
            signal = (signal - mean) / std
        else:
            signal = signal - mean
        
        # 길이가 모델 요구와 다르면 선형 보간으로 리샘플링
        if signal.shape[0] != self.target_len:
            x = np.linspace(0, 1, signal.shape[0])
            x_new = np.linspace(0, 1, self.target_len)
            signal = np.interp(x_new, x, signal)
        
        return signal.reshape(-1, 1)
    
    def _bandpass_filter(self, signal, lowcut=0.7, highcut=4.0, order=4):
        """
        Phase 1-Step 2: Butterworth 밴드패스 필터
        
        Args:
            signal: 입력 신호
            lowcut: 하한 주파수 (Hz) - 기본값 0.7 Hz
            highcut: 상한 주파수 (Hz) - 기본값 4.0 Hz
            order: 필터 차수 - 기본값 4
            
        Returns:
            필터링된 신호
        """
        # Nyquist 주파수 계산
        nyq = 0.5 * self.fps
        
        # 정규화된 주파수
        low = lowcut / nyq
        high = highcut / nyq
        
        # 범위 확인 (0 < low < high < 1)
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)
        
        if low >= high:
            # 필터 범위가 유효하지 않으면 원본 신호 반환
            return signal
        
        # Butterworth 필터 설계
        b, a = butter(order, [low, high], btype='band')
        
        # 양방향 필터링 (phase distortion 없음)
        try:
            filtered_signal = filtfilt(b, a, signal)
        except Exception:
            # 필터링 실패 시 원본 신호 반환
            filtered_signal = signal
        
        return filtered_signal
    
    def is_buffer_full(self):
        """
        신호 버퍼가 가득 찼는지 확인
        """
        return len(self.signal_buffer) == self.window_size


def load_model(model_path):
    """
    사전 학습된 모델을 로드합니다
    """
    print(f"모델 로드 중: {model_path}")
    
    dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel
    }
    
    model = ks.models.load_model(model_path, custom_objects=dependencies)
    print("모델 로드 완료!")
    
    return model


def predict_bp(model, signal):
    """
    신호로부터 혈압을 예측합니다
    
    Args:
        model: 학습된 모델
        signal: PPG 신호 (shape: (875, 1))
    
    Returns:
        (SBP, DBP) 튜플
    """
    # 배치 차원 추가
    input_data = np.expand_dims(signal, axis=0)  # (1, 875, 1)
    
    prediction = model.predict(input_data, verbose=0)
    
    # 경우 1: 단일 텐서 출력 (shape: (1, 2))
    if hasattr(prediction, 'shape') and prediction.ndim >= 2 and prediction.shape[-1] == 2:
        sbp = float(prediction[0, 0])
        dbp = float(prediction[0, 1])
        return sbp, dbp
    # 경우 2: 두 개의 출력 리스트 ([sbp_batch, dbp_batch])
    elif isinstance(prediction, (list, tuple)) and len(prediction) == 2:
        sbp_batch, dbp_batch = prediction
        sbp_val = np.squeeze(sbp_batch)
        dbp_val = np.squeeze(dbp_batch)
        # 스칼라인 경우 직접 변환, 배열이면 첫 요소
        sbp = float(sbp_val) if sbp_val.ndim == 0 else float(sbp_val[0])
        dbp = float(dbp_val) if dbp_val.ndim == 0 else float(dbp_val[0])
        return sbp, dbp
    else:
        raise ValueError(f"예상하지 못한 모델 출력 형태: type={type(prediction)}, shape={getattr(prediction, 'shape', None)}")


def main():
    parser = argparse.ArgumentParser(description='카메라로부터 rPPG 신호 추출 및 혈압 예측')
    parser.add_argument('--model', type=str, default='data/resnet_ppg_nonmixed.h5',
                        help='모델 파일 경로 (기본값: ResNet - Phase 1-Step 4)')
    parser.add_argument('--camera', type=int, default=0,
                        help='카메라 ID (기본값: 0 - 기본 카메라)')
    parser.add_argument('--backend', type=str, choices=['default','dshow','msmf'], default='default',
                        help='카메라 백엔드 선택 (Windows: dshow 또는 msmf)')
    parser.add_argument('--width', type=int, default=640, help='카메라 해상도 가로')
    parser.add_argument('--height', type=int, default=480, help='카메라 해상도 세로')
    parser.add_argument('--fps', type=int, default=30, help='요청 프레임레이트')
    parser.add_argument('--list', action='store_true', help='사용 가능한 카메라 나열 후 종료')
    parser.add_argument('--duration', type=int, default=7,
                        help='신호 수집 시간 (초, 기본값: 7)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("카메라 기반 rPPG 혈압 예측")
    print("="*80)

    # 간단한 카메라 열기 함수
    def open_camera(index, backend_name):
        if backend_name == 'dshow':
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        elif backend_name == 'msmf':
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(index)
        return cap

    # 카메라 나열 모드: 모델 로드 없이 장치 탐색
    if args.list:
        print("\n사용 가능한 카메라 탐색 중...")
        found = []
        for idx in range(0, 10):
            for backend in ['dshow','msmf','default']:
                cap = open_camera(idx, backend)
                if cap.isOpened():
                    # 프레임 읽기 테스트
                    ok, frame = cap.read()
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if ok:
                        found.append((idx, backend, int(width), int(height), int(fps) if fps and fps>0 else None))
                        print(f"- 카메라 {idx} (backend={backend}): {int(width)}x{int(height)} fps={int(fps) if fps and fps>0 else 'N/A'}")
                        break
        if not found:
            print("카메라를 찾지 못했습니다. 다른 백엔드/인덱스를 직접 지정해 시도하세요.")
        return

    # GPU 비활성화로 CUDA 관련 경고 회피 (CPU 강제 사용)
    try:
        tf.config.experimental.set_visible_devices([], 'GPU')
    except Exception:
        pass
    
    # 1. 모델 로드
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 2. rPPG 추출기 초기화는 카메라 시작 후 FPS 확인 뒤에 수행
    
    # 3. 카메라 시작
    print(f"\n카메라 {args.camera} 시작 중... (backend={args.backend})")

    # 선택된 백엔드로 시도 후, 다른 백엔드로 폴백
    def open_camera(index, backend_name):
        if backend_name == 'dshow':
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        elif backend_name == 'msmf':
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(index)
        return cap

    cap = open_camera(args.camera, args.backend)
    if not cap.isOpened():
        # 폴백 시도
        for b in ['dshow','msmf','default']:
            if b == args.backend:
                continue
            cap = open_camera(args.camera, b)
            if cap.isOpened():
                print(f"백엔드 폴백 성공: {b}")
                break
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다!")
        return
    
    # 카메라 해상도/프레임레이트 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # 카메라 FPS 읽기 (기본 30)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = float(args.fps)
    # 수집할 윈도우 크기 계산 (duration 초 * fps), 예측 입력은 875로 리샘플링
    window_size = max(int(args.duration * fps), 30)
    rppg = RPPGExtractor(window_size=window_size, fps=fps, target_len=875)

    print("카메라 준비 완료!")
    print("\n얼굴을 카메라에 정면으로 맞춰주세요.")
    print(f"{args.duration}초 동안 신호를 수집합니다 (FPS {fps:.0f}, 수집 샘플 {window_size}).")
    print("Ctrl+C를 눌러 중단할 수 있습니다.\n")
    
    frame_count = 0
    signal_collected = False
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("프레임을 읽을 수 없습니다!")
                break
            
            # 프레임 처리
            signal_value = rppg.process_frame(frame)
            
            # Phase 1-Step 3: 단일 얼굴 박스만 표시
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # 개선된 파라미터로 얼굴 감지
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,      # Step 1에서 최적화됨
                minSize=(100, 100),
                maxSize=(400, 400)
            )
            
            # Step 3: 가장 큰 얼굴만 표시 (단일 박스)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 진행 상황 표시
            buffer_ratio = len(rppg.signal_buffer) / rppg.window_size
            cv2.putText(frame, f"Progress: {buffer_ratio*100:.1f}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            if signal_value is not None:
                cv2.putText(frame, f"Signal: {signal_value:.1f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
            
            # 화면 출력
            cv2.imshow('rPPG 신호 수집', frame)
            
            frame_count += 1
            
            # 신호 수집 완료 확인
            if rppg.is_buffer_full() and not signal_collected:
                print("\n신호 수집 완료! 혈압 예측 중...")
                signal_collected = True
                
                # 혈압 예측
                signal = rppg.get_signal()
                sbp, dbp = predict_bp(model, signal)
                
                print("\n" + "="*80)
                print("예측 결과")
                print("="*80)
                print(f"수축기 혈압 (SBP): {sbp:.1f} mmHg")
                print(f"이완기 혈압 (DBP): {dbp:.1f} mmHg")
                print("="*80)
                
                # 추가 신호 수집 여부 확인
                print("\n추가 예측을 원하면 계속 버튼을 누르세요.")
                print("종료하려면 Ctrl+C를 누르세요.")
                
                # 버퍼 초기화하여 새로운 신호 수집 시작
                rppg.signal_buffer.clear()
                signal_collected = False
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n사용자 중단")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("카메라 종료")


if __name__ == '__main__':
    main()
