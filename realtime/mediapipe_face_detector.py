"""
mediapipe_face_detector.py - MediaPipe 기반 얼굴 감지기 (폴백: Haar Cascade)

MediaPipe Face Detection을 사용하여 더 정확한 얼굴 감지 제공
호환성 문제 시 자동으로 Haar Cascade로 폴백
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple, List

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class MediaPipeFaceDetector:
    """
    MediaPipe Face Detection 기반 얼굴 감지기
    
    MediaPipe 미설치 시 자동으로 Haar Cascade 사용
    """
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Args:
            min_detection_confidence: 감지 신뢰도 임계값 (0.0~1.0)
                                     0.5: 균형 (1m 환경에서 최적)
                                     0.7: 높은 정확도 (오감지 감소)
        """
        self.min_detection_confidence = min_detection_confidence
        self.last_face_rect = None  # 마지막 얼굴 좌표 저장
        self.last_confidence = 0.0  # 마지막 신뢰도 저장
        
        if MEDIAPIPE_AVAILABLE:
            try:
                mp_face_detection = mp.solutions.face_detection
                self.face_detection = mp_face_detection.FaceDetection(
                    model_selection=0,  # 0: 2m 이내 (빠름, 1m 환경 최적), 1: 5m 이내
                    min_detection_confidence=min_detection_confidence
                )
                self.use_mediapipe = True
                print("✓ MediaPipe Face Detection 활성화")
            except Exception as e:
                print(f"⚠️  MediaPipe 초기화 실패 ({e}), Haar Cascade 사용")
                self._init_haar_cascade()
                self.use_mediapipe = False
        else:
            print("⚠️  MediaPipe 미설치, Haar Cascade 사용")
            self._init_haar_cascade()
            self.use_mediapipe = False
    
    def _init_haar_cascade(self):
        """Haar Cascade 초기화 (폴백)"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.min_neighbors = 8
    
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        프레임에서 가장 신뢰도 높은 얼굴 영역 추출
        
        Args:
            frame: BGR 형식의 입력 프레임
        
        Returns:
            얼굴 ROI (numpy array) 또는 None
        """
        if self.use_mediapipe:
            return self._detect_mediapipe(frame)
        else:
            return self._detect_haar(frame)
    
    def _detect_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """MediaPipe로 얼굴 감지"""
        # BGR to RGB (MediaPipe 요구사항)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if not results.detections:
            return None
        
        # 가장 신뢰도 높은 얼굴 선택
        detection = max(results.detections, key=lambda d: d.score[0])
        
        # Bounding box 추출 (normalized coordinates → pixel values)
        h, w = frame.shape[:2]
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)
        
        # 경계 체크
        x = max(0, x)
        y = max(0, y)
        w_box = min(w_box, w - x)
        h_box = min(h_box, h - y)
        
        # Confidence score 저장
        self.last_confidence = float(detection.score[0])
        self.last_face_rect = (x, y, w_box, h_box)
        roi = frame[y:y+h_box, x:x+w_box]
        
        return roi
    
    def findFaces(self, img: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float]]]:
        """
        얼굴 감지 및 시각화 (Haar Cascade 호환 인터페이스)
        
        Args:
            img: BGR 입력 이미지
            draw: 바운딩 박스 및 신뢰도 그리기 여부
        
        Returns:
            (이미지, [(x, y, w, h, confidence), ...])
        """
        img_copy = img.copy() if draw else img
        faces = []
        
        if self.use_mediapipe:
            # MediaPipe 사용
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                h, w = img.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    w_box = min(int(bbox.width * w), w - x)
                    h_box = min(int(bbox.height * h), h - y)
                    confidence = float(detection.score[0])
                    
                    faces.append((x, y, w_box, h_box, confidence))
                    
                    if draw:
                        # 바운딩 박스 그리기
                        cv2.rectangle(img_copy, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                        
                        # 신뢰도 표시
                        label = f"Face: {confidence:.2f}"
                        cv2.putText(img_copy, label, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Haar Cascade 폴백
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            detections = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4,
                flags=cv2.CASCADE_SCALE_IMAGE, minSize=(80, 80), maxSize=(450, 450)
            )
            
            for (x, y, w_box, h_box) in detections:
                confidence = 1.0  # Haar Cascade는 confidence 제공 안함
                faces.append((x, y, w_box, h_box, confidence))
                
                if draw:
                    cv2.rectangle(img_copy, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                    cv2.putText(img_copy, "Face", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return img_copy, faces
    
    def _detect_haar(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Haar Cascade로 얼굴 감지 (폴백)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(80, 80),
            maxSize=(450, 450)
        )
        
        if len(faces) == 0:
            return None
        
        # 가장 큰 얼굴 선택
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        self.last_face_rect = (x, y, w, h)
        roi = frame[y:y+h, x:x+w]
        
        return roi
    
    def detect_with_landmarks(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        얼굴 영역 반환 (Haar Cascade는 랜드마크 미지원)
        
        Args:
            frame: BGR 형식의 입력 프레임
        
        Returns:
            (roi, None) - Haar Cascade는 랜드마크 미지원
        """
        roi = self.detect(frame)
        return roi, None
    
    def get_skin_mask_from_landmarks(self, frame: np.ndarray, 
                                     landmarks: np.ndarray) -> np.ndarray:
        """
        Haar Cascade는 랜드마크 미지원 - 기본 마스크 반환
        
        Args:
            frame: 원본 프레임
            landmarks: 미사용
        
        Returns:
            전체 마스크
        """
        h, w = frame.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return mask
    
    def process_with_roi_margin(self, frame: np.ndarray, 
                                margin: float = 0.1) -> Optional[np.ndarray]:
        """
        마진을 포함한 ROI 추출
        
        Args:
            frame: BGR 프레임
            margin: 마진 비율 (0-1), 기본값 0.1 (10%)
        
        Returns:
            마진을 포함한 ROI
        """
        h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 명암비 증가로 감지율 개선
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(80, 80),
            maxSize=(450, 450)
        )
        
        if len(faces) == 0:
            return None
        
        x, y, w_face, h_face = max(faces, key=lambda f: f[2] * f[3])
        
        # 마진 추가
        x = max(0, int(x - w_face * margin))
        y = max(0, int(y - h_face * margin))
        w_margin = min(int(w_face * (1 + 2*margin)), w - x)
        h_margin = min(int(h_face * (1 + 2*margin)), h - y)
        
        roi = frame[y:y+h_margin, x:x+w_margin]
        
        return roi


class HaarCascadeFaceDetector:
    """
    Haar Cascade 기반 얼굴 감지기 (기존 방식, 비교용)
    """
    
    def __init__(self, min_neighbors: int = 8):
        """
        Args:
            min_neighbors: 최소 이웃 수, 클수록 정확하지만 느림
        """
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.min_neighbors = min_neighbors
        self.last_face_rect = None  # 마지막 감지된 얼굴 위치
    
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Haar Cascade로 얼굴 감지
        
        Args:
            frame: BGR 프레임
        
        Returns:
            얼굴 ROI 또는 None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 명암비 증가로 감지율 개선
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(80, 80),
            maxSize=(450, 450)
        )
        
        if len(faces) == 0:
            return None
        
        # 가장 큰 얼굴
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        self.last_face_rect = (x, y, w, h)  # 좌표 저장
        roi = frame[y:y+h, x:x+w]
        
        return roi
    
    def get_last_face_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        마지막 감지된 얼굴의 좌표 반환
        
        Returns:
            (x, y, w, h) 또는 None
        """
        return self.last_face_rect


# 성능 비교용 테스트
if __name__ == "__main__":
    print("MediaPipe 얼굴 감지기 로드됨")
    print("\n사용 예시:")
    print("  from realtime.mediapipe_face_detector import MediaPipeFaceDetector")
    print("  detector = MediaPipeFaceDetector()")
    print("  roi = detector.detect(frame)")
