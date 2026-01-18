"""
mediapipe_face_detector.py - MediaPipe 기반 얼굴 감지기 (폴백: Haar Cascade)

주의: MediaPipe 0.10+는 Python 3.8 호환성 문제 있음
현재는 Haar Cascade 사용, 추후 Python 3.9+ 환경에서 MediaPipe로 교체 가능
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple, List


class MediaPipeFaceDetector:
    """
    MediaPipe 대신 Haar Cascade 사용 (Python 3.8 호환성)
    
    향후 Python 3.9+에서는 MediaPipe로 교체 가능
    """
    
    def __init__(self, min_detection_confidence: float = 0.7):
        """
        Args:
            min_detection_confidence: 감지 신뢰도 임계값 (미사용, Haar Cascade용)
        """
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.min_neighbors = 8  # Haar Cascade 파라미터
    
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        프레임에서 가장 신뢰도 높은 얼굴 영역 추출 (Haar Cascade 사용)
        
        Args:
            frame: BGR 형식의 입력 프레임
        
        Returns:
            얼굴 ROI (numpy array) 또는 None
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
        
        # 가장 큰 얼굴 선택
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
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
    print("  from mediapipe_face_detector import MediaPipeFaceDetector")
    print("  detector = MediaPipeFaceDetector()")
    print("  roi = detector.detect(frame)")
