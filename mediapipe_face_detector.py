"""
mediapipe_face_detector.py - MediaPipe 기반 얼굴 감지기

Haar Cascade보다 더 정확한 얼굴 감지를 제공합니다.
- 더 정확한 감지
- 더 빠른 처리
- 얼굴 랜드마크 지원 (추후 피부 영역 정제에 사용 가능)
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import mediapipe as mp


class MediaPipeFaceDetector:
    """
    MediaPipe Face Detection을 사용한 얼굴 감지기
    """
    
    def __init__(self, min_detection_confidence: float = 0.7):
        """
        Args:
            min_detection_confidence: 감지 신뢰도 임계값 (0-1)
                                     클수록 높은 신뢰도의 얼굴만 감지
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0: 카메라로부터 2m 이내, 1: 5m 이내
            min_detection_confidence=min_detection_confidence
        )
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
    
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        프레임에서 가장 신뢰도 높은 얼굴 영역 추출
        
        Args:
            frame: BGR 형식의 입력 프레임
        
        Returns:
            얼굴 ROI (numpy array) 또는 None
        """
        # BGR → RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        
        if not results.detections:
            return None
        
        # 가장 신뢰도 높은 얼굴 선택
        best_detection = max(results.detections, 
                            key=lambda d: d.score[0])
        
        # 상대 좌표 → 절대 좌표 변환
        h, w = frame.shape[:2]
        bbox = best_detection.location_data.relative_bounding_box
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # 경계 확인
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        # ROI 추출
        roi = frame[y:y+height, x:x+width]
        
        return roi
    
    def detect_with_landmarks(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        얼굴 영역과 랜드마크 추출 (피부 영역 정제용)
        
        Args:
            frame: BGR 형식의 입력 프레임
        
        Returns:
            (roi, landmarks) 튜플
            - roi: 얼굴 ROI
            - landmarks: (468, 2) 얼굴 랜드마크 좌표
        """
        # BGR → RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        # ROI는 Face Detection으로 추출
        roi = self.detect(frame)
        
        if not results.multi_face_landmarks or not roi is not None:
            return roi, None
        
        # 랜드마크 추출
        h, w = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        
        # (x, y) 좌표로 변환
        landmark_coords = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in landmarks],
            dtype=np.int32
        )
        
        return roi, landmark_coords
    
    def get_skin_mask_from_landmarks(self, frame: np.ndarray, 
                                     landmarks: np.ndarray) -> np.ndarray:
        """
        랜드마크로부터 피부 영역 마스크 생성
        
        MediaPipe Face Mesh는 468개의 랜드마크를 제공합니다.
        얼굴 윤곽 랜드마크를 사용하여 피부 영역을 정의합니다.
        
        Args:
            frame: 원본 프레임
            landmarks: (468, 2) 얼굴 랜드마크
        
        Returns:
            피부 영역 마스크 (0 또는 255)
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 얼굴 윤곽 랜드마크 인덱스 (대략적)
        face_contour_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        if landmarks.shape[0] < 468:
            return mask
        
        # 얼굴 윤곽 좌표
        contour_points = landmarks[face_contour_indices]
        
        # 다각형 내부 채우기
        cv2.fillPoly(mask, [contour_points], 255)
        
        return mask
    
    def process_with_roi_margin(self, frame: np.ndarray, 
                                margin: float = 0.1) -> Optional[np.ndarray]:
        """
        마진을 포함한 ROI 추출 (신호 추출 안정성 향상)
        
        Args:
            frame: BGR 프레임
            margin: 마진 비율 (0-1), 기본값 0.1 (10%)
        
        Returns:
            마진을 포함한 ROI
        """
        h, w = frame.shape[:2]
        
        # 기본 ROI 추출
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        
        if not results.detections:
            return None
        
        best_detection = max(results.detections, 
                            key=lambda d: d.score[0])
        
        bbox = best_detection.location_data.relative_bounding_box
        
        # 상대 좌표 → 절대 좌표
        x = bbox.xmin * w
        y = bbox.ymin * h
        width = bbox.width * w
        height = bbox.height * h
        
        # 마진 추가
        x = max(0, int(x - width * margin))
        y = max(0, int(y - height * margin))
        width = min(int(width * (1 + 2*margin)), w - x)
        height = min(int(height * (1 + 2*margin)), h - y)
        
        roi = frame[y:y+height, x:x+width]
        
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
    
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Haar Cascade로 얼굴 감지
        
        Args:
            frame: BGR 프레임
        
        Returns:
            얼굴 ROI 또는 None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=self.min_neighbors,
            minSize=(100, 100),
            maxSize=(400, 400)
        )
        
        if len(faces) == 0:
            return None
        
        # 가장 큰 얼굴
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        roi = frame[y:y+h, x:x+w]
        
        return roi


# 성능 비교용 테스트
if __name__ == "__main__":
    print("MediaPipe 얼굴 감지기 로드됨")
    print("\n사용 예시:")
    print("  from mediapipe_face_detector import MediaPipeFaceDetector")
    print("  detector = MediaPipeFaceDetector()")
    print("  roi = detector.detect(frame)")
