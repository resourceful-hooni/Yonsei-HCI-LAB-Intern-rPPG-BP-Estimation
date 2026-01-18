"""
debug_face_detection.py - 얼굴인식 디버그 스크립트

얼굴이 제대로 인식되는지 실시간으로 확인합니다.
"""

import cv2
import numpy as np
from mediapipe_face_detector import HaarCascadeFaceDetector

def main():
    print("=" * 80)
    print("얼굴인식 디버그 - 카메라 테스트")
    print("=" * 80)
    
    # 카메라 1번 열기
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("❌ 카메라 1번을 열 수 없습니다.")
        return
    
    print("✓ 카메라 1번 준비 완료")
    
    # 얼굴 감지기 초기화
    detector = HaarCascadeFaceDetector(min_neighbors=8)
    
    frame_count = 0
    detect_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ 프레임 읽기 실패")
            break
        
        frame_count += 1
        
        # 얼굴 감지
        roi = detector.detect(frame)
        
        if roi is not None:
            detect_count += 1
            print(f"Frame {frame_count}: ✓ 얼굴 감지됨 - ROI 크기: {roi.shape}")
            
            # 화면에 표시
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(100, 100),
                maxSize=(400, 400)
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{w}x{h}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            print(f"Frame {frame_count}: ✗ 얼굴 미감지")
        
        # 화면에 표시
        cv2.imshow("Face Detection Debug", frame)
        cv2.putText(frame, f"Detect: {detect_count}/{frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 종료: q 키
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("\n사용자 중단")
            break
        elif key == ord('c'):
            print("\n설정 변경 모드")
            neighbors = int(input("minNeighbors 값 (기본값 8): ") or "8")
            detector.min_neighbors = neighbors
            print(f"✓ minNeighbors 변경: {neighbors}")
    
    print(f"\n총 {frame_count}개 프레임 중 {detect_count}개에서 얼굴 감지 ({100*detect_count/frame_count:.1f}%)")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
