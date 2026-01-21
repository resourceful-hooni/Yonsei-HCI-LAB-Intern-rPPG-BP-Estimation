"""
debug_face_detection.py - ?¼êµ´?¸ì‹ ?”ë²„ê·??¤í¬ë¦½íŠ¸

?¼êµ´???œë?ë¡??¸ì‹?˜ëŠ”ì§€ ?¤ì‹œê°„ìœ¼ë¡??•ì¸?©ë‹ˆ??
"""

import cv2
import numpy as np
from realtime.mediapipe_face_detector import HaarCascadeFaceDetector

def main():
    print("=" * 80)
    print("?¼êµ´?¸ì‹ ?”ë²„ê·?- ì¹´ë©”???ŒìŠ¤??)
    print("=" * 80)
    
    # ì¹´ë©”??1ë²??´ê¸°
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("??ì¹´ë©”??1ë²ˆì„ ?????†ìŠµ?ˆë‹¤.")
        return
    
    print("??ì¹´ë©”??1ë²?ì¤€ë¹??„ë£Œ")
    
    # ?¼êµ´ ê°ì?ê¸?ì´ˆê¸°??
    detector = HaarCascadeFaceDetector(min_neighbors=8)
    
    frame_count = 0
    detect_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("???„ë ˆ???½ê¸° ?¤íŒ¨")
            break
        
        frame_count += 1
        
        # ?¼êµ´ ê°ì?
        roi = detector.detect(frame)
        
        if roi is not None:
            detect_count += 1
            print(f"Frame {frame_count}: ???¼êµ´ ê°ì???- ROI ?¬ê¸°: {roi.shape}")
            
            # ?”ë©´???œì‹œ
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
            print(f"Frame {frame_count}: ???¼êµ´ ë¯¸ê°ì§€")
        
        # ?”ë©´???œì‹œ
        cv2.imshow("Face Detection Debug", frame)
        cv2.putText(frame, f"Detect: {detect_count}/{frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ì¢…ë£Œ: q ??
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("\n?¬ìš©??ì¤‘ë‹¨")
            break
        elif key == ord('c'):
            print("\n?¤ì • ë³€ê²?ëª¨ë“œ")
            neighbors = int(input("minNeighbors ê°?(ê¸°ë³¸ê°?8): ") or "8")
            detector.min_neighbors = neighbors
            print(f"??minNeighbors ë³€ê²? {neighbors}")
    
    print(f"\nì´?{frame_count}ê°??„ë ˆ??ì¤?{detect_count}ê°œì—???¼êµ´ ê°ì? ({100*detect_count/frame_count:.1f}%)")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
