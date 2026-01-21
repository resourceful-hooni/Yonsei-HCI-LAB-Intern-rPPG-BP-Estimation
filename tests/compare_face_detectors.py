"""
?ºÍµ¥ Í∞êÏ? ÎπÑÍµê: MediaPipe vs Haar Cascade

?§Î≥¥???®Ï∂ï??
  'm' - MediaPipeÎ°??ÑÌôò
  'h' - Haar CascadeÎ°??ÑÌôò
  'q' - Ï¢ÖÎ£å
"""

import cv2
import time
from realtime.mediapipe_face_detector import MediaPipeFaceDetector, HaarCascadeFaceDetector

def main():
    print("="*70)
    print("?ºÍµ¥ Í∞êÏ? ÎπÑÍµê: MediaPipe vs Haar Cascade")
    print("="*70)
    print("\n?§Î≥¥???®Ï∂ï??")
    print("  [m] - MediaPipe Face Detection")
    print("  [h] - Haar Cascade Detection")
    print("  [q] - Ï¢ÖÎ£å\n")
    
    # Í∞êÏ?Í∏?Ï¥àÍ∏∞??
    mediapipe_detector = MediaPipeFaceDetector(min_detection_confidence=0.7)
    haar_detector = HaarCascadeFaceDetector(min_neighbors=4)
    
    # ?ÑÏû¨ ?¨Ïö© Ï§ëÏù∏ Í∞êÏ?Í∏?
    use_mediapipe = True
    current_detector = mediapipe_detector
    
    # Ïπ¥Î©î???¥Í∏∞
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("??Ïπ¥Î©î?ºÎ? ?????ÜÏäµ?àÎã§")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("??Ïπ¥Î©î??Ï§ÄÎπ??ÑÎ£å\n")
    
    # ?±Îä• Ï∏°Ï†ï
    frame_times = []
    detection_count = 0
    total_frames = 0
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frames += 1
            
            # ?ºÍµ¥ Í∞êÏ? (findFaces Î©îÏÑú???¨Ïö©)
            if use_mediapipe:
                img_with_faces, faces = current_detector.findFaces(frame, draw=True)
            else:
                # Haar Cascade ?¨Ïö©
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                
                detections = haar_detector.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=4,
                    flags=cv2.CASCADE_SCALE_IMAGE, minSize=(80, 80)
                )
                
                img_with_faces = frame.copy()
                faces = []
                for (x, y, w, h) in detections:
                    faces.append((x, y, w, h, 1.0))
                    cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(img_with_faces, "Face", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if len(faces) > 0:
                detection_count += 1
            
            # ?ÑÎ†à???úÍ∞Ñ Ï∏°Ï†ï
            elapsed = time.time() - start_time
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            avg_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # ?ïÎ≥¥ ?úÏãú
            detector_name = "MediaPipe" if use_mediapipe else "Haar Cascade"
            color = (0, 255, 0) if use_mediapipe else (255, 0, 0)
            
            # ?ÅÎã® ?ïÎ≥¥ ?®ÎÑê
            cv2.rectangle(img_with_faces, (0, 0), (640, 120), (0, 0, 0), -1)
            cv2.rectangle(img_with_faces, (0, 0), (640, 120), color, 2)
            
            cv2.putText(img_with_faces, f"Detector: {detector_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(img_with_faces, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_faces, f"Faces: {len(faces)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # MediaPipe??Í≤ΩÏö∞ ?†Î¢∞???úÏãú
            if use_mediapipe and len(faces) > 0:
                conf_text = f"Confidence: {faces[0][4]:.2f}"
                cv2.putText(img_with_faces, conf_text, (200, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # ?®Ï∂ï???àÎÇ¥
            cv2.putText(img_with_faces, "[M]ediaPipe  [H]aar  [Q]uit", (10, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Face Detection Comparison', img_with_faces)
            
            # ???ÖÎ†• Ï≤òÎ¶¨
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                use_mediapipe = True
                current_detector = mediapipe_detector
                print("\n?°Ô∏è  MediaPipe Face Detection ?úÏÑ±??)
            elif key == ord('h'):
                use_mediapipe = False
                current_detector = haar_detector
                print("\n?°Ô∏è  Haar Cascade Detection ?úÏÑ±??)
    
    except KeyboardInterrupt:
        print("\n\n?¨Ïö©??Ï§ëÎã®")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # ?µÍ≥Ñ Ï∂úÎ†•
        detection_rate = (detection_count / total_frames * 100) if total_frames > 0 else 0
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        
        print("\n" + "="*70)
        print("?±Îä• ?µÍ≥Ñ")
        print("="*70)
        print(f"Ï¥??ÑÎ†à?? {total_frames}")
        print(f"?ºÍµ¥ Í∞êÏ?: {detection_count} ({detection_rate:.1f}%)")
        print(f"?âÍ∑† FPS: {avg_fps:.1f}")
        print("="*70)


if __name__ == "__main__":
    main()
