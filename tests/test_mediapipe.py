"""MediaPipe ?åÏä§???§ÌÅ¨Î¶ΩÌä∏"""
import cv2
from realtime.mediapipe_face_detector import MediaPipeFaceDetector

print("="*60)
print("MediaPipe Í∞êÏ?Í∏??åÏä§??)
print("="*60)

# Í∞êÏ?Í∏?Ï¥àÍ∏∞??
detector = MediaPipeFaceDetector(min_detection_confidence=0.7)

print("\nÏπ¥Î©î???¥Í∏∞ Ï§?..")
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Ïπ¥Î©î??1Î≤??§Ìå®, 0Î≤àÏúºÎ°??úÎèÑ...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("??Ïπ¥Î©î?ºÎ? ?????ÜÏäµ?àÎã§")
    exit(1)

print("??Ïπ¥Î©î??Ï§ÄÎπ??ÑÎ£å")
print("\n?ºÍµ¥??Ïπ¥Î©î???ûÏóê ?ÑÏπò?úÌÇ§?∏Ïöî...")
print("'q' ?§Î? ?åÎü¨ Ï¢ÖÎ£å\n")

frame_count = 0
detection_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # ?ºÍµ¥ Í∞êÏ?
        roi = detector.detect(frame)
        
        if roi is not None:
            detection_count += 1
            
            # ?ºÍµ¥ ?ÅÏó≠ ?úÏãú
            if detector.last_face_rect:
                x, y, w, h = detector.last_face_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # ?ïÎ≥¥ ?úÏãú
                cv2.putText(frame, f"Face Detected ({w}x{h})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
        
        # ?µÍ≥Ñ ?úÏãú
        detection_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0
        cv2.putText(frame, f"Frames: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {detection_count} ({detection_rate:.1f}%)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # MediaPipe ?¨Ïö© ?¨Î? ?úÏãú
        if hasattr(detector, 'use_mediapipe') and detector.use_mediapipe:
            cv2.putText(frame, "MediaPipe: ON", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Haar Cascade: ON", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        cv2.imshow('MediaPipe Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n\n?¨Ïö©??Ï§ëÎã®")

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("?åÏä§??Í≤∞Í≥º")
    print("="*60)
    print(f"Ï¥??ÑÎ†à?? {frame_count}")
    print(f"?ºÍµ¥ Í∞êÏ?: {detection_count} ({detection_rate:.1f}%)")
    
    if hasattr(detector, 'use_mediapipe'):
        if detector.use_mediapipe:
            print("??MediaPipe Face Detection ?¨Ïö©??)
        else:
            print("?†Ô∏è  Haar Cascade ?¨Ïö©??(MediaPipe ?¥Î∞±)")
    
    print("="*60)
