"""
ê°„ë‹¨???µí•© ?ŒìŠ¤??- ë¹ ë¥¸ ê²€ì¦ìš©
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*70)
print("ë¹ ë¥¸ ?µí•© ?ŒìŠ¤??)
print("="*70)

# Test 1: Imports
print("\n[1/5] ?¼ì´ë¸ŒëŸ¬ë¦?Import ?ŒìŠ¤??..")
try:
    import tensorflow as tf
    import mediapipe as mp
    import numpy as np
    import cv2
    print(f"  ??TensorFlow {tf.__version__}")
    print(f"  ??MediaPipe {mp.__version__}")
    print(f"  ??NumPy {np.__version__}")
    print(f"  ??OpenCV {cv2.__version__}")
except Exception as e:
    print(f"  ??Import ?¤íŒ¨: {e}")
    exit(1)

# Test 2: MediaPipe Detector
print("\n[2/5] MediaPipe Detector ì´ˆê¸°??..")
try:
    from realtime.mediapipe_face_detector import MediaPipeFaceDetector
    detector = MediaPipeFaceDetector()
    
    if hasattr(detector, 'use_mediapipe') and detector.use_mediapipe:
        print("  ??MediaPipe Face Detection ?œì„±??)
    else:
        print("  ? ï¸  Haar Cascade ?´ë°± ?¬ìš©")
except Exception as e:
    print(f"  ??Detector ì´ˆê¸°???¤íŒ¨: {e}")
    exit(1)

# Test 3: Model Loading
print("\n[3/5] Transformer ëª¨ë¸ ë¡œë“œ...")
try:
    from realtime.integrated_pipeline import IntegratedRPPGPipeline
    
    pipeline = IntegratedRPPGPipeline(
        model_path='data/transformer_bp_model.h5',
        enable_bbox_filter=True
    )
    print("  ??Pipeline ì´ˆê¸°???„ë£Œ")
except Exception as e:
    print(f"  ??Pipeline ì´ˆê¸°???¤íŒ¨: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Frame Processing
print("\n[4/5] ?„ë ˆ??ì²˜ë¦¬ ?ŒìŠ¤??..")
try:
    # ?”ë? ?„ë ˆ???ì„±
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    status = pipeline.process_frame(test_frame)
    print(f"  ???„ë ˆ??ì²˜ë¦¬ ?±ê³µ")
    print(f"    - Face detected: {status['face_detected']}")
    print(f"    - Signal collected: {status['signal_collected']}")
except Exception as e:
    print(f"  ???„ë ˆ??ì²˜ë¦¬ ?¤íŒ¨: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Full Pipeline (with synthetic data)
print("\n[5/5] ?„ì²´ ?Œì´?„ë¼???ŒìŠ¤??..")
try:
    # 7ì´?ë¶„ëŸ‰???”ë? ?„ë ˆ???ì„±
    print("  210ê°??„ë ˆ???ì„± ì¤?..")
    for i in range(210):
        frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        
        # ?¼êµ´ ?ì—­ ì¶”ê? (?¤ë²„?Œë¡œ??ë°©ì?)
        face_x, face_y = 220, 140
        face_w, face_h = 200, 200
        pulse = int(15 * np.sin(2 * np.pi * 1.2 * i / 30))
        
        roi = frame[face_y:face_y+face_h, face_x:face_x+face_w, :].astype(np.int16)
        roi += pulse
        roi = np.clip(roi, 0, 255).astype(np.uint8)
        frame[face_y:face_y+face_h, face_x:face_x+face_w, :] = roi
        
        pipeline.process_frame(frame)
        
        if (i + 1) % 50 == 0:
            print(f"    ì§„í–‰: {i+1}/210")
    
    print("  ?ˆì¸¡ ?¤í–‰ ì¤?..")
    results = pipeline.extract_and_predict()
    
    if results:
        print("  ???ˆì¸¡ ?±ê³µ!")
        print(f"    - SBP: {results['sbp']:.1f} mmHg")
        print(f"    - DBP: {results['dbp']:.1f} mmHg")
        print(f"    - HR:  {results['hr']:.1f} bpm")
        print(f"    - Quality: {results['quality_score']:.3f}")
        print(f"    - Confidence: {results['confidence']:.3f}")
    else:
        print("  ? ï¸  ?ˆì¸¡ ê²°ê³¼ ?†ìŒ")
        
except Exception as e:
    print(f"  ???Œì´?„ë¼???¤íŒ¨: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("??ëª¨ë“  ?ŒìŠ¤???µê³¼!")
print("="*70)
print("\n?¤ì‹œê°?ëª¨ë‹ˆ?°ë§ ?¤í–‰:")
print("  python run_integrated_bp_monitor.py --camera 1")
print()
