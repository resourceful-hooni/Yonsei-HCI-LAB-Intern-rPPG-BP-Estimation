"""
?œìŠ¤???íƒœ ?•ì¸ ë°??¤í–‰ ê°€?´ë“œ
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*70)
print("rPPG BP ëª¨ë‹ˆ?°ë§ ?œìŠ¤???íƒœ ?•ì¸")
print("="*70)

# 1. ?˜ê²½ ?•ì¸
print("\n[?˜ê²½ ?•ì¸]")
try:
    import tensorflow as tf
    import mediapipe as mp
    import numpy as np
    import cv2
    
    print(f"  ??TensorFlow: {tf.__version__}")
    print(f"  ??MediaPipe: {mp.__version__}")
    print(f"  ??NumPy: {np.__version__}")
    print(f"  ??OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"  ???¼ì´ë¸ŒëŸ¬ë¦??¤ë¥˜: {e}")
    exit(1)

# 2. ëª¨ë¸ ?Œì¼ ?•ì¸
print("\n[ëª¨ë¸ ?Œì¼ ?•ì¸]")
import os
models = {
    'Transformer': 'data/transformer_bp_model.h5',
    'ResNet': 'data/resnet_ppg_nonmixed.h5',
}

for name, path in models.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  ??{name}: {path} ({size:.1f} MB)")
    else:
        print(f"  ??{name}: {path} (?†ìŒ)")

# 3. MediaPipe ?íƒœ
print("\n[MediaPipe ?íƒœ]")
try:
    from realtime.mediapipe_face_detector import MediaPipeFaceDetector
    detector = MediaPipeFaceDetector()
    
    if hasattr(detector, 'use_mediapipe') and detector.use_mediapipe:
        print("  ??MediaPipe Face Detection ?œì„±??)
        print("    - model_selection: 1 (5m ?´ë‚´)")
        print("    - min_detection_confidence: 0.7")
    else:
        print("  ? ï¸  Haar Cascade ?´ë°± ëª¨ë“œ")
except Exception as e:
    print(f"  ??Detector ?¤ë¥˜: {e}")

# 4. ì¹´ë©”???•ì¸
print("\n[ì¹´ë©”???•ì¸]")
for cam_id in [0, 1]:
    cap = cv2.VideoCapture(cam_id)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  ??Camera {cam_id}: {w}x{h} @ {fps:.0f} FPS")
        cap.release()
    else:
        print(f"  ??Camera {cam_id}: ?¬ìš© ë¶ˆê?")

# 5. ?µí•© ?Œì´?„ë¼???íƒœ
print("\n[?µí•© ?Œì´?„ë¼???íƒœ]")
try:
    from realtime.integrated_pipeline import IntegratedRPPGPipeline
    print("  ??IntegratedRPPGPipeline ëª¨ë“ˆ ë¡œë“œ ?„ë£Œ")
    print("    - Face Detection ??ROI ??POS ??Quality ??Model ??Kalman")
except Exception as e:
    print(f"  ??Pipeline ?¤ë¥˜: {e}")

print("\n" + "="*70)
print("?¤í–‰ ëª…ë ¹??)
print("="*70)

print("\n1. ë¹ ë¥¸ ?ŒìŠ¤??(?©ì„± ?°ì´??:")
print("   .\\env\\Scripts\\Activate.ps1; python test_quick.py")

print("\n2. ?¤ì‹œê°?ëª¨ë‹ˆ?°ë§ (ì¹´ë©”???„ìš”):")
print("   .\\env\\Scripts\\Activate.ps1; python run_integrated_bp_monitor.py --camera 1")

print("\n3. ê¸°ì¡´ ë°©ì‹ (?¸í™˜???ŒìŠ¤??:")
print("   .\\env\\Scripts\\Activate.ps1; python camera_rppg_advanced.py --model data/transformer_bp_model.h5 --camera 1")

print("\n" + "="*70)
print("???œìŠ¤??ì¤€ë¹??„ë£Œ")
print("="*70)
