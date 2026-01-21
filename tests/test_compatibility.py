"""
Transformer ëª¨ë¸ + MediaPipe ?¸í™˜???ŒìŠ¤??
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*70)
print("Transformer ëª¨ë¸ + MediaPipe ?¸í™˜???ŒìŠ¤??)
print("="*70)

# 1. MediaPipe ?ŒìŠ¤??
print("\n[1/4] MediaPipe ë¡œë“œ ?ŒìŠ¤??..")
try:
    from realtime.mediapipe_face_detector import MediaPipeFaceDetector
    detector = MediaPipeFaceDetector(min_detection_confidence=0.7)
    
    if hasattr(detector, 'use_mediapipe') and detector.use_mediapipe:
        print("??MediaPipe Face Detection ?œì„±?”ë¨")
    else:
        print("? ï¸  Haar Cascade ?´ë°± ëª¨ë“œ")
except Exception as e:
    print(f"??MediaPipe ë¡œë“œ ?¤íŒ¨: {e}")
    exit(1)

# 2. TensorFlow/Keras ?ŒìŠ¤??
print("\n[2/4] TensorFlow ë¡œë“œ ?ŒìŠ¤??..")
try:
    import tensorflow as tf
    import tensorflow.keras as ks
    print(f"??TensorFlow {tf.__version__}")
    print(f"??Keras {ks.__version__}")
except Exception as e:
    print(f"??TensorFlow ë¡œë“œ ?¤íŒ¨: {e}")
    exit(1)

# 3. Transformer ëª¨ë¸ ë¡œë“œ ?ŒìŠ¤??
print("\n[3/4] Transformer ëª¨ë¸ ë¡œë“œ ?ŒìŠ¤??..")
try:
    from models.transformer_model import MultiHeadAttention, EncoderLayer, TransformerEncoder
    from kapre import STFT, Magnitude, MagnitudeToDecibel
    
    custom_objects = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel,
        'MultiHeadAttention': MultiHeadAttention,
        'EncoderLayer': EncoderLayer,
        'TransformerEncoder': TransformerEncoder
    }
    
    model_path = 'data/transformer_bp_model.h5'
    model = ks.models.load_model(model_path, custom_objects=custom_objects)
    print(f"??ëª¨ë¸ ë¡œë“œ ?±ê³µ: {model_path}")
    print(f"  - Input shape: {model.input_shape}")
    
    # ì¶œë ¥ ?•ì‹ ?•ì¸
    if isinstance(model.output_shape, list):
        print(f"  - Output shape: {len(model.output_shape)} outputs")
        for i, shape in enumerate(model.output_shape):
            print(f"    Output {i+1}: {shape}")
    else:
        print(f"  - Output shape: {model.output_shape}")
    
    # ?”ë? ?ˆì¸¡ ?ŒìŠ¤??
    import numpy as np
    dummy_input = np.random.randn(1, 875, 1).astype(np.float32)
    prediction = model.predict(dummy_input, verbose=0)
    
    # ?ˆì¸¡ ê²°ê³¼ ?Œì‹±
    if isinstance(prediction, list):
        print(f"  - ?ˆì¸¡ ì¶œë ¥: {len(prediction)} ê°œì˜ ì¶œë ¥")
        sbp = float(prediction[0][0, 0])
        dbp = float(prediction[1][0, 0])
    else:
        print(f"  - ?ˆì¸¡ ì¶œë ¥: {prediction.shape}")
        sbp = float(prediction[0, 0])
        dbp = float(prediction[0, 1])
    
    print(f"  - ?ˆì¸¡ê°? SBP={sbp:.2f}, DBP={dbp:.2f}")
    
except Exception as e:
    print(f"??ëª¨ë¸ ë¡œë“œ ?¤íŒ¨: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. ?µí•© ?ŒìŠ¤??
print("\n[4/4] ?µí•© ?¸í™˜???ŒìŠ¤??..")
try:
    import cv2
    import numpy as np
    
    # ?”ë? ?„ë ˆ???ì„±
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # MediaPipe ?¼êµ´ ê°ì?
    roi = detector.detect(dummy_frame)
    print(f"??MediaPipe ?¼êµ´ ê°ì?: {'?±ê³µ' if roi is not None else '?¤íŒ¨ (?•ìƒ - ?”ë? ?„ë ˆ??'}")
    
    # ?”ë? ? í˜¸ë¡?ëª¨ë¸ ?ˆì¸¡
    dummy_signal = np.random.randn(875, 1).astype(np.float32)
    input_data = np.expand_dims(dummy_signal, axis=0)
    prediction = model.predict(input_data, verbose=0)
    print(f"??Transformer ëª¨ë¸ ?ˆì¸¡ ?±ê³µ")
    
    print("\n" + "="*70)
    print("??ëª¨ë“  ?ŒìŠ¤???µê³¼!")
    print("="*70)
    print("\n?„ì¬ ?˜ê²½?ì„œ Transformer ëª¨ë¸ + MediaPipe ?¬ìš© ê°€?¥í•©?ˆë‹¤!")
    print("\n?¤í–‰ ëª…ë ¹??")
    print("  python camera_rppg_advanced.py --model data/transformer_bp_model.h5 --camera 1")
    
except Exception as e:
    print(f"???µí•© ?ŒìŠ¤???¤íŒ¨: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
