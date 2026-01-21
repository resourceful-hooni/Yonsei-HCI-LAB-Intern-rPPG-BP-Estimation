"""
test_phase2_step3.py - Phase 2-Step 3 ?ŒìŠ¤??

POS ?Œê³ ë¦¬ì¦˜ê³?MediaPipe ê¸°ë³¸ ê¸°ëŠ¥ ?ŒìŠ¤??
"""

import numpy as np
from realtime.pos_algorithm import POSExtractor
from realtime.mediapipe_face_detector import MediaPipeFaceDetector, HaarCascadeFaceDetector

print("="*60)
print("Phase 2-Step 3 ?ŒìŠ¤?? POS + MediaPipe ëª¨ë“ˆ")
print("="*60)

# 1. POS ?Œê³ ë¦¬ì¦˜ ì´ˆê¸°??
print("\n[1] POS ?Œê³ ë¦¬ì¦˜ ì´ˆê¸°??..")
pos = POSExtractor(fs=30, window_size=1.6)
print(f"??POS ì´ˆê¸°???„ë£Œ")
print(f"  - ?˜í”Œë§?ì£¼íŒŒ?? 30 Hz")
print(f"  - ?ˆë„???¬ê¸°: {pos.window_samples} ?˜í”Œ (1.6ì´?")

# 2. ê°€ì§?RGB ? í˜¸ ?ì„± (?ŒìŠ¤?¸ìš©)
print("\n[2] ?ŒìŠ¤??RGB ? í˜¸ ?ì„±...")
N = 300  # 10ì´?ë¶„ëŸ‰ (30fps)
t = np.arange(N) / 30.0

# ?¬ë°•????75 bpm = 1.25 Hz
hr_freq = 1.25
pulse = np.sin(2 * np.pi * hr_freq * t)
noise = 0.1 * np.random.randn(N)

# RGB ? í˜¸ ?ì„± (heart rate modulation)
rgb = np.zeros((N, 3))
rgb[:, 0] = 100 + 5 * pulse + noise  # R (?ê²Œ ë³€??
rgb[:, 1] = 120 + 10 * pulse + noise  # G (ë§ì´ ë³€??
rgb[:, 2] = 110 + 3 * pulse + noise  # B (ì¤‘ê°„)
print(f"??RGB ? í˜¸ ?ì„± ?„ë£Œ: {rgb.shape}")

# 3. POS ?Œê³ ë¦¬ì¦˜ ?ŒìŠ¤??
print("\n[3] POS ?Œê³ ë¦¬ì¦˜ ?¤í–‰...")
try:
    pulse_signal = pos.pos_algorithm(rgb)
    print(f"??POS ?Œê³ ë¦¬ì¦˜ ?„ë£Œ")
    print(f"  - ì¶œë ¥ ? í˜¸ ê¸¸ì´: {len(pulse_signal)}")
    print(f"  - ? í˜¸ ë²”ìœ„: [{pulse_signal.min():.4f}, {pulse_signal.max():.4f}]")
except Exception as e:
    print(f"??POS ?Œê³ ë¦¬ì¦˜ ?¤íŒ¨: {e}")

# 4. ë°´ë“œ?¨ìŠ¤ ?„í„° ?ŒìŠ¤??
print("\n[4] ë°´ë“œ?¨ìŠ¤ ?„í„° ?ŒìŠ¤??..")
try:
    filtered_signal = pos.bandpass_filter(pulse_signal, lowcut=0.7, highcut=4.0)
    print(f"???„í„°ë§??„ë£Œ")
    print(f"  - ?„í„° ë²”ìœ„: 0.7-4.0 Hz (42-240 bpm)")
    print(f"  - ?„í„°ë§???? í˜¸ ë²”ìœ„: [{filtered_signal.min():.4f}, {filtered_signal.max():.4f}]")
except Exception as e:
    print(f"???„í„°ë§??¤íŒ¨: {e}")

# 5. ?¬ë°•??ì¶”ì • ?ŒìŠ¤??
print("\n[5] ?¬ë°•??ì¶”ì •...")
try:
    hr, freqs = pos.estimate_heart_rate(filtered_signal)
    print(f"???¬ë°•??ì¶”ì • ?„ë£Œ")
    print(f"  - ?ˆìƒ ?¬ë°•?? 75 bpm")
    print(f"  - ì¶”ì • ?¬ë°•?? {hr:.1f} bpm")
    print(f"  - ?¤ì°¨: {abs(hr - 75):.1f} bpm")
except Exception as e:
    print(f"???¬ë°•??ì¶”ì • ?¤íŒ¨: {e}")

# 6. MediaPipe ?¼êµ´ ê°ì?ê¸?ì´ˆê¸°??
print("\n[6] MediaPipe ?¼êµ´ ê°ì?ê¸?ì´ˆê¸°??..")
try:
    mediapipe_detector = MediaPipeFaceDetector(min_detection_confidence=0.7)
    print(f"??MediaPipe ê°ì?ê¸?ì´ˆê¸°???„ë£Œ")
except Exception as e:
    print(f"??MediaPipe ì´ˆê¸°???¤íŒ¨: {e}")

# 7. Haar Cascade ë¹„êµ (?´ë°±)
print("\n[7] Haar Cascade ê°ì?ê¸?ì´ˆê¸°??..")
try:
    haar_detector = HaarCascadeFaceDetector(min_neighbors=8)
    print(f"??Haar Cascade ê°ì?ê¸?ì´ˆê¸°???„ë£Œ")
except Exception as e:
    print(f"??Haar Cascade ì´ˆê¸°???¤íŒ¨: {e}")

print("\n" + "="*60)
print("?ŒìŠ¤???„ë£Œ!")
print("="*60)
print("\n?¤ìŒ ?¨ê³„: ?¤ì œ ì¹´ë©”?¼ë¡œ camera_rppg_advanced.py ?ŒìŠ¤??)
