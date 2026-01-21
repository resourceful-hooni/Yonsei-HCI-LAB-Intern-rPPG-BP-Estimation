"""
test_pos_only.py - POS ?Œê³ ë¦¬ì¦˜ë§??ŒìŠ¤??(MediaPipe ?œì™¸)
"""

import numpy as np
from realtime.pos_algorithm import POSExtractor

print("="*60)
print("Phase 2-Step 3 ?ŒìŠ¤?? POS ?Œê³ ë¦¬ì¦˜")
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
print(f"  - R ë²”ìœ„: [{rgb[:, 0].min():.1f}, {rgb[:, 0].max():.1f}]")
print(f"  - G ë²”ìœ„: [{rgb[:, 1].min():.1f}, {rgb[:, 1].max():.1f}]")
print(f"  - B ë²”ìœ„: [{rgb[:, 2].min():.1f}, {rgb[:, 2].max():.1f}]")

# 3. POS ?Œê³ ë¦¬ì¦˜ ?ŒìŠ¤??
print("\n[3] POS ?Œê³ ë¦¬ì¦˜ ?¤í–‰...")
try:
    pulse_signal = pos.pos_algorithm(rgb)
    print(f"??POS ?Œê³ ë¦¬ì¦˜ ?„ë£Œ")
    print(f"  - ì¶œë ¥ ? í˜¸ ê¸¸ì´: {len(pulse_signal)}")
    print(f"  - ? í˜¸ ë²”ìœ„: [{pulse_signal.min():.4f}, {pulse_signal.max():.4f}]")
    print(f"  - ? í˜¸ ?œì??¸ì°¨: {np.std(pulse_signal):.4f}")
except Exception as e:
    print(f"??POS ?Œê³ ë¦¬ì¦˜ ?¤íŒ¨: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. ë°´ë“œ?¨ìŠ¤ ?„í„° ?ŒìŠ¤??
print("\n[4] ë°´ë“œ?¨ìŠ¤ ?„í„° ?ŒìŠ¤??..")
try:
    filtered_signal = pos.bandpass_filter(pulse_signal, lowcut=0.7, highcut=4.0)
    print(f"???„í„°ë§??„ë£Œ")
    print(f"  - ?„í„° ë²”ìœ„: 0.7-4.0 Hz (42-240 bpm)")
    print(f"  - ?„í„°ë§???? í˜¸ ë²”ìœ„: [{filtered_signal.min():.4f}, {filtered_signal.max():.4f}]")
except Exception as e:
    print(f"???„í„°ë§??¤íŒ¨: {e}")
    import traceback
    traceback.print_exc()

# 5. ?¬ë°•??ì¶”ì • ?ŒìŠ¤??
print("\n[5] ?¬ë°•??ì¶”ì •...")
try:
    hr, freqs = pos.estimate_heart_rate(filtered_signal)
    print(f"???¬ë°•??ì¶”ì • ?„ë£Œ")
    print(f"  - ?ˆìƒ ?¬ë°•?? 75 bpm (1.25 Hz)")
    print(f"  - ì¶”ì • ?¬ë°•?? {hr:.1f} bpm ({hr/60:.3f} Hz)")
    print(f"  - ?¤ì°¨: {abs(hr - 75):.1f} bpm ({abs(hr-75)/75*100:.1f}%)")
except Exception as e:
    print(f"???¬ë°•??ì¶”ì • ?¤íŒ¨: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("??POS ?Œê³ ë¦¬ì¦˜ ?ŒìŠ¤???„ë£Œ!")
print("="*60)
print("\n?¤ìŒ ?¨ê³„:")
print("1. MediaPipe ?¸í™˜??ë¬¸ì œ ?´ê²°")
print("2. camera_rppg_advanced.py ?ŒìŠ¤??)
