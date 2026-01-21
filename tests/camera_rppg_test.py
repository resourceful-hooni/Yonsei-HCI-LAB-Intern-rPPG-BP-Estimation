"""
ì¹´ë©”?¼ë? ?¬ìš©?˜ì—¬ rPPG (remote Photoplethysmography) ? í˜¸ë¥?ì¶”ì¶œ?˜ê³  ?ˆì••???ˆì¸¡?˜ëŠ” ?¤í¬ë¦½íŠ¸

?„ìš”???¨í‚¤ì§€:
    - opencv-python
    - tensorflow-gpu==2.4.1
    - kapre
    - scipy (?„í„°ë§ìš©)

Phase 2 ê°œì„ ?¬í•­:
    - POS ?Œê³ ë¦¬ì¦˜ ?µí•© ê°€??(pos_algorithm.py)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU ?„ì „ ë¹„í™œ?±í™”
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from kapre import STFT, Magnitude, MagnitudeToDecibel
from collections import deque
import argparse
from scipy.signal import butter, filtfilt


class RPPGExtractor:
    """
    ì¹´ë©”?¼ë¡œë¶€??rPPG (?ê²© PPG) ? í˜¸ë¥?ì¶”ì¶œ?˜ëŠ” ?´ë˜??
    
    ë°©ë²•: Green ì±„ë„???‰ê·  ê°•ë„ ë³€??ì¶”ì¶œ (ê°„ë‹¨??rPPG ë°©ë²•)
    ???•êµ??ë°©ë²•: Wang et al. 2017??Plane-Orthogonal-to-Skin ?Œê³ ë¦¬ì¦˜
    """
    
    def __init__(self, window_size=875, fps=30, target_len=875):
        """
        Args:
            window_size: PPG ? í˜¸ ?ˆë„???¬ê¸° (?˜í”Œ ??
            fps: ì¹´ë©”???„ë ˆ???ˆì´??(Hz)
        """
        self.window_size = window_size
        self.fps = fps
        self.target_len = target_len  # ëª¨ë¸ ?…ë ¥ ê¸¸ì´ (875)
        self.duration = window_size / max(fps, 1)  # ì´??¨ìœ„ ?œê°„
        self.signal_buffer = deque(maxlen=window_size)
        
    def extract_face_region(self, frame):
        """
        ?„ë ˆ?„ì—???¼êµ´ ?ì—­??ì¶”ì¶œ?©ë‹ˆ??(ê°œì„ ???Œë¼ë¯¸í„°)
        
        Phase 1-Step 1: Haar Cascade ?Œë¼ë¯¸í„° ìµœì ??
        - minNeighbors: 4 ??8 (ê±°ì§“ ê°ì? ê°ì†Œ)
        - minSize/maxSize ì§€??(ë¶ˆí•„?”í•œ ê°ì? ?œê±°)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # ê°œì„ ???Œë¼ë¯¸í„°
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,      # 4 ??8 (ê±°ì§“ ê°ì? ê°ì†Œ)
            minSize=(100, 100),  # ìµœì†Œ ?¬ê¸° ì§€??
            maxSize=(400, 400)   # ìµœë? ?¬ê¸° ì§€??
        )
        
        if len(faces) == 0:
            return None
        
        # ê°€?????¼êµ´ ? íƒ
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return frame[y:y+h, x:x+w]
    
    def extract_skin_color(self, face_region):
        """
        ?¼êµ´ ?ì—­?ì„œ ?¼ë????½ì???ì¶”ì¶œ?©ë‹ˆ??(ê°„ë‹¨??ë°©ë²•)
        
        HSV ??ê³µê°„?ì„œ ?¼ë???ë²”ìœ„ë¥??•ì˜
        """
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # ?¼ë???ë²”ìœ„ ?•ì˜ (HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = face_region[mask == 255]
        
        return skin_pixels
    
    def extract_green_channel_mean(self, face_region):
        """
        ?¼êµ´ ?ì—­??Green ì±„ë„ ?‰ê· ê°’ì„ ì¶”ì¶œ?©ë‹ˆ??
        
        ê°„ë‹¨??rPPG ? í˜¸ ì¶”ì¶œ ë°©ë²•:
        - Green ì±„ë„??Red, Blue ì±„ë„ë³´ë‹¤ ?¼ë????ˆë¥˜ ë³€?”ì— ??ë¯¼ê°??
        """
        if face_region is None or face_region.size == 0:
            return None
        
        # BGR ?•ì‹?ì„œ Green?€ ?¸ë±??1
        green_channel = face_region[:, :, 1]
        mean_value = np.mean(green_channel)
        
        return mean_value
    
    def process_frame(self, frame):
        """
        ?„ë ˆ?„ì„ ì²˜ë¦¬?˜ì—¬ rPPG ? í˜¸ ê°’ì„ ì¶”ì¶œ?©ë‹ˆ??
        
        Returns:
            ì¶”ì¶œ??? í˜¸ ê°?(?ëŠ” None)
        """
        # 1. ?¼êµ´ ?ì—­ ì¶”ì¶œ
        face_region = self.extract_face_region(frame)
        if face_region is None:
            return None
        
        # 2. Green ì±„ë„ ?‰ê· ê°?ì¶”ì¶œ (ê°„ë‹¨??rPPG ? í˜¸)
        signal_value = self.extract_green_channel_mean(face_region)
        
        if signal_value is not None:
            self.signal_buffer.append(signal_value)
        
        return signal_value
    
    def get_signal(self):
        """
        ?„ì¬ê¹Œì? ì¶”ì¶œ??? í˜¸ë¥?ë°˜í™˜?©ë‹ˆ??(Phase 1-Step 2: ë°´ë“œ?¨ìŠ¤ ?„í„° ?ìš©)
        
        Phase 1-Step 2: Butterworth ë°´ë“œ?¨ìŠ¤ ?„í„° ì¶”ê?
        - ?¬ë°•??ë²”ìœ„: 0.7-4 Hz (42-240 bpm)
        - Order: 4 (ê¸°ë³¸ê°?
        - ?„í„°ë§? scipy.signal.filtfilt (?‘ë°©???„í„°ë§?
        
        Returns:
            ? í˜¸ ë°°ì—´ (shape: (n_samples, 1))
        """
        if len(self.signal_buffer) == 0:
            return None
        
        signal = np.array(list(self.signal_buffer), dtype=np.float32)
        
        # Phase 1-Step 2: ë°´ë“œ?¨ìŠ¤ ?„í„° ?ìš©
        # ? í˜¸ê°€ ì¶©ë¶„??ê¸¸ì–´???„í„°ë§?ê°€??
        if len(signal) > 10:
            signal = self._bandpass_filter(signal, lowcut=0.7, highcut=4.0, order=4)
        
        # ?•ê·œ??(?œì??¸ì°¨ê°€ 0??ê²½ìš° ë°©ì?)
        std = np.std(signal)
        mean = np.mean(signal)
        if std > 1e-8:
            signal = (signal - mean) / std
        else:
            signal = signal - mean
        
        # ê¸¸ì´ê°€ ëª¨ë¸ ?”êµ¬?€ ?¤ë¥´ë©?? í˜• ë³´ê°„?¼ë¡œ ë¦¬ìƒ˜?Œë§
        if signal.shape[0] != self.target_len:
            x = np.linspace(0, 1, signal.shape[0])
            x_new = np.linspace(0, 1, self.target_len)
            signal = np.interp(x_new, x, signal)
        
        return signal.reshape(-1, 1)
    
    def _bandpass_filter(self, signal, lowcut=0.7, highcut=4.0, order=4):
        """
        Phase 1-Step 2: Butterworth ë°´ë“œ?¨ìŠ¤ ?„í„°
        
        Args:
            signal: ?…ë ¥ ? í˜¸
            lowcut: ?˜í•œ ì£¼íŒŒ??(Hz) - ê¸°ë³¸ê°?0.7 Hz
            highcut: ?í•œ ì£¼íŒŒ??(Hz) - ê¸°ë³¸ê°?4.0 Hz
            order: ?„í„° ì°¨ìˆ˜ - ê¸°ë³¸ê°?4
            
        Returns:
            ?„í„°ë§ëœ ? í˜¸
        """
        # Nyquist ì£¼íŒŒ??ê³„ì‚°
        nyq = 0.5 * self.fps
        
        # ?•ê·œ?”ëœ ì£¼íŒŒ??
        low = lowcut / nyq
        high = highcut / nyq
        
        # ë²”ìœ„ ?•ì¸ (0 < low < high < 1)
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)
        
        if low >= high:
            # ?„í„° ë²”ìœ„ê°€ ? íš¨?˜ì? ?Šìœ¼ë©??ë³¸ ? í˜¸ ë°˜í™˜
            return signal
        
        # Butterworth ?„í„° ?¤ê³„
        b, a = butter(order, [low, high], btype='band')
        
        # ?‘ë°©???„í„°ë§?(phase distortion ?†ìŒ)
        try:
            filtered_signal = filtfilt(b, a, signal)
        except Exception:
            # ?„í„°ë§??¤íŒ¨ ???ë³¸ ? í˜¸ ë°˜í™˜
            filtered_signal = signal
        
        return filtered_signal
    
    def is_buffer_full(self):
        """
        ? í˜¸ ë²„í¼ê°€ ê°€??ì°¼ëŠ”ì§€ ?•ì¸
        """
        return len(self.signal_buffer) == self.window_size


def load_model(model_path):
    """
    ?¬ì „ ?™ìŠµ??ëª¨ë¸??ë¡œë“œ?©ë‹ˆ??
    """
    print(f"ëª¨ë¸ ë¡œë“œ ì¤? {model_path}")
    
    dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel
    }
    
    model = ks.models.load_model(model_path, custom_objects=dependencies)
    print("ëª¨ë¸ ë¡œë“œ ?„ë£Œ!")
    
    return model


def predict_bp(model, signal):
    """
    ? í˜¸ë¡œë????ˆì••???ˆì¸¡?©ë‹ˆ??
    
    Args:
        model: ?™ìŠµ??ëª¨ë¸
        signal: PPG ? í˜¸ (shape: (875, 1))
    
    Returns:
        (SBP, DBP) ?œí”Œ
    """
    # ë°°ì¹˜ ì°¨ì› ì¶”ê?
    input_data = np.expand_dims(signal, axis=0)  # (1, 875, 1)
    
    prediction = model.predict(input_data, verbose=0)
    
    # ê²½ìš° 1: ?¨ì¼ ?ì„œ ì¶œë ¥ (shape: (1, 2))
    if hasattr(prediction, 'shape') and prediction.ndim >= 2 and prediction.shape[-1] == 2:
        sbp = float(prediction[0, 0])
        dbp = float(prediction[0, 1])
        return sbp, dbp
    # ê²½ìš° 2: ??ê°œì˜ ì¶œë ¥ ë¦¬ìŠ¤??([sbp_batch, dbp_batch])
    elif isinstance(prediction, (list, tuple)) and len(prediction) == 2:
        sbp_batch, dbp_batch = prediction
        sbp_val = np.squeeze(sbp_batch)
        dbp_val = np.squeeze(dbp_batch)
        # ?¤ì¹¼?¼ì¸ ê²½ìš° ì§ì ‘ ë³€?? ë°°ì—´?´ë©´ ì²??”ì†Œ
        sbp = float(sbp_val) if sbp_val.ndim == 0 else float(sbp_val[0])
        dbp = float(dbp_val) if dbp_val.ndim == 0 else float(dbp_val[0])
        return sbp, dbp
    else:
        raise ValueError(f"?ˆìƒ?˜ì? ëª»í•œ ëª¨ë¸ ì¶œë ¥ ?•íƒœ: type={type(prediction)}, shape={getattr(prediction, 'shape', None)}")


def main():
    parser = argparse.ArgumentParser(description='ì¹´ë©”?¼ë¡œë¶€??rPPG ? í˜¸ ì¶”ì¶œ ë°??ˆì•• ?ˆì¸¡')
    parser.add_argument('--model', type=str, default='data/resnet_ppg_nonmixed.h5',
                        help='ëª¨ë¸ ?Œì¼ ê²½ë¡œ (ê¸°ë³¸ê°? ResNet - Phase 1-Step 4)')
    parser.add_argument('--camera', type=int, default=0,
                        help='ì¹´ë©”??ID (ê¸°ë³¸ê°? 0 - ê¸°ë³¸ ì¹´ë©”??')
    parser.add_argument('--backend', type=str, choices=['default','dshow','msmf'], default='default',
                        help='ì¹´ë©”??ë°±ì—”??? íƒ (Windows: dshow ?ëŠ” msmf)')
    parser.add_argument('--width', type=int, default=640, help='ì¹´ë©”???´ìƒ??ê°€ë¡?)
    parser.add_argument('--height', type=int, default=480, help='ì¹´ë©”???´ìƒ???¸ë¡œ')
    parser.add_argument('--fps', type=int, default=30, help='?”ì²­ ?„ë ˆ?„ë ˆ?´íŠ¸')
    parser.add_argument('--list', action='store_true', help='?¬ìš© ê°€?¥í•œ ì¹´ë©”???˜ì—´ ??ì¢…ë£Œ')
    parser.add_argument('--duration', type=int, default=7,
                        help='? í˜¸ ?˜ì§‘ ?œê°„ (ì´? ê¸°ë³¸ê°? 7)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ì¹´ë©”??ê¸°ë°˜ rPPG ?ˆì•• ?ˆì¸¡")
    print("="*80)

    # ê°„ë‹¨??ì¹´ë©”???´ê¸° ?¨ìˆ˜
    def open_camera(index, backend_name):
        if backend_name == 'dshow':
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        elif backend_name == 'msmf':
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(index)
        return cap

    # ì¹´ë©”???˜ì—´ ëª¨ë“œ: ëª¨ë¸ ë¡œë“œ ?†ì´ ?¥ì¹˜ ?ìƒ‰
    if args.list:
        print("\n?¬ìš© ê°€?¥í•œ ì¹´ë©”???ìƒ‰ ì¤?..")
        found = []
        for idx in range(0, 10):
            for backend in ['dshow','msmf','default']:
                cap = open_camera(idx, backend)
                if cap.isOpened():
                    # ?„ë ˆ???½ê¸° ?ŒìŠ¤??
                    ok, frame = cap.read()
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if ok:
                        found.append((idx, backend, int(width), int(height), int(fps) if fps and fps>0 else None))
                        print(f"- ì¹´ë©”??{idx} (backend={backend}): {int(width)}x{int(height)} fps={int(fps) if fps and fps>0 else 'N/A'}")
                        break
        if not found:
            print("ì¹´ë©”?¼ë? ì°¾ì? ëª»í–ˆ?µë‹ˆ?? ?¤ë¥¸ ë°±ì—”???¸ë±?¤ë? ì§ì ‘ ì§€?•í•´ ?œë„?˜ì„¸??")
        return

    # GPU ë¹„í™œ?±í™”ë¡?CUDA ê´€??ê²½ê³  ?Œí”¼ (CPU ê°•ì œ ?¬ìš©)
    try:
        tf.config.experimental.set_visible_devices([], 'GPU')
    except Exception:
        pass
    
    # 1. ëª¨ë¸ ë¡œë“œ
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"ëª¨ë¸ ë¡œë“œ ?¤íŒ¨: {e}")
        return
    
    # 2. rPPG ì¶”ì¶œê¸?ì´ˆê¸°?”ëŠ” ì¹´ë©”???œì‘ ??FPS ?•ì¸ ?¤ì— ?˜í–‰
    
    # 3. ì¹´ë©”???œì‘
    print(f"\nì¹´ë©”??{args.camera} ?œì‘ ì¤?.. (backend={args.backend})")

    # ? íƒ??ë°±ì—”?œë¡œ ?œë„ ?? ?¤ë¥¸ ë°±ì—”?œë¡œ ?´ë°±
    def open_camera(index, backend_name):
        if backend_name == 'dshow':
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        elif backend_name == 'msmf':
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(index)
        return cap

    cap = open_camera(args.camera, args.backend)
    if not cap.isOpened():
        # ?´ë°± ?œë„
        for b in ['dshow','msmf','default']:
            if b == args.backend:
                continue
            cap = open_camera(args.camera, b)
            if cap.isOpened():
                print(f"ë°±ì—”???´ë°± ?±ê³µ: {b}")
                break
    
    if not cap.isOpened():
        print("ì¹´ë©”?¼ë? ?????†ìŠµ?ˆë‹¤!")
        return
    
    # ì¹´ë©”???´ìƒ???„ë ˆ?„ë ˆ?´íŠ¸ ?¤ì •
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # ì¹´ë©”??FPS ?½ê¸° (ê¸°ë³¸ 30)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = float(args.fps)
    # ?˜ì§‘???ˆë„???¬ê¸° ê³„ì‚° (duration ì´?* fps), ?ˆì¸¡ ?…ë ¥?€ 875ë¡?ë¦¬ìƒ˜?Œë§
    window_size = max(int(args.duration * fps), 30)
    rppg = RPPGExtractor(window_size=window_size, fps=fps, target_len=875)

    print("ì¹´ë©”??ì¤€ë¹??„ë£Œ!")
    print("\n?¼êµ´??ì¹´ë©”?¼ì— ?•ë©´?¼ë¡œ ë§ì¶°ì£¼ì„¸??")
    print(f"{args.duration}ì´??™ì•ˆ ? í˜¸ë¥??˜ì§‘?©ë‹ˆ??(FPS {fps:.0f}, ?˜ì§‘ ?˜í”Œ {window_size}).")
    print("Ctrl+Cë¥??ŒëŸ¬ ì¤‘ë‹¨?????ˆìŠµ?ˆë‹¤.\n")
    
    frame_count = 0
    signal_collected = False
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("?„ë ˆ?„ì„ ?½ì„ ???†ìŠµ?ˆë‹¤!")
                break
            
            # ?„ë ˆ??ì²˜ë¦¬
            signal_value = rppg.process_frame(frame)
            
            # Phase 1-Step 3: ?¨ì¼ ?¼êµ´ ë°•ìŠ¤ë§??œì‹œ
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # ê°œì„ ???Œë¼ë¯¸í„°ë¡??¼êµ´ ê°ì?
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,      # Step 1?ì„œ ìµœì ?”ë¨
                minSize=(100, 100),
                maxSize=(400, 400)
            )
            
            # Step 3: ê°€?????¼êµ´ë§??œì‹œ (?¨ì¼ ë°•ìŠ¤)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # ì§„í–‰ ?í™© ?œì‹œ
            buffer_ratio = len(rppg.signal_buffer) / rppg.window_size
            cv2.putText(frame, f"Progress: {buffer_ratio*100:.1f}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            if signal_value is not None:
                cv2.putText(frame, f"Signal: {signal_value:.1f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
            
            # ?”ë©´ ì¶œë ¥
            cv2.imshow('rPPG ? í˜¸ ?˜ì§‘', frame)
            
            frame_count += 1
            
            # ? í˜¸ ?˜ì§‘ ?„ë£Œ ?•ì¸
            if rppg.is_buffer_full() and not signal_collected:
                print("\n? í˜¸ ?˜ì§‘ ?„ë£Œ! ?ˆì•• ?ˆì¸¡ ì¤?..")
                signal_collected = True
                
                # ?ˆì•• ?ˆì¸¡
                signal = rppg.get_signal()
                sbp, dbp = predict_bp(model, signal)
                
                print("\n" + "="*80)
                print("?ˆì¸¡ ê²°ê³¼")
                print("="*80)
                print(f"?˜ì¶•ê¸??ˆì•• (SBP): {sbp:.1f} mmHg")
                print(f"?´ì™„ê¸??ˆì•• (DBP): {dbp:.1f} mmHg")
                print("="*80)
                
                # ì¶”ê? ? í˜¸ ?˜ì§‘ ?¬ë? ?•ì¸
                print("\nì¶”ê? ?ˆì¸¡???í•˜ë©?ê³„ì† ë²„íŠ¼???„ë¥´?¸ìš”.")
                print("ì¢…ë£Œ?˜ë ¤ë©?Ctrl+Cë¥??„ë¥´?¸ìš”.")
                
                # ë²„í¼ ì´ˆê¸°?”í•˜???ˆë¡œ??? í˜¸ ?˜ì§‘ ?œì‘
                rppg.signal_buffer.clear()
                signal_collected = False
            
            # 'q' ?¤ë? ?„ë¥´ë©?ì¢…ë£Œ
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n?¬ìš©??ì¤‘ë‹¨")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("ì¹´ë©”??ì¢…ë£Œ")


if __name__ == '__main__':
    main()
