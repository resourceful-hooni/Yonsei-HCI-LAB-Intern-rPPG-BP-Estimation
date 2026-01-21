"""
ê°„ë‹¨??ëª¨ë¸ ?ŒìŠ¤???ˆì œ
- ê°€??ê¸°ë³¸?ì¸ ?¬ìš©ë²•ì„ ë³´ì—¬ì£¼ëŠ” ?¤í¬ë¦½íŠ¸?…ë‹ˆ??
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU ?„ì „ ë¹„í™œ?±í™”
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from kapre import STFT, Magnitude, MagnitudeToDecibel


def simple_test():
    """
    ?¬ì „ ?™ìŠµ??ëª¨ë¸ë¡?ê°„ë‹¨???ŒìŠ¤?¸ë? ?˜í–‰?©ë‹ˆ??
    """
    print("="*80)
    print("ê°„ë‹¨??ëª¨ë¸ ?ŒìŠ¤???ˆì œ")
    print("="*80)

    # GPUë¥??¬ìš©?˜ì? ?Šë„ë¡??¤ì • (CUDA ê²½ê³  ?Œí”¼ ë°?CPU ê°•ì œ)
    try:
        tf.config.experimental.set_visible_devices([], 'GPU')
    except Exception:
        pass
    
    # 1. ëª¨ë¸ ë¡œë“œ
    model_path = 'data/alexnet_ppg_nonmixed.h5'  # ?í•˜??ëª¨ë¸ë¡?ë³€ê²?ê°€??
    print(f"\n1?¨ê³„: ëª¨ë¸ ë¡œë“œ - {model_path}")
    
    dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel
    }
    
    try:
        model = ks.models.load_model(model_path, custom_objects=dependencies)
        print("??ëª¨ë¸ ë¡œë“œ ?±ê³µ!")
    except Exception as e:
        print(f"??ëª¨ë¸ ë¡œë“œ ?¤íŒ¨: {e}")
        return
    
    # 2. ?”ë? ?°ì´???ì„± (?¤ì œë¡œëŠ” ?¤ì œ PPG ? í˜¸ë¥??¬ìš©?´ì•¼ ?©ë‹ˆ??
    print("\n2?¨ê³„: ?ŒìŠ¤???°ì´??ì¤€ë¹?)
    # ?•íƒœ: (?˜í”Œ ?? 875, 1) - 875??7ì´?* 125Hz
    dummy_ppg_signal = np.random.randn(5, 875, 1).astype(np.float32)
    print(f"?ŒìŠ¤???°ì´???•íƒœ: {dummy_ppg_signal.shape}")
    
    # 3. ?ˆì•• ?ˆì¸¡
    print("\n3?¨ê³„: ?ˆì•• ?ˆì¸¡")
    predictions = model.predict(dummy_ppg_signal, verbose=0)
    
    # 4. ê²°ê³¼ ì¶œë ¥ (ëª¨ë¸ ì¶œë ¥ ?•íƒœ???°ë¼ ì²˜ë¦¬)
    print("\n4?¨ê³„: ?ˆì¸¡ ê²°ê³¼")
    print("="*80)
    
    # ê²½ìš° 1: ?¨ì¼ ?ì„œ ì¶œë ¥ (shape: (batch, 2))
    if hasattr(predictions, 'shape') and predictions.ndim >= 2 and predictions.shape[-1] == 2:
        for i in range(predictions.shape[0]):
            sbp = float(predictions[i, 0])
            dbp = float(predictions[i, 1])
            print(f"?˜í”Œ {i+1}: SBP = {sbp:.1f} mmHg, DBP = {dbp:.1f} mmHg")
    # ê²½ìš° 2: ??ê°œì˜ ì¶œë ¥ ë¦¬ìŠ¤??([sbp_batch, dbp_batch])
    elif isinstance(predictions, (list, tuple)) and len(predictions) == 2:
        sbp_batch, dbp_batch = predictions
        # ê°?ë°°ì¹˜??ì°¨ì› ?•ë¦¬ (?? (batch, 1) -> (batch,))
        sbp_batch = np.squeeze(sbp_batch)
        dbp_batch = np.squeeze(dbp_batch)
        for i in range(len(sbp_batch)):
            sbp = float(sbp_batch[i])
            dbp = float(dbp_batch[i])
            print(f"?˜í”Œ {i+1}: SBP = {sbp:.1f} mmHg, DBP = {dbp:.1f} mmHg")
    else:
        print(f"?ˆìƒ?˜ì? ëª»í•œ ì¶œë ¥ ?•íƒœ: {type(predictions)}")
        try:
            print(f"ì¶œë ¥ ?”ì•½: shape={getattr(predictions, 'shape', None)}; len={len(predictions) if hasattr(predictions, '__len__') else 'N/A'}")
        except Exception:
            pass
    print("="*80)
    
    print("\nì°¸ê³ : ???ˆì¸¡ ê°’ì? ?œë¤ ?°ì´?°ë¡œ ?ŒìŠ¤?¸í•œ ê²ƒì´ë¯€ë¡??˜ë?ê°€ ?†ìŠµ?ˆë‹¤.")
    print("?¤ì œ PPG ? í˜¸ë¥??¬ìš©?˜ë ¤ë©?test_model.py ?¤í¬ë¦½íŠ¸ë¥??¬ìš©?˜ì„¸??")


if __name__ == '__main__':
    simple_test()
