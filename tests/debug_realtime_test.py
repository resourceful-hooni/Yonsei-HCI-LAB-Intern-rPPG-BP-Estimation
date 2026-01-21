"""
?¤ì‹œê°??Œì´?„ë¼???”ë²„ê¹??ŒìŠ¤??
BP?€ HR ì¶œë ¥ ?•ìƒ??ê²€ì¦?
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from time import perf_counter as timer
from realtime.integrated_pipeline import IntegratedRPPGPipeline

# ?•ìƒ ë²”ìœ„ ?•ì˜
BP_RANGES = {
    'sbp': (70, 200),   # ?˜ì¶•ê¸??ˆì••
    'dbp': (40, 130),   # ?´ì™„ê¸??ˆì••
}
HR_RANGE = (40, 180)    # ?¬ë°•??

class DebugMonitor:
    """?¤ì‹œê°??”ë²„ê¹?ëª¨ë‹ˆ??""
    
    def __init__(self):
        self.frame_count = 0
        self.face_detected_count = 0
        self.prediction_count = 0
        self.bp_history = {'sbp': [], 'dbp': []}
        self.hr_history = []
        self.quality_history = []
        self.stage_times_history = {
            'detection': [],
            'roi': [],
            'pos': [],
            'quality': [],
            'preprocessing': [],
            'inference': [],
            'postprocessing': []
        }
        self.anomalies = []
    
    def check_bp_validity(self, sbp, dbp):
        """BP ê°?? íš¨??ê²€??""
        issues = []
        
        # ë²”ìœ„ ì²´í¬
        if not (BP_RANGES['sbp'][0] <= sbp <= BP_RANGES['sbp'][1]):
            issues.append(f"SBP ë²”ìœ„ ?´íƒˆ: {sbp:.1f} (?•ìƒ: {BP_RANGES['sbp']})")
        
        if not (BP_RANGES['dbp'][0] <= dbp <= BP_RANGES['dbp'][1]):
            issues.append(f"DBP ë²”ìœ„ ?´íƒˆ: {dbp:.1f} (?•ìƒ: {BP_RANGES['dbp']})")
        
        # ?ë¦¬?™ì  ?€?¹ì„± (SBP > DBP)
        if sbp <= dbp:
            issues.append(f"?ë¦¬?™ì  ?¤ë¥˜: SBP({sbp:.1f}) <= DBP({dbp:.1f})")
        
        # ë§¥ì•• ì²´í¬ (?•ìƒ: 30-50 mmHg)
        pulse_pressure = sbp - dbp
        if pulse_pressure < 20 or pulse_pressure > 80:
            issues.append(f"ë§¥ì•• ë¹„ì •?? {pulse_pressure:.1f} mmHg (?•ìƒ: 30-50)")
        
        return issues
    
    def check_hr_validity(self, hr):
        """HR ê°?? íš¨??ê²€??""
        issues = []
        
        if not (HR_RANGE[0] <= hr <= HR_RANGE[1]):
            issues.append(f"HR ë²”ìœ„ ?´íƒˆ: {hr:.1f} (?•ìƒ: {HR_RANGE})")
        
        return issues
    
    def check_stability(self):
        """ê°??ˆì •??ê²€??""
        issues = []
        
        # SBP ë³€?™ì„±
        if len(self.bp_history['sbp']) >= 3:
            recent_sbp = self.bp_history['sbp'][-3:]
            std_sbp = np.std(recent_sbp)
            if std_sbp > 20:
                issues.append(f"SBP ë³€??ê³¼ë‹¤: std={std_sbp:.1f} mmHg")
        
        # DBP ë³€?™ì„±
        if len(self.bp_history['dbp']) >= 3:
            recent_dbp = self.bp_history['dbp'][-3:]
            std_dbp = np.std(recent_dbp)
            if std_dbp > 15:
                issues.append(f"DBP ë³€??ê³¼ë‹¤: std={std_dbp:.1f} mmHg")
        
        # HR ë³€?™ì„±
        if len(self.hr_history) >= 3:
            recent_hr = self.hr_history[-3:]
            std_hr = np.std(recent_hr)
            if std_hr > 30:
                issues.append(f"HR ë³€??ê³¼ë‹¤: std={std_hr:.1f} bpm")
        
        return issues
    
    def log_pipeline_status(self, status, results=None):
        """?Œì´?„ë¼???íƒœ ë¡œê¹…"""
        self.frame_count += 1
        
        # ?¼êµ´ ê°ì? ?µê³„
        if status['face_detected']:
            self.face_detected_count += 1
        
        detection_rate = (self.face_detected_count / self.frame_count) * 100
        
        print(f"\n{'='*70}")
        print(f"Frame #{self.frame_count} | Detection Rate: {detection_rate:.1f}%")
        print(f"{'='*70}")
        
        # Stage 1: Face Detection
        if status['face_detected']:
            bbox = status['bbox']
            print(f"??[Stage 1] Face Detected: {bbox}")
            if status['bbox_filtered']:
                print(f"  ??Filtered BBox: {status['bbox_filtered']}")
        else:
            print(f"??[Stage 1] No Face Detected")
            return
        
        # ?ˆì¸¡ ê²°ê³¼ê°€ ?ˆìœ¼ë©??ì„¸ ë¶„ì„
        if results:
            self.prediction_count += 1
            
            print(f"\n{'?€'*70}")
            print(f"?ˆì¸¡ #{self.prediction_count}")
            print(f"{'?€'*70}")
            
            # BP ê²€ì¦?
            sbp = results['sbp']
            dbp = results['dbp']
            hr = results['hr']
            quality = results['quality_score']
            confidence = results['confidence']
            sbp_raw = results.get('sbp_raw', sbp)
            dbp_raw = results.get('dbp_raw', dbp)
            sbp_raw_model = results.get('sbp_raw_model', sbp_raw)
            dbp_raw_model = results.get('dbp_raw_model', dbp_raw)
            stab = results.get('stabilization', {})
            
            print(f"\n[BP ê²°ê³¼]")
            print(f"  SBP: {sbp:.1f} mmHg")
            print(f"  DBP: {dbp:.1f} mmHg")
            print(f"  ë§¥ì••: {sbp - dbp:.1f} mmHg")
            print(f"  Raw (after inverse): {sbp_raw:.3f} / {dbp_raw:.3f}")
            print(f"  Raw (model output):  {sbp_raw_model:.6f} / {dbp_raw_model:.6f}")
            if stab:
                print(f"  Stabilization: method={stab.get('method')} sbp_outlier={stab.get('sbp_outlier')} dbp_outlier={stab.get('dbp_outlier')}")
            
            bp_issues = self.check_bp_validity(sbp, dbp)
            if bp_issues:
                print(f"  ? ï¸  BP ë¬¸ì œ:")
                for issue in bp_issues:
                    print(f"     - {issue}")
                    self.anomalies.append(f"Frame {self.frame_count}: {issue}")
            else:
                print(f"  ??BP ?•ìƒ ë²”ìœ„")
            
            # HR ê²€ì¦?
            print(f"\n[HR ê²°ê³¼]")
            print(f"  HR: {hr:.1f} bpm")
            
            hr_issues = self.check_hr_validity(hr)
            if hr_issues:
                print(f"  ? ï¸  HR ë¬¸ì œ:")
                for issue in hr_issues:
                    print(f"     - {issue}")
                    self.anomalies.append(f"Frame {self.frame_count}: {issue}")
            else:
                print(f"  ??HR ?•ìƒ ë²”ìœ„")
            
            # ? í˜¸ ?ˆì§ˆ
            print(f"\n[? í˜¸ ?ˆì§ˆ]")
            print(f"  Quality Score: {quality:.3f}")
            print(f"  Confidence: {confidence:.3f}")
            
            if quality < 0.5:
                print(f"  ? ï¸  ? í˜¸ ?ˆì§ˆ ??Œ")
                self.anomalies.append(f"Frame {self.frame_count}: ??? ?ˆì§ˆ ({quality:.3f})")
            
            if 'quality_metrics' in results:
                metrics = results['quality_metrics']
                print(f"  SNR: {metrics.get('snr', 0):.2f} dB")
                print(f"  Peak Regularity: {metrics.get('peak_regularity', 0):.3f}")
            
            # ?€?´ë° ë¶„ì„
            if 'timings' in results:
                print(f"\n[?Œì´?„ë¼???€?´ë°]")
                timings = results['timings']
                total_time = 0
                for stage, timing in timings.items():
                    mean_time = timing['mean']
                    total_time += mean_time
                    print(f"  {stage:15s}: {mean_time:6.2f} ms")
                    
                    # ?€?´ë° ?´ìƒ ê°ì?
                    if stage == 'detection' and mean_time > 50:
                        self.anomalies.append(f"Frame {self.frame_count}: ?ë¦° ?¼êµ´ ê°ì? ({mean_time:.2f}ms)")
                    elif stage == 'inference' and mean_time > 100:
                        self.anomalies.append(f"Frame {self.frame_count}: ?ë¦° ì¶”ë¡  ({mean_time:.2f}ms)")
                
                print(f"  {'TOTAL':15s}: {total_time:6.2f} ms")
                print(f"  ?ˆìƒ FPS: {1000/total_time:.1f}")
            
            # ?ˆìŠ¤? ë¦¬ ?…ë°?´íŠ¸
            self.bp_history['sbp'].append(sbp)
            self.bp_history['dbp'].append(dbp)
            self.hr_history.append(hr)
            self.quality_history.append(quality)
            
            # ?ˆì •??ê²€??
            stability_issues = self.check_stability()
            if stability_issues:
                print(f"\n[?ˆì •??ê²½ê³ ]")
                for issue in stability_issues:
                    print(f"  ? ï¸  {issue}")
                    self.anomalies.append(f"Frame {self.frame_count}: {issue}")
    
    def print_summary(self):
        """ìµœì¢… ?”ì•½ ì¶œë ¥"""
        print(f"\n{'='*70}")
        print(f"?ŒìŠ¤???”ì•½")
        print(f"{'='*70}")
        
        print(f"\n[?µê³„]")
        print(f"  ì´??„ë ˆ?? {self.frame_count}")
        print(f"  ?¼êµ´ ê°ì?: {self.face_detected_count} ({self.face_detected_count/max(self.frame_count,1)*100:.1f}%)")
        print(f"  ?ˆì¸¡ ?Ÿìˆ˜: {self.prediction_count}")
        
        if self.bp_history['sbp']:
            print(f"\n[BP ?µê³„]")
            print(f"  SBP: {np.mean(self.bp_history['sbp']):.1f} Â± {np.std(self.bp_history['sbp']):.1f} mmHg")
            print(f"       ë²”ìœ„: {np.min(self.bp_history['sbp']):.1f} - {np.max(self.bp_history['sbp']):.1f} mmHg")
            print(f"  DBP: {np.mean(self.bp_history['dbp']):.1f} Â± {np.std(self.bp_history['dbp']):.1f} mmHg")
            print(f"       ë²”ìœ„: {np.min(self.bp_history['dbp']):.1f} - {np.max(self.bp_history['dbp']):.1f} mmHg")
        
        if self.hr_history:
            print(f"\n[HR ?µê³„]")
            print(f"  HR: {np.mean(self.hr_history):.1f} Â± {np.std(self.hr_history):.1f} bpm")
            print(f"      ë²”ìœ„: {np.min(self.hr_history):.1f} - {np.max(self.hr_history):.1f} bpm")
        
        if self.quality_history:
            print(f"\n[?ˆì§ˆ ?µê³„]")
            print(f"  ?‰ê·  ?ˆì§ˆ: {np.mean(self.quality_history):.3f}")
            print(f"  ?ˆì§ˆ ë²”ìœ„: {np.min(self.quality_history):.3f} - {np.max(self.quality_history):.3f}")
        
        if self.anomalies:
            print(f"\n[ê°ì????´ìƒ ({len(self.anomalies)}ê°?]")
            for anomaly in self.anomalies[-10:]:  # ìµœê·¼ 10ê°œë§Œ
                print(f"  ? ï¸  {anomaly}")
            if len(self.anomalies) > 10:
                print(f"  ... ??{len(self.anomalies) - 10}ê°?)
        else:
            print(f"\n???´ìƒ ?†ìŒ")
        
        print(f"\n{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='?¤ì‹œê°??Œì´?„ë¼???”ë²„ê¹?)
    parser.add_argument('--camera', type=int, default=1, help='ì¹´ë©”???¸ë±??)
    parser.add_argument('--model', type=str, default='data/transformer_bp_model.h5',
                       help='ëª¨ë¸ ê²½ë¡œ')
    parser.add_argument('--duration', type=int, default=30,
                       help='?ŒìŠ¤???œê°„ (ì´?')
    args = parser.parse_args()
    
    print("="*70)
    print("?¤ì‹œê°??Œì´?„ë¼???”ë²„ê¹??ŒìŠ¤??)
    print("="*70)
    print(f"ëª¨ë¸: {args.model}")
    print(f"ì¹´ë©”?? {args.camera}")
    print(f"?ŒìŠ¤???œê°„: {args.duration}ì´?)
    print("="*70)
    
    # ?Œì´?„ë¼??ì´ˆê¸°??
    pipeline = IntegratedRPPGPipeline(args.model)
    monitor = DebugMonitor()
    
    # ì¹´ë©”???´ê¸°
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"??ì¹´ë©”??{args.camera} ?´ê¸° ?¤íŒ¨")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n??ì¹´ë©”??ì¤€ë¹??„ë£Œ")
    print("Press 'q' to quit\n")
    
    start_time = timer()
    last_result = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # ?œê°„ ?œí•œ
            if timer() - start_time > args.duration:
                print("\n?±ï¸  ?ŒìŠ¤???œê°„ ì¢…ë£Œ")
                break
            
            # ?„ë ˆ??ì²˜ë¦¬
            status = pipeline.process_frame(frame)
            
            # ?ˆì¸¡ ì¤€ë¹„ë˜ë©??¤í–‰
            results = None
            if status['signal_collected']:
                results = pipeline.extract_and_predict()
                pipeline.reset()
            
            # ?”ë²„ê¹?ë¡œê·¸ (?ˆì¸¡???ˆì„ ?Œë§Œ ?ì„¸)
            if results or monitor.frame_count % 30 == 0:  # 30 ?„ë ˆ?„ë§ˆ???ëŠ” ?ˆì¸¡??
                monitor.log_pipeline_status(status, results)
                last_result = results
            
            # ?œê°??
            vis_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # ?¼êµ´ ë°•ìŠ¤
            if status['face_detected']:
                bbox = status['bbox_filtered'] if status['bbox_filtered'] else status['bbox']
                if bbox:
                    x, y, w_box, h_box = bbox
                    cv2.rectangle(vis_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            
            # ?•ë³´ ?¨ë„
            panel = np.zeros((120, w, 3), dtype=np.uint8)
            
            # ?„ë¡œê·¸ë ˆ??
            progress = len(pipeline.frame_buffer) / pipeline.window_size
            cv2.rectangle(panel, (10, 10), (w - 10, 25), (50, 50, 50), -1)
            cv2.rectangle(panel, (10, 10), (int(10 + (w - 20) * progress), 25), (0, 255, 0), -1)
            
            # ê²°ê³¼ ?œì‹œ
            if last_result:
                cv2.putText(panel, f"BP: {last_result['sbp']:.0f}/{last_result['dbp']:.0f} mmHg",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(panel, f"HR: {last_result['hr']:.0f} bpm",
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(panel, f"Q: {last_result['quality_score']:.2f} C: {last_result['confidence']:.2f}",
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            combined = np.vstack([vis_frame, panel])
            cv2.imshow('Debug Monitor', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n?›‘ ?¬ìš©??ì¤‘ë‹¨")
                break
    
    except KeyboardInterrupt:
        print("\n?›‘ ?¸í„°?½íŠ¸")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # ìµœì¢… ?”ì•½
        monitor.print_summary()


if __name__ == '__main__':
    main()
