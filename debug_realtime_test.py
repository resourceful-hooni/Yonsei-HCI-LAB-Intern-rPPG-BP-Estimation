"""
실시간 파이프라인 디버깅 테스트
BP와 HR 출력 정상성 검증
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from time import perf_counter as timer
from realtime.integrated_pipeline import IntegratedRPPGPipeline

# 정상 범위 정의
BP_RANGES = {
    'sbp': (70, 200),   # 수축기 혈압
    'dbp': (40, 130),   # 이완기 혈압
}
HR_RANGE = (40, 180)    # 심박수

class DebugMonitor:
    """실시간 디버깅 모니터"""
    
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
        """BP 값 유효성 검사"""
        issues = []
        
        # 범위 체크
        if not (BP_RANGES['sbp'][0] <= sbp <= BP_RANGES['sbp'][1]):
            issues.append(f"SBP 범위 이탈: {sbp:.1f} (정상: {BP_RANGES['sbp']})")
        
        if not (BP_RANGES['dbp'][0] <= dbp <= BP_RANGES['dbp'][1]):
            issues.append(f"DBP 범위 이탈: {dbp:.1f} (정상: {BP_RANGES['dbp']})")
        
        # 생리학적 타당성 (SBP > DBP)
        if sbp <= dbp:
            issues.append(f"생리학적 오류: SBP({sbp:.1f}) <= DBP({dbp:.1f})")
        
        # 맥압 체크 (정상: 30-50 mmHg)
        pulse_pressure = sbp - dbp
        if pulse_pressure < 20 or pulse_pressure > 80:
            issues.append(f"맥압 비정상: {pulse_pressure:.1f} mmHg (정상: 30-50)")
        
        return issues
    
    def check_hr_validity(self, hr):
        """HR 값 유효성 검사"""
        issues = []
        
        if not (HR_RANGE[0] <= hr <= HR_RANGE[1]):
            issues.append(f"HR 범위 이탈: {hr:.1f} (정상: {HR_RANGE})")
        
        return issues
    
    def check_stability(self):
        """값 안정성 검사"""
        issues = []
        
        # SBP 변동성
        if len(self.bp_history['sbp']) >= 3:
            recent_sbp = self.bp_history['sbp'][-3:]
            std_sbp = np.std(recent_sbp)
            if std_sbp > 20:
                issues.append(f"SBP 변동 과다: std={std_sbp:.1f} mmHg")
        
        # DBP 변동성
        if len(self.bp_history['dbp']) >= 3:
            recent_dbp = self.bp_history['dbp'][-3:]
            std_dbp = np.std(recent_dbp)
            if std_dbp > 15:
                issues.append(f"DBP 변동 과다: std={std_dbp:.1f} mmHg")
        
        # HR 변동성
        if len(self.hr_history) >= 3:
            recent_hr = self.hr_history[-3:]
            std_hr = np.std(recent_hr)
            if std_hr > 30:
                issues.append(f"HR 변동 과다: std={std_hr:.1f} bpm")
        
        return issues
    
    def log_pipeline_status(self, status, results=None):
        """파이프라인 상태 로깅"""
        self.frame_count += 1
        
        # 얼굴 감지 통계
        if status['face_detected']:
            self.face_detected_count += 1
        
        detection_rate = (self.face_detected_count / self.frame_count) * 100
        
        print(f"\n{'='*70}")
        print(f"Frame #{self.frame_count} | Detection Rate: {detection_rate:.1f}%")
        print(f"{'='*70}")
        
        # Stage 1: Face Detection
        if status['face_detected']:
            bbox = status['bbox']
            print(f"✓ [Stage 1] Face Detected: {bbox}")
            if status['bbox_filtered']:
                print(f"  → Filtered BBox: {status['bbox_filtered']}")
        else:
            print(f"✗ [Stage 1] No Face Detected")
            return
        
        # 예측 결과가 있으면 상세 분석
        if results:
            self.prediction_count += 1
            
            print(f"\n{'─'*70}")
            print(f"예측 #{self.prediction_count}")
            print(f"{'─'*70}")
            
            # BP 검증
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
            
            print(f"\n[BP 결과]")
            print(f"  SBP: {sbp:.1f} mmHg")
            print(f"  DBP: {dbp:.1f} mmHg")
            print(f"  맥압: {sbp - dbp:.1f} mmHg")
            print(f"  Raw (after inverse): {sbp_raw:.3f} / {dbp_raw:.3f}")
            print(f"  Raw (model output):  {sbp_raw_model:.6f} / {dbp_raw_model:.6f}")
            if stab:
                print(f"  Stabilization: method={stab.get('method')} sbp_outlier={stab.get('sbp_outlier')} dbp_outlier={stab.get('dbp_outlier')}")
            
            bp_issues = self.check_bp_validity(sbp, dbp)
            if bp_issues:
                print(f"  ⚠️  BP 문제:")
                for issue in bp_issues:
                    print(f"     - {issue}")
                    self.anomalies.append(f"Frame {self.frame_count}: {issue}")
            else:
                print(f"  ✓ BP 정상 범위")
            
            # HR 검증
            print(f"\n[HR 결과]")
            print(f"  HR: {hr:.1f} bpm")
            
            hr_issues = self.check_hr_validity(hr)
            if hr_issues:
                print(f"  ⚠️  HR 문제:")
                for issue in hr_issues:
                    print(f"     - {issue}")
                    self.anomalies.append(f"Frame {self.frame_count}: {issue}")
            else:
                print(f"  ✓ HR 정상 범위")
            
            # 신호 품질
            print(f"\n[신호 품질]")
            print(f"  Quality Score: {quality:.3f}")
            print(f"  Confidence: {confidence:.3f}")
            
            if quality < 0.5:
                print(f"  ⚠️  신호 품질 낮음")
                self.anomalies.append(f"Frame {self.frame_count}: 낮은 품질 ({quality:.3f})")
            
            if 'quality_metrics' in results:
                metrics = results['quality_metrics']
                print(f"  SNR: {metrics.get('snr', 0):.2f} dB")
                print(f"  Peak Regularity: {metrics.get('peak_regularity', 0):.3f}")
            
            # 타이밍 분석
            if 'timings' in results:
                print(f"\n[파이프라인 타이밍]")
                timings = results['timings']
                total_time = 0
                for stage, timing in timings.items():
                    mean_time = timing['mean']
                    total_time += mean_time
                    print(f"  {stage:15s}: {mean_time:6.2f} ms")
                    
                    # 타이밍 이상 감지
                    if stage == 'detection' and mean_time > 50:
                        self.anomalies.append(f"Frame {self.frame_count}: 느린 얼굴 감지 ({mean_time:.2f}ms)")
                    elif stage == 'inference' and mean_time > 100:
                        self.anomalies.append(f"Frame {self.frame_count}: 느린 추론 ({mean_time:.2f}ms)")
                
                print(f"  {'TOTAL':15s}: {total_time:6.2f} ms")
                print(f"  예상 FPS: {1000/total_time:.1f}")
            
            # 히스토리 업데이트
            self.bp_history['sbp'].append(sbp)
            self.bp_history['dbp'].append(dbp)
            self.hr_history.append(hr)
            self.quality_history.append(quality)
            
            # 안정성 검사
            stability_issues = self.check_stability()
            if stability_issues:
                print(f"\n[안정성 경고]")
                for issue in stability_issues:
                    print(f"  ⚠️  {issue}")
                    self.anomalies.append(f"Frame {self.frame_count}: {issue}")
    
    def print_summary(self):
        """최종 요약 출력"""
        print(f"\n{'='*70}")
        print(f"테스트 요약")
        print(f"{'='*70}")
        
        print(f"\n[통계]")
        print(f"  총 프레임: {self.frame_count}")
        print(f"  얼굴 감지: {self.face_detected_count} ({self.face_detected_count/max(self.frame_count,1)*100:.1f}%)")
        print(f"  예측 횟수: {self.prediction_count}")
        
        if self.bp_history['sbp']:
            print(f"\n[BP 통계]")
            print(f"  SBP: {np.mean(self.bp_history['sbp']):.1f} ± {np.std(self.bp_history['sbp']):.1f} mmHg")
            print(f"       범위: {np.min(self.bp_history['sbp']):.1f} - {np.max(self.bp_history['sbp']):.1f} mmHg")
            print(f"  DBP: {np.mean(self.bp_history['dbp']):.1f} ± {np.std(self.bp_history['dbp']):.1f} mmHg")
            print(f"       범위: {np.min(self.bp_history['dbp']):.1f} - {np.max(self.bp_history['dbp']):.1f} mmHg")
        
        if self.hr_history:
            print(f"\n[HR 통계]")
            print(f"  HR: {np.mean(self.hr_history):.1f} ± {np.std(self.hr_history):.1f} bpm")
            print(f"      범위: {np.min(self.hr_history):.1f} - {np.max(self.hr_history):.1f} bpm")
        
        if self.quality_history:
            print(f"\n[품질 통계]")
            print(f"  평균 품질: {np.mean(self.quality_history):.3f}")
            print(f"  품질 범위: {np.min(self.quality_history):.3f} - {np.max(self.quality_history):.3f}")
        
        if self.anomalies:
            print(f"\n[감지된 이상 ({len(self.anomalies)}개)]")
            for anomaly in self.anomalies[-10:]:  # 최근 10개만
                print(f"  ⚠️  {anomaly}")
            if len(self.anomalies) > 10:
                print(f"  ... 외 {len(self.anomalies) - 10}개")
        else:
            print(f"\n✅ 이상 없음")
        
        print(f"\n{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='실시간 파이프라인 디버깅')
    parser.add_argument('--camera', type=int, default=1, help='카메라 인덱스')
    parser.add_argument('--model', type=str, default='data/transformer_bp_model.h5',
                       help='모델 경로')
    parser.add_argument('--duration', type=int, default=30,
                       help='테스트 시간 (초)')
    parser.add_argument('--skip-preproc', action='store_true',
                       help='전처리(detrend/필터/스무딩) 건너뛰기')
    args = parser.parse_args()
    
    print("="*70)
    print("실시간 파이프라인 디버깅 테스트")
    print("="*70)
    print(f"모델: {args.model}")
    print(f"카메라: {args.camera}")
    print(f"테스트 시간: {args.duration}초")
    print("="*70)
    
    # 파이프라인 초기화
    pipeline = IntegratedRPPGPipeline(
        args.model,
        use_quality_filters=not args.skip_preproc
    )
    monitor = DebugMonitor()
    
    # 카메라 열기
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ 카메라 {args.camera} 열기 실패")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n✓ 카메라 준비 완료")
    print("Press 'q' to quit\n")
    
    start_time = timer()
    last_result = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 시간 제한
            if timer() - start_time > args.duration:
                print("\n⏱️  테스트 시간 종료")
                break
            
            # 프레임 처리
            status = pipeline.process_frame(frame)
            
            # 예측 준비되면 실행
            results = None
            if status['signal_collected']:
                results = pipeline.extract_and_predict()
                pipeline.reset()
            
            # 디버깅 로그 (예측이 있을 때만 상세)
            if results or monitor.frame_count % 30 == 0:  # 30 프레임마다 또는 예측시
                monitor.log_pipeline_status(status, results)
                last_result = results
            
            # 시각화
            vis_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # 얼굴 박스
            if status['face_detected']:
                bbox = status['bbox_filtered'] if status['bbox_filtered'] else status['bbox']
                if bbox:
                    x, y, w_box, h_box = bbox
                    cv2.rectangle(vis_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            
            # 정보 패널
            panel = np.zeros((120, w, 3), dtype=np.uint8)
            
            # 프로그레스
            progress = len(pipeline.frame_buffer) / pipeline.window_size
            cv2.rectangle(panel, (10, 10), (w - 10, 25), (50, 50, 50), -1)
            cv2.rectangle(panel, (10, 10), (int(10 + (w - 20) * progress), 25), (0, 255, 0), -1)
            
            # 결과 표시
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
                print("\n🛑 사용자 중단")
                break
    
    except KeyboardInterrupt:
        print("\n🛑 인터럽트")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # 최종 요약
        monitor.print_summary()


if __name__ == '__main__':
    main()
