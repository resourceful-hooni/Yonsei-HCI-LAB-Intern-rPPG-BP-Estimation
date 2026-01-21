"""
Production-Ready Real-Time BP Monitoring
========================================

Integrated pipeline with MediaPipe Face Detection
Optimized for 30 FPS real-time performance
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import argparse
from .integrated_pipeline import IntegratedRPPGPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Integrated rPPG BP Monitoring with MediaPipe'
    )
    parser.add_argument('--model', type=str, default='data/transformer_bp_model.h5',
                       help='Model path')
    parser.add_argument('--camera', type=int, default=1, help='Camera index')
    parser.add_argument('--enable-bbox-filter', action='store_true', default=True,
                       help='Enable bounding box Kalman filtering')
    parser.add_argument('--no-bbox-filter', dest='enable_bbox_filter', action='store_false',
                       help='Disable bounding box filtering')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Integrated rPPG BP Monitoring - Production Mode")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Camera: {args.camera}")
    print(f"Bounding Box Filter: {'Enabled' if args.enable_bbox_filter else 'Disabled'}")
    print("="*70 + "\n")
    
    # Initialize pipeline
    pipeline = IntegratedRPPGPipeline(
        model_path=args.model,
        enable_bbox_filter=args.enable_bbox_filter
    )
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}, trying camera 0...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera ready")
    print("\nPress 'q' to quit, 'r' to reset\n")
    
    # State variables
    last_results = None
    frame_count = 0
    import time
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_fps = frame_count / (time.time() - start_time + 1e-6)
            
            # Process frame
            status = pipeline.process_frame(frame)
            
            # Create visualization
            vis_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Draw face detection
            if status['face_detected']:
                bbox = status['bbox_filtered'] if status['bbox_filtered'] else status['bbox']
                if bbox:
                    x, y, w_box, h_box = bbox
                    
                    # Color based on filter status
                    color = (0, 255, 0) if args.enable_bbox_filter else (255, 0, 0)
                    cv2.rectangle(vis_frame, (x, y), (x + w_box, y + h_box), color, 2)
                    
                    # Label
                    label = "Face (Filtered)" if args.enable_bbox_filter else "Face (Raw)"
                    cv2.putText(vis_frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Info panel
            panel_h = 180
            panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
            
            # Progress bar
            progress = len(pipeline.frame_buffer) / pipeline.window_size
            cv2.rectangle(panel, (10, 10), (w - 10, 30), (50, 50, 50), -1)
            cv2.rectangle(panel, (10, 10), (int(10 + (w - 20) * progress), 30), (0, 255, 0), -1)
            cv2.putText(panel, f"Signal: {progress*100:.0f}%", (15, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # FPS
            cv2.putText(panel, f"FPS: {current_fps:.1f}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Detector type
            detector_type = "MediaPipe" if pipeline.use_mediapipe else "Haar Cascade"
            color = (0, 255, 0) if pipeline.use_mediapipe else (0, 165, 255)
            cv2.putText(panel, f"Detector: {detector_type}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # BP results
            if last_results:
                cv2.putText(panel, f"BP: {last_results['sbp']:.0f}/{last_results['dbp']:.0f} mmHg",
                           (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(panel, f"HR: {last_results['hr']:.0f} bpm",
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(panel, f"Quality: {last_results['quality_score']:.2f}",
                           (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                conf_color = (0, 255, 0) if last_results['confidence'] > 0.7 else (0, 165, 255)
                cv2.putText(panel, f"Confidence: {last_results['confidence']:.2f}",
                           (300, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
            
            # Combine
            combined = np.vstack([vis_frame, panel])
            cv2.imshow('Integrated rPPG BP Monitor', combined)
            
            # Check if ready for prediction
            if status['signal_collected']:
                print("\n✓ Signal collection complete, predicting...")
                results = pipeline.extract_and_predict()
                
                if results:
                    last_results = results
                    
                    print("="*70)
                    print("BP Measurement Results")
                    print("="*70)
                    print(f"  SBP: {results['sbp']:.1f} mmHg")
                    print(f"  DBP: {results['dbp']:.1f} mmHg")
                    print(f"  HR:  {results['hr']:.1f} bpm")
                    print(f"  Quality: {results['quality_score']:.3f}")
                    print(f"  Confidence: {results['confidence']:.3f}")
                    print("="*70)
                    
                    # Performance report
                    perf = pipeline.get_performance_report()
                    print(f"\nPipeline Performance:")
                    print(f"  Total time: {perf['total_ms']:.2f} ms")
                    print(f"  Max FPS: {perf['max_fps']:.1f}")
                    print("\nPress 'c' to continue measuring...\n")
                
                # Reset for next measurement
                pipeline.reset()
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting pipeline...")
                pipeline.reset()
                last_results = None
    
    except KeyboardInterrupt:
        print("\n\nUser interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final performance report
        print("\n" + "="*70)
        print("Session Summary")
        print("="*70)
        perf = pipeline.get_performance_report()
        for stage, timing in perf.items():
            if isinstance(timing, dict):
                print(f"  {stage:15s}: {timing['avg_ms']:.2f} ms")
            else:
                print(f"  {stage:15s}: {timing:.2f}")
        print("="*70)


if __name__ == "__main__":
    main()
