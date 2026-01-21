"""System-level validation script (imports, models, pipeline smoke tests)."""

import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 70)
print("System Validation")
print("=" * 70)

issues = []
warnings = []

# 1) Environment check
print("\n[1/6] Environment check...")
try:
    import tensorflow as tf
    import mediapipe as mp
    import cv2
    from time import perf_counter as timer  # noqa: F401

    print(f"  ✓ TensorFlow: {tf.__version__}")
    print(f"  ✓ MediaPipe: {mp.__version__}")
    print(f"  ✓ NumPy: {np.__version__}")
    print(f"  ✓ OpenCV: {cv2.__version__}")
except Exception as e:  # pragma: no cover - diagnostic
    issues.append(f"Library import failed: {e}")

# 2) Model file presence
print("\n[2/6] Model files...")
model_paths = {
    'Transformer': 'data/transformer_bp_model.h5',
    'Multi-Task': 'models/multi_task_bp_model.h5',
    'ResNet-Adapted': 'models/resnet_rppg_adapted.h5',
    'ResNet-Original': 'data/resnet_ppg_nonmixed.h5',
}
for name, path in model_paths.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  ✓ {name}: {size_mb:.1f} MB ({path})")
    else:
        warnings.append(f"Missing model: {path}")

# 3) Core module imports
print("\n[3/6] Core modules...")
try:
    from realtime.mediapipe_face_detector import MediaPipeFaceDetector
    from realtime.integrated_pipeline import IntegratedRPPGPipeline, timer as pipeline_timer
    from realtime.pos_algorithm import POSExtractor
    from realtime.signal_quality import SignalQualityAssessor
    from realtime.bp_stability import BPStabilizer
    from models.transformer_model import MultiHeadAttention, EncoderLayer, TransformerEncoder

    print("  ✓ Core modules imported")
except Exception as e:  # pragma: no cover - diagnostic
    issues.append(f"Module import failed: {e}")

# 4) MediaPipe configuration
print("\n[4/6] MediaPipe configuration...")
try:
    detector = MediaPipeFaceDetector()
    if getattr(detector, 'use_mediapipe', False):
        print("  ✓ MediaPipe enabled")
        if hasattr(detector, 'min_detection_confidence'):
            print(f"  ✓ min_detection_confidence: {detector.min_detection_confidence}")
    else:
        warnings.append("MediaPipe disabled (Haar fallback in use)")
except Exception as e:  # pragma: no cover - diagnostic
    warnings.append(f"MediaPipe init failed: {e}")

# 5) Pipeline smoke test
print("\n[5/6] Pipeline smoke test...")
try:
    pipeline = IntegratedRPPGPipeline('data/transformer_bp_model.h5')
    print("  ✓ IntegratedRPPGPipeline initialized")
    print(f"  ✓ High-resolution timer: {pipeline_timer.__name__}")

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    status = pipeline.process_frame(dummy)
    print(f"  ✓ Frame processed; face_detected={status.get('face_detected', False)}")
except Exception as e:  # pragma: no cover - diagnostic
    issues.append(f"Pipeline init/process failed: {e}")

# 6) Timing sanity
print("\n[6/6] Timing sanity...")
try:
    from time import perf_counter as timer
    import time  # noqa: F401

    samples = []
    for _ in range(10):
        t0 = timer()
        _ = np.random.rand(100, 100)
        samples.append((timer() - t0) * 1000)  # ms

    avg_ms = float(np.mean(samples)) if samples else 0.0
    print(f"  ✓ perf_counter avg: {avg_ms:.4f} ms")
except Exception as e:  # pragma: no cover - diagnostic
    issues.append(f"Timing check failed: {e}")

# Results
print("\n" + "=" * 70)
print("Validation Summary")
print("=" * 70)

if issues:
    print(f"\n{len(issues)} blocking issue(s):")
    for item in issues:
        print(f"  - {item}")
else:
    print("\nNo blocking issues detected.")

if warnings:
    print(f"\n⚠️  {len(warnings)} warning(s):")
    for item in warnings:
        print(f"  - {item}")
else:
    print("⚠️  No warnings.")

if not issues:
    print("\n" + "=" * 70)
    print("System ready for runtime")
    print("=" * 70)
    print("\nRun:")
    print("  .\\env\\Scripts\\Activate.ps1; python -m realtime.run_integrated_bp_monitor --camera 1")
else:
    print("\nResolve blocking issues before running real-time pipeline.")

print("=" * 70)
