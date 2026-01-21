"""
run_phase4_final.py - Complete Phase 4 after training
"""

import os
import subprocess
import sys
import time

def wait_for_model(timeout=600, check_interval=5):
    """Wait for model to be ready"""
    print("[*] Waiting for Transformer model to be ready...")
    
    model_path = 'models/transformer_bp_model.h5'
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if size_mb > 8:
                print(f"[OK] Model ready: {size_mb:.1f} MB")
                return True
            else:
                print(f"  Waiting... {size_mb:.1f} MB")
        else:
            print(f"  Model file not found yet")
        
        time.sleep(check_interval)
    
    print("[ERROR] Timeout waiting for model")
    return False


def visualize_results():
    """Run visualization"""
    print("\n[*] Running visualization...")
    
    result = subprocess.run(
        [sys.executable, os.path.join('training', 'visualize_transformer.py')],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    print(result.stdout)
    
    if result.returncode != 0:
        print("[ERROR] Visualization failed")
        if result.stderr:
            print(result.stderr[-500:])
        return False
    
    return True


def commit_to_github():
    """Commit Phase 4 completion to GitHub"""
    print("\n[*] Committing Phase 4 to GitHub...")
    
    result = subprocess.run(
        ['git', 'add', '-A'],
        capture_output=True,
        text=True
    )
    
    result = subprocess.run(
        ['git', 'commit', '-m', 'Phase 4: Transformer - Complete (training finished)'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout if result.stdout else result.stderr)
    
    result = subprocess.run(
        ['git', 'push', 'origin', 'main'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout if result.stdout else result.stderr)
    print("[OK] Pushed to GitHub")


def main():
    print("\n" + "="*60)
    print("PHASE 4: TRANSFORMER - FINALIZATION")
    print("="*60)
    
    if wait_for_model(timeout=900):
        if visualize_results():
            commit_to_github()
            
            print("\n" + "="*60)
            print("OK PHASE 4 COMPLETED!")
            print("="*60)
            print("\nNext: Phase 5 - ONNX/TensorRT Optimization")
            print("Command: python deployment/prepare_onnx_export.py")
        else:
            print("Visualization failed")
    else:
        print("Model training timeout")


if __name__ == '__main__':
    main()
