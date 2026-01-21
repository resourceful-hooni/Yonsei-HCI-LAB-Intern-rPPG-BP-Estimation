# 호환성 문제 자동 수정 스크립트
# TensorFlow 2.13 + MediaPipe 0.10.5 기반

Write-Host "=" -NoNewline
Write-Host ("="*69)
Write-Host "환경 호환성 자동 수정 스크립트"
Write-Host "=" -NoNewline
Write-Host ("="*69)
Write-Host ""

# 현재 환경 백업
Write-Host "[1/6] 현재 패키지 목록 백업 중..." -ForegroundColor Yellow
pip freeze > requirements_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt
Write-Host "✓ 백업 완료" -ForegroundColor Green
Write-Host ""

# 충돌하는 패키지 제거
Write-Host "[2/6] 충돌 패키지 제거 중..." -ForegroundColor Yellow
Write-Host "  - TensorFlow 제거..."
pip uninstall tensorflow tensorflow-gpu tensorflow-intel tensorflow-estimator tensorboard -y 2>&1 | Out-Null
Write-Host "  - 의존성 패키지 제거..."
pip uninstall numpy absl-py flatbuffers -y 2>&1 | Out-Null
Write-Host "✓ 제거 완료" -ForegroundColor Green
Write-Host ""

# 핵심 패키지 설치 (호환 버전)
Write-Host "[3/6] 호환 패키지 설치 중..." -ForegroundColor Yellow

$packages = @(
    "tensorflow==2.13.0",
    "numpy==1.23.5",
    "protobuf==3.20.3",
    "absl-py==1.4.0",
    "flatbuffers==23.5.26"
)

foreach ($pkg in $packages) {
    Write-Host "  - $pkg 설치 중..."
    pip install $pkg --no-deps 2>&1 | Out-Null
}

Write-Host "✓ 핵심 패키지 설치 완료" -ForegroundColor Green
Write-Host ""

# MediaPipe 재설치
Write-Host "[4/6] MediaPipe 재설치 중..." -ForegroundColor Yellow
pip install mediapipe==0.10.5 --force-reinstall 2>&1 | Out-Null
Write-Host "✓ MediaPipe 설치 완료" -ForegroundColor Green
Write-Host ""

# 기존 requirements.txt 패키지 재설치
Write-Host "[5/6] 기존 프로젝트 패키지 재설치 중..." -ForegroundColor Yellow
$requirements = @(
    "h5py==2.10.0",
    "heartpy==1.2.7",
    "wfdb==3.4.0",
    "kapre==0.3.5",
    "natsort==7.1.1",
    "pandas==1.3.1",
    "matplotlib==3.4.2",
    "scikit-learn==0.24.2",
    "opencv-python==4.8.0.76",
    "scipy"
)

foreach ($pkg in $requirements) {
    try {
        Write-Host "  - $pkg"
        pip install $pkg 2>&1 | Out-Null
    } catch {
        Write-Host "    ⚠️ $pkg 설치 실패 (건너뜀)" -ForegroundColor Yellow
    }
}

Write-Host "✓ 프로젝트 패키지 설치 완료" -ForegroundColor Green
Write-Host ""

# 검증
Write-Host "[6/6] 설치 검증 중..." -ForegroundColor Yellow
Write-Host ""

$tf_version = python -c "import tensorflow as tf; print(tf.__version__)" 2>&1
$np_version = python -c "import numpy as np; print(np.__version__)" 2>&1
$mp_check = python -c "import mediapipe; print('OK')" 2>&1

Write-Host "  TensorFlow: $tf_version" -ForegroundColor Cyan
Write-Host "  NumPy: $np_version" -ForegroundColor Cyan
Write-Host "  MediaPipe: $mp_check" -ForegroundColor Cyan
Write-Host ""

# 최종 충돌 확인
Write-Host "의존성 충돌 확인 중..."
$conflicts = pip check 2>&1 | Select-String -Pattern "has requirement" -Context 0
if ($conflicts.Count -eq 0) {
    Write-Host "✓ 의존성 충돌 없음!" -ForegroundColor Green
} else {
    Write-Host "⚠️ 일부 경고 발견:" -ForegroundColor Yellow
    $conflicts | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
}

Write-Host ""
Write-Host "=" -NoNewline
Write-Host ("="*69)
Write-Host "✓ 호환성 수정 완료!" -ForegroundColor Green
Write-Host "=" -NoNewline
Write-Host ("="*69)
Write-Host ""
Write-Host "다음 명령어로 테스트하세요:"
Write-Host "  python test_mediapipe.py" -ForegroundColor Cyan
Write-Host "  python camera_rppg_advanced.py --model models/transformer_bp_model.h5 --camera 1" -ForegroundColor Cyan
Write-Host ""
