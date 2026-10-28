# VisiVital 배포 번들 가이드

## 포함 파일
- backend/: Flask API 서버 (얼굴 랜드마크 모델 `models/assets/face_landmarker.task` 포함)
- frontend/: React 앱 소스(Vite) + build
- docs/: 구현 문서
- .env.example: 환경변수 예시
- RUN_BACKEND.bat / RUN_FRONTEND.bat: 실행 스크립트 (버전이 낮으면 안내 메시지를 띄우고 종료)

## 타 PC에서 실행
1) Python 3.12 이상 설치 (설치 시 "Add python.exe to PATH" 체크)
2) Node.js 20.19 이상 설치 (LTS 권장)
3) backend 의존성 설치
   - `cd backend`
   - `pip install -r requirements.txt`
   - 이전 버전을 설치했던 환경이면 먼저 `pip uninstall opencv-python` (MediaPipe가 쓰는 `opencv-contrib-python`과 충돌)
   - 연구용 BP 모델(.h5)을 쓸 때만 추가로 `pip install -r requirements-research.txt` (TensorFlow)
4) frontend 의존성 설치
   - `cd ../frontend`
   - `npm install` (RUN_FRONTEND.bat이 처음 실행 시 자동으로 해줍니다)
5) 루트의 `.env.example`을 `.env`로 복사 후 경로 수정
6) 서버 실행
   - 백엔드: RUN_BACKEND.bat
   - 프론트: RUN_FRONTEND.bat (Vite 개발 서버, `/api`는 백엔드 5000번으로 자동 프록시)

## 체크 포인트
- 백엔드 헬스: http://127.0.0.1:5000/api/health
- 프론트: http://127.0.0.1:3000

## 주의
- API_KEY는 프론트/백엔드가 동일해야 합니다. 프론트 키는 `frontend/.env`의 `REACT_APP_API_KEY`로 지정합니다(미지정 시 기본값끼리 일치).
- `RESEARCH_BP_MODEL_PATH`는 대상 PC 실제 경로로 수정해야 합니다. 파일이 없으면 기본 추정식으로 동작합니다.
