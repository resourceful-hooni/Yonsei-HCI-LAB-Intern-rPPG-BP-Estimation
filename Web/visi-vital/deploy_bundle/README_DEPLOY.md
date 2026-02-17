# VisiVital 배포 번들 가이드

## 포함 파일
- backend/: Flask API 서버
- frontend/: React 앱 소스 + build
- docs/: 구현 문서
- .env.example: 환경변수 예시
- RUN_BACKEND.bat / RUN_FRONTEND.bat: 실행 스크립트

## 타 PC에서 실행
1) Python 3.8+ 설치
2) Node.js 18+ 설치
3) backend 의존성 설치
   - `cd backend`
   - `pip install -r requirements.txt`
4) frontend 의존성 설치
   - `cd ../frontend`
   - `npm install`
5) 루트의 `.env.example`을 `.env`로 복사 후 경로 수정
6) 서버 실행
   - 백엔드: RUN_BACKEND.bat
   - 프론트: RUN_FRONTEND.bat

## 체크 포인트
- 백엔드 헬스: http://127.0.0.1:5000/api/health
- 프론트: http://127.0.0.1:3000

## 주의
- API_KEY는 프론트/백엔드가 동일해야 합니다.
- `RESEARCH_BP_MODEL_PATH`는 대상 PC 실제 경로로 수정해야 합니다.
