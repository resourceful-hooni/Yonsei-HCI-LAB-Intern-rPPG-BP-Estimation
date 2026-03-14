# VisiVital 구현/검증 종합 문서 (2026-02-17)

## 1) 프로젝트 개요

VisiVital은 rPPG 기반 혈압/혈당 참고 모니터링 웹앱으로, 아래 구조로 구현됨.

- Backend: Flask API, SQLite DB, rPPG 신호처리, BP/Glucose 추정
- Frontend: React, 측정/요약/생활관리 화면
- Security: API Key, Rate Limit, Request Size 제한

현재 사용자 요청에 따라 **로그인은 `demo-user` 고정**, **혈압은 연구 모델 우선**, **혈당은 더미 데이터 사용**으로 동작함.

---

## 2) 이번 수정 핵심 (이슈 대응)

### A. 측정 시 `Payload Too Large` 에러 수정

원인:
- `300` 프레임을 원본 해상도에 가깝게 base64 전송하면서 요청 바디가 제한(`MAX_CONTENT_LENGTH`)을 초과.

수정:
1. 프론트 캡처 전송량 축소
   - 캡처 프레임 리사이즈: `320px` 폭
   - JPEG 품질: `0.55`
   - 카메라 ideal 해상도: `640x480`
2. 백엔드 허용 용량 상향
   - `MAX_CONTENT_LENGTH=80` MB
3. 413 에러 UX 개선
   - 프론트에서 413을 사용자 안내 문구로 변환

적용 파일:
- frontend: `src/components/Measurement/CameraView.jsx`
- frontend: `src/services/apiService.js`
- backend/env: `.env`
- backend/config: `backend/config.py`

### B. 혈당은 더미 데이터 모드로 전환

요구사항 반영:
- 혈당은 모델 기반 실측이 아니라, 더미 데이터로 제공.

수정:
- `USE_DUMMY_GLUCOSE=True` 설정 추가
- `BloodGlucoseEstimator`에 더미 모드 구현
- 결과 JSON에 `glucose_source` 추가

적용 파일:
- backend: `backend/models/glucose_model.py`
- backend: `backend/routes/measurement.py`
- backend/config/env:
  - `backend/config.py`
  - `.env`
- frontend 결과 표시:
  - `frontend/src/components/Measurement/ResultDisplay.jsx`

---

## 3) 주요 아키텍처/구현 상태

### 3.1 Backend

#### 서버/설정
- Flask 앱 생성, CORS, 에러 핸들러(400/404/500), Health endpoint 구현.
- 고정 사용자: `DEMO_USER_ID=demo-user`.

#### 측정 API
- `POST /api/measurement/start`
  - 요청 user_id 무시, 내부적으로 demo-user 고정.
- `POST /api/measurement/process`
  - base64 frame 디코딩 → rPPG 처리 → BP/Glucose 산출 → DB 저장.
  - 응답에 `bp_source`, `glucose_source` 포함.
- `GET /api/measurement/result/<id>`

#### 요약/추세 API
- `GET /api/summary/daily`
- `GET /api/summary/trends`

#### 생활관리 API
- `GET /api/lifestyle/comparison`
- `GET /api/lifestyle/recommendations`

#### 모델
- BP:
  - 연구 `.h5` 모델 브릿지 우선 사용 (`research_model`)
  - 실패 시 경험식 폴백 (`empirical_fallback`)
- Glucose:
  - 현재 설정상 더미 모드 (`dummy`)

#### 보안
- API Key 인증
- 분당 rate limit
- 요청 사이즈 제한 (`MAX_CONTENT_LENGTH`)

### 3.2 Database

#### 스키마
- `users`
- `measurements`
- `status_history`
- 인덱스: `idx_measurements_user_time`

#### DB 매니저 제공 기능
- DB 초기화
- 사용자 보장(`ensure_user`)
- 측정 저장/최신/최근/평균/ID 조회

### 3.3 Frontend

#### 페이지/네비
- 측정, 요약, 생활관리 라우팅
- 상단 탭 네비 + 현대적 UI 스타일

#### 측정 화면
- 카메라 접근
- 얼굴 감지 오버레이
- 10초 프레임 수집/진행률
- 압축 전송 및 에러 가이드 문구

#### 결과 화면
- 혈압/혈당/신뢰도 표시
- 혈압 소스(`research_model` 등), 혈당 소스(`dummy`) 표시

#### 요약/생활관리
- 상태 라벨/요약 텍스트
- 추세 차트(Recharts)
- 비교 차트/습관 카드/모달

---

## 4) 실행/검증 결과

### 4.1 백엔드 통합 검증 (자동 스모크)
검증 항목 모두 PASS:
- `demo_user_fixed`
- `dummy_glucose_enabled`
- `max_content_len_80mb`
- `health_200`
- `measurement_start_demo_user`
- DB CRUD/평균/스키마 존재
- `bp_source_available`
- `glucose_source_dummy`
- `summary_daily_200`
- `summary_trends_200`
- `lifestyle_comparison_200`
- `lifestyle_recommendations_200`

### 4.2 프론트 검증
- `npm install` 완료
- `npm run build` 성공
- 런타임 경고(react-scripts/webpack deprecation)는 기능 동작에는 영향 없음

---

## 5) 환경 고정값

`.env` 기준:
- `DEMO_USER_ID=demo-user`
- `USE_DUMMY_GLUCOSE=True`
- `MAX_CONTENT_LENGTH=80`
- `RESEARCH_BP_MODEL_PATH=.../resnet_rppg_adapted.h5`
- `RESEARCH_SCALER_INFO_PATH=.../rppg_info.txt`
- `RESEARCH_MODEL_TARGET_LEN=875`

---

## 6) 현재 동작 요약

1. 측정 시작 시 항상 demo-user 세션 생성
2. 프론트는 압축된 프레임(320px, jpeg 0.55) 300장 전송
3. 백엔드는 rPPG → BP(연구 모델 우선) + Glucose(더미) 계산
4. 결과 DB 저장 후 요약/추세/생활관리 API에서 재활용

---

## 7) 남은 개선 권장 사항 (선택)

- 측정 업로드 구조를 단일 대용량 JSON 대신 chunk 업로드로 개선
- Glucose 더미를 사용자 시나리오별 프로파일 더미(안정형/변동형)로 확장
- `status_history` 자동 기록 로직 추가
- 프론트 e2e 테스트(Playwright/Cypress) 자동화

---

## 8) 참고

본 서비스는 의료인의 진단/치료를 대체하지 않으며, 자가 건강관리를 위한 참고 정보 제공 목적임.
