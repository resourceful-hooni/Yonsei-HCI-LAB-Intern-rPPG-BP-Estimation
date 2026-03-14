# VisiVital

rPPG 기반 비접촉 건강 모니터링(혈압/혈당 참고값) 웹앱입니다.

## 1) 구현 범위 요약

- 측정 페이지
	- 카메라 기반 10초 측정, MediaPipe 얼굴 감지 오버레이
	- 프레임 압축/다운스케일 전송, 대용량 업로드 대응
	- 실시간 측정 품질 체크리스트(조명/움직임/얼굴정렬)
- 요약 페이지
	- 카드 순서: 혈압 → 혈당 → 신뢰도 추이 → 통합 추이 → 품질 체크리스트
	- 신뢰도 추이 미니차트
	- 혈압/혈당 통합 주간 추이(동시 비교)
	- 체크리스트 출처/점수 표시(실측 우선, 없으면 규칙기반 fallback)
- 생활관리 페이지
	- 오늘 vs 최근 7일 평균 (혈압/혈당)
	- 오늘 한 줄 코멘트
	- 추천 실천률 체크/개인 루틴 목표
	- 주간 변화 전/후 비교
	- 알림 설정
	- 습관 제안 카드/상세 모달
- 백엔드
	- Flask API(측정/요약/생활관리), API Key, Rate limit, 요청 크기 검증
	- SQLite 저장(측정값 + 습관 체크 + 알림 + 품질 지표)
	- 연구 모델 연동(BP), 모델 실패 시 fallback

## 2) 기술 스택

- Backend: Flask, SQLite, NumPy, SciPy, OpenCV, TensorFlow
- Frontend: React, Recharts, MediaPipe FaceMesh
- 환경: Windows 기준 개발/검증

## 3) 실행 방법

### Backend
1. [backend](backend) 이동
2. 가상환경 생성/활성화
3. `pip install -r requirements.txt`
4. 루트 [.env](.env) 설정 확인
5. `python app.py`

### Frontend
1. [frontend](frontend) 이동
2. `npm install`
3. `npm start`

## 4) 환경 변수(.env)

주요 값:
- `DEMO_USER_ID=demo-user` (고정 사용자)
- `API_KEY` (프론트 요청 키와 동일해야 함)
- `RESEARCH_BP_MODEL_PATH` (MS-TCN 가중치 파일 경로)
- `RESEARCH_SCALER_INFO_PATH`
- `USE_DUMMY_GLUCOSE=True` (현재 혈당은 참고용 더미 모드)

배포 시 실제 경로에 맞게 수정하세요.

## 5) 배포 번들

다른 PC 배포용 정리 폴더:
- [deploy_bundle](deploy_bundle)

포함 내용:
- backend 소스
- frontend 소스 + build 결과
- docs
- 실행 스크립트/가이드(배포 폴더 내 README 참고)

## 6) GitHub 백업(권장 절차)

프로젝트 루트에서:
1. `git init`
2. `git add .`
3. `git commit -m "backup: visivital full implementation"`
4. GitHub 새 저장소 생성 후 remote 연결
5. `git push -u origin main`

## 7) 문제 해결

- 프론트가 안 뜰 때: Node PATH 확인, `npm start` 재실행
- API 실패: `http://127.0.0.1:5000/api/health` 확인
- 체크리스트 fallback 고정: 최신 측정 저장 여부/백엔드 재시작 확인
- 시간 라벨 미표시: 브라우저 캐시 삭제 후 새로고침

## 8) 면책

본 서비스는 의료인의 진단/치료를 대체하지 않으며, 자가 건강관리 참고용입니다.

## 9) 자동 유지보수(권장)

다음 자동화 설정이 포함되어 있습니다.

- 의존성 자동 업데이트(PR 생성)
	- [.github/dependabot.yml](.github/dependabot.yml)
	- 대상: `frontend`(npm), `backend`(pip), GitHub Actions
	- 주기: 매주 월요일
- 기본 CI 점검
	- [.github/workflows/ci-maintenance.yml](.github/workflows/ci-maintenance.yml)
	- Frontend: `npm install --legacy-peer-deps` + `npm run build` + `npm audit(고위험, prod)`
	- Backend: `requirements.txt` 해석(dry-run) + `app.py` 문법 점검
- 브라우저 호환 DB 자동 갱신(PR 생성)
	- [.github/workflows/browserslist-update.yml](.github/workflows/browserslist-update.yml)
	- `caniuse-lite`(Browserslist DB) 주간 업데이트
- Dependabot PR 자동 병합
	- [.github/workflows/dependabot-auto-merge.yml](.github/workflows/dependabot-auto-merge.yml)
	- 조건: CI 통과 시 `patch/minor/major` 자동 squash merge

### 운영 루틴

1. 자동 생성된 의존성 PR은 CI 통과 시 자동 병합
2. 월 1회 릴리즈 브랜치에서 실서버 스모크 테스트
3. 주요 패키지(`react`, `react-scripts`, `tensorflow`, `flask`)는
	 보안 공지(CVE) 발생 시 주간 주기와 무관하게 즉시 업데이트

### GitHub에서 자동화 활성화 전 확인

- 저장소를 GitHub 원격에 푸시해야 워크플로가 실행됩니다.
- Actions/Dependabot 권한이 저장소 설정에서 활성화되어 있어야 합니다.
