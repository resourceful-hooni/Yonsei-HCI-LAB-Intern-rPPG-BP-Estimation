# VisiVital 배포 가이드 (운영)

라이브: **https://yonseihci.kro.kr/**

## 1) 운영 아키텍처

- **호스트**: Oracle Cloud Ubuntu VM (`ubuntu@<HOST>`)
- **앱 경로**: `/home/ubuntu/Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation/Web/visi-vital`
- **Docker Compose 3 컨테이너**
  - `visivital-backend` — Flask API (`python:3.10-slim`), 포트 5000
  - `visivital-frontend` — React 빌드 → nginx 정적 서빙 (`node:20-alpine` 빌드 → `nginx:1.27-alpine`), 포트 3000
  - `visivital-caddy` — 리버스 프록시 + TLS(도메인 처리)
- 프론트는 **컨테이너 빌드 시 내부에서 `npm run build`** 하므로, 소스만 올리고 `docker compose build` 하면 됩니다(서버에서 별도 빌드 불필요).
- DB(SQLite)는 `./data` 볼륨에 영속 — 재배포로 사라지지 않습니다.

> 참고: GitHub 레포 구조(루트에 `frontend/`·`backend/`)와 서버 구조(`…/Web/visi-vital/…`)가 다릅니다. 서버는 GitHub을 `git pull` 하지 않고 **소스 동기화 + Docker 재빌드**로 갱신됩니다.

## 2) 배포 방법 A — GitHub Actions (권장, 쉬움)

워크플로우: [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml)

- **수동 1‑클릭**: GitHub → Actions → "Deploy (server)" → *Run workflow*
- **자동**: `main`에 `frontend/src/**` 또는 `backend/**`(코드) 변경이 푸시되면 자동 실행
- 하는 일: 앱 소스(`frontend/src`, `frontend/package.json`+lock, 백엔드 python)를 SSH로 동기화 → 프론트 Dockerfile에 `--legacy-peer-deps` 보장(멱등) → `docker compose build frontend backend && up -d` → `/api/health` 확인
- **하지 않는 일**(인프라는 수동): `backend/Dockerfile`·`requirements.txt`, `docker-compose.yml`, `nginx`/`caddy` 설정은 동기화하지 않습니다. 백엔드 의존성/인프라가 바뀌면 방법 B 또는 서버에서 수동 처리하세요.

> 참고: 프론트가 `react-helmet`을 import 하는데 `package.json`에 빠져 있거나, Docker 빌드의 `npm install`에 `--legacy-peer-deps`가 없으면 `Module not found: react-helmet`로 빌드가 실패합니다. 위 동기화/패치가 이를 자동 처리합니다(서버 `package.json`의 `proxy` 필드는 레포 `package.json`에도 포함되어 보존됨).

### 필요한 Secrets (Settings → Secrets and variables → Actions)

| 이름 | 값 예시 |
|---|---|
| `DEPLOY_SSH_KEY` | 서버 접속 개인키(PEM 전체 내용) |
| `DEPLOY_HOST` | `168.107.10.60` |
| `DEPLOY_USER` | `ubuntu` |
| `DEPLOY_APP_DIR` | `/home/ubuntu/Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation/Web/visi-vital` |
| `HEALTHCHECK_URL` | (선택) `https://yonseihci.kro.kr/api/health` |

```bash
# gh CLI로 한 번에 등록 (개인키 파일 경로/호스트는 실제 값으로)
gh secret set DEPLOY_SSH_KEY  < /path/to/ssh-key.key
gh secret set DEPLOY_HOST     --body "168.107.10.60"
gh secret set DEPLOY_USER     --body "ubuntu"
gh secret set DEPLOY_APP_DIR  --body "/home/ubuntu/Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation/Web/visi-vital"
```

> 자동 푸시 배포를 끄고 수동(1‑클릭)만 쓰려면 `deploy.yml`의 `push:` 블록을 제거하세요.

## 3) 배포 방법 B — 수동 스크립트

로컬 클론에서 직접 배포(인프라 변경 포함 시 유용). Git Bash:

```bash
KEY=/path/to/ssh-key.key bash scripts/deploy.sh
```

`scripts/deploy.sh`는 앱 소스를 tar로 묶어 scp 업로드 → 서버에서 **적용 전 백업** → 복사 → `docker compose build && up -d` → health 확인까지 수행합니다. (rsync 불필요, Windows 친화적)

## 4) 롤백

매 배포 직전 서버에 백업 tar가 생성됩니다: `/tmp/vv_backup_<timestamp>.tgz`

```bash
ssh -i <key> ubuntu@<HOST>
cd /home/ubuntu/Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation/Web/visi-vital
tar xzf /tmp/vv_backup_<timestamp>.tgz          # frontend/src + backend 복원
docker compose build frontend backend && docker compose up -d
```

## 5) 검증 체크리스트

- 컨테이너 상태: `docker ps` 에서 frontend/backend/caddy 가 `Up`
- 헬스: `curl -fsS https://yonseihci.kro.kr/api/health` → `{"status":"ok"}`
- 프론트 번들 해시 변경 확인: `curl -s https://yonseihci.kro.kr/ | grep -o 'main\.[a-z0-9]*\.js'`
- 백엔드 변경은 **컨테이너 재시작 시 적용**(위 `up -d`에 포함)

## 6) 연구 BP 모델(MS-TCN) 활성화 (선택)

기본은 가중치가 없어 **혈압=경험식, 혈당=더미**입니다. 실제 연구 모델을 쓰려면 `docker-compose.yml`(서버)의 backend 환경에 다음을 설정 후 재배포:

- `RESEARCH_BP_MODEL_PATH` = 가중치 `.h5` 경로(컨테이너 내부, 볼륨 마운트 필요)
- `RESEARCH_SCALER_INFO_PATH` = 스케일러 통계 파일
- `RESEARCH_MODEL_FS` = 학습 샘플링레이트(Hz). 설정 시 주파수 정합 리샘플 활성화

## 7) 신호 파이프라인 검증(로컬, 그라운드트루스 불필요)

```bash
python backend/tests/test_signal_pipeline.py
```
합성 신호로 심박 복원 오차·SNR·NaN 안전성·POS vs CHROM 회귀를 점검합니다.
