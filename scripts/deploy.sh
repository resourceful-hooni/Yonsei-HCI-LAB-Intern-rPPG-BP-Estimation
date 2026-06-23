#!/usr/bin/env bash
#
# Manual deploy of app source to the live server (alternative to the GitHub
# Actions "Deploy (server)" workflow). Surgical + safe: backs up on the server
# before applying, then rebuilds the Docker containers. Windows-friendly
# (tar + scp, no rsync required).
#
#   KEY=/path/to/ssh-key.key bash scripts/deploy.sh
#
# Optional overrides:
#   SERVER=ubuntu@168.107.10.60
#   APP=/home/ubuntu/Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation/Web/visi-vital
#   HEALTHCHECK_URL=https://yonseihci.kro.kr/api/health
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${KEY:?set KEY=/path/to/ssh-key.key}"
SERVER="${SERVER:-ubuntu@168.107.10.60}"
APP="${APP:-/home/ubuntu/Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation/Web/visi-vital}"
HEALTHCHECK_URL="${HEALTHCHECK_URL:-https://yonseihci.kro.kr/api/health}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -i "$KEY")

echo ">> staging app source"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/frontend" "$STAGE/backend"
cp -r "$ROOT/frontend/src" "$STAGE/frontend/src"
# package.json/lock are synced too so dependencies (e.g. react-helmet) stay in
# sync with the source — drift here is what breaks the in-container build.
cp "$ROOT/frontend/package.json" "$STAGE/frontend/package.json"
[ -f "$ROOT/frontend/package-lock.json" ] && cp "$ROOT/frontend/package-lock.json" "$STAGE/frontend/package-lock.json" || true
for p in models routes utils database config.py app.py; do
  cp -r "$ROOT/backend/$p" "$STAGE/backend/$p"
done
( cd "$STAGE" && find . -name __pycache__ -type d -prune -exec rm -rf {} + ; tar -czf payload.tgz frontend backend )

echo ">> uploading"
ssh "${SSH_OPTS[@]}" "$SERVER" 'rm -rf /tmp/vv_deploy && mkdir -p /tmp/vv_deploy'
scp "${SSH_OPTS[@]}" "$STAGE/payload.tgz" "$SERVER:/tmp/vv_deploy/payload.tgz"

echo ">> applying + rebuilding on server"
ssh "${SSH_OPTS[@]}" "$SERVER" APP="$APP" bash -s <<'REMOTE'
set -euo pipefail
cd "$APP"
TS=$(date +%Y%m%d_%H%M%S)
echo "   backup -> /tmp/vv_backup_$TS.tgz"
tar -czf "/tmp/vv_backup_$TS.tgz" frontend/src frontend/package.json frontend/Dockerfile backend
tar -xzf /tmp/vv_deploy/payload.tgz -C /tmp/vv_deploy
cp -r /tmp/vv_deploy/frontend/. frontend/
cp -r /tmp/vv_deploy/backend/. backend/
# react-helmet@6 declares a React 16/17 peer; React 18 needs --legacy-peer-deps
# in the Docker frontend build. Patch the Dockerfile once (idempotent).
if ! grep -q -- '--legacy-peer-deps' frontend/Dockerfile; then
  cp frontend/Dockerfile "frontend/Dockerfile.bak.$TS"
  sed -i 's/^RUN npm install$/RUN npm install --legacy-peer-deps/' frontend/Dockerfile
  echo "   patched frontend/Dockerfile -> npm install --legacy-peer-deps"
fi
docker compose build frontend backend
docker compose up -d
docker ps --format '{{.Names}}\t{{.Status}}'
REMOTE

echo ">> health"
sleep 5
curl -fsS "$HEALTHCHECK_URL" || { echo "health check failed"; exit 1; }
echo; echo ">> done"
