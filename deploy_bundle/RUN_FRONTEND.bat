@echo off
cd /d %~dp0frontend
node -e "const [a,b]=process.versions.node.split('.').map(Number); process.exit(a>20||(a===20&&b>=19)?0:1)" 2>nul
if errorlevel 1 (
  echo [VisiVital] Node.js 20.19 or newer is required. Current:
  node --version
  echo Install the LTS version from https://nodejs.org/
  pause
  exit /b 1
)
if not exist node_modules call npm install
npm start
