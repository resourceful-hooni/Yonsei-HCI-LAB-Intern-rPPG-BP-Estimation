# GitHub 백업 푸시 방법

현재 로컬 커밋까지 완료되었습니다.

- 브랜치: `main`
- 최근 커밋: `backup: full visivital implementation + deploy bundle`

## 1) GitHub 저장소 생성
GitHub 웹에서 새 빈 저장소를 만드세요.

## 2) 원격 연결
```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
```
예: `https://github.com/<user>/visi-vital.git`

## 3) 푸시
```bash
git push -u origin main
```

인증이 필요하면 GitHub PAT(토큰) 또는 Git Credential Manager를 사용하세요.
