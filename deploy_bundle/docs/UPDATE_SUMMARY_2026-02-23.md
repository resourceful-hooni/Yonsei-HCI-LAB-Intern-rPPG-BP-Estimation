# VisiVital 업데이트 정리 (2026-02-23)

## 1) 반영 범위
이번 업데이트는 아래 4개 축으로 진행됨.

1. 다국어(i18n) 기반 구조 도입 (KO/EN 토글)
2. 측정/요약/생활관리 UI 텍스트 및 표시 체계 개선
3. 모바일 UX 버그 수정 (측정 FAB 위치, 터치 애니메이션, 모달 표시)
4. 서버 반영 및 프론트 배포 검증

---

## 2) 주요 기능 업데이트

### 2.1 언어 토글 + 전역 번역 구조
- 헤더 우측 상단 언어 토글 버튼 추가
- KO/EN 번역 사전 및 컨텍스트 도입
- 앱 전반 텍스트를 `t(key)` 기반으로 치환

**신규 파일**
- `frontend/src/contexts/LangContext.jsx`
- `frontend/src/i18n/translations.js`

**적용 파일**
- `frontend/src/index.js` (LangProvider 적용)
- `frontend/src/App.jsx` (토글 버튼, 상/하단 네비 문구)
- `frontend/src/pages/MeasurementPage.jsx`
- `frontend/src/pages/SummaryPage.jsx`
- `frontend/src/pages/LifestyleGuidePage.jsx`
- `frontend/src/components/Measurement/CameraView.jsx`
- `frontend/src/components/Measurement/ResultDisplay.jsx`
- `frontend/src/components/Summary/VitalCard.jsx`

---

### 2.2 측정 페이지 메시지 체계 고도화
- 얼굴 감지 안내 문구를 다중 상태 기반으로 확장
- 문구 형식을 일관된 `[상태] · [안내]` 스타일로 정리
- 상태 우선순위(얼굴 미감지/정렬/조명/움직임/측정중/준비완료)로 표시

**핵심 반영점**
- `CameraView`의 품질 상태값을 문자열 라벨에서 코드(`good/mid/bad/wait`) 기반으로 정리
- 실시간 체크리스트 상태 표시/경고 배너/버튼 문구 i18n 연동

---

### 2.3 혈압 표기 개선 (수축기/이완기)
- 혈압 수치를 `SBP/DBP` 값만 보여주던 방식에서
- 수치 아래에 `수축기 / 이완기` 서브라벨을 추가해 의미를 명확화

**적용 파일**
- `frontend/src/components/Measurement/ResultDisplay.jsx`
- `frontend/src/components/Summary/VitalCard.jsx`

**스타일 추가**
- `frontend/src/styles/globalStyles.css`
  - `.bp-sublabels`

---

## 3) 버그 수정 내역

### 3.1 측정 시작 버튼(FAB) 측정중 상태에서 오른쪽 치우침
**원인**
- 측정중 애니메이션(`btn-measuring`)이 `transform`을 덮어쓰면서,
  고정 위치용 `translateX(-50%)`가 깨짐.

**조치**
- 측정중 애니메이션 transform을 위치 보존 형태로 정리
- 모바일/데스크톱에서 transform 충돌이 없도록 보강

**파일**
- `frontend/src/styles/globalStyles.css`

---

### 3.2 요약 페이지 시각 정확도 문제
**원인**
- 타임스탬프 문자열 처리 시 타임존/정렬 일관성이 부족하여,
  차트 축/최근 측정 시각이 실제와 어긋날 수 있었음.

**조치**
- 타임스탬프 파싱 로직 정리
- `Asia/Seoul` 기준 포맷 적용
- 통합 추이 데이터 시간순 정렬 후 최신 시각 계산

**파일**
- `frontend/src/pages/SummaryPage.jsx`

---

### 3.3 모바일 터치 애니메이션 미표시
**원인**
- 기존 리플 로직이 버튼 위주로 동작하여 링크/터처블 카드에서 미적용 케이스 발생

**조치**
- 리플 이벤트를 `button`, `a`, `.touchable`까지 확장
- 모바일 터치 이벤트 경로 보강 (`touchstart` 포함)
- 리플 스타일 선택자 범위 확장

**파일**
- `frontend/src/App.jsx`
- `frontend/src/styles/globalStyles.css`

---

### 3.4 생활관리 “이 행동이 왜 필요한가요?” 모달 접근성
**원인**
- 모바일에서 레이아웃/스크롤 컨텍스트 영향으로 모달 체감 위치가 하단으로 밀려 보이는 문제

**조치**
- 모달을 `createPortal(..., document.body)`로 렌더링하도록 변경
- 현재 화면 기준 즉시 오버레이 표시되도록 개선

**파일**
- `frontend/src/components/LifestyleGuide/WhyNeededModal.jsx`

---

## 4) 스타일 업데이트
- 언어 토글 버튼 스타일 추가: `.lang-toggle`
- 혈압 서브라벨 스타일 추가: `.bp-sublabels`
- 리플 적용 범위 및 터처블 요소 스타일 정리
- FAB 측정중 애니메이션/transform 충돌 보정

**파일**
- `frontend/src/styles/globalStyles.css`

---

## 5) 배포 반영 사항

### 5.1 서버 반영 경로
- 서버: `ubuntu@168.107.10.60`
- 대상 소스: `/home/ubuntu/Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation/Web/visi-vital/frontend/src`
- 임시 업로드: `/tmp/vv_upload/src`

### 5.2 반영 파일
- `App.jsx`, `index.js`
- `components/Measurement/CameraView.jsx`
- `components/Measurement/ResultDisplay.jsx`
- `components/Summary/VitalCard.jsx`
- `components/LifestyleGuide/WhyNeededModal.jsx`
- `pages/MeasurementPage.jsx`
- `pages/SummaryPage.jsx`
- `pages/LifestyleGuidePage.jsx`
- `styles/globalStyles.css`
- `contexts/LangContext.jsx`
- `i18n/translations.js`

### 5.3 배포 검증 결과
- 프론트 컨테이너 재생성/재시작 완료
- 컨테이너 시작 시각 갱신 확인
- 번들 해시 변경 확인 (`main.b5a7ae1f.js`)
- 신규 키워드(`fl_align_critical`, `lang-toggle`) 포함 여부 확인

---

## 6) 확인 권장 체크리스트 (운영 화면)
1. 측정 시작 후 FAB가 중앙에서 유지되는지
2. 요약 차트 X축 시각과 `최근 측정 시각`이 일관적인지
3. 모바일 탭/버튼/카드 터치 시 리플이 보이는지
4. 생활관리에서 “이 행동이 왜 필요한가요?” 탭 시 즉시 모달이 보이는지
5. 언어 토글 시 핵심 페이지 문구가 정상 전환되는지

---

작성일: 2026-02-23
