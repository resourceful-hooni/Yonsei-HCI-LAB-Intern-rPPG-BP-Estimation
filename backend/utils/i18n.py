"""Server-side i18n for generated text (status, summary, quality, comments,
recommendations). The frontend i18n only covers static UI labels; the strings
the backend *generates* are localized here based on a `lang` query param.

Usage: every text-producing helper takes a `lang` ("ko" | "en")."""

from __future__ import annotations


def normalize_lang(lang) -> str:
    return "en" if str(lang or "").lower().startswith("en") else "ko"


def _pick(table: dict, key, lang: str, default_key=None):
    entry = table.get(key, table.get(default_key, {})) if default_key is not None else table.get(key, {})
    return entry.get(lang, entry.get("ko", ""))


# ── Status (level 1-4) ───────────────────────────────────────────────────────
STATUS_LABEL = {
    1: {"ko": "안정적", "en": "Stable"},
    2: {"ko": "관심 필요", "en": "Watch"},
    3: {"ko": "집중 관리", "en": "Manage closely"},
    4: {"ko": "관리 강화", "en": "Take action"},
}

SUMMARY_TEXT = {
    1: {"ko": "오늘 측정된 혈당과 혈압 수치가 최근 기록 범위 내에서 비교적 안정적으로 나타났어요.",
        "en": "Today's blood pressure and glucose readings look fairly stable within your recent range."},
    2: {"ko": "최근 기록에서 평소보다 변동이 조금 늘어난 흐름이 보여요.",
        "en": "Your recent readings show a little more variation than usual."},
    3: {"ko": "최근 기록의 변동 폭이 크게 나타나는 구간이 있어요. 생활 루틴 점검이 도움이 될 수 있어요.",
        "en": "Some recent readings vary quite a bit - reviewing your daily routine may help."},
    4: {"ko": "최근 기록에서 큰 변동이 반복되거나 흐름이 불규칙하게 이어지는 모습이 보여요. 당분간 기록과 루틴 관리를 강화해보세요.",
        "en": "Recent readings vary a lot or look irregular - consider measuring more often and tightening your routine for a while."},
}


def status_label(level: int, lang: str) -> str:
    return _pick(STATUS_LABEL, int(level), lang, default_key=1)


def summary_text(level: int, lang: str) -> str:
    return _pick(SUMMARY_TEXT, int(level), lang, default_key=1)


# ── Quality checklist ────────────────────────────────────────────────────────
QUALITY_LABEL = {
    "lighting":  {"ko": "조명", "en": "Lighting"},
    "movement":  {"ko": "움직임", "en": "Movement"},
    "alignment": {"ko": "얼굴정렬", "en": "Face alignment"},
}
QUALITY_STATUS = {
    "good": {"ko": "좋음", "en": "Good"},
    "fair": {"ko": "보통", "en": "Fair"},
    "poor": {"ko": "개선 필요", "en": "Needs work"},
}
QUALITY_TIP = {
    "lighting":  {"ko": "얼굴 좌우 밝기 차이를 줄이고 그림자가 생기지 않도록 조명을 맞춰주세요.",
                  "en": "Even out the lighting on your face and avoid shadows."},
    "movement":  {"ko": "측정 중 고개·시선 이동을 줄이고 자연스럽게 정면을 유지해주세요.",
                  "en": "Minimize head/eye movement and face the camera naturally."},
    "alignment": {"ko": "얼굴 중심이 프레임 중앙에 오도록 위치를 맞추고 턱선이 잘 보이게 해주세요.",
                  "en": "Center your face in the frame and keep your jawline visible."},
}
QUALITY_TIP_FALLBACK = {
    "lighting":  {"ko": "얼굴 정면에 균일한 빛을 유지해주세요.", "en": "Keep even light on your face."},
    "movement":  {"ko": "측정 중 고개/상체 움직임을 줄여주세요.", "en": "Reduce head/upper-body movement while measuring."},
    "alignment": {"ko": "얼굴을 화면 중앙에 유지해주세요.", "en": "Keep your face centered in the frame."},
}


def quality_status_code(score) -> str:
    if score is None:
        return "fair"
    if score >= 0.75:
        return "good"
    if score >= 0.55:
        return "fair"
    return "poor"


def quality_label(qid: str, lang: str) -> str:
    return _pick(QUALITY_LABEL, qid, lang)


def quality_status_text(code: str, lang: str) -> str:
    return _pick(QUALITY_STATUS, code, lang, default_key="fair")


def quality_tip(qid: str, lang: str, measured: bool) -> str:
    table = QUALITY_TIP if measured else QUALITY_TIP_FALLBACK
    return _pick(table, qid, lang)


# ── Daily comment ────────────────────────────────────────────────────────────
DAILY_COMMENT = {
    "stable": {"ko": "오늘은 최근 흐름 대비 변동 폭이 비교적 안정적으로 보여요.",
               "en": "Today looks fairly stable compared with your recent trend."},
    "small":  {"ko": "최근 기록에서 작은 변동이 보여요. 측정 환경을 일정하게 유지해보세요.",
               "en": "Small variations recently - try to keep your measurement conditions consistent."},
    "large":  {"ko": "최근 변동 폭이 다소 큰 편이에요. 조명과 자세를 일정하게 맞춰 측정해보세요.",
               "en": "Variation has been a bit large recently - keep lighting and posture consistent."},
}


def daily_comment(variant: str, lang: str) -> str:
    return _pick(DAILY_COMMENT, variant, lang, default_key="stable")


# ── Re-guide + vital labels + percentile range ───────────────────────────────
REGUIDE = {
    "title":       {"ko": "다시 측정 가이드", "en": "Re-measure guide"},
    "description": {"ko": "조명/자세/정렬 팁을 확인하고 다시 측정해보세요.",
                    "en": "Check the lighting / posture / alignment tips and measure again."},
}
VITAL_LABELS = {
    "bp":      {"ko": "혈압", "en": "Blood pressure"},
    "glucose": {"ko": "혈당", "en": "Glucose"},
}
RANGE_LABEL = {
    "low":    {"ko": "낮음", "en": "Low"},
    "high":   {"ko": "높음", "en": "High"},
    "normal": {"ko": "일반적인 범위", "en": "Typical range"},
}


def reguide(lang: str) -> dict:
    return {"title": _pick(REGUIDE, "title", lang), "description": _pick(REGUIDE, "description", lang), "path": "/measurement"}


def vital_labels(lang: str) -> dict:
    return {"bp": _pick(VITAL_LABELS, "bp", lang), "glucose": _pick(VITAL_LABELS, "glucose", lang)}


def range_label(percentile: float, lang: str) -> str:
    code = "low" if percentile < 30 else "high" if percentile > 70 else "normal"
    return _pick(RANGE_LABEL, code, lang)


# ── Recommendations (with interpolated numbers) ──────────────────────────────
RECO_TEXT = {
    "sleep": {
        "title":  {"ko": "수면 시간 1시간 늘리기", "en": "Sleep one hour more"},
        "icon": "🌙",
        "description": {"ko": "수면 습관은 혈당·혈압 관리에 영향을 줄 수 있는 생활 요소 중 하나로 알려져 있어요.",
                        "en": "Sleep habits are one lifestyle factor that can affect blood pressure and glucose."},
        "detail": {"ko": "취침/기상 시간을 일정하게 맞추면 일중 변동 관리에 참고가 될 수 있어요.",
                   "en": "Consistent bedtimes and wake times can help with day-to-day variation."},
        "reason": {"ko": "최근 7일 변동성(BP {bp_cv:.1f}%, 혈당 {gl_cv:.1f}%)을 기준으로 수면 루틴을 우선 추천해요.",
                   "en": "Based on the last 7 days' variability (BP {bp_cv:.1f}%, glucose {gl_cv:.1f}%), a sleep routine comes first."},
    },
    "salt": {
        "title":  {"ko": "염분 섭취 줄이기", "en": "Cut back on salt"},
        "icon": "🧂",
        "description": {"ko": "염분 섭취는 혈압 관리와 연관된 생활 요인으로 알려져 있어요.",
                        "en": "Salt intake is a lifestyle factor linked to blood pressure."},
        "detail": {"ko": "국물류 섭취 빈도를 조금 줄이면 혈압 변동 완화에 도움이 될 수 있어요.",
                   "en": "Eating fewer soups/broths may help ease blood-pressure swings."},
        "reason": {"ko": "최근 평균 수축기 혈압이 {bp_mean:.1f}로 나타나 염분 관리 실천을 함께 권장해요.",
                   "en": "Your recent average systolic BP is {bp_mean:.1f}, so managing salt is recommended too."},
    },
    "water": {
        "title":  {"ko": "하루 하나 더 수분 마시기", "en": "Drink one more glass of water"},
        "icon": "💧",
        "description": {"ko": "수분 섭취는 일상적인 혈압 관리에 참고되는 요소 중 하나예요.",
                        "en": "Hydration is one factor to consider for everyday blood-pressure care."},
        "detail": {"ko": "하루 1회 물 섭취 알림을 설정해 루틴을 만드는 방법을 추천해요.",
                   "en": "Try one daily water reminder to build the habit."},
        "reason": {"ko": "최근 혈당 평균 {gl_mean:.1f} 흐름을 참고해 수분 루틴을 함께 관리해보세요.",
                   "en": "Given your recent average glucose of {gl_mean:.1f}, manage a hydration routine too."},
    },
}


def recommendation(rid: str, lang: str, **fmt) -> dict:
    spec = RECO_TEXT[rid]
    return {
        "id": rid,
        "icon": spec["icon"],
        "title": spec["title"].get(lang, spec["title"]["ko"]),
        "description": spec["description"].get(lang, spec["description"]["ko"]),
        "detail": spec["detail"].get(lang, spec["detail"]["ko"]),
        "reason_context": spec["reason"].get(lang, spec["reason"]["ko"]).format(**fmt),
    }
