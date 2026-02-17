from __future__ import annotations

STATUS_TEXT = {
    1: "오늘 측정된 혈당과 혈압 수치가 최근 기록 범위 내에서 비교적 안정적으로 나타났어요.",
    2: "최근 기록에서 평소보다 변동이 조금 늘어난 흐름이 보여요.",
    3: "최근 기록의 변동 폭이 크게 나타나는 구간이 있어요. 생활 루틴 점검이 도움이 될 수 있어요.",
    4: "최근 기록에서 큰 변동이 반복되거나 흐름이 불규칙하게 이어지는 모습이 보여요. 당분간 기록과 루틴 관리를 강화해보세요.",
}


def generate_summary(status_result: dict):
    level = int(status_result.get("status_level", 1))
    return STATUS_TEXT.get(level, STATUS_TEXT[1])
