from __future__ import annotations

from utils.i18n import summary_text


def generate_summary(status_result: dict, lang: str = "ko"):
    level = int(status_result.get("status_level", 1))
    return summary_text(level, lang)
