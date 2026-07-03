from __future__ import annotations

import re


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"ignore\s+(as\s+)?instru[cç][oõ]es\s+anteriores",
    r"desconsidere\s+(as\s+)?instru[cç][oõ]es",
    r"system\s*:",
    r"developer\s*:",
    r"assistant\s*:",
    r"you\s+are\s+now",
    r"voce\s+agora\s+e",
    r"você\s+agora\s+é",
]


def sanitize_user_query(query: str, max_chars: int = 500) -> str:
    """Clean user input before retrieval and prompt construction."""
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", query or "")
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    for pattern in INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[instrucao removida]", sanitized, flags=re.IGNORECASE)
    if len(sanitized) > max_chars:
        sanitized = sanitized[:max_chars].rsplit(" ", 1)[0].strip()
    return sanitized


def was_sanitized(original: str, sanitized: str) -> bool:
    return (original or "").strip() != (sanitized or "").strip()
