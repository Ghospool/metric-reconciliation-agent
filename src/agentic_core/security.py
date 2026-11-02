"""Governance helpers: PII redaction, prompt-injection screening, audit logging."""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PII = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}(?!\d)"),
}

_INJECTION = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"you are now\b",
    r"reveal (your|the) (system )?prompt",
    r"developer mode",
]
_INJECTION_RE = re.compile("|".join(_INJECTION), re.I)


@dataclass(frozen=True)
class Redaction:
    text: str
    counts: dict[str, int]


def redact_pii(text: str) -> Redaction:
    counts: dict[str, int] = {}
    for label, pat in _PII.items():
        text, n = pat.subn(f"[{label}]", text)
        if n:
            counts[label] = n
    return Redaction(text, counts)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


class AuditLog:
    """Append-only JSONL audit trail. Never log raw PII: pass redacted text only."""

    def __init__(self, path: str | Path | None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        if not self._path:
            return
        rec = {"ts": time.time(), "event": event, **fields}
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")


_SECRETS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
]
_SECRET_ASSIGN = re.compile(
    r"""(?P<pre>(?:password|passwd|secret|api[_-]?key|token|access[_-]?key)\w*\s*[:=]\s*)(?P<q>["'])(?P<val>[^"']{4,})(?P=q)""",
    re.I)


def redact_secrets(text: str) -> Redaction:
    """Mask credentials in source code before it is sent to a model or written to a report."""
    counts: dict[str, int] = {}
    for pat in _SECRETS:
        text, n = pat.subn("[REDACTED_SECRET]", text)
        if n:
            counts["SECRET"] = counts.get("SECRET", 0) + n
    extra = 0

    def _mask(m: re.Match) -> str:
        nonlocal extra
        if m["val"] == "[REDACTED_SECRET]":  # already masked by a stronger pattern
            return m[0]
        extra += 1
        return f'{m["pre"]}{m["q"]}[REDACTED_SECRET]{m["q"]}'

    text = _SECRET_ASSIGN.sub(_mask, text)
    if extra:
        counts["SECRET"] = counts.get("SECRET", 0) + extra
    return Redaction(text, counts)
