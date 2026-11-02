"""All money is integer cents: no float drift, so a bridge can reconcile to the cent."""
from __future__ import annotations

import re
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

_STRIP = re.compile(r"[^\d.\-()]")


def parse_cents(raw: object) -> int | None:
    s = str(raw).strip()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")") or s.startswith("-")
    s = _STRIP.sub("", s).replace("(", "").replace(")", "").lstrip("-")
    if s.count(".") > 1 or not s:
        return None
    try:
        v = Decimal(s).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None
    c = int(v * 100)
    return -c if neg else c


def parse_day(raw: object) -> date | None:
    s = str(raw).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def fmt(cents: int, sign: bool = False) -> str:
    q = Decimal(abs(cents)) / 100
    s = f"{q:,.2f}"
    if cents < 0:
        return "-" + s
    return ("+" + s) if sign and cents > 0 else s
