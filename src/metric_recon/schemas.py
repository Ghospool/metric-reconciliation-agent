from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class Mapping:
    """Which columns of a source mean what. Declared by the agent, validated against the real columns."""
    key: str
    amount: str
    date: str | None = None
    status: str | None = None
    currency: str | None = None


@dataclass
class Row:
    key: str
    cents: int
    day: date | None
    status: str | None
    currency: str | None
    line: int


@dataclass
class Step:
    id: str
    probe: str
    family: str          # duplicates | status_scope | cutoff | missing_records | value_basis
    side: str            # A | B | AB
    description: str
    delta_cents: int     # contribution to bridging A -> B (signed)
    rows: int
    samples: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def no_effect(self) -> bool:
        return self.delta_cents == 0 and self.rows == 0
