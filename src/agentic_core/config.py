"""Environment-driven configuration. No secrets in code."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    fast_model: str
    smart_model: str
    log_level: str
    audit_log_path: str
    max_retries: int = 3
    timeout_s: float = 60.0


def get_settings() -> Settings:
    return Settings(
        api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        fast_model=os.getenv("FAST_MODEL", "claude-haiku-5-5"),
        smart_model=os.getenv("SMART_MODEL", "claude-sonnet-5-5"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        audit_log_path=os.getenv("AUDIT_LOG_PATH", "audit.jsonl"),
    )
