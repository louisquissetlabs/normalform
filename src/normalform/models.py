"""Pydantic models for captured API call payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CapturedRequest(BaseModel):
    """A captured OpenAI API request payload."""

    model_config = ConfigDict(extra="allow")

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    method: str
    url: str
    base_url: str
    endpoint: str
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    messages: list[dict[str, Any]] | None = None
    body: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
