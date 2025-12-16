"""Tracked OpenAI client that captures API call payloads."""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from typing import Any, Mapping

import httpx
from openai import OpenAI, AsyncOpenAI

from .models import CapturedRequest


class TrackedOpenAI(OpenAI):
    """OpenAI client wrapper that captures the last N API call payloads.

    Args:
        history_size: Maximum number of requests to keep in history. Defaults to 3.
        **kwargs: All standard OpenAI client arguments.

    Example:
        >>> client = TrackedOpenAI(history_size=5)
        >>> client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        >>> print(client.history[-1].model)
        'gpt-4'
    """

    def __init__(self, *, history_size: int = 3, **kwargs: Any) -> None:
        self._history: deque[CapturedRequest] = deque(maxlen=history_size)
        self._history_size = history_size

        # Create a custom HTTP client with request hooks
        http_client = kwargs.pop("http_client", None)
        if http_client is None:
            http_client = httpx.Client(
                event_hooks={"request": [self._capture_request]}
            )
        else:
            # Add our hook to existing client's hooks
            existing_hooks = http_client.event_hooks.get("request", [])
            http_client.event_hooks["request"] = [self._capture_request] + list(existing_hooks)

        super().__init__(http_client=http_client, **kwargs)

    @property
    def history(self) -> list[CapturedRequest]:
        """List of captured requests, oldest first."""
        return list(self._history)

    @property
    def last_request(self) -> CapturedRequest | None:
        """The most recent captured request, or None if no requests have been made."""
        return self._history[-1] if self._history else None

    def clear_history(self) -> None:
        """Clear all captured requests from history."""
        self._history.clear()

    def _capture_request(self, request: httpx.Request) -> None:
        """Event hook to capture request details before sending."""
        # Parse the request body
        body: dict[str, Any] = {}
        if request.content:
            try:
                body = json.loads(request.content.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # Extract the endpoint from the URL path
        url_str = str(request.url)
        base_url = str(self.base_url).rstrip("/")
        endpoint = url_str.replace(base_url, "").lstrip("/")

        # Build captured request with common fields extracted
        captured = CapturedRequest(
            timestamp=datetime.utcnow(),
            method=request.method,
            url=url_str,
            base_url=base_url,
            endpoint=endpoint,
            model=body.get("model"),
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
            messages=body.get("messages"),
            body=body,
            headers={k: v for k, v in request.headers.items() if k.lower() != "authorization"},
        )

        self._history.append(captured)


class AsyncTrackedOpenAI(AsyncOpenAI):
    """Async OpenAI client wrapper that captures the last N API call payloads.

    Args:
        history_size: Maximum number of requests to keep in history. Defaults to 3.
        **kwargs: All standard AsyncOpenAI client arguments.

    Example:
        >>> client = AsyncTrackedOpenAI(history_size=5)
        >>> await client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        >>> print(client.history[-1].model)
        'gpt-4'
    """

    def __init__(self, *, history_size: int = 3, **kwargs: Any) -> None:
        self._history: deque[CapturedRequest] = deque(maxlen=history_size)
        self._history_size = history_size

        # Create a custom HTTP client with request hooks
        http_client = kwargs.pop("http_client", None)
        if http_client is None:
            http_client = httpx.AsyncClient(
                event_hooks={"request": [self._capture_request]}
            )
        else:
            # Add our hook to existing client's hooks
            existing_hooks = http_client.event_hooks.get("request", [])
            http_client.event_hooks["request"] = [self._capture_request] + list(existing_hooks)

        super().__init__(http_client=http_client, **kwargs)

    @property
    def history(self) -> list[CapturedRequest]:
        """List of captured requests, oldest first."""
        return list(self._history)

    @property
    def last_request(self) -> CapturedRequest | None:
        """The most recent captured request, or None if no requests have been made."""
        return self._history[-1] if self._history else None

    def clear_history(self) -> None:
        """Clear all captured requests from history."""
        self._history.clear()

    async def _capture_request(self, request: httpx.Request) -> None:
        """Event hook to capture request details before sending."""
        # Parse the request body
        body: dict[str, Any] = {}
        if request.content:
            try:
                body = json.loads(request.content.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # Extract the endpoint from the URL path
        url_str = str(request.url)
        base_url = str(self.base_url).rstrip("/")
        endpoint = url_str.replace(base_url, "").lstrip("/")

        # Build captured request with common fields extracted
        captured = CapturedRequest(
            timestamp=datetime.utcnow(),
            method=request.method,
            url=url_str,
            base_url=base_url,
            endpoint=endpoint,
            model=body.get("model"),
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
            messages=body.get("messages"),
            body=body,
            headers={k: v for k, v in request.headers.items() if k.lower() != "authorization"},
        )

        self._history.append(captured)
