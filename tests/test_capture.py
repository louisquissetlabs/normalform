"""Minimal test to verify request capture functionality."""

import json
from unittest.mock import patch, MagicMock

import httpx
import pytest

from normalform import TrackedOpenAI, CapturedRequest

TEST_MODEL="gpt-4o-mini"

def test_single_prompt_captured():
    """Test that a single chat completion request is captured correctly."""
    # Create a mock response
    mock_response = httpx.Response(
        200,
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        },
    )

    # Create a transport that returns our mock response
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return mock_response

    mock_transport = httpx.MockTransport(mock_handler)
    http_client = httpx.Client(transport=mock_transport)

    # Create the tracked client with our mock
    client = TrackedOpenAI(
        api_key="test-key",
        http_client=http_client,
        history_size=3,
    )

    # Make a request
    client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0.7,
    )

    # Verify the request was captured
    assert len(client.history) == 1

    captured = client.last_request
    assert captured is not None
    assert isinstance(captured, CapturedRequest)
    assert captured.model == TEST_MODEL
    assert captured.temperature == 0.7
    assert captured.messages == [{"role": "user", "content": "Say hello"}]
    assert captured.method == "POST"
    assert "chat/completions" in captured.endpoint
    assert captured.base_url == "https://api.openai.com/v1"


def test_history_limit():
    """Test that history respects the size limit."""
    mock_response = httpx.Response(
        200,
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": TEST_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        },
    )

    mock_transport = httpx.MockTransport(lambda r: mock_response)
    http_client = httpx.Client(transport=mock_transport)

    client = TrackedOpenAI(
        api_key="test-key",
        http_client=http_client,
        history_size=2,
    )

    # Make 3 requests
    for i in range(3):
        client.chat.completions.create(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": f"Message {i}"}],
        )

    assert len(client.history) == 2
    
    
def test_clear_history():
    """Test that clear_history removes all captured requests."""
    mock_response = httpx.Response(
        200,
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": TEST_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        },
    )

    mock_transport = httpx.MockTransport(lambda r: mock_response)
    http_client = httpx.Client(transport=mock_transport)

    client = TrackedOpenAI(
        api_key="test-key",
        http_client=http_client,
    )

    client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert len(client.history) == 1
    client.clear_history()
    assert len(client.history) == 0
    assert client.last_request is None
