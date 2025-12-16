# OpenAI Capture

A minimal wrapper over the OpenAI Python SDK that captures API call payloads for debugging and inspection.

## Installation

```bash
pip install openai-capture
```

Or install from source:

```bash
pip install -e .
```

## Usage

```python
from normalform import TrackedOpenAI

# Create a tracked client (keeps last 3 requests by default)
client = TrackedOpenAI(history_size=3)

# Use it just like the regular OpenAI client
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
)

# Access the captured request payload
last_request = client.last_request
print(f"Model: {last_request.model}")
print(f"Temperature: {last_request.temperature}")
print(f"Messages: {last_request.messages}")
print(f"Base URL: {last_request.base_url}")
print(f"Full body: {last_request.body}")

# Access full history
for req in client.history:
    print(f"{req.timestamp}: {req.model} - {req.endpoint}")

# Clear history when needed
client.clear_history()
```

### Async Client

```python
from normalform import AsyncTrackedOpenAI

client = AsyncTrackedOpenAI(history_size=5)

response = await client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(client.last_request.model)
```

## Captured Fields

Each `CapturedRequest` contains:

- `timestamp`: When the request was made
- `method`: HTTP method (e.g., "POST")
- `url`: Full request URL
- `base_url`: API base URL
- `endpoint`: API endpoint path
- `model`: Model name (if applicable)
- `temperature`: Temperature setting (if applicable)
- `max_tokens`: Max tokens setting (if applicable)
- `messages`: Chat messages (if applicable)
- `body`: Full request body as dict
- `headers`: Request headers (excluding Authorization)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT
