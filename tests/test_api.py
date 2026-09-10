from risk_reasoning.models.anthropic_client import AnthropicClient
from risk_reasoning.prompts.templates import Message

cfg = {
    "name": "test",
    "api_model_id": "claude-sonnet-4-5-20250929",
    "serving": {},
    "generation_defaults": {"temperature": 0.0, "max_new_tokens": 32},
}

client = AnthropicClient(cfg)

out = client.generate([
    Message(role="user", content="Say hello in one sentence.")
])

print(out[0].text)
# Output received: Hello, it's nice to meet you!