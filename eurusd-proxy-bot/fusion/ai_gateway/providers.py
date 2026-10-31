"""Provider adapters — provider-specific API clients behind one interface.

Each adapter knows how to:
  - List available models for a given API key
  - Get model capabilities/metadata
  - Test model availability
  - Construct chat completion requests
  - Parse responses

All HTTP calls use stdlib urllib (consistent with existing codebase).
No API key is ever logged or included in error messages.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
# Provider Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class ProviderConfig:
    """Trusted provider configuration — endpoints are internal, never user-supplied."""
    name: str
    display_name: str
    base_url: str
    models_endpoint: str
    chat_endpoint: str
    api_key_header: str
    api_key_prefix: str  # For detection: prefix pattern in the key
    key_patterns: list[str] = field(default_factory=list)  # Regex patterns for detection
    docs_url: str = ""


# Registry of trusted providers — endpoints are determined internally
PROVIDERS: dict[str, ProviderConfig] = {
    "nvidia_nim": ProviderConfig(
        name="nvidia_nim",
        display_name="NVIDIA NIM",
        base_url="https://integrate.api.nvidia.com",
        models_endpoint="/v1/models",
        chat_endpoint="/v1/chat/completions",
        api_key_header="Authorization",
        api_key_prefix="nvapi-",
        key_patterns=["^nvapi-[A-Za-z0-9_-]+$"],
        docs_url="https://docs.nvidia.com/nim",
    ),
    "openai": ProviderConfig(
        name="openai",
        display_name="OpenAI",
        base_url="https://api.openai.com",
        models_endpoint="/v1/models",
        chat_endpoint="/v1/chat/completions",
        api_key_header="Authorization",
        api_key_prefix="sk-",
        key_patterns=["^sk-(?!ant-|or-|proj-)[A-Za-z0-9_-]{20,}$"],
        docs_url="https://platform.openai.com/docs",
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        display_name="Anthropic",
        base_url="https://api.anthropic.com",
        models_endpoint="/v1/models",
        chat_endpoint="/v1/messages",
        api_key_header="x-api-key",
        api_key_prefix="sk-ant-",
        key_patterns=["^sk-ant-[A-Za-z0-9_-]+$"],
        docs_url="https://docs.anthropic.com",
    ),
    "gemini": ProviderConfig(
        name="gemini",
        display_name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com",
        models_endpoint="/v1beta/models",
        chat_endpoint="/v1beta/models/{model}:generateContent",
        api_key_header="x-goog-api-key",
        api_key_prefix="AIza",
        key_patterns=["^AIza[A-Za-z0-9_-]+$"],
        docs_url="https://ai.google.dev",
    ),
    "groq": ProviderConfig(
        name="groq",
        display_name="Groq",
        base_url="https://api.groq.com",
        models_endpoint="/openai/v1/models",
        chat_endpoint="/openai/v1/chat/completions",
        api_key_header="Authorization",
        api_key_prefix="gsk_",
        key_patterns=["^gsk_[A-Za-z0-9_-]+$"],
        docs_url="https://console.groq.com",
    ),
    "mistral": ProviderConfig(
        name="mistral",
        display_name="Mistral AI",
        base_url="https://api.mistral.ai",
        models_endpoint="/v1/models",
        chat_endpoint="/v1/chat/completions",
        api_key_header="Authorization",
        api_key_prefix="sk-",
        key_patterns=["^[a-zA-Z0-9]{32,}$"],  # Mistral uses 32+ char alphanumeric
        docs_url="https://docs.mistral.ai",
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        display_name="DeepSeek",
        base_url="https://api.deepseek.com",
        models_endpoint="/models",
        chat_endpoint="/chat/completions",
        api_key_header="Authorization",
        api_key_prefix="sk-",
        key_patterns=["^sk-[A-Za-z0-9]{32,}$"],  # DeepSeek uses sk- + 32+ chars
        docs_url="https://api-docs.deepseek.com",
    ),
    "cohere": ProviderConfig(
        name="cohere",
        display_name="Cohere",
        base_url="https://api.cohere.ai",
        models_endpoint="/v1/models",
        chat_endpoint="/v1/chat",
        api_key_header="Authorization",
        api_key_prefix="CohereAPIKey-",
        key_patterns=["^[A-Za-z0-9]{40,}$"],
        docs_url="https://docs.cohere.com",
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        display_name="OpenRouter",
        base_url="https://openrouter.ai",
        models_endpoint="/api/v1/models",
        chat_endpoint="/api/v1/chat/completions",
        api_key_header="Authorization",
        api_key_prefix="sk-or-",
        key_patterns=["^sk-or-[A-Za-z0-9_-]+$"],
        docs_url="https://openrouter.ai/docs",
    ),
}


# ─────────────────────────────────────────────────────────────
# IPv4 urlopen helper (reuse pattern from provider.py)
# ─────────────────────────────────────────────────────────────

def _ipv4_urlopen(request, timeout=10.0):
    """urllib opener that forces IPv4 resolution (avoids IPv6 hangs)."""
    host = request.host
    if ':' in host:
        host_only = host.rsplit(':', 1)[0]
    else:
        host_only = host
    try:
        addrinfo = socket.getaddrinfo(host_only, None, family=socket.AF_INET)
        if addrinfo:
            ipv4_addr = addrinfo[0][4][0]
            original_host = request.host
            port_part = ''
            if ':' in original_host:
                port_part = original_host.rsplit(':', 1)[1]
            new_host = f"{ipv4_addr}:{port_part}" if port_part else ipv4_addr
            request.host = new_host
            request.add_header("Host", original_host)
    except socket.gaierror:
        pass
    return urllib.request.urlopen(request, timeout=timeout)


# ─────────────────────────────────────────────────────────────
# Provider Adapter Base
# ─────────────────────────────────────────────────────────────

class ProviderAdapter:
    """Base adapter — all provider-specific logic lives here or in subclasses."""

    def __init__(self, config: ProviderConfig, api_key: str, opener=_ipv4_urlopen):
        self.config = config
        self._api_key = api_key  # Stored server-side only
        self._opener = opener
        self.last_status: int | None = None
        self.calls = 0
        self.failures = 0

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def _headers(self) -> dict[str, str]:
        h = {
            "User-Agent": "saviour-ai-gateway/1.0",
            "Accept": "application/json",
        }
        if self.config.name == "anthropic":
            h[self.config.api_key_header] = self._api_key
            h["anthropic-version"] = "2023-06-01"
        elif self.config.name == "gemini":
            h[self.config.api_key_header] = self._api_key
        else:
            h[self.config.api_key_header] = f"Bearer {self._api_key}"
        return h

    def _request(self, path: str, params: dict | None = None, method: str = "GET") -> dict:
        """Make an authenticated request to the provider."""
        if not self.configured:
            raise ProviderError("No API key configured", status=None, exhausted=True)

        url = f"{self.config.base_url}{path}"
        data = None
        headers = self._headers()
        if params:
            if method == "GET":
                query = urllib.parse.urlencode(params)
                url = f"{url}?{query}"
            else:
                data = json.dumps(params).encode("utf-8")
                headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with self._opener(request, timeout=15.0) as response:
                status = int(getattr(response, "status", 200) or 200)
                body = response.read().decode("utf-8")
            self.last_status = status
            self.calls += 1
            payload = json.loads(body) if body else {}
            if isinstance(payload, dict) and payload.get("status") == "error":
                msg = str(payload.get("message", "provider error"))
                self.failures += 1
                raise ProviderError(msg, status=status)
            if status >= 400:
                self.failures += 1
                raise ProviderError(f"HTTP {status}", status=status)
            return payload
        except urllib.error.HTTPError as error:
            self.failures += 1
            self.last_status = int(error.code)
            raise ProviderError(f"HTTP {error.code}", status=int(error.code)) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            self.failures += 1
            raise ProviderError(f"Network error: {error}", status=None) from error
        except json.JSONDecodeError as error:
            self.failures += 1
            raise ProviderError("Provider returned non-JSON response", status=None) from error

    def list_models(self) -> list[dict]:
        """Fetch all models accessible to this API key.

        Raises ProviderError if the API call fails (auth error, rate
        limit, network issue). Returns [] only if the provider responds
        successfully but has no models.
        """
        path = self.config.models_endpoint
        payload = self._request(path)

        models = []
        if self.config.name == "gemini":
            models_data = payload.get("models", [])
            for m in models_data:
                models.append({
                    "id": m.get("name", "").split("/")[-1],
                    "name": m.get("name", "").split("/")[-1],
                    "provider": self.config.name,
                    "availability": "available",
                    **self._extract_gemini_caps(m),
                })
        elif self.config.name == "anthropic":
            models_data = payload.get("data", [])
            for m in models_data:
                models.append({
                    "id": m.get("id", ""),
                    "name": m.get("id", ""),
                    "provider": self.config.name,
                    "availability": "available" if m.get("archived") is False else "deprecated",
                    **self._extract_anthropic_caps(m),
                })
        else:
            # OpenAI-compatible
            models_data = payload.get("data", [])
            for m in models_data:
                models.append({
                    "id": m.get("id", ""),
                    "name": m.get("id", ""),
                    "provider": self.config.name,
                    "availability": "available",
                    **self._extract_openai_caps(m, payload),
                })

        return [m for m in models if m["id"]]

    def test_model(self, model_id: str) -> dict:
        """Test if a specific model is available and responding."""
        try:
            if self.config.name == "anthropic":
                self._request(
                    self.config.chat_endpoint.replace("{model}", model_id),
                    params={
                        "model": model_id,
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    method="POST",
                )
            elif self.config.name == "gemini":
                self._request(
                    self.config.chat_endpoint.replace("{model}", model_id),
                    params={"contents": [{"parts": [{"text": "hi"}]}]},
                    method="POST",
                )
            else:
                self._request(
                    self.config.chat_endpoint,
                    params={
                        "model": model_id,
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    method="POST",
                )
            return {"available": True, "latency_ms": None}
        except ProviderError as e:
            return {"available": False, "error": str(e), "latency_ms": None}

    def _extract_openai_caps(self, model: dict, payload: dict) -> dict:
        """Extract capabilities from OpenAI-compatible model response."""
        caps = {}
        if "context_length" in model or "context_window" in model:
            caps["context_length"] = model.get("context_length", model.get("context_window"))
        if "owner" in model:
            caps["owner"] = model.get("owner")
        if "created" in model:
            caps["created"] = model.get("created")
        if "deprecated" in model:
            caps["deprecated"] = model.get("deprecated")
        if "vision" in model or "supports_vision" in model:
            caps["vision"] = model.get("vision", model.get("supports_vision"))
        return caps

    def _extract_anthropic_caps(self, model: dict) -> dict:
        caps = {}
        if "display_name" in model:
            caps["display_name"] = model.get("display_name")
        if "created_at" in model:
            caps["created"] = model.get("created_at")
        if "archived" in model:
            caps["deprecated"] = model.get("archived")
        return caps

    def _extract_gemini_caps(self, model: dict) -> dict:
        caps = {}
        if "displayName" in model:
            caps["display_name"] = model.get("displayName")
        if "description" in model:
            caps["description"] = model.get("description")
        if "inputTokenLimit" in model:
            caps["context_length"] = model.get("inputTokenLimit")
        if "outputTokenLimit" in model:
            caps["max_output_tokens"] = model.get("outputTokenLimit")
        if "thinkingConfig" in model:
            caps["reasoning"] = True
        # Gemini models generally support vision
        if model.get("name", "").startswith("gemini-") and "vision" not in caps:
            caps["vision"] = True
        return caps


class ProviderError(Exception):
    """A provider call that did not return usable data."""
    def __init__(self, message: str, status: int | None = None, exhausted: bool = False):
        super().__init__(message)
        self.status = status
        self.exhausted = exhausted


def get_adapter(provider_name: str, api_key: str, opener=None) -> ProviderAdapter:
    """Get a provider adapter for the given provider name."""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}")
    if opener is None:
        opener = _ipv4_urlopen
    return ProviderAdapter(PROVIDERS[provider_name], api_key, opener)
