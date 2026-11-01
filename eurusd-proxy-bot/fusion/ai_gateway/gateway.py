"""Universal Model Gateway — unified interface for all AI providers.

Provides a single routing interface that:
- Routes requests to the correct provider adapter
- Selects models based on capability requirements
- Handles retries and failover
- Tracks usage and latency

The gateway never exposes API keys to callers — it manages
credential retrieval internally via CredentialStore.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from .providers import get_adapter, ProviderError
from .registry import ModelRegistry
from .store import CredentialStore
from .discovery import validate_and_discover


@dataclass
class GatewayStatus:
    """Gateway operational status."""
    providers_connected: int = 0
    models_available: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    last_request: str = ""
    active: bool = False

    def to_dict(self) -> dict:
        return {
            "providers_connected": self.providers_connected,
            "models_available": self.models_available,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "last_request": self.last_request,
            "active": self.active,
        }


class AIGateway:
    """Unified gateway for AI model routing.

    Routes requests to the correct provider based on:
    - Requested model ID (if specified)
    - Capability requirements (vision, reasoning, tool calling)
    - Availability and health
    """

    def __init__(
        self,
        registry: ModelRegistry,
        credential_store: CredentialStore,
    ):
        self.registry = registry
        self.credentials = credential_store
        self.status = GatewayStatus()
        self._latency_samples: list[float] = []

    def connect(self, provider_name: str, api_key: str) -> dict:
        """Connect a provider: validate key, store it, discover models.

        Returns a result dict with validation status and discovered models.
        """
        # Validate the key against the provider
        result = validate_and_discover(provider_name, api_key)

        if result["valid"]:
            # Store the key securely
            self.credentials.store(provider_name, api_key)

            # Register provider and models
            masked = result["models"][0].get("provider", "") if result["models"] else ""
            from .store import mask_key
            self.registry.register_provider(
                provider_name=provider_name,
                display_name=result["display_name"],
                api_key_masked=mask_key(api_key),
                models_count=len(result["models"]),
                calls=result.get("calls", 0),
                failures=result.get("failures", 0),
                docs_url="",
            )
            self.registry.register_models(provider_name, result["models"])

            self.status.providers_connected = len(self.credentials.list_credentials())
            self.status.models_available = self.registry.get_model_count()
            self.status.active = True

            return {
                "ok": True,
                "provider": provider_name,
                "display_name": result["display_name"],
                "models_discovered": len(result["models"]),
                "models": result["models"],
            }
        else:
            return {
                "ok": False,
                "provider": provider_name,
                "error": result["error"],
                "models": [],
            }

    def disconnect(self, provider_name: str) -> bool:
        """Disconnect a provider and revoke stored credentials."""
        self.credentials.remove(provider_name)
        self.registry.disconnect_provider(provider_name)
        self.status.providers_connected = len(self.credentials.list_credentials())
        self.status.models_available = self.registry.get_model_count()
        return True

    def list_available_models(self, provider: str | None = None) -> list[dict]:
        """List all available models, optionally filtered by provider."""
        return self.registry.get_models(provider=provider, available_only=True)

    def select_model(self, requirements: dict | None = None) -> dict | None:
        """Select a model based on capability requirements.

        Requirements can include:
        - provider: specific provider to use
        - vision: needs vision capability
        - reasoning: needs reasoning capability
        - tool_calling: needs tool calling capability
        - min_context: minimum context length
        - max_latency_ms: maximum acceptable latency

        Returns the best matching model info, or None if no match.
        """
        requirements = requirements or {}
        candidates = self.registry.get_models(
            provider=requirements.get("provider"),
            available_only=True,
        )

        best = None
        best_score = -1

        for model in candidates:
            score = 0

            # Capability matching
            if requirements.get("vision") and not model.get("vision"):
                continue
            if requirements.get("reasoning") and not model.get("reasoning"):
                continue
            if requirements.get("tool_calling") and not model.get("tool_calling", True):
                continue

            if requirements.get("min_context"):
                ctx = model.get("context_length") or 0
                if ctx < requirements["min_context"]:
                    continue
                score += ctx

            if requirements.get("max_latency_ms"):
                latency = model.get("latency_ms") or 0
                if latency > requirements["max_latency_ms"]:
                    continue
                score += 10000 - latency  # Prefer lower latency

            # Prefer models with known latency
            if model.get("latency_ms") is not None:
                score += 100

            # Prefer models with more capabilities
            if model.get("vision"):
                score += 50
            if model.get("reasoning"):
                score += 50

            if score > best_score:
                best_score = score
                best = model

        return best

    def chat(self, model_id: str, messages: list[dict], **kwargs) -> dict:
        """Send a chat completion request through the unified gateway.

        Routes to the correct provider based on the model_id prefix
        or by searching the registry.
        """
        start = time.monotonic()
        self.status.total_requests += 1

        # Find the model in the registry
        model = None
        provider_name = None
        for key, m in self.registry.models.items():
            if m.model_id == model_id or key.endswith(f":{model_id}"):
                model = m
                provider_name = m.provider
                break

        if not provider_name:
            # Try to infer provider from model_id prefix
            provider_name = self._infer_provider(model_id)

        if not provider_name:
            self.status.failed_requests += 1
            return {"ok": False, "error": f"Model not found in registry: {model_id}"}

        api_key = self.credentials.get_api_key(provider_name)
        if not api_key:
            self.status.failed_requests += 1
            return {"ok": False, "error": f"No API key stored for provider: {provider_name}"}

        try:
            adapter = get_adapter(provider_name, api_key)

            if provider_name == "anthropic":
                payload = {
                    "model": model_id,
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "messages": messages,
                }
            elif provider_name == "gemini":
                # Gemini uses a different format
                payload = {
                    "contents": [
                        {"role": "user", "parts": [{"text": m.get("content", "")}]}
                        for m in messages
                    ],
                }
            else:
                payload = {
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 1024),
                }
                if kwargs.get("temperature") is not None:
                    payload["temperature"] = kwargs["temperature"]

            if provider_name == "gemini":
                url_path = adapter.config.chat_endpoint.replace("{model}", model_id)
                result = adapter._request(url_path, params=payload, method="POST")
            else:
                result = adapter._request(adapter.config.chat_endpoint, params=payload, method="POST")

            latency = (time.monotonic() - start) * 1000
            self._update_latency(latency)
            self.status.last_request = datetime.now(timezone.utc).isoformat()

            # Update model health
            if model:
                model.last_checked = datetime.now(timezone.utc).isoformat()
                model.latency_ms = latency

            return {
                "ok": True,
                "provider": provider_name,
                "model": model_id,
                "latency_ms": round(latency, 1),
                "response": result,
            }

        except ProviderError as e:
            self.status.failed_requests += 1
            latency = (time.monotonic() - start) * 1000
            self._update_latency(latency)
            return {
                "ok": False,
                "provider": provider_name,
                "model": model_id,
                "error": str(e),
                "latency_ms": round(latency, 1),
            }
        except Exception as e:
            self.status.failed_requests += 1
            return {
                "ok": False,
                "provider": provider_name,
                "model": model_id,
                "error": str(e),
            }

    def _infer_provider(self, model_id: str) -> str | None:
        """Try to infer the provider from a model ID."""
        # Check connected providers
        for provider_name in self.credentials.list_credentials():
            api_key = self.credentials.get_api_key(provider_name)
            if api_key:
                # Quick check: try listing models
                try:
                    adapter = get_adapter(provider_name, api_key)
                    models = adapter.list_models()
                    for m in models:
                        if m.get("id") == model_id:
                            return provider_name
                except ProviderError:
                    continue
        return None

    def _update_latency(self, latency_ms: float) -> None:
        """Update average latency tracking."""
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 100:
            self._latency_samples = self._latency_samples[-100:]
        self.status.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def get_status(self) -> dict:
        """Get gateway status."""
        return {
            **self.status.to_dict(),
            "providers": self.registry.get_connected_providers(),
        }
