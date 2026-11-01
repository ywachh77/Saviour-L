"""AI Model Registry — tracks discovered models, capabilities, and health.

The registry is populated by provider discovery and continuously updated
by the monitor. It tracks:
- Model name and ID
- Provider
- Availability status
- Context capacity
- Input/output limits
- Reasoning capabilities
- Vision support
- Tool calling
- Streaming
- Latency
- Pricing/free-tier information when verifiable

No model information is ever fabricated — only what the provider reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ModelInfo:
    """Information about a single AI model."""
    model_id: str
    model_name: str
    provider: str
    availability: str = "unknown"  # available, deprecated, unavailable
    context_length: int | None = None
    max_output_tokens: int | None = None
    reasoning: bool = False
    vision: bool = False
    tool_calling: bool = False
    streaming: bool = True
    latency_ms: float | None = None
    pricing: dict[str, Any] = field(default_factory=dict)
    discovered_at: str = ""
    last_checked: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "provider": self.provider,
            "availability": self.availability,
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "reasoning": self.reasoning,
            "vision": self.vision,
            "tool_calling": self.tool_calling,
            "streaming": self.streaming,
            "latency_ms": self.latency_ms,
            "pricing": self.pricing,
            "discovered_at": self.discovered_at,
            "last_checked": self.last_checked,
            "capabilities": self.capabilities,
        }


@dataclass
class ProviderInfo:
    """Information about a connected provider."""
    name: str
    display_name: str
    connected: bool = False
    api_key_masked: str = ""
    models_count: int = 0
    last_sync: str = ""
    last_error: str = ""
    calls_total: int = 0
    failures_total: int = 0
    docs_url: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "connected": self.connected,
            "api_key_masked": self.api_key_masked,
            "models_count": self.models_count,
            "last_sync": self.last_sync,
            "last_error": self.last_error,
            "calls_total": self.calls_total,
            "failures_total": self.failures_total,
            "docs_url": self.docs_url,
        }


class ModelRegistry:
    """Persistent registry of discovered AI models.

    Stores model metadata in a local JSON file. The registry is
    populated by discovery and continuously updated by the monitor.
    """

    def __init__(self, registry_path: str | Path = "neural/ai_registry.json"):
        self.registry_path = Path(registry_path)
        self.providers: dict[str, ProviderInfo] = {}
        self.models: dict[str, ModelInfo] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            with open(self.registry_path) as f:
                data = json.load(f)
            for pname, pdata in data.get("providers", {}).items():
                self.providers[pname] = ProviderInfo(**pdata)
            for mid, mdata in data.get("models", {}).items():
                self.models[mid] = ModelInfo(**mdata)
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            pass

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "providers": {k: v.__dict__ for k, v in self.providers.items()},
            "models": {k: v.__dict__ for k, v in self.models.items()},
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def register_provider(self, provider_name: str, display_name: str,
                          api_key_masked: str, models_count: int,
                          calls: int, failures: int, docs_url: str = "") -> ProviderInfo:
        """Register or update a connected provider."""
        info = ProviderInfo(
            name=provider_name,
            display_name=display_name,
            connected=True,
            api_key_masked=api_key_masked,
            models_count=models_count,
            last_sync=datetime.now(timezone.utc).isoformat(),
            last_error="",
            calls_total=calls,
            failures_total=failures,
            docs_url=docs_url,
        )
        self.providers[provider_name] = info
        self._save()
        return info

    def register_models(self, provider_name: str, models: list[dict]) -> int:
        """Register discovered models for a provider.

        Returns the number of models registered.
        """
        count = 0
        now = datetime.now(timezone.utc).isoformat()
        for m in models:
            model_id = m.get("id", "")
            if not model_id:
                continue
            # Deduplicate: use provider/model_id as key
            key = f"{provider_name}:{model_id}"
            existing = self.models.get(key)
            if existing:
                # Update existing model
                existing.availability = m.get("availability", "available")
                existing.last_checked = now
                existing.latency_ms = m.get("latency_ms", existing.latency_ms)
                existing.capabilities = {**existing.capabilities, **m}
                # Update context length if available
                if m.get("context_length"):
                    existing.context_length = m["context_length"]
                if m.get("max_output_tokens"):
                    existing.max_output_tokens = m["max_output_tokens"]
                # Update vision/reasoning
                existing.vision = m.get("vision", existing.vision)
                existing.reasoning = m.get("reasoning", existing.reasoning)
            else:
                # New model
                info = ModelInfo(
                    model_id=model_id,
                    model_name=m.get("name", model_id),
                    provider=provider_name,
                    availability=m.get("availability", "available"),
                    context_length=m.get("context_length"),
                    max_output_tokens=m.get("max_output_tokens"),
                    reasoning=m.get("reasoning", False),
                    vision=m.get("vision", False),
                    tool_calling=m.get("tool_calling", True),
                    streaming=m.get("streaming", True),
                    latency_ms=m.get("latency_ms"),
                    capabilities=m,
                    discovered_at=now,
                    last_checked=now,
                )
                self.models[key] = info
            count += 1

        # Update provider model count
        if provider_name in self.providers:
            self.providers[provider_name].models_count = count
        self._save()
        return count

    def mark_unavailable(self, model_key: str, reason: str) -> None:
        """Mark a model as unavailable."""
        if model_key in self.models:
            self.models[model_key].availability = "unavailable"
            self.models[model_key].last_checked = datetime.now(timezone.utc).isoformat()
            self._save()

    def mark_available(self, model_key: str, latency_ms: float | None = None) -> None:
        """Mark a model as available with optional latency."""
        if model_key in self.models:
            self.models[model_key].availability = "available"
            self.models[model_key].last_checked = datetime.now(timezone.utc).isoformat()
            if latency_ms is not None:
                self.models[model_key].latency_ms = latency_ms
            self._save()

    def disconnect_provider(self, provider_name: str) -> bool:
        """Disconnect a provider and remove its models from the registry."""
        if provider_name not in self.providers:
            return False
        # Remove all models for this provider
        keys_to_remove = [k for k in self.models if k.startswith(f"{provider_name}:")]
        for k in keys_to_remove:
            del self.models[k]
        self.providers[provider_name].connected = False
        self.providers[provider_name].api_key_masked = ""
        self.providers[provider_name].models_count = 0
        self._save()
        return True

    def get_models(self, provider: str | None = None, available_only: bool = False) -> list[dict]:
        """Get registered models, optionally filtered by provider or availability."""
        result = []
        for key, model in self.models.items():
            if provider and model.provider != provider:
                continue
            if available_only and model.availability != "available":
                continue
            result.append(model.to_dict())
        return result

    def get_providers(self) -> list[dict]:
        """Get all registered providers."""
        return [p.to_dict() for p in self.providers.values()]

    def get_connected_providers(self) -> list[dict]:
        """Get only connected providers."""
        return [p.to_dict() for p in self.providers.values() if p.connected]

    def get_model_count(self) -> int:
        return len(self.models)

    def get_provider_count(self) -> int:
        return len(self.providers)

    def stats(self) -> dict:
        """Registry statistics."""
        available = sum(1 for m in self.models.values() if m.availability == "available")
        unavailable = sum(1 for m in self.models.values() if m.availability == "unavailable")
        deprecated = sum(1 for m in self.models.values() if m.availability == "deprecated")
        return {
            "total_models": len(self.models),
            "available": available,
            "unavailable": unavailable,
            "deprecated": deprecated,
            "total_providers": len(self.providers),
            "connected_providers": sum(1 for p in self.providers.values() if p.connected),
        }
