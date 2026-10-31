"""Continuous monitoring for AI providers and models.

Monitors:
- API authentication (periodic validation)
- Provider connectivity
- Model availability (test each model)
- Rate limits (track 429 responses)
- Quota exhaustion
- Response latency

Automatically synchronizes the model registry and detects new or
discontinued models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from .providers import get_adapter, ProviderError
from .registry import ModelRegistry
from .store import CredentialStore


@dataclass
class MonitorStatus:
    """Status of the monitoring system."""
    last_check: str = ""
    last_sync: str = ""
    checks_total: int = 0
    failures_total: int = 0
    providers_monitored: int = 0
    models_checked: int = 0
    active: bool = False

    def to_dict(self) -> dict:
        return {
            "last_check": self.last_check,
            "last_sync": self.last_sync,
            "checks_total": self.checks_total,
            "failures_total": self.failures_total,
            "providers_monitored": self.providers_monitored,
            "models_checked": self.models_checked,
            "active": self.active,
        }


class AIMonitor:
    """Continuously monitors connected AI providers and their models.

    The monitor runs health checks on demand (called by the dashboard)
    rather than as a background thread, to avoid interfering with the
    trading operations daemon.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        credential_store: CredentialStore,
        check_interval_minutes: int = 10,
        model_check_interval_minutes: int = 30,
    ):
        self.registry = registry
        self.credentials = credential_store
        self.check_interval = timedelta(minutes=check_interval_minutes)
        self.model_check_interval = timedelta(minutes=model_check_interval_minutes)
        self.status = MonitorStatus()
        self._last_model_check: dict[str, datetime] = {}
        self._last_provider_check: dict[str, datetime] = {}

    def check_provider(self, provider_name: str, full_check: bool = False) -> dict:
        """Check a single provider's health and sync its models.

        If full_check is True, also tests each model's availability.
        Returns a status dict with details.
        """
        api_key = self.credentials.get_api_key(provider_name)
        if not api_key:
            return {
                "provider": provider_name,
                "ok": False,
                "error": "No API key stored for this provider",
            }

        try:
            adapter = get_adapter(provider_name, api_key)
            start = time.monotonic()
            models = adapter.list_models()
            latency = (time.monotonic() - start) * 1000

            # Sync models to registry
            registered = self.registry.register_models(provider_name, models)

            # Update provider info
            if provider_name in self.registry.providers:
                self.registry.providers[provider_name].last_sync = datetime.now(timezone.utc).isoformat()
                self.registry.providers[provider_name].calls_total = adapter.calls
                self.registry.providers[provider_name].failures_total = adapter.failures
                self.registry.providers[provider_name].last_error = ""

            self._last_provider_check[provider_name] = datetime.now(timezone.utc)

            result = {
                "provider": provider_name,
                "ok": True,
                "models_found": len(models),
                "models_registered": registered,
                "latency_ms": round(latency, 1),
                "calls": adapter.calls,
                "failures": adapter.failures,
            }

            # Full check: test each model
            if full_check and models:
                tested = 0
                for m in models[:20]:  # Test up to 20 models
                    model_id = m.get("id", "")
                    if not model_id:
                        continue
                    key = f"{provider_name}:{model_id}"
                    test_start = time.monotonic()
                    try:
                        adapter.test_model(model_id)
                        test_latency = (time.monotonic() - test_start) * 1000
                        self.registry.mark_available(key, test_latency)
                        tested += 1
                    except ProviderError:
                        self.registry.mark_unavailable(key, "test failed")
                    self._last_model_check[key] = datetime.now(timezone.utc)
                result["models_tested"] = tested

            return result

        except ProviderError as e:
            # Update provider error
            if provider_name in self.registry.providers:
                self.registry.providers[provider_name].last_error = str(e)
            self.status.failures_total += 1
            return {
                "provider": provider_name,
                "ok": False,
                "error": str(e),
                "status": e.status,
            }
        except Exception as e:
            if provider_name in self.registry.providers:
                self.registry.providers[provider_name].last_error = str(e)
            self.status.failures_total += 1
            return {
                "provider": provider_name,
                "ok": False,
                "error": str(e),
            }

    def check_all(self, full_check: bool = False) -> dict:
        """Check all connected providers. Returns aggregate status."""
        connected = self.credentials.list_credentials()
        results = []
        for cred in connected:
            result = self.check_provider(cred["provider"], full_check)
            results.append(result)

        self.status.last_check = datetime.now(timezone.utc).isoformat()
        self.status.checks_total += 1
        self.status.providers_monitored = len(results)
        self.status.models_checked = sum(r.get("models_found", 0) for r in results if r.get("ok"))
        self.status.active = True

        return {
            "ok": True,
            "checked_at": self.status.last_check,
            "providers": results,
            "stats": self.status.to_dict(),
            "registry": self.registry.stats(),
        }

    def should_check(self, provider_name: str, full_check: bool = False) -> bool:
        """Check if a provider is due for a health check."""
        interval = self.model_check_interval if full_check else self.check_interval
        key = f"{provider_name}:{'full' if full_check else 'quick'}"
        last = self._last_provider_check.get(key)
        if last is None:
            return True
        return (datetime.now(timezone.utc) - last) >= interval

    def get_status(self) -> dict:
        """Get monitoring system status."""
        return {
            **self.status.to_dict(),
            "check_interval_minutes": self.check_interval.total_seconds() / 60,
            "model_check_interval_minutes": self.model_check_interval.total_seconds() / 60,
        }
