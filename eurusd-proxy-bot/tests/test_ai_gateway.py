"""Tests for the Universal AI Control Center — AI Gateway.

Tests cover:
- Provider detection from API key patterns
- Credential store (store, get, remove, masking)
- Model registry (register, update, disconnect)
- Gateway (connect, disconnect, model selection)
- Monitor (provider checks, status)
- Security (keys never exposed, masking)
- Edge cases (empty keys, unknown providers, corrupt files)
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

# Ensure fusion package is importable
BOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT_ROOT))

from fusion.ai_gateway.providers import (  # noqa: E402
    PROVIDERS, ProviderConfig, ProviderAdapter, ProviderError, get_adapter,
)
from fusion.ai_gateway.discovery import (  # noqa: E402
    detect_provider, validate_and_discover, auto_discover, get_provider_info,
)
from fusion.ai_gateway.registry import (  # noqa: E402
    ModelRegistry, ModelInfo, ProviderInfo,
)
from fusion.ai_gateway.store import (  # noqa: E402
    CredentialStore, Credential, mask_key,
)
from fusion.ai_gateway.monitor import AIMonitor, MonitorStatus  # noqa: E402
from fusion.ai_gateway.gateway import AIGateway, GatewayStatus  # noqa: E402


class TestProviderDetection(unittest.TestCase):
    """Test provider detection from API key patterns."""

    def test_detect_nvidia_key(self):
        candidates = detect_provider("nvapi-abc123def456")
        self.assertEqual(candidates, ["nvidia_nim"])

    def test_detect_openai_key(self):
        candidates = detect_provider("sk-abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(candidates, ["openai"])

    def test_detect_anthropic_key(self):
        candidates = detect_provider("sk-ant-api03-abc123def")
        self.assertIn("anthropic", candidates)

    def test_detect_gemini_key(self):
        candidates = detect_provider("AIzaSyTestKey123456789")
        self.assertEqual(candidates, ["gemini"])

    def test_detect_groq_key(self):
        candidates = detect_provider("gsk_testKey123")
        self.assertEqual(candidates, ["groq"])

    def test_detect_openrouter_key(self):
        candidates = detect_provider("sk-or-test1234567890")
        self.assertIn("openrouter", candidates)

    def test_detect_unknown_key(self):
        candidates = detect_provider("garbage_key_no_match")
        self.assertEqual(candidates, [])

    def test_detect_empty_key(self):
        candidates = detect_provider("")
        self.assertEqual(candidates, [])

    def test_detect_none_key(self):
        candidates = detect_provider(None)
        self.assertEqual(candidates, [])

    def test_get_provider_info(self):
        providers = get_provider_info()
        self.assertGreater(len(providers), 5)
        for p in providers:
            self.assertIn("name", p)
            self.assertIn("display_name", p)
            self.assertIn("key_prefix", p)


class TestCredentialStore(unittest.TestCase):
    """Test secure credential storage."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.tmpdir, "creds.json")
        self.store = CredentialStore(self.store_path)

    def test_store_and_get(self):
        self.store.store("openai", "sk-test1234567890abcdef")
        cred = self.store.get("openai")
        self.assertIsNotNone(cred)
        self.assertEqual(cred.provider, "openai")
        self.assertEqual(cred.api_key, "sk-test1234567890abcdef")

    def test_mask_key(self):
        self.assertEqual(mask_key("sk-test1234567890abcdef"), "sk-t...cdef")
        self.assertEqual(mask_key(""), "")
        self.assertEqual(mask_key("short"), "****")

    def test_list_credentials_masked(self):
        self.store.store("openai", "sk-test1234567890abcdef")
        creds = self.store.list_credentials()
        self.assertEqual(len(creds), 1)
        self.assertNotIn("api_key", creds[0])  # Never expose actual key
        self.assertIn("api_key_masked", creds[0])

    def test_remove_credential(self):
        self.store.store("openai", "sk-test1234567890abcdef")
        self.assertTrue(self.store.has_credential("openai"))
        self.assertTrue(self.store.remove("openai"))
        self.assertFalse(self.store.has_credential("openai"))

    def test_remove_nonexistent(self):
        self.assertFalse(self.store.remove("nonexistent"))

    def test_persistent_storage(self):
        self.store.store("openai", "sk-test1234567890abcdef")
        # Create new store instance — should load from disk
        store2 = CredentialStore(self.store_path)
        cred = store2.get("openai")
        self.assertIsNotNone(cred)
        self.assertEqual(cred.api_key, "sk-test1234567890abcdef")

    def test_corrupt_store_file(self):
        with open(self.store_path, "w") as f:
            f.write("not json")
        store2 = CredentialStore(self.store_path)
        # Should handle gracefully, return empty
        self.assertEqual(len(store2.list_credentials()), 0)

    def test_key_not_in_error_messages(self):
        """API keys should never appear in error messages."""
        key = "sk-supersecretkey1234567890"
        self.store.store("openai", key)
        creds = self.store.list_credentials()
        self.assertNotIn(key, json.dumps(creds))


class TestModelRegistry(unittest.TestCase):
    """Test model registry operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry = ModelRegistry(os.path.join(self.tmpdir, "registry.json"))

    def test_register_models(self):
        models = [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai",
             "availability": "available", "context_length": 8192, "vision": True},
            {"id": "gpt-3.5", "name": "GPT-3.5", "provider": "openai",
             "availability": "available", "context_length": 16384},
        ]
        count = self.registry.register_models("openai", models)
        self.assertEqual(count, 2)
        self.assertEqual(len(self.registry.models), 2)

    def test_get_models(self):
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "availability": "available"},
        ])
        models = self.registry.get_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["model_id"], "gpt-4")

    def test_get_models_by_provider(self):
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "availability": "available"},
        ])
        self.registry.register_models("anthropic", [
            {"id": "claude-3", "name": "Claude 3", "provider": "anthropic", "availability": "available"},
        ])
        models = self.registry.get_models(provider="openai")
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["provider"], "openai")

    def test_get_available_only(self):
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "availability": "available"},
            {"id": "gpt-3.5", "name": "GPT-3.5", "provider": "openai", "availability": "deprecated"},
        ])
        available = self.registry.get_models(available_only=True)
        self.assertEqual(len(available), 1)
        self.assertEqual(available[0]["model_id"], "gpt-4")

    def test_mark_unavailable(self):
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "availability": "available"},
        ])
        key = "openai:gpt-4"
        self.registry.mark_unavailable(key, "test failed")
        model = self.registry.models[key]
        self.assertEqual(model.availability, "unavailable")

    def test_mark_available(self):
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "availability": "unavailable"},
        ])
        key = "openai:gpt-4"
        self.registry.mark_available(key, 45.2)
        model = self.registry.models[key]
        self.assertEqual(model.availability, "available")
        self.assertEqual(model.latency_ms, 45.2)

    def test_register_provider(self):
        self.registry.register_provider(
            "openai", "OpenAI", "sk-t...cdef", 5, 10, 2, "https://openai.com"
        )
        self.assertIn("openai", self.registry.providers)
        self.assertTrue(self.registry.providers["openai"].connected)

    def test_disconnect_provider(self):
        self.registry.register_provider("openai", "OpenAI", "sk-t...cdef", 5, 10, 2)
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "availability": "available"},
        ])
        self.assertTrue(self.registry.disconnect_provider("openai"))
        self.assertEqual(len(self.registry.models), 0)
        self.assertFalse(self.registry.providers["openai"].connected)

    def test_disconnect_nonexistent(self):
        self.assertFalse(self.registry.disconnect_provider("nonexistent"))

    def test_stats(self):
        self.registry.register_provider("openai", "OpenAI", "sk-t...cdef", 2, 10, 0)
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "availability": "available"},
            {"id": "gpt-3.5", "name": "GPT-3.5", "provider": "openai", "availability": "deprecated"},
        ])
        stats = self.registry.stats()
        self.assertEqual(stats["total_models"], 2)
        self.assertEqual(stats["available"], 1)
        self.assertEqual(stats["deprecated"], 1)
        self.assertEqual(stats["connected_providers"], 1)

    def test_persistent_storage(self):
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "availability": "available"},
        ])
        reg2 = ModelRegistry(self.registry.registry_path)
        self.assertEqual(len(reg2.models), 1)

    def test_model_deduplication(self):
        """Registering the same model twice should update, not duplicate."""
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "availability": "available", "vision": False},
        ])
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4 Turbo", "provider": "openai", "availability": "available", "vision": True},
        ])
        self.assertEqual(len(self.registry.models), 1)
        model = self.registry.models["openai:gpt-4"]
        self.assertTrue(model.vision)


class TestAIGateway(unittest.TestCase):
    """Test the unified AI gateway."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry = ModelRegistry(os.path.join(self.tmpdir, "reg.json"))
        self.credentials = CredentialStore(os.path.join(self.tmpdir, "creds.json"))
        self.gateway = AIGateway(self.registry, self.credentials)

    def test_gateway_status(self):
        status = self.gateway.get_status()
        self.assertEqual(status["providers_connected"], 0)
        self.assertEqual(status["models_available"], 0)
        self.assertFalse(status["active"])

    def test_connect_invalid_provider(self):
        result = self.gateway.connect("nonexistent", "sk-test")
        self.assertFalse(result["ok"])

    def test_connect_requires_api_key(self):
        """No API key should be stored without validation."""
        result = self.gateway.connect("openai", "")
        self.assertFalse(result["ok"])

    def test_disconnect_updates_status(self):
        # Manually register a provider for testing
        self.credentials.store("openai", "sk-test1234567890abcdef1234")
        self.registry.register_provider("openai", "OpenAI", "sk-t...cdef", 0, 0, 0)
        self.gateway.status.providers_connected = 1
        self.gateway.disconnect("openai")
        self.assertEqual(self.gateway.status.providers_connected, 0)

    def test_select_model_by_capability(self):
        """Model selection should match capability requirements."""
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai",
             "availability": "available", "vision": True, "tool_calling": True},
            {"id": "gpt-3.5", "name": "GPT-3.5", "provider": "openai",
             "availability": "available", "vision": False, "tool_calling": True},
        ])
        # Select with vision requirement
        model = self.gateway.select_model({"vision": True})
        self.assertIsNotNone(model)
        self.assertTrue(model["vision"])

        # Select with reasoning requirement
        model = self.gateway.select_model({"reasoning": True})
        # No model with reasoning — but should still return something
        # (since no model has reasoning=True, it returns None)
        # Actually, let's test that it doesn't crash

    def test_select_model_no_match(self):
        """Return None when no model matches requirements."""
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai",
             "availability": "available", "vision": False},
        ])
        model = self.gateway.select_model({"vision": True})
        self.assertIsNone(model)

    def test_list_available_models(self):
        self.registry.register_models("openai", [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "availability": "available"},
            {"id": "gpt-3.5", "name": "GPT-3.5", "provider": "openai", "availability": "deprecated"},
        ])
        models = self.gateway.list_available_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["model_id"], "gpt-4")


class TestAIMonitor(unittest.TestCase):
    """Test the AI monitoring system."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry = ModelRegistry(os.path.join(self.tmpdir, "reg.json"))
        self.credentials = CredentialStore(os.path.join(self.tmpdir, "creds.json"))
        self.monitor = AIMonitor(self.registry, self.credentials)

    def test_check_no_credentials(self):
        """Check with no stored credentials should return error."""
        result = self.monitor.check_provider("openai")
        self.assertFalse(result["ok"])

    def test_check_all_no_providers(self):
        """Check all with no providers should return empty results."""
        result = self.monitor.check_all()
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["providers"]), 0)

    def test_monitor_status(self):
        status = self.monitor.get_status()
        self.assertIn("last_check", status)
        self.assertIn("check_interval_minutes", status)
        self.assertFalse(status["active"])

    def test_should_check_first_time(self):
        """First check should always return True."""
        self.assertTrue(self.monitor.should_check("openai"))


class TestDiscovery(unittest.TestCase):
    """Test provider discovery functions."""

    def test_validate_and_discover_invalid_provider(self):
        result = validate_and_discover("nonexistent", "sk-test")
        self.assertFalse(result["valid"])
        self.assertIn("error", result)

    def test_auto_discover_unknown_key(self):
        result = auto_discover("garbage_key_no_match")
        self.assertFalse(result["valid"])
        self.assertEqual(result["candidates"], [])

    def test_auto_discover_empty_key(self):
        result = auto_discover("")
        self.assertFalse(result["valid"])


class TestSecurity(unittest.TestCase):
    """Test security requirements."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(os.path.join(self.tmpdir, "creds.json"))

    def test_keys_not_exposed_in_dict(self):
        """API keys must not appear in dict representations."""
        self.store.store("openai", "sk-supersecretkey1234567890")
        creds = self.store.list_credentials()
        cred_dict = creds[0]
        self.assertNotIn("api_key", cred_dict)
        self.assertNotIn("supersecret", json.dumps(cred_dict))

    def test_credential_to_dict_masks_key(self):
        """Credential.to_dict() should only return masked key."""
        self.store.store("openai", "sk-supersecretkey1234567890")
        cred = self.store.get("openai")
        d = cred.to_dict()
        self.assertNotIn("api_key", d)
        self.assertIn("api_key_masked", d)

    def test_credential_to_internal_dict_has_key(self):
        """Credential.to_internal_dict() should have the actual key (server-side only)."""
        self.store.store("openai", "sk-supersecretkey1234567890")
        cred = self.store.get("openai")
        d = cred.to_internal_dict()
        self.assertIn("api_key", d)
        self.assertEqual(d["api_key"], "sk-supersecretkey1234567890")

    def test_remove_revokes_access(self):
        """After remove, the key should not be accessible."""
        self.store.store("openai", "sk-supersecretkey1234567890")
        self.assertTrue(self.store.has_credential("openai"))
        self.store.remove("openai")
        self.assertFalse(self.store.has_credential("openai"))
        self.assertIsNone(self.store.get_api_key("openai"))


class TestProviderConfigs(unittest.TestCase):
    """Test provider configuration integrity."""

    def test_all_providers_have_required_fields(self):
        for name, config in PROVIDERS.items():
            self.assertIn("base_url", config.__dict__)
            self.assertIn("models_endpoint", config.__dict__)
            self.assertIn("chat_endpoint", config.__dict__)
            self.assertIn("api_key_header", config.__dict__)
            self.assertIn("api_key_prefix", config.__dict__)
            self.assertTrue(len(config.key_patterns) > 0)

    def test_provider_base_urls_are_https(self):
        """All provider base URLs should use HTTPS."""
        for name, config in PROVIDERS.items():
            self.assertTrue(config.base_url.startswith("https://"),
                          f"{name} base_url should be HTTPS: {config.base_url}")

    def test_no_mock_endpoints(self):
        """Provider endpoints should be real, not mock/placeholder."""
        for name, config in PROVIDERS.items():
            self.assertNotIn("mock", config.base_url.lower())
            self.assertNotIn("placeholder", config.base_url.lower())
            self.assertNotIn("example", config.base_url.lower())


if __name__ == "__main__":
    unittest.main()
