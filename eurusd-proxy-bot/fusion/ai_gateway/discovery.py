"""Provider detection and model discovery.

When a user enters an API key, this module:
1. Identifies the provider using key pattern matching
2. Authenticates the key by calling the provider's models endpoint
3. Fetches all models accessible to that key
4. Returns discovered models with capabilities

If provider detection is ambiguous, the caller should ask the user
to select the provider. Never guess.
"""

from __future__ import annotations

import re
from typing import Any

from .providers import PROVIDERS, ProviderAdapter, ProviderError, get_adapter


def detect_provider(api_key: str) -> list[str]:
    """Detect provider(s) from an API key using pattern matching.

    Returns a list of candidate provider names. If empty, the key
    doesn't match any known provider pattern. If multiple match,
    the caller should ask the user which provider.
    """
    if not api_key or not api_key.strip():
        return []

    key = api_key.strip()
    candidates = []
    for name, config in PROVIDERS.items():
        for pattern in config.key_patterns:
            if re.match(pattern, key):
                candidates.append(name)
                break

    return candidates


def validate_and_discover(provider_name: str, api_key: str, opener=None) -> dict:
    """Validate an API key against a specific provider and discover models.

    Returns a dict with:
    - provider: provider name
    - display_name: human-readable provider name
    - valid: whether the key authenticated successfully
    - error: error message if validation failed
    - models: list of discovered models with capabilities
    """
    if provider_name not in PROVIDERS:
        return {
            "provider": provider_name,
            "valid": False,
            "error": f"Unknown provider: {provider_name}",
            "models": [],
        }

    config = PROVIDERS[provider_name]
    adapter = get_adapter(provider_name, api_key, opener)

    try:
        models = adapter.list_models()
        return {
            "provider": provider_name,
            "display_name": config.display_name,
            "valid": True,
            "error": None,
            "models": models,
            "calls": adapter.calls,
            "failures": adapter.failures,
        }
    except ProviderError as e:
        return {
            "provider": provider_name,
            "display_name": config.display_name,
            "valid": False,
            "error": str(e),
            "models": [],
            "calls": adapter.calls,
            "failures": adapter.failures,
        }
    except Exception as e:
        return {
            "provider": provider_name,
            "display_name": config.display_name,
            "valid": False,
            "error": str(e),
            "models": [],
            "calls": adapter.calls,
            "failures": adapter.failures,
        }


def auto_discover(api_key: str, opener=None) -> dict:
    """Auto-detect provider from API key and discover models.

    If only one provider matches the key pattern, uses it.
    If multiple match, tries each in order until one validates.
    If none match, returns an error asking the user to select.
    """
    key = api_key.strip() if api_key else ""
    candidates = detect_provider(key)

    if not candidates:
        return {
            "provider": None,
            "valid": False,
            "error": "API key doesn't match any known provider. Please select a provider.",
            "models": [],
            "candidates": [],
        }

    # If only one candidate, use it directly
    if len(candidates) == 1:
        result = validate_and_discover(candidates[0], key, opener)
        result["candidates"] = candidates
        return result

    # Multiple candidates — try each in order
    tried = []
    for provider in candidates:
        result = validate_and_discover(provider, key, opener)
        if result["valid"]:
            result["candidates"] = candidates
            result["tried"] = tried
            return result
        tried.append(provider)

    # None validated
    return {
        "provider": None,
        "valid": False,
        "error": "API key did not validate against any matching provider. Please select a provider.",
        "models": [],
        "candidates": candidates,
        "tried": tried,
    }


def get_provider_info() -> list[dict]:
    """Return metadata about all supported providers (no secrets)."""
    return [
        {
            "name": config.name,
            "display_name": config.display_name,
            "docs_url": config.docs_url,
            "key_prefix": config.api_key_prefix,
        }
        for config in PROVIDERS.values()
    ]
