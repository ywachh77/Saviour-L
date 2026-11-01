"""Auto-loader for stored AI credentials.

Loads stored API keys and integrates them into the AI gateway automatically.
This ensures the NVIDIA API key (and any other stored keys) are used
continuously without requiring manual reconnection.

The system does not permanently depend on any single key — if a key
becomes invalid or is removed, the gateway continues to function with
other available providers or falls back to non-AI features.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .store import CredentialStore
from .registry import ModelRegistry
from .gateway import AIGateway
from .discovery import auto_discover


def initialize_ai_gateway(
    state_dir: str | None = None,
    auto_connect: bool = True,
) -> AIGateway | None:
    """Initialize AI gateway with auto-connect for stored credentials.

    Args:
        state_dir: Directory to store credentials (default: neural/)
        auto_connect: Automatically connect stored credentials

    Returns:
        AIGateway instance with stored credentials connected, or None if
        no credentials available or auto_connect is False.
    """
    if not auto_connect:
        return None

    # Initialize credential store
    store_path = f"{state_dir}/ai_credentials.json" if state_dir else "neural/ai_credentials.json"
    store = CredentialStore(store_path=store_path)

    # Check for stored credentials
    credentials = store.list_credentials()
    if not credentials:
        return None

    # Initialize registry and gateway
    registry = ModelRegistry()
    gateway = AIGateway(registry=registry, credential_store=store)

    # Auto-connect each stored credential
    connected_providers = []
    for cred_dict in credentials:
        provider_name = cred_dict.get("provider", "")
        api_key = store.get_api_key(provider_name)
        if api_key:
            try:
                result = gateway.connect(provider_name, api_key)
                if result.get("valid", False):
                    connected_providers.append(provider_name)
                    print(f"[AI Gateway] Auto-connected to {provider_name}")
            except Exception as e:
                # Log error but don't fail — system continues without this provider
                print(f"[AI Gateway] Auto-connect failed for {provider_name}: {e}")

    if connected_providers:
        print(f"[AI Gateway] Connected to {len(connected_providers)} provider(s): {connected_providers}")
        return gateway

    return None


def get_stored_credentials_info(state_dir: str | None = None) -> dict[str, Any]:
    """Get information about stored credentials without exposing keys.

    Args:
        state_dir: Directory containing credentials

    Returns:
        Dict with credential info (masked keys only)
    """
    store_path = f"{state_dir}/ai_credentials.json" if state_dir else "neural/ai_credentials.json"
    store = CredentialStore(store_path=store_path)

    credentials = store.list_credentials()
    result = {
        "total_credentials": len(credentials),
        "credentials": [],
        "auto_connect_enabled": True,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    # credentials is already a list of dicts with masked keys
    for cred in credentials:
        # Map the internal key name to the public one
        if "api_key_masked" in cred:
            cred["masked_key"] = cred["api_key_masked"]
    result["credentials"] = credentials

    return result


def refresh_credentials(state_dir: str | None = None) -> dict[str, Any]:
    """Refresh all stored credentials by re-validating them.

    Args:
        state_dir: Directory containing credentials

    Returns:
        Dict with refresh results for each provider
    """
    store_path = f"{state_dir}/ai_credentials.json" if state_dir else "neural/ai_credentials.json"
    store = CredentialStore(store_path=store_path)

    credentials = store.list_credentials()
    results = {
        "refreshed": [],
        "failed": [],
        "total": len(credentials),
    }

    for provider in credentials:
        api_key = store.get_api_key(provider)
        if api_key:
            try:
                result = auto_discover(api_key)
                if result.get("valid", False):
                    results["refreshed"].append(provider)
                    print(f"[AI Gateway] Refreshed {provider}")
                else:
                    results["failed"].append({"provider": provider, "reason": "invalid"})
            except Exception as e:
                results["failed"].append({"provider": provider, "reason": str(e)})

    return results
