"""Secure credential storage for AI providers.

API keys are stored server-side only, never exposed to frontend.
Keys are stored in a local JSON file with restricted permissions.
All access goes through this module — no other module should read
or write API keys directly.

Security principles:
- Keys are never logged
- Keys are never included in error messages
- Keys are masked in all API responses
- Users can disconnect and revoke stored credentials
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Credential:
    """A stored API key for a provider."""
    provider: str
    api_key: str  # Stored server-side only
    api_key_masked: str  # For display purposes
    connected_at: str = ""
    last_used: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        """Return only masked key for external use."""
        return {
            "provider": self.provider,
            "api_key_masked": self.api_key_masked,
            "connected_at": self.connected_at,
            "last_used": self.last_used,
            "notes": self.notes,
        }

    def to_internal_dict(self) -> dict:
        """Return full credential including actual key (server-side only)."""
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "api_key_masked": self.api_key_masked,
            "connected_at": self.connected_at,
            "last_used": self.last_used,
            "notes": self.notes,
        }


def mask_key(key: str) -> str:
    """Mask an API key for display. Shows first 4 and last 4 chars."""
    if not key or len(key) <= 12:
        return "****" if key else ""
    return f"{key[:4]}...{key[-4:]}"


class CredentialStore:
    """Persistent store for provider API keys.

    Keys are stored in a local JSON file with restricted permissions.
    The file is created with 0600 permissions to prevent other users
    from reading the credentials.
    """

    def __init__(self, store_path: str | Path = "neural/ai_credentials.json"):
        self.store_path = Path(store_path)
        self.credentials: dict[str, Credential] = {}
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            with open(self.store_path) as f:
                data = json.load(f)
            for pname, cdata in data.get("credentials", {}).items():
                self.credentials[pname] = Credential(
                    provider=cdata["provider"],
                    api_key=cdata["api_key"],
                    api_key_masked=cdata.get("api_key_masked", mask_key(cdata["api_key"])),
                    connected_at=cdata.get("connected_at", ""),
                    last_used=cdata.get("last_used", ""),
                    notes=cdata.get("notes", ""),
                )
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            pass

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "credentials": {k: v.to_internal_dict() for k, v in self.credentials.items()},
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        # Write with restricted permissions
        fd = os.open(str(self.store_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        # Ensure permissions are set (in case file already existed)
        try:
            os.chmod(str(self.store_path), stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    def store(self, provider: str, api_key: str) -> Credential:
        """Store an API key for a provider."""
        key = api_key.strip()
        cred = Credential(
            provider=provider,
            api_key=key,
            api_key_masked=mask_key(key),
            connected_at=datetime.now(timezone.utc).isoformat(),
        )
        self.credentials[provider] = cred
        self._save()
        return cred

    def get(self, provider: str) -> Credential | None:
        """Get a stored credential for a provider."""
        cred = self.credentials.get(provider)
        if cred:
            cred.last_used = datetime.now(timezone.utc).isoformat()
            self._save()
        return cred

    def get_api_key(self, provider: str) -> str | None:
        """Get the actual API key for a provider (server-side only)."""
        cred = self.credentials.get(provider)
        if cred:
            cred.last_used = datetime.now(timezone.utc).isoformat()
            self._save()
            return cred.api_key
        return None

    def remove(self, provider: str) -> bool:
        """Remove a stored credential for a provider."""
        if provider not in self.credentials:
            return False
        del self.credentials[provider]
        self._save()
        return True

    def list_credentials(self) -> list[dict]:
        """List all stored credentials with masked keys (for display)."""
        return [c.to_dict() for c in self.credentials.values()]

    def has_credential(self, provider: str) -> bool:
        return provider in self.credentials
