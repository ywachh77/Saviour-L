"""SAVIOUR Universal AI Control Center — Provider-agnostic AI gateway.

Connect any supported AI provider with just an API key. The system
auto-detects the provider, discovers all accessible models, monitors
health continuously, and exposes a unified gateway for model routing.

Supported providers: NVIDIA NIM, OpenAI, Anthropic, Gemini, Groq,
Mistral, Cohere, DeepSeek, and any OpenAI-compatible endpoint.

Security: API keys are stored server-side only, never exposed to
frontend. Keys are masked in all API responses.
"""

from __future__ import annotations

from .gateway import AIGateway, GatewayStatus
from .registry import ModelRegistry, ModelInfo, ProviderInfo
from .discovery import detect_provider, validate_and_discover, auto_discover
from .store import CredentialStore
from .monitor import AIMonitor

__all__ = [
    "AIGateway",
    "GatewayStatus",
    "ModelRegistry",
    "ModelInfo",
    "ProviderInfo",
    "detect_provider",
    "validate_and_discover",
    "auto_discover",
    "CredentialStore",
    "AIMonitor",
]
