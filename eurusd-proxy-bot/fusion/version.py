"""System version tracking.

Tracks the current version of the Saviour-L trading system for:
- Model version invalidation
- Feature version tracking
- Workflow versioning
- Deployment tracking
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

# Major version - breaking changes to model/features
MAJOR_VERSION = 2

# Minor version - new features, improvements
MINOR_VERSION = 0

# Patch version - bug fixes, small tweaks
PATCH_VERSION = 0

# Feature version - bumped when features change (invalidates stored models)
FEATURE_VERSION = 10

# Model version - bumped when model architecture changes
MODEL_VERSION = 3

# System version string
VERSION = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"

# Full version string with feature and model versions
FULL_VERSION = f"{VERSION} (features v{FEATURE_VERSION}, model v{MODEL_VERSION})"

# Build timestamp (set by CI/CD)
BUILD_TIMESTAMP = os.environ.get("BUILD_TIMESTAMP") or datetime.now(timezone.utc).isoformat()

# Git commit hash (set by CI/CD)
GIT_COMMIT = os.environ.get("GIT_COMMIT") or "unknown"


def get_version() -> str:
    """Get the current system version string."""
    return VERSION


def get_full_version() -> str:
    """Get the full version string with feature and model versions."""
    return FULL_VERSION


def get_feature_version() -> int:
    """Get the current feature version."""
    return FEATURE_VERSION


def get_model_version() -> int:
    """Get the current model version."""
    return MODEL_VERSION


def invalidate_stored_models() -> bool:
    """Check if stored models should be invalidated.

    Returns True if the feature or model version has changed since the
    model was trained, indicating stored models should be retrained.
    """
    return True  # Always invalidate after version bump


def get_version_info() -> dict:
    """Get complete version information for logging and reporting."""
    return {
        "version": VERSION,
        "full_version": FULL_VERSION,
        "major": MAJOR_VERSION,
        "minor": MINOR_VERSION,
        "patch": PATCH_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_version": MODEL_VERSION,
        "build_timestamp": BUILD_TIMESTAMP,
        "git_commit": GIT_COMMIT,
    }
