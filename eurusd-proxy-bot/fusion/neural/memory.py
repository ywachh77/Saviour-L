"""Persistent Memory — Supabase integration for neural AI state.

Stores and retrieves:
  - Signal history
  - Prediction outcomes
  - Training metadata
  - Model versions
  - Performance metrics
  - Learning history
  - System health
  - Upgrade history

Uses the existing Supabase project for persistence.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..store import SupabaseStore


# ---------------------------------------------------------------------------
# Memory tables (idempotent — no migrations)
# ---------------------------------------------------------------------------

NEURAL_SIGNALS_TABLE = "neural_signals"
NEURAL_OUTCOMES_TABLE = "neural_outcomes"
NEURAL_TRAINING_TABLE = "neural_training"
NEURAL_MODEL_VERSIONS_TABLE = "neural_model_versions"
NEURAL_HEALTH_TABLE = "neural_health"
NEURAL_UPGRADES_TABLE = "neural_upgrades"


# ---------------------------------------------------------------------------
# Memory Manager
# ---------------------------------------------------------------------------

class NeuralMemory:
    """Persistent memory for the neural AI system.

    Uses Supabase for cloud persistence. Falls back to local JSON files
    when Supabase is unavailable.
    """

    def __init__(self, store: SupabaseStore | None = None, local_dir: str = "neural/memory"):
        self.store = store
        self.local_dir = local_dir
        os.makedirs(local_dir, exist_ok=True)

    # ------------------------------------------------------------------ signals

    def save_signal(self, prediction: dict) -> str:
        """Save a prediction signal."""
        row = {
            "id": prediction.get("signal_id", ""),
            "pair": prediction.get("pair", "EUR/USD"),
            "prediction_time": prediction.get("prediction_time", ""),
            "entry_price": prediction.get("entry_reference_price", 0.0),
            "horizon_minutes": prediction.get("horizon_minutes", 1),
            "action": prediction.get("action", "NO_TRADE"),
            "direction_probs": json.dumps(prediction.get("direction_probs", {})),
            "expected_return_pips": prediction.get("expected_return_pips", 0.0),
            "uncertainty": prediction.get("uncertainty", 0.0),
            "regime": prediction.get("regime", "UNKNOWN"),
            "model_version": prediction.get("model_version", "unknown"),
            "validation_status": prediction.get("validation_status", "PASSED"),
            "reason": prediction.get("reason", ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return self._insert(NEURAL_SIGNALS_TABLE, row, signal_id=row["id"])

    def get_signal(self, signal_id: str) -> dict | None:
        """Retrieve a signal by ID."""
        return self._select_one(NEURAL_SIGNALS_TABLE, "id", signal_id)

    def get_recent_signals(self, limit: int = 100, pair: str | None = None) -> list[dict]:
        """Get recent signals."""
        return self._select(NEURAL_SIGNALS_TABLE, limit, pair=pair)

    # ------------------------------------------------------------------ outcomes

    def save_outcome(self, signal_id: str, settlement: dict) -> str:
        """Save a prediction outcome."""
        row = {
            "id": signal_id,
            "signal_id": signal_id,
            "outcome": settlement.get("outcome", "UNRESOLVED"),
            "entry_price": settlement.get("entry_price", 0.0),
            "exit_price": settlement.get("exit_price", 0.0),
            "gross_pips": settlement.get("gross_pips", 0.0),
            "net_pips": settlement.get("net_pips", 0.0),
            "spread_pips": settlement.get("spread_pips", 1.2),
            "resolved_at": settlement.get("settled_at", datetime.now(timezone.utc).isoformat()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return self._insert(NEURAL_OUTCOMES_TABLE, row, signal_id=signal_id)

    def get_outcome(self, signal_id: str) -> dict | None:
        """Retrieve an outcome by signal ID."""
        return self._select_one(NEURAL_OUTCOMES_TABLE, "signal_id", signal_id)

    def get_unresolved_outcomes(self, limit: int = 100) -> list[dict]:
        """Get unresolved outcomes (pending settlement)."""
        signals = self.get_recent_signals(limit)
        outcomes = []
        for s in signals:
            outcome = self.get_outcome(s["id"])
            if outcome is None or outcome["outcome"] == "UNRESOLVED":
                outcomes.append(s)
        return outcomes

    # ------------------------------------------------------------------ training

    def save_training_report(self, report: dict) -> str:
        """Save a training report."""
        run_id = report.get("run_id", "")
        row = {
            "id": run_id,
            "run_id": run_id,
            "model_version": report.get("model_version", ""),
            "started_at": report.get("started_at", ""),
            "finished_at": report.get("finished_at", ""),
            "total_epochs": report.get("total_epochs", 0),
            "best_epoch": report.get("best_epoch", 0),
            "best_val_loss": report.get("best_val_loss", 0.0),
            "best_val_direction_acc": report.get("best_val_direction_acc", 0.0),
            "test_metrics": json.dumps(report.get("test_metrics", {})),
            "config": json.dumps(report.get("config", {})),
            "neural_config": json.dumps(report.get("neural_config", {})),
            "notes": report.get("notes", ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return self._insert(NEURAL_TRAINING_TABLE, row, signal_id=run_id)

    def get_training_reports(self, limit: int = 50) -> list[dict]:
        """Get recent training reports."""
        return self._select(NEURAL_TRAINING_TABLE, limit)

    # ------------------------------------------------------------------ model versions

    def save_model_version(self, version: dict) -> str:
        """Save model version metadata."""
        v = version.get("version", "")
        row = {
            "id": v,
            "version": v,
            "checkpoint_path": version.get("checkpoint_path", ""),
            "status": version.get("status", "CANDIDATE"),
            "created_at": version.get("created_at", datetime.now(timezone.utc).isoformat()),
            "metrics": json.dumps(version.get("metrics", {})),
            "is_production": version.get("is_production", False),
            "parent_version": version.get("parent_version", ""),
            "notes": version.get("notes", ""),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        return self._insert(NEURAL_MODEL_VERSIONS_TABLE, row, signal_id=v)

    def get_model_versions(self, limit: int = 50) -> list[dict]:
        """Get model versions."""
        return self._select(NEURAL_MODEL_VERSIONS_TABLE, limit)

    # ------------------------------------------------------------------ health

    def save_health(self, health: dict) -> str:
        """Save system health snapshot."""
        ts = datetime.now(timezone.utc).isoformat()
        row = {
            "id": ts,
            "timestamp": ts,
            "is_healthy": health.get("is_healthy", True),
            "health_score": health.get("health_score", 1.0),
            "total_predictions": health.get("total_predictions", 0),
            "error_count": health.get("error_count", 0),
            "consecutive_errors": health.get("consecutive_errors", 0),
            "last_error": health.get("last_error", ""),
            "last_prediction_age": health.get("last_prediction_age_seconds", 0.0),
            "created_at": ts,
        }
        return self._insert(NEURAL_HEALTH_TABLE, row, signal_id=ts)

    def get_recent_health(self, limit: int = 100) -> list[dict]:
        """Get recent health snapshots."""
        return self._select(NEURAL_HEALTH_TABLE, limit)

    # ------------------------------------------------------------------ upgrades

    def save_upgrade(self, upgrade: dict) -> str:
        """Save an upgrade event."""
        ts = datetime.now(timezone.utc).isoformat()
        row = {
            "id": ts,
            "timestamp": ts,
            "from_version": upgrade.get("from_version", ""),
            "to_version": upgrade.get("to_version", ""),
            "status": upgrade.get("status", ""),
            "metrics": json.dumps(upgrade.get("metrics", {})),
            "notes": upgrade.get("notes", ""),
            "created_at": ts,
        }
        return self._insert(NEURAL_UPGRADES_TABLE, row, signal_id=ts)

    def get_upgrade_history(self, limit: int = 50) -> list[dict]:
        """Get upgrade history."""
        return self._select(NEURAL_UPGRADES_TABLE, limit)

    # ------------------------------------------------------------------ low-level

    def _insert(self, table: str, row: dict, signal_id: str = "") -> str:
        """Insert a row into a table, with local fallback."""
        if self.store:
            try:
                # Try insert with on_conflict update
                self.store.supabase.table(table).upsert(row, on_conflict="id").execute()
                return signal_id
            except Exception:
                pass
        # Local fallback
        self._save_local(table, signal_id, row)
        return signal_id

    def _select(self, table: str, limit: int = 100, pair: str | None = None) -> list[dict]:
        """Select rows from a table, with local fallback."""
        if self.store:
            try:
                query = self.store.supabase.table(table).select("*").limit(limit)
                if pair:
                    query = query.eq("pair", pair)
                result = query.execute()
                if result.data:
                    return result.data
            except Exception:
                pass
        return self._load_local(table, limit)

    def _select_one(self, table: str, key: str, value: str) -> dict | None:
        """Select one row by key=value, with local fallback."""
        if self.store:
            try:
                result = self.store.supabase.table(table).select("*").eq(key, value).limit(1).execute()
                if result.data and len(result.data) > 0:
                    return result.data[0]
            except Exception:
                pass
        rows = self._load_local(table, 1000)
        for r in rows:
            if r.get(key) == value:
                return r
        return None

    def _save_local(self, table: str, key: str, row: dict):
        """Save to local JSON file."""
        path = os.path.join(self.local_dir, f"{table}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _load_local(self, table: str, limit: int = 100) -> list[dict]:
        """Load from local JSON file."""
        path = os.path.join(self.local_dir, f"{table}.jsonl")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            lines = f.readlines()
        rows = []
        for line in lines[-limit:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows
