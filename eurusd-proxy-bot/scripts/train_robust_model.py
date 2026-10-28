#!/usr/bin/env python3
"""
Multi-Scenario Robust Training Pipeline
========================================
Generates diverse market patterns and trains GBDT ensembles against
various conditions to produce a model that handles:

  1. Strong trends (up & down)
  2. Range-bound / sideways markets
  3. High volatility bursts
  4. Low volatility / quiet periods
  5. Mean-reversion patterns
  6. Breakout / impulse patterns
  7. London vs NY session patterns
  8. Volatility clustering (calm → storm → calm)

Each scenario is generated with distinct price action characteristics,
then the ensemble is trained and evaluated per-scenario.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES, FEATURE_VERSION
from fusion.gbdt import GradientBoostingModel

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
OUTPUT_JSON = CANDLE_DIR / "ensemble_model.json"
REPORT_JSON = CANDLE_DIR / "training_report.json"

HORIZON = 5
MIN_BARS = 30
N_MODELS = 5
CANDLESTERS_PER_SCENARIO = 500  # 10 scenarios × 500 = 5,000 candles (same scale as fast model)


# ─── Scenario Generators ──────────────────────────────────────────────

MIN_PRICE = 0.5000  # never let prices drop below this


def make_candle(ts, o, c, spread_frac=0.4):
    """Create a valid candle with proper OHLC relationships."""
    o = max(o, MIN_PRICE)
    c = max(c, MIN_PRICE)
    body = abs(c - o)
    wick = body * random.uniform(0.3, 1.5) if body > 0 else random.uniform(2, 8) * 0.0001
    high = max(o, c) + wick * random.uniform(0.2, spread_frac)
    low = max(min(o, c) - wick * random.uniform(0.2, spread_frac), MIN_PRICE)
    vol = random.randint(1000, 10000)
    return (int(ts), f"{o:.6f}", f"{high:.6f}", f"{low:.6f}", f"{c:.6f}", str(vol))


def gen_strong_trend(base_price, n, direction, pip=0.0001):
    """Strong directional trend with micro-pullbacks."""
    candles = []
    price = base_price
    start = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=random.randint(1, 48))
    ts = int(start.timestamp())
    step = direction * random.uniform(8, 25) * pip
    for i in range(n):
        o = price
        c = price + step + random.uniform(-3, 3) * pip
        # occasional pullback
        if random.random() < 0.15:
            c = o - step * 0.5
        c = max(c, MIN_PRICE)
        candles.append(make_candle(ts, o, c))
        price = c
        ts += 60
    return candles


def gen_range_bound(base_price, n, pip=0.0001):
    """Oscillating range-bound market."""
    candles = []
    price = base_price
    start = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=random.randint(1, 48))
    ts = int(start.timestamp())
    amp = random.uniform(15, 35) * pip
    period = random.randint(30, 80)
    for i in range(n):
        o = price
        phase = (i % period) / period
        c = base_price + amp * math.sin(2 * math.pi * phase) + random.uniform(-2, 2) * pip
        c = max(c, MIN_PRICE)
        candles.append(make_candle(ts, o, c))
        price = c
        ts += 60
    return candles


def gen_high_volatility(base_price, n, pip=0.0001):
    """High volatility with large swings."""
    candles = []
    price = base_price
    start = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=random.randint(1, 48))
    ts = int(start.timestamp())
    for i in range(n):
        o = price
        move = random.gauss(0, 30) * pip
        c = max(o + move, MIN_PRICE)
        candles.append(make_candle(ts, o, c, spread_frac=0.8))
        price = c
        ts += 60
    return candles


def gen_low_volatility(base_price, n, pip=0.0001):
    """Quiet market with tiny moves."""
    candles = []
    price = base_price
    start = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=random.randint(1, 48))
    ts = int(start.timestamp())
    for i in range(n):
        o = price
        move = random.gauss(0, 3) * pip
        c = max(o + move, MIN_PRICE)
        candles.append(make_candle(ts, o, c, spread_frac=0.2))
        price = c
        ts += 60
    return candles


def gen_mean_reversion(base_price, n, pip=0.0001):
    """Mean-reverting pattern — price pulls back to center."""
    candles = []
    price = max(base_price + random.uniform(-20, 20) * pip, MIN_PRICE)
    start = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=random.randint(1, 48))
    ts = int(start.timestamp())
    for i in range(n):
        o = price
        dist = price - base_price
        pull = -dist * random.uniform(0.1, 0.3) + random.uniform(-2, 2) * pip
        c = max(o + pull, MIN_PRICE)
        candles.append(make_candle(ts, o, c))
        price = c
        ts += 60
    return candles


def gen_breakout(base_price, n, pip=0.0001):
    """Breakout pattern — consolidation then impulse."""
    candles = []
    price = base_price
    start = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=random.randint(1, 48))
    ts = int(start.timestamp())
    consolidation = n // 2
    direction = random.choice([1, -1])
    for i in range(n):
        o = price
        if i < consolidation:
            # tight consolidation
            c = max(o + random.uniform(-2, 2) * pip, MIN_PRICE)
            candles.append(make_candle(ts, o, c, spread_frac=0.15))
        else:
            # breakout impulse
            impulse = direction * random.uniform(15, 30) * pip
            c = max(o + impulse, MIN_PRICE)
            candles.append(make_candle(ts, o, c, spread_frac=0.7))
        price = c
        ts += 60
    return candles


def gen_volatility_cluster(base_price, n, pip=0.0001):
    """Calm → storm → calm transition."""
    candles = []
    price = base_price
    start = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=random.randint(1, 48))
    ts = int(start.timestamp())
    third = n // 3
    for i in range(n):
        o = price
        if i < third:
            move = random.gauss(0, 3) * pip  # calm
        elif i < 2 * third:
            move = random.gauss(0, 35) * pip  # storm
        else:
            move = random.gauss(0, 5) * pip  # calming
        c = max(o + move, MIN_PRICE)
        frac = 0.2 if i < third else (0.8 if i < 2 * third else 0.3)
        candles.append(make_candle(ts, o, c, spread_frac=frac))
        price = c
        ts += 60
    return candles


def gen_session_london(base_price, n, pip=0.0001):
    """London session pattern — higher activity, trending."""
    candles = []
    price = base_price
    start = datetime(2024, 1, 15, 7, 0, 0, tzinfo=timezone.utc) + timedelta(hours=random.randint(0, 4))
    ts = int(start.timestamp())
    direction = random.choice([1, -1])
    for i in range(n):
        o = price
        trend = direction * random.uniform(5, 15) * pip
        noise = random.uniform(-4, 4) * pip
        c = max(o + trend + noise, MIN_PRICE)
        candles.append(make_candle(ts, o, c, spread_frac=0.5))
        price = c
        ts += 60
    return candles


def gen_session_ny(base_price, n, pip=0.0001):
    """NY session pattern — different character, more reversals."""
    candles = []
    price = base_price
    start = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc) + timedelta(hours=random.randint(0, 4))
    ts = int(start.timestamp())
    for i in range(n):
        o = price
        if i % 20 < 10:
            move = random.uniform(5, 15) * pip
        else:
            move = -random.uniform(5, 15) * pip
        noise = random.uniform(-3, 3) * pip
        c = max(o + move + noise, MIN_PRICE)
        candles.append(make_candle(ts, o, c, spread_frac=0.4))
        price = c
        ts += 60
    return candles


SCENARIOS = [
    ("strong_trend_up",    lambda p: gen_strong_trend(p, CANDLESTERS_PER_SCENARIO,  1)),
    ("strong_trend_down",  lambda p: gen_strong_trend(p, CANDLESTERS_PER_SCENARIO, -1)),
    ("range_bound",        lambda p: gen_range_bound(p, CANDLESTERS_PER_SCENARIO)),
    ("high_volatility",    lambda p: gen_high_volatility(p, CANDLESTERS_PER_SCENARIO)),
    ("low_volatility",     lambda p: gen_low_volatility(p, CANDLESTERS_PER_SCENARIO)),
    ("mean_reversion",     lambda p: gen_mean_reversion(p, CANDLESTERS_PER_SCENARIO)),
    ("breakout",           lambda p: gen_breakout(p, CANDLESTERS_PER_SCENARIO)),
    ("volatility_cluster", lambda p: gen_volatility_cluster(p, CANDLESTERS_PER_SCENARIO)),
    ("session_london",     lambda p: gen_session_london(p, CANDLESTERS_PER_SCENARIO)),
    ("session_ny",         lambda p: gen_session_ny(p, CANDLESTERS_PER_SCENARIO)),
]


# ─── Training ─────────────────────────────────────────────────────────

def load_all_candles():
    """Load all candles from all scenario batch files, tracking scenario index."""
    candles = []
    scenario_idx = []
    for batch_idx, batch_file in enumerate(sorted(CANDLE_DIR.glob("batch_*.txt"))):
        for line in batch_file.read_text().strip().split(";"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            ts = int(float(parts[0]))
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            candles.append(Candle(
                timestamp=dt,
                open=float(parts[1]),
                high=float(parts[2]),
                low=float(parts[3]),
                close=float(parts[4]),
                volume=float(parts[5]) if len(parts) > 5 else 0.0,
            ))
            scenario_idx.append(batch_idx)
    # Sort by timestamp but keep scenario mapping
    combined = list(zip(candles, scenario_idx))
    combined.sort(key=lambda x: x[0].timestamp)
    candles = [c for c, _ in combined]
    scenario_idx = [s for _, s in combined]
    return candles, scenario_idx


def compute_features_all(candles):
    """Compute features for all candles."""
    feat_dicts = compute_features(candles)
    return feat_dicts


def build_samples(candles, feat_dicts, scenario_idx=None):
    """Build (features, label) samples with optional scenario tracking."""
    samples = []
    sample_scenarios = []
    for i in range(MIN_BARS, len(candles) - HORIZON):
        feat = feat_dicts[i]
        if feat is None:
            continue
        future_close = candles[i + HORIZON].close
        current_close = candles[i].close
        label = 1 if future_close > current_close else 0
        samples.append((feat, label))
        if scenario_idx:
            sample_scenarios.append(scenario_idx[i])
    return samples, sample_scenarios


def split_by_scenario(samples, sample_scenarios):
    """Split samples by scenario index."""
    scenario_samples = {name: [] for name, _ in SCENARIOS}
    for (feat, label), sidx in zip(samples, sample_scenarios):
        if sidx < len(SCENARIOS):
            scenario_samples[SCENARIOS[sidx][0]].append((feat, label))
    return scenario_samples


def train_model(samples, config, seed):
    """Train a single GBDT model."""
    model = GradientBoostingModel(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config.get("subsample", 0.7),
    )
    model.fit(samples, seed=seed)
    return model


def evaluate_model(model, samples):
    """Evaluate a model on samples."""
    correct = 0
    for feat, label in samples:
        pred = model.predict_up(feat)
        if (pred > 0.5) == bool(label):
            correct += 1
    return correct / len(samples) if samples else 0


def main():
    print("=" * 60)
    print("MULTI-SCENARIO ROBUST TRAINING PIPELINE")
    print("=" * 60)

    # ─── Step 1: Generate diverse candles ───
    print("\n[1] Generating diverse candle patterns...")
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    # Clear old batch files
    for f in CANDLE_DIR.glob("batch_*.txt"):
        f.unlink()

    all_candles_raw = []
    base_price = 1.1000
    for idx, (name, gen_fn) in enumerate(SCENARIOS):
        candles = gen_fn(base_price)
        all_candles_raw.extend(candles)
        # Save to batch file
        out_file = CANDLE_DIR / f"batch_{idx:03d}.txt"
        # Write as semicolon-separated
        content = ";".join(",".join(str(x) for x in c) for c in candles)
        out_file.write_text(content)
        print(f"  {name}: {len(candles)} candles")

    print(f"  Total: {len(all_candles_raw)} candles across {len(SCENARIOS)} scenarios")

    # ─── Step 2: Load and compute features ───
    print("\n[2] Computing features...")
    candles, scenario_idx = load_all_candles()
    print(f"  Loaded {len(candles)} candles")
    feat_dicts = compute_features_all(candles)
    samples, sample_scenarios = build_samples(candles, feat_dicts, scenario_idx)
    print(f"  Samples: {len(samples)}")

    # ─── Step 3: Split by scenario ───
    print("\n[3] Splitting samples by scenario...")
    scenario_samples = split_by_scenario(samples, sample_scenarios)
    for name, s_samples in scenario_samples.items():
        if s_samples:
            pos = sum(l for _, l in s_samples)
            print(f"  {name}: {len(s_samples)} samples (pos={pos}, neg={len(s_samples)-pos})")

    # ─── Step 4: Train ensemble with multiple configurations ───
    print("\n[4] Training ensemble (5 models, varied configs)...")
    configs = [
        {"n_estimators": 50,  "max_depth": 3, "learning_rate": 0.10, "subsample": 0.7},
        {"n_estimators": 60,  "max_depth": 3, "learning_rate": 0.15, "subsample": 0.8},
        {"n_estimators": 40,  "max_depth": 4, "learning_rate": 0.12, "subsample": 0.6},
        {"n_estimators": 80,  "max_depth": 3, "learning_rate": 0.08, "subsample": 0.9},
        {"n_estimators": 30,  "max_depth": 2, "learning_rate": 0.20, "subsample": 0.5},
    ]

    models = []
    for i, cfg in enumerate(configs):
        print(f"  Model {i+1}/{len(configs)}: depth={cfg['max_depth']} est={cfg['n_estimators']} lr={cfg['learning_rate']}")
        sys.stdout.flush()
        model = train_model(samples, cfg, seed=42 + i)
        models.append(model)
        print(f"    -> done")
        sys.stdout.flush()

    # ─── Step 5: Evaluate per-scenario ───
    print("\n[5] Per-scenario evaluation...")
    scenario_report = {}
    for name, s_samples in scenario_samples.items():
        if not s_samples:
            continue
        ensemble_correct = 0
        for feat, label in s_samples:
            pred = sum(m.predict_up(feat) for m in models) / len(models)
            if (pred > 0.5) == bool(label):
                ensemble_correct += 1
        acc = ensemble_correct / len(s_samples)
        scenario_report[name] = {"samples": len(s_samples), "accuracy": round(acc, 4)}
        print(f"  {name}: {acc:.1%} ({len(s_samples)} samples)")

    # ─── Step 6: Overall evaluation ───
    print("\n[6] Overall evaluation...")
    overall_correct = 0
    for feat, label in samples:
        pred = sum(m.predict_up(feat) for m in models) / len(models)
        if (pred > 0.5) == bool(label):
            overall_correct += 1
    overall_acc = overall_correct / len(samples) if samples else 0
    print(f"  Overall accuracy: {overall_acc:.1%} ({overall_correct}/{len(samples)})")

    # ─── Step 7: Save model ───
    print("\n[7] Saving model...")
    model_data = {
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "horizon": HORIZON,
        "accuracy": overall_acc,
        "n_samples": len(samples),
        "n_models": len(models),
        "configs": configs,
        "scenarios": scenario_report,
        "models": [],
    }
    for m in models:
        model_data["models"].append(m.to_dict())

    OUTPUT_JSON.write_text(json.dumps(model_data, indent=2))
    print(f"  Saved to {OUTPUT_JSON}")

    # Save training report
    REPORT_JSON.write_text(json.dumps({
        "overall_accuracy": overall_acc,
        "total_samples": len(samples),
        "n_models": len(models),
        "scenarios": scenario_report,
        "configs": configs,
    }, indent=2))
    print(f"  Report saved to {REPORT_JSON}")

    # ─── Summary ───
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Overall Accuracy: {overall_acc:.1%}")
    print(f"Models: {len(models)}")
    print(f"Samples: {len(samples)}")
    print(f"Scenarios: {len(scenario_report)}")
    avg_scenario = sum(s["accuracy"] for s in scenario_report.values()) / len(scenario_report) if scenario_report else 0
    print(f"Average scenario accuracy: {avg_scenario:.1%}")


if __name__ == "__main__":
    main()
