#!/usr/bin/env python3
"""
Win Streak Model Trainer
=======================
Trains a SEPARATE model specifically for the Win Streak compounding system.

This model differs from the main signal generator in:
  - Objective: predicts WIN/LOSS (not direction) — "will this trade be profitable?"
  - Training data: only high-confidence setups (ML_CONF >= 0.68)
  - Hyperparameters: tuned for precision on confident setups
  - Separate JSON file: ensemble_model_winstreak.json

The main signal generator (ensemble_model_fb.json) continues to work
independently. The win streak system loads this model exclusively.
"""

import sys, json, math, random
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES
from fusion.gbdt import GradientBoostingModel
from backtest_engine import load_candles, load_models

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
OUTPUT_JSON = CANDLE_DIR / "ensemble_model_winstreak.json"

PIPS_PER_MOVE = 0.0001
SPREAD_PIPS = 1.2
HORIZON = 5  # 5-bar = 30-min expiry
ML_CONF_THRESHOLD = 0.68
N_MODELS = 3  # Smaller ensemble for the win streak model


def main():
    print("=" * 60)
    print("WIN STREAK MODEL TRAINER")
    print("=" * 60)
    print(f"Output: {OUTPUT_JSON}")
    print(f"ML Conf Threshold: {ML_CONF_THRESHOLD}")
    print(f"Horizon: {HORIZON} bars")
    print()

    candles = load_candles(2000)
    feats = compute_features(candles)
    main_models = load_models(30)

    print(f"Loaded {len(candles)} candles, {len(main_models)} main model members")

    # Step 1: Get main model predictions for all candles
    ml_probs = []
    for i in range(len(candles)):
        feat = feats[i]
        if feat is None:
            ml_probs.append(None)
            continue
        prob = sum(m.predict_up(feat) for m in main_models) / len(main_models)
        ml_probs.append(prob)

    # Step 2: Build training samples
    # Label: 1 = trade wins (net_pips > 0), 0 = trade loses
    # Only include high-confidence setups
    samples = []
    for i in range(60, len(candles) - HORIZON):
        feat = feats[i]
        if feat is None:
            continue
        ml_up = ml_probs[i]
        if ml_up is None:
            continue

        confidence = max(ml_up, 1 - ml_up)
        if confidence < ML_CONF_THRESHOLD:
            continue

        # Determine trade outcome
        direction = 1 if ml_up > 0.5 else -1
        future_close = candles[i + HORIZON].close
        entry = candles[i].close
        gross_pips = direction * (future_close - entry) / PIPS_PER_MOVE
        net_pips = gross_pips - SPREAD_PIPS
        target = 1 if net_pips > 0 else 0

        vec = [feat[name] for name in FEATURE_NAMES]
        samples.append((vec, target, confidence, ml_up))

    print(f"High-confidence samples: {len(samples)}")
    wins = sum(1 for _, t, _, _ in samples if t == 1)
    print(f"Wins: {wins} ({wins/len(samples)*100:.1f}%)")
    print()

    if len(samples) < 30:
        print("ERROR: Too few samples. Lowering threshold...")
        ml_conf = 0.65
        samples = []
        for i in range(60, len(candles) - HORIZON):
            feat = feats[i]
            if feat is None:
                continue
            ml_up = ml_probs[i]
            if ml_up is None:
                continue
            confidence = max(ml_up, 1 - ml_up)
            if confidence < ml_conf:
                continue
            direction = 1 if ml_up > 0.5 else -1
            future_close = candles[i + HORIZON].close
            entry = candles[i].close
            gross_pips = direction * (future_close - entry) / PIPS_PER_MOVE
            net_pips = gross_pips - SPREAD_PIPS
            target = 1 if net_pips > 0 else 0
            vec = [feat[name] for name in FEATURE_NAMES]
            samples.append((vec, target, confidence, ml_up))
        print(f"With threshold {ml_conf}: {len(samples)} samples, "
              f"{sum(1 for _, t, _, _ in samples if t == 1)} wins "
              f"({sum(1 for _, t, _, _ in samples if t == 1)/len(samples)*100:.1f}%)")

    # Step 3: Train multiple models with different feature subsamples
    random.seed(42)
    all_models = []
    feature_subsets = []

    for model_idx in range(N_MODELS):
        # Use 70% of features per model (feature bagging)
        n_features = len(FEATURE_NAMES)
        subsample_size = int(n_features * 0.7)
        indices = list(range(n_features))
        random.shuffle(indices)
        subset = sorted(indices[:subsample_size])
        subset_names = [FEATURE_NAMES[i] for i in subset]
        feature_subsets.append(subset_names)

        # Build training data in dict format for this subset
        model = GradientBoostingModel(
            feature_names=tuple(subset_names),
            feature_version=9,
            n_estimators=80,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.8,
        )

        # Build dict-format samples
        dict_samples = []
        for s in samples:
            feat_vec = s[0]
            target = s[1]
            feat_dict = {subset_names[k]: feat_vec[subset[k]] for k in range(len(subset_names))}
            dict_samples.append((feat_dict, target))

        model.fit(dict_samples)
        all_models.append(model)

        # Quick validation
        val_correct = 0
        for feat_dict, target in dict_samples:
            pred = model.predict_up(feat_dict)
            pred_class = 1 if pred > 0.5 else 0
            if pred_class == target:
                val_correct += 1
        acc = val_correct / len(dict_samples) * 100
        print(f"Model {model_idx+1}/{N_MODELS}: {len(subset_names)} features, train acc {acc:.1f}%")

    # Step 4: Save ensemble
    payload = {
        "ensemble_type": "feature_bagging",
        "full_feature_names": list(FEATURE_NAMES),
        "feature_subsets": [list(subset) for subset in feature_subsets],
        "members": [m.to_dict() for m in all_models],
        "n_members": N_MODELS,
        "version": "winstreak_v1",
        "label": "win_streak",
        "horizon": HORIZON,
        "accuracy": None,
        "ml_conf_threshold": ML_CONF_THRESHOLD,
        "spread_pips": SPREAD_PIPS,
        "pips_per_move": PIPS_PER_MOVE,
        "config": {
            "n_estimators": 80,
            "learning_rate": 0.08,
            "max_depth": 4,
            "subsample": 0.8,
            "feature_subsample": 0.7,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {N_MODELS} models to {OUTPUT_JSON}")
    print(f"Training samples: {len(samples)}")
    print(f"Base win rate: {wins/len(samples)*100:.1f}%")


if __name__ == "__main__":
    main()
