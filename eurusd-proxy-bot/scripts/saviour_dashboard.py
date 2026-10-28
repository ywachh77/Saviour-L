#!/usr/bin/env python3
"""
SAVIOUR DASHBOARD — Multi-Page Trading Bot Interface

A web dashboard with 5 pages, each showing a different aspect of the
signal prediction system. This is how people build bots that predict
future signals: data → indicators → ML model → combiner → signals.

Pages:
  /            Home — overview, stats, pipeline explanation
  /chart       Combiner Chart — multi-indicator fusion visualization
  /signals     Signals — ML prediction table with outcomes
  /indicators  Indicators — all indicator values + documentation
  /generator   Signal Generator — predicts FUTURE signals

Run: python3 scripts/saviour_dashboard.py
Then open: http://localhost:8765
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES, FEATURE_LABELS
from fusion.indicators import (
    ema, rsi, atr, bollinger, supertrend_window, vwap, stochastic,
    choppiness_index, smc_structure, keltner_squeeze, adx, cci,
    ichimoku_cloud, hurst_exponent, aroon, donchian_channel, pivot_points,
)
from fusion.gbdt import GradientBoostingModel

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
PORT = 8765
N_CHART = 200
HORIZON = 5
MIN_BARS = 30


# =====================================================================
# DATA LOADING
# =====================================================================

def load_candles() -> list[Candle]:
    candles = []
    for batch_file in sorted(CANDLE_DIR.glob("batch_*.txt")):
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
    candles.sort(key=lambda c: c.timestamp)
    return candles


def load_ensemble() -> list[GradientBoostingModel]:
    ensemble_path = CANDLE_DIR / "ensemble_model.json"
    if not ensemble_path.exists():
        return []
    data = json.loads(ensemble_path.read_text())
    models = []
    for entry in data.get("ensemble", []):
        model = GradientBoostingModel.from_dict(entry["model"])
        if model:
            models.append(model)
    return models


def compute_all_indicators(candles: list[Candle]) -> dict:
    """Compute all indicators for the dashboard."""
    closes = [c.close for c in candles]
    n = len(candles)

    st_dirs, st_flips = supertrend_window(candles, period=10, multiplier=3.0, window=52)

    bb_lower, bb_middle, bb_upper = [None]*n, [None]*n, [None]*n
    for i in range(19, n):
        lo, mid, up = bollinger(closes[max(0, i-19):i+1], period=20, multiplier=2.0)
        bb_lower[i], bb_middle[i], bb_upper[i] = lo, mid, up

    vwap_vals = vwap(candles, period=30)
    rsi_vals = rsi(closes, period=14)
    atr_vals = atr(candles, period=14)
    stoch_k, stoch_d = stochastic(candles, period=14, smooth=3)
    chop_vals = choppiness_index(candles, period=14)
    atr_for_smc = [v if v is not None else 0.0 for v in atr_vals]
    fvg_bull, fvg_bear, ob_bull, ob_bear = smc_structure(candles, atr_for_smc)
    squeeze_on, keltner_mom = keltner_squeeze(candles)
    ema_20 = ema(closes, 20)
    ema_50 = ema(closes, 50)
    adx_vals, di_plus, di_minus = adx(candles, period=14)
    cci_vals = cci(candles, period=20)

    # NEW INDICATORS
    tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku_cloud(candles)
    hurst_vals = hurst_exponent(closes, max_lag=20)
    aroon_up, aroon_down = aroon(candles, period=25)
    donch_upper, donch_lower, donch_mid = donchian_channel(candles, period=20)
    pivot, r1, r2, s1, s2 = pivot_points(candles, period=14)

    return {
        "supertrend_dir": st_dirs, "supertrend_flip": st_flips,
        "bb_lower": bb_lower, "bb_middle": bb_middle, "bb_upper": bb_upper,
        "vwap": vwap_vals, "rsi": rsi_vals, "atr": atr_vals,
        "stoch_k": stoch_k, "stoch_d": stoch_d, "choppiness": chop_vals,
        "fvg_bull": fvg_bull, "fvg_bear": fvg_bear,
        "ob_bull": ob_bull, "ob_bear": ob_bear,
        "squeeze": squeeze_on, "keltner_mom": keltner_mom,
        "ema_20": ema_20, "ema_50": ema_50,
        "adx": adx_vals, "di_plus": di_plus, "di_minus": di_minus,
        "cci": cci_vals,
        "tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a,
        "senkou_b": senkou_b, "chikou": chikou,
        "hurst": hurst_vals,
        "aroon_up": aroon_up, "aroon_down": aroon_down,
        "donch_upper": donch_upper, "donch_lower": donch_lower, "donch_mid": donch_mid,
        "pivot": pivot, "r1": r1, "r2": r2, "s1": s1, "s2": s2,
    }


def compute_ml_signals(candles, models) -> list[dict]:
    if not models:
        return []
    feature_dicts = compute_features(candles)
    signals = []
    for i in range(len(candles) - N_CHART - HORIZON, len(candles) - HORIZON):
        if i < MIN_BARS:
            continue
        feat = feature_dicts[i]
        if feat is None:
            continue
        probs = [m.predict_up(feat) for m in models]
        avg_prob = sum(probs) / len(probs)
        current_close = candles[i].close
        future_close = candles[i + HORIZON].close
        actual_up = future_close > current_close
        predicted_up = avg_prob > 0.5
        signals.append({
            "ts": candles[i].timestamp.timestamp(),
            "close": current_close,
            "prob": avg_prob,
            "predicted_up": predicted_up,
            "actual_up": actual_up,
            "correct": predicted_up == actual_up,
            "future_close": future_close,
            "pips": (future_close - current_close) * 10000,
        })
    return signals


def compute_combiner_score(candles, ind, ml_signals) -> list[dict]:
    """Combine all indicators into a single signal score."""
    n = len(candles)
    signals = {}
    for s in ml_signals:
        signals[s["ts"]] = s

    results = []
    for i in range(N_CHART, n):
        ts = candles[i].timestamp.timestamp()
        close = candles[i].close
        score = 0.0
        components = []
        direction = "FLAT"

        ml = signals.get(ts)
        if ml:
            ml_conf = ml["prob"] if ml["predicted_up"] else 1 - ml["prob"]
            score += ml_conf * 0.30
            direction = "BUY" if ml["predicted_up"] else "SELL"
            components.append(f"ML:{ml_conf:.2f}")

        st = ind["supertrend_dir"][i]
        if st > 0:
            score += 0.15 if direction == "BUY" else -0.15
            components.append("ST:+")
        elif st < 0:
            score += 0.15 if direction == "SELL" else -0.15
            components.append("ST:-")

        r = ind["rsi"][i]
        if r is not None:
            if r < 30:
                score += 0.15; components.append(f"RSI:{r:.0f}(OS)")
            elif r > 70:
                score += 0.15; components.append(f"RSI:{r:.0f}(OB)")

        sk = ind["stoch_k"][i]
        sd = ind["stoch_d"][i]
        if sk is not None and sd is not None:
            if sk < 20 and sk > sd:
                score += 0.10; components.append("Stoch:OS")
            elif sk > 80 and sk < sd:
                score += 0.10; components.append("Stoch:OB")

        v = ind["vwap"][i]
        if v is not None:
            if close > v:
                score += 0.05; components.append("VWAP:+")
            else:
                score += 0.05; components.append("VWAP:-")

        ch = ind["choppiness"][i]
        if ch is not None:
            if ch < 38.2:
                score += 0.05; components.append("Chop:Trend")
            elif ch > 61.8:
                score -= 0.05; components.append("Chop:Range")

        if ind["fvg_bull"][i] > 0 or ind["ob_bull"][i] > 0:
            score += 0.10; components.append("SMC:Bull")
            if direction == "FLAT": direction = "BUY"
        if ind["fvg_bear"][i] > 0 or ind["ob_bear"][i] > 0:
            score += 0.10; components.append("SMC:Bear")
            if direction == "FLAT": direction = "SELL"

        if ind["squeeze"][i]:
            score += 0.05; components.append("Sqz:On")

        # NEW: Ichimoku cloud
        tenkan = ind["tenkan"][i]
        kijun = ind["kijun"][i]
        if tenkan is not None and kijun is not None:
            if tenkan > kijun:
                score += 0.05; components.append("Ichi:+")
                if direction == "FLAT": direction = "BUY"
            elif tenkan < kijun:
                score += 0.05; components.append("Ichi:-")
                if direction == "FLAT": direction = "SELL"

        # NEW: Aroon
        au = ind["aroon_up"][i]
        ad = ind["aroon_down"][i]
        if au is not None and ad is not None:
            if au > 70 and au > ad:
                score += 0.05; components.append("Aroon:+")
            elif ad > 70 and ad > au:
                score += 0.05; components.append("Aroon:-")

        # NEW: Hurst
        h = ind["hurst"][i]
        if h is not None:
            if h > 0.55:
                score += 0.03; components.append(f"Hurst:{h:.2f}(T)")
            elif h < 0.45:
                score -= 0.03; components.append(f"Hurst:{h:.2f}(MR)")

        score = max(-1.0, min(1.0, score))

        if score > 0.4:
            sig_dir, sig_type = "BUY", "STRONG"
        elif score > 0.2:
            sig_dir, sig_type = "BUY", "MODERATE"
        elif score < -0.4:
            sig_dir, sig_type = "SELL", "STRONG"
        elif score < -0.2:
            sig_dir, sig_type = "SELL", "MODERATE"
        else:
            sig_dir, sig_type = "FLAT", "NONE"

        results.append({
            "ts": ts, "close": close, "score": round(score, 3),
            "direction": sig_dir, "strength": sig_type,
            "components": " ".join(components),
        })

    return results


def predict_future_signals(candles, ind, models, n_future=10) -> list[dict]:
    """Predict signals for the NEXT n_future candles.

    This is the core of a signal prediction bot:
    1. Take the last candle's indicator values
    2. Predict ML probability for the next candle
    3. Compute combiner score based on current conditions
    4. Display predicted signal with confidence
    """
    if not models:
        return []

    feature_dicts = compute_features(candles)
    last_idx = len(candles) - 1
    last_feat = feature_dicts[last_idx]
    if last_feat is None:
        return []

    predictions = []
    last_close = candles[last_idx].close

    for h in range(1, n_future + 1):
        # Predict ML probability
        probs = [m.predict_up(last_feat) for m in models]
        ml_prob = sum(probs) / len(probs)
        ml_conf = ml_prob if ml_prob > 0.5 else 1 - ml_prob
        ml_dir = "BUY" if ml_prob > 0.5 else "SELL"

        # Combiner score based on current indicator values
        score = ml_conf * 0.30
        components = [f"ML:{ml_conf:.2f}"]
        direction = ml_dir

        st = ind["supertrend_dir"][last_idx]
        if st > 0:
            score += 0.15 if direction == "BUY" else -0.15
            components.append("ST:+")
        elif st < 0:
            score += 0.15 if direction == "SELL" else -0.15
            components.append("ST:-")

        r = ind["rsi"][last_idx]
        if r is not None:
            if r < 30:
                score += 0.15; components.append(f"RSI:{r:.0f}(OS)")
            elif r > 70:
                score += 0.15; components.append(f"RSI:{r:.0f}(OB)")

        sk = ind["stoch_k"][last_idx]
        sd = ind["stoch_d"][last_idx]
        if sk is not None and sd is not None:
            if sk < 20 and sk > sd:
                score += 0.10; components.append("Stoch:OS")
            elif sk > 80 and sk < sd:
                score += 0.10; components.append("Stoch:OB")

        v = ind["vwap"][last_idx]
        if v is not None:
            if last_close > v:
                score += 0.05; components.append("VWAP:+")
            else:
                score += 0.05; components.append("VWAP:-")

        ch = ind["choppiness"][last_idx]
        if ch is not None:
            if ch < 38.2:
                score += 0.05; components.append("Chop:Trend")
            elif ch > 61.8:
                score -= 0.05; components.append("Chop:Range")

        if ind["fvg_bull"][last_idx] > 0 or ind["ob_bull"][last_idx] > 0:
            score += 0.10; components.append("SMC:Bull")
            if direction == "FLAT": direction = "BUY"
        if ind["fvg_bear"][last_idx] > 0 or ind["ob_bear"][last_idx] > 0:
            score += 0.10; components.append("SMC:Bear")
            if direction == "FLAT": direction = "SELL"

        if ind["squeeze"][last_idx]:
            score += 0.05; components.append("Sqz:On")

        tenkan = ind["tenkan"][last_idx]
        kijun = ind["kijun"][last_idx]
        if tenkan is not None and kijun is not None:
            if tenkan > kijun:
                score += 0.05; components.append("Ichi:+")
            elif tenkan < kijun:
                score += 0.05; components.append("Ichi:-")

        au = ind["aroon_up"][last_idx]
        ad = ind["aroon_down"][last_idx]
        if au is not None and ad is not None:
            if au > 70 and au > ad:
                score += 0.05; components.append("Aroon:+")
            elif ad > 70 and ad > au:
                score += 0.05; components.append("Aroon:-")

        h_val = ind["hurst"][last_idx]
        if h_val is not None:
            if h_val > 0.55:
                score += 0.03; components.append(f"Hurst:{h_val:.2f}(T)")
            elif h_val < 0.45:
                score -= 0.03; components.append(f"Hurst:{h_val:.2f}(MR)")

        score = max(-1.0, min(1.0, score))

        if score > 0.4:
            sig_dir, sig_type = "BUY", "STRONG"
        elif score > 0.2:
            sig_dir, sig_type = "BUY", "MODERATE"
        elif score < -0.4:
            sig_dir, sig_type = "SELL", "STRONG"
        elif score < -0.2:
            sig_dir, sig_type = "SELL", "MODERATE"
        else:
            sig_dir, sig_type = "FLAT", "NONE"

        # Estimate future price
        est_move = (ml_prob - 0.5) * 2 * ind["atr"][last_idx] if ind["atr"][last_idx] else 0.0
        est_price = last_close + est_move

        predictions.append({
            "bars_ahead": h,
            "direction": sig_dir,
            "strength": sig_type,
            "score": round(score, 3),
            "ml_prob": round(ml_prob, 4),
            "est_price": round(est_price, 5),
            "est_pips": round(est_move * 10000, 2),
            "components": " ".join(components),
        })

    return predictions


# =====================================================================
# HTML GENERATION
# =====================================================================

NAV_HTML = """
<nav class="nav">
  <div class="nav-brand">⛩️ SAVIOUR</div>
  <div class="nav-links">
    <a href="/" class="{home}">Home</a>
    <a href="/chart" class="{chart}">Chart</a>
    <a href="/signals" class="{signals}">Signals</a>
    <a href="/indicators" class="{indicators}">Indicators</a>
    <a href="/generator" class="{generator}">Generator</a>
  </div>
</nav>
"""

def nav_active(page: str) -> str:
    pages = {"home": "", "chart": "", "signals": "", "indicators": "", "generator": ""}
    pages[page] = "active"
    return NAV_HTML.format(**pages)


CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: #0a0a0f; color: #e0e0e0;
  font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh;
}
.nav {
  background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%);
  padding: 12px 30px; border-bottom: 2px solid #ff6b35;
  display: flex; justify-content: space-between; align-items: center;
  position: sticky; top: 0; z-index: 100;
}
.nav-brand { font-size: 20px; font-weight: 700; color: #ff6b35; }
.nav-links { display: flex; gap: 20px; }
.nav-links a {
  color: #888; text-decoration: none; font-size: 14px; font-weight: 600;
  padding: 6px 12px; border-radius: 6px; transition: all 0.2s;
}
.nav-links a:hover { color: #e0e0e0; background: #1a1a2e; }
.nav-links a.active { color: #ff6b35; background: rgba(255,107,53,0.1); }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { font-size: 24px; color: #ff6b35; margin-bottom: 4px; }
h2 { font-size: 18px; color: #e0e0e0; margin-bottom: 12px; margin-top: 20px; }
.subtitle { font-size: 12px; color: #666; margin-bottom: 20px; }
.stats-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin-bottom: 20px;
}
.stat-card {
  background: #12121e; border: 1px solid #2a2a3e; border-radius: 10px;
  padding: 14px; text-align: center;
}
.stat-card .label { font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.stat-card .value { font-size: 22px; font-weight: 700; }
.green { color: #2da44e; } .red { color: #f85149; }
.orange { color: #ff6b35; } .yellow { color: #d29922; } .blue { color: #58a6ff; }
.panel { background: #12121e; border: 1px solid #2a2a3e; border-radius: 10px; padding: 15px; margin-bottom: 15px; }
.panel h3 { font-size: 13px; color: #888; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
table { width: 100%; border-collapse: collapse; }
th { background: #1a1a2e; padding: 10px 14px; text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #888; border-bottom: 1px solid #2a2a3e; }
td { padding: 8px 14px; font-size: 12px; border-bottom: 1px solid #1a1a2e; }
tr:hover { background: #1a1a2e; }
.signal-buy { color: #2da44e; } .signal-sell { color: #f85149; } .signal-flat { color: #666; }
.strength-strong { color: #ff6b35; font-weight: 700; } .strength-moderate { color: #d29922; }
.score-bar { display: inline-block; width: 50px; height: 6px; background: #2a2a3e; border-radius: 3px; overflow: hidden; vertical-align: middle; margin-right: 6px; }
.score-fill { height: 100%; border-radius: 3px; }
canvas { width: 100%; display: block; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
@media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
.pipeline-step {
  display: flex; align-items: flex-start; gap: 15px; padding: 12px;
  border-bottom: 1px solid #1a1a2e;
}
.pipeline-num {
  width: 32px; height: 32px; border-radius: 50%; background: #ff6b35;
  color: #0a0a0f; display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 14px; flex-shrink: 0;
}
.pipeline-content h4 { font-size: 14px; color: #e0e0e0; margin-bottom: 4px; }
.pipeline-content p { font-size: 12px; color: #888; line-height: 1.5; }
.indicator-row {
  display: grid; grid-template-columns: 180px 1fr 1fr; gap: 10px;
  padding: 10px 0; border-bottom: 1px solid #1a1a2e; align-items: center;
}
.indicator-name { font-weight: 600; font-size: 13px; color: #e0e0e0; }
.indicator-value { font-size: 14px; font-family: monospace; }
.indicator-desc { font-size: 11px; color: #666; }
.footer { text-align: center; padding: 20px; color: #444; font-size: 11px; }
"""


def generate_home(stats: dict, candles: list[Candle], ml_signals: list[dict], combiner: list[dict], models: list) -> str:
    total = len(ml_signals)
    correct = sum(1 for s in ml_signals if s["correct"])
    accuracy = correct / total if total else 0
    buys = [c for c in combiner if c["direction"] == "BUY"]
    sells = [c for c in combiner if c["direction"] == "SELL"]
    strong = [c for c in combiner if c["strength"] == "STRONG"]
    last_close = candles[-1].close if candles else 0
    last_ts = candles[-1].timestamp.strftime("%Y-%m-%d %H:%M UTC") if candles else "N/A"

    pipeline = [
        ("Data Collection", "Fetch candles from Supabase or exchange. Each candle has OHLC + volume. Stored in batch files for offline training."),
        ("Feature Engineering", "Compute 30+ features from raw candles: returns, volatility, momentum, pattern detection. All causal — no look-ahead."),
        ("Indicator Computation", "Calculate 20+ technical indicators: RSI, ATR, Bollinger, Supertrend, Ichimoku, Hurst, Aroon, Donchian, Pivot Points, SMC Structure."),
        ("Model Training", "Train 3-model GBDT ensemble. Each model is a gradient-boosted decision tree. Ensemble averages predictions for robustness."),
        ("ML Prediction", "For each candle, the ensemble outputs P(price goes up in next 5 bars). This is the ML signal probability."),
        ("Combiner Scoring", "Fuse ML prediction + 12 indicator votes into one score (-1 to +1). Weights: ML 30%, Supertrend 15%, RSI 15%, Stochastic 10%, VWAP 10%, SMC 10%, Ichimoku 5%, Aroon 5%, Choppiness 5%, Squeeze 5%, Hurst 3%."),
        ("Signal Generation", "Score > 0.4 = STRONG BUY, > 0.2 = MODERATE BUY, < -0.4 = STRONG SELL, < -0.2 = MODERATE SELL. Future signals are predicted from the last candle's indicator state."),
        ("Signal Output", "Display on dashboard, send to Telegram, or trigger broker orders. Charter enforced: late signals discarded, spread charged on every trade."),
    ]

    pipeline_html = ""
    for i, (title, desc) in enumerate(pipeline):
        pipeline_html += f"""
        <div class="pipeline-step">
          <div class="pipeline-num">{i+1}</div>
          <div class="pipeline-content">
            <h4>{title}</h4>
            <p>{desc}</p>
          </div>
        </div>"""

    recent_signals_html = ""
    for s in ml_signals[-10:]:
        sig_class = f"signal-{'buy' if s['predicted_up'] else 'sell'}"
        actual_class = f"signal-{'buy' if s['actual_up'] else 'sell'}"
        prob_pct = s["prob"] * 100
        prob_color = "#2da44e" if s["prob"] > 0.5 else "#f85149"
        dt = datetime.fromtimestamp(s["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
        recent_signals_html += f"""
        <tr>
          <td>{dt}</td>
          <td>{s['close']:.5f}</td>
          <td class="{sig_class}">{"BUY" if s['predicted_up'] else "SELL"}</td>
          <td><div class="score-bar"><div class="score-fill" style="width:{prob_pct}%;background:{prob_color}"></div></div>{prob_pct:.1f}%</td>
          <td class="{actual_class}">{"UP" if s['actual_up'] else "DOWN"}</td>
          <td class="{'signal-buy' if s['correct'] else 'signal-sell'}">{"✓" if s['correct'] else "✗"}</td>
          <td class="{'signal-buy' if s['correct'] else 'signal-sell'}">{s['pips']:+.2f}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAVIOUR — Trading Bot Dashboard</title>
<style>{CSS}</style>
</head>
<body>
{nav_active("home")}
<div class="container">
  <h1>⛩️ SAVIOUR FUSION DASHBOARD</h1>
  <div class="subtitle">Multi-indicator fusion trading bot · {last_ts} · EUR/USD M1 · Last Close: {last_close:.5f}</div>

  <div class="stats-grid">
    <div class="stat-card"><div class="label">Candles</div><div class="value orange">{len(candles)}</div></div>
    <div class="stat-card"><div class="label">ML Signals</div><div class="value orange">{total}</div></div>
    <div class="stat-card"><div class="label">ML Accuracy</div><div class="value {'green' if accuracy > 0.5 else 'red'}">{accuracy:.1%}</div></div>
    <div class="stat-card"><div class="label">Combiner BUY</div><div class="value green">{len(buys)}</div></div>
    <div class="stat-card"><div class="label">Combiner SELL</div><div class="value red">{len(sells)}</div></div>
    <div class="stat-card"><div class="label">Strong Signals</div><div class="value orange">{len(strong)}</div></div>
    <div class="stat-card"><div class="label">Ensemble Models</div><div class="value blue">{len(models)}</div></div>
    <div class="stat-card"><div class="label">Indicators</div><div class="value yellow">21</div></div>
  </div>

  <div class="panel">
    <h3>How a Signal Prediction Bot Works</h3>
    {pipeline_html}
  </div>

  <div class="panel">
    <h3>Recent ML Signals (Last 10)</h3>
    <table>
      <thead><tr><th>Time</th><th>Close</th><th>Pred</th><th>Prob</th><th>Actual</th><th>Result</th><th>Pips</th></tr></thead>
      <tbody>{recent_signals_html}</tbody>
    </table>
  </div>
</div>
<div class="footer">SAVIOUR FUSION — Research/Paper Only · No Real-Money Orders · Honesty Charter Enforced</div>
</body>
</html>"""


def generate_chart(candles, ind, ml_signals, combiner) -> str:
    chart_candles = candles[-N_CHART:]
    offset = len(candles) - len(chart_candles)
    highs = [c.high for c in chart_candles]
    lows = [c.low for c in chart_candles]
    price_high = max(highs)
    price_low = min(lows)
    price_range = price_high - price_low or 0.001

    candle_data = []
    for idx, c in enumerate(chart_candles):
        gi = offset + idx
        candle_data.append({
            "ts": c.timestamp.isoformat(),
            "o": c.open, "h": c.high, "l": c.low, "c": c.close,
            "dt": c.timestamp.strftime("%m-%d %H:%M"),
            "bb_lower": ind["bb_lower"][gi], "bb_middle": ind["bb_middle"][gi], "bb_upper": ind["bb_upper"][gi],
            "vwap": ind["vwap"][gi], "ema20": ind["ema_20"][gi], "ema50": ind["ema_50"][gi],
            "st_dir": ind["supertrend_dir"][gi],
            "rsi": ind["rsi"][gi], "atr": ind["atr"][gi],
            "tenkan": ind["tenkan"][gi], "kijun": ind["kijun"][gi],
            "senkou_a": ind["senkou_a"][gi], "senkou_b": ind["senkou_b"][gi],
        })

    ml_chart = [s for s in ml_signals if s["ts"] >= chart_candles[0].timestamp.timestamp()]
    ml_data = [{
        "ts": s["ts"], "close": s["close"], "prob": round(s["prob"], 4),
        "predicted_up": s["predicted_up"], "actual_up": s["actual_up"],
        "correct": s["correct"], "pips": round(s["pips"], 2),
        "dt": datetime.fromtimestamp(s["ts"], tz=timezone.utc).strftime("%m-%d %H:%M"),
    } for s in ml_chart]

    combiner_chart = [c for c in combiner if c["ts"] >= chart_candles[0].timestamp.timestamp()]
    combiner_data = [{
        "ts": c["ts"], "close": c["close"], "score": c["score"],
        "direction": c["direction"], "strength": c["strength"],
        "components": c["components"],
        "dt": datetime.fromtimestamp(c["ts"], tz=timezone.utc).strftime("%m-%d %H:%M"),
    } for c in combiner_chart]

    total = len(ml_signals)
    correct = sum(1 for s in ml_signals if s["correct"])
    accuracy = correct / total if total else 0
    buys = [c for c in combiner if c["direction"] == "BUY"]
    sells = [c for c in combiner if c["direction"] == "SELL"]
    strong = [c for c in combiner if c["strength"] == "STRONG"]

    # Build combiner signal table
    ctable = ""
    for c in combiner_data[-20:]:
        sig_class = f"signal-{c['direction'].lower()}"
        strength_class = f"strength-{c['strength'].lower()}" if c['strength'] != 'NONE' else ''
        score_pct = abs(c["score"]) * 100
        score_color = "#2da44e" if c["score"] > 0 else "#f85149"
        ctable += f"""
        <tr>
          <td>{c['dt']}</td>
          <td>{c['close']:.5f}</td>
          <td class="{sig_class} {strength_class}">{c['direction']}</td>
          <td><div class="score-bar"><div class="score-fill" style="width:{score_pct}%;background:{score_color}"></div></div>{c['score']:+.3f}</td>
          <td style="font-size:10px;color:#666">{c['components']}</td>
        </tr>"""

    # Build ML signal table
    mtable = ""
    for s in ml_data[-20:]:
        sig_class = f"signal-{'buy' if s['predicted_up'] else 'sell'}"
        actual_class = f"signal-{'buy' if s['actual_up'] else 'sell'}"
        prob_pct = s["prob"] * 100
        prob_color = "#2da44e" if s["prob"] > 0.5 else "#f85149"
        mtable += f"""
        <tr>
          <td>{s['dt']}</td>
          <td>{s['close']:.5f}</td>
          <td class="{sig_class}">{"BUY" if s['predicted_up'] else "SELL"}</td>
          <td><div class="score-bar"><div class="score-fill" style="width:{prob_pct}%;background:{prob_color}"></div></div>{prob_pct:.1f}%</td>
          <td class="{actual_class}">{"UP" if s['actual_up'] else "DOWN"}</td>
          <td class="{'signal-buy' if s['correct'] else 'signal-sell'}">{s['pips']:+.2f}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAVIOUR — Combiner Chart</title>
<style>{CSS}</style>
</head>
<body>
{nav_active("chart")}
<div class="container">
  <h1>⛩️ SAVIOUR COMBINER CHART</h1>
  <div class="subtitle">Multi-indicator fusion · Last {N_CHART} candles · EUR/USD M1</div>

  <div class="stats-grid">
    <div class="stat-card"><div class="label">ML Accuracy</div><div class="value {'green' if accuracy > 0.5 else 'red'}">{accuracy:.1%}</div></div>
    <div class="stat-card"><div class="label">BUYs</div><div class="value green">{len(buys)}</div></div>
    <div class="stat-card"><div class="label">SELLs</div><div class="value red">{len(sells)}</div></div>
    <div class="stat-card"><div class="label">Strong</div><div class="value orange">{len(strong)}</div></div>
  </div>

  <div class="panel">
    <h3>Candlestick + Multi-Indicator Fusion</h3>
    <canvas id="mainChart" width="1300" height="450"></canvas>
    <canvas id="rsiChart" width="1300" height="80"></canvas>
  </div>

  <div class="grid-2">
    <div class="panel">
      <h3>Combiner Signals (Last 20)</h3>
      <table>
        <thead><tr><th>Time</th><th>Close</th><th>Dir</th><th>Score</th><th>Components</th></tr></thead>
        <tbody>{ctable}</tbody>
      </table>
    </div>
    <div class="panel">
      <h3>ML Predictions (Last 20)</h3>
      <table>
        <thead><tr><th>Time</th><th>Close</th><th>Pred</th><th>Prob</th><th>Actual</th><th>Pips</th></tr></thead>
        <tbody>{mtable}</tbody>
      </table>
    </div>
  </div>
</div>
<div class="footer">SAVIOUR FUSION — Research/Paper Only · No Real-Money Orders · Honesty Charter Enforced</div>

<script>
const candles = {json.dumps(candle_data)};
const mlSignals = {json.dumps(ml_data)};
const combiner = {json.dumps(combiner_data)};
const priceHigh = {price_high};
const priceLow = {price_low};

function drawMain() {{
  const canvas = document.getElementById('mainChart');
  canvas.height = 450;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = 450;
  const padL = 70, padR = 20, padT = 20, padB = 50;
  const cw = W - padL - padR, ch = H - padT - padB;
  const n = candles.length, gap = cw / n, cw2 = Math.max(2, gap * 0.7);
  const yS = p => padT + ch - ((p - priceLow) / (priceHigh - priceLow)) * ch;
  const xS = i => padL + i * gap + gap / 2;

  ctx.fillStyle = '#12121e'; ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = '#1a1a2e'; ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {{
    const y = padT + (ch * i / 5);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '11px monospace';
    ctx.fillText((priceHigh - ((priceHigh - priceLow) * i / 5)).toFixed(5), 5, y + 4);
  }}
  ctx.fillStyle = '#666'; ctx.font = '10px monospace';
  for (let i = 0; i < n; i += 40) ctx.fillText(candles[i].dt, xS(i) - 30, H - padB + 18);

  // BB
  for (let i = 0; i < n; i++) {{
    const c = candles[i];
    if (c.bb_upper !== null) {{
      ctx.strokeStyle = 'rgba(88,166,255,0.15)';
      ctx.beginPath(); ctx.moveTo(xS(i), yS(c.bb_upper)); ctx.lineTo(xS(i), yS(c.bb_lower)); ctx.stroke();
    }}
  }}

  // EMA
  for (let emaKey of ['ema20', 'ema50']) {{
    ctx.strokeStyle = emaKey === 'ema20' ? 'rgba(163,113,247,0.5)' : 'rgba(255,107,53,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    let first = true;
    for (let i = 0; i < n; i++) {{
      const v = candles[i][emaKey];
      if (v !== null) {{
        if (first) {{ ctx.moveTo(xS(i), yS(v)); first = false; }}
        else ctx.lineTo(xS(i), yS(v));
      }}
    }}
    ctx.stroke();
  }}

  // VWAP
  ctx.strokeStyle = 'rgba(88,166,255,0.7)'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  let firstV = true;
  for (let i = 0; i < n; i++) {{
    const v = candles[i].vwap;
    if (v !== null) {{
      if (firstV) {{ ctx.moveTo(xS(i), yS(v)); firstV = false; }}
      else ctx.lineTo(xS(i), yS(v));
    }}
  }}
  ctx.stroke();

  // Ichimoku cloud (senkou A/B)
  ctx.strokeStyle = 'rgba(210,153,34,0.3)'; ctx.lineWidth = 1;
  ctx.beginPath();
  let firstA = true;
  for (let i = 0; i < n; i++) {{
    const v = candles[i].senkou_a;
    if (v !== null) {{
      if (firstA) {{ ctx.moveTo(xS(i), yS(v)); firstA = false; }}
      else ctx.lineTo(xS(i), yS(v));
    }}
  }}
  ctx.stroke();
  ctx.strokeStyle = 'rgba(210,153,34,0.2)';
  ctx.beginPath();
  let firstB = true;
  for (let i = 0; i < n; i++) {{
    const v = candles[i].senkou_b;
    if (v !== null) {{
      if (firstB) {{ ctx.moveTo(xS(i), yS(v)); firstB = false; }}
      else ctx.lineTo(xS(i), yS(v));
    }}
  }}
  ctx.stroke();

  // Candles
  for (let i = 0; i < n; i++) {{
    const c = candles[i];
    const x = xS(i);
    const isUp = c.c >= c.o;
    const col = isUp ? '#2da44e' : '#f85149';
    ctx.strokeStyle = col; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, yS(c.h)); ctx.lineTo(x, yS(c.l)); ctx.stroke();
    const bt = yS(Math.max(c.o, c.c)), bb = yS(Math.min(c.o, c.c));
    ctx.fillStyle = col;
    ctx.fillRect(x - cw2 / 2, bt, cw2, Math.max(1, bb - bt));
  }}

  // ML signals
  const mlMap = {{}};
  mlSignals.forEach(s => {{ mlMap[Math.round(s.ts/60000)*60000] = s; }});
  for (let i = 0; i < n; i++) {{
    const c = candles[i];
    const key = Math.round(new Date(c.ts).getTime()/60000)*60000;
    const sig = mlMap[key];
    if (sig && sig.prob >= 0.7) {{
      const y = yS(c.l) - 12;
      if (sig.predicted_up) {{
        ctx.fillStyle = sig.correct ? '#2da44e' : '#d29922';
        ctx.beginPath();
        ctx.moveTo(xS(i), y - 6); ctx.lineTo(xS(i) - 4, y); ctx.lineTo(xS(i) + 4, y);
        ctx.closePath(); ctx.fill();
      }} else {{
        ctx.fillStyle = sig.correct ? '#f85149' : '#d29922';
        ctx.beginPath();
        ctx.moveTo(xS(i), y + 6); ctx.lineTo(xS(i) - 4, y); ctx.lineTo(xS(i) + 4, y);
        ctx.closePath(); ctx.fill();
      }}
    }}
  }}

  // Combiner strong
  const cmMap = {{}};
  combiner.forEach(c => {{ cmMap[Math.round(c.ts/60000)*60000] = c; }});
  for (let i = 0; i < n; i++) {{
    const c = candles[i];
    const key = Math.round(new Date(c.ts).getTime()/60000)*60000;
    const csig = cmMap[key];
    if (csig && csig.strength === 'STRONG') {{
      ctx.fillStyle = '#ff6b35';
      ctx.font = 'bold 10px sans-serif';
      ctx.fillText(csig.direction === 'BUY' ? '▲' : '▼', xS(i) - 4, yS(c.h) + 12);
    }}
  }}

  ctx.fillStyle = '#888'; ctx.font = '11px monospace';
  ctx.fillText('Price', 5, padT - 5);
  ctx.fillText('EUR/USD M1', padL, H - 5);
}}

function drawRSI() {{
  const canvas = document.getElementById('rsiChart');
  canvas.height = 80;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = 80;
  const padL = 70, padR = 20, padT = 5, padB = 15;
  const cw = W - padL - padR, ch = H - padT - padB;
  const n = candles.length, gap = cw / n;
  const rsiVals = candles.map(c => c.rsi);
  const rsiMax = 100, rsiMin = 0;
  const yS = v => padT + ch - ((v - rsiMin) / (rsiMax - rsiMin)) * ch;

  ctx.fillStyle = '#12121e'; ctx.fillRect(0, 0, W, H);

  // 30/70 lines
  ctx.strokeStyle = 'rgba(248,81,73,0.3)';
  ctx.beginPath(); ctx.moveTo(padL, yS(70)); ctx.lineTo(W - padR, yS(70)); ctx.stroke();
  ctx.strokeStyle = 'rgba(45,164,78,0.3)';
  ctx.beginPath(); ctx.moveTo(padL, yS(30)); ctx.lineTo(W - padR, yS(30)); ctx.stroke();

  // Grid
  ctx.strokeStyle = '#1a1a2e'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {{
    const y = padT + (ch * i / 4);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '9px monospace';
    ctx.fillText((rsiMax - ((rsiMax - rsiMin) * i / 4)).toFixed(0), 5, y + 4);
  }}

  // RSI line
  ctx.strokeStyle = '#ff6b35'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  let first = true;
  for (let i = 0; i < n; i++) {{
    if (rsiVals[i] !== null) {{
      if (first) {{ ctx.moveTo(padL + i * gap + gap/2, yS(rsiVals[i])); first = false; }}
      else ctx.lineTo(padL + i * gap + gap/2, yS(rsiVals[i]));
    }}
  }}
  ctx.stroke();

  ctx.fillStyle = '#666'; ctx.font = '9px sans-serif';
  ctx.fillText('RSI(14)', padL, H - 2);
}}

drawMain();
drawRSI();
</script>
</body>
</html>"""


def generate_signals(candles, ml_signals, combiner) -> str:
    total = len(ml_signals)
    correct = sum(1 for s in ml_signals if s["correct"])
    accuracy = correct / total if total else 0

    # ML signals table
    ml_html = ""
    for s in ml_signals[-50:]:
        sig_class = f"signal-{'buy' if s['predicted_up'] else 'sell'}"
        actual_class = f"signal-{'buy' if s['actual_up'] else 'sell'}"
        prob_pct = s["prob"] * 100
        prob_color = "#2da44e" if s["prob"] > 0.5 else "#f85149"
        dt = datetime.fromtimestamp(s["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
        ml_html += f"""
        <tr>
          <td>{dt}</td>
          <td>{s['close']:.5f}</td>
          <td class="{sig_class}">{"BUY" if s['predicted_up'] else "SELL"}</td>
          <td><div class="score-bar"><div class="score-fill" style="width:{prob_pct}%;background:{prob_color}"></div></div>{prob_pct:.1f}%</td>
          <td class="{actual_class}">{"UP" if s['actual_up'] else "DOWN"}</td>
          <td class="{'signal-buy' if s['correct'] else 'signal-sell'}">{"✓ CORRECT" if s['correct'] else "✗ WRONG"}</td>
          <td class="{'signal-buy' if s['correct'] else 'signal-sell'}">{s['pips']:+.2f}</td>
        </tr>"""

    # Combiner signals table
    c_html = ""
    for c in combiner[-50:]:
        sig_class = f"signal-{c['direction'].lower()}"
        strength_class = f"strength-{c['strength'].lower()}" if c['strength'] != 'NONE' else ''
        score_pct = abs(c["score"]) * 100
        score_color = "#2da44e" if c["score"] > 0 else "#f85149"
        dt = datetime.fromtimestamp(c["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
        c_html += f"""
        <tr>
          <td>{dt}</td>
          <td>{c['close']:.5f}</td>
          <td class="{sig_class} {strength_class}">{c['direction']}</td>
          <td><div class="score-bar"><div class="score-fill" style="width:{score_pct}%;background:{score_color}"></div></div>{c['score']:+.3f}</td>
          <td style="font-size:10px;color:#666">{c['components']}</td>
        </tr>"""

    buys = [c for c in combiner if c["direction"] == "BUY"]
    sells = [c for c in combiner if c["direction"] == "SELL"]
    strong = [c for c in combiner if c["strength"] == "STRONG"]
    hc = [s for s in ml_signals if s["prob"] >= 0.80 or s["prob"] <= 0.20]
    hc_correct = sum(1 for s in hc if s["correct"])
    hc_acc = hc_correct / len(hc) if hc else 0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAVIOUR — Signals</title>
<style>{CSS}</style>
</head>
<body>
{nav_active("signals")}
<div class="container">
  <h1>⛩️ SIGNAL PREDICTIONS</h1>
  <div class="subtitle">ML ensemble predictions + combiner scores · Last 50 signals</div>

  <div class="stats-grid">
    <div class="stat-card"><div class="label">ML Accuracy</div><div class="value {'green' if accuracy > 0.5 else 'red'}">{accuracy:.1%}</div></div>
    <div class="stat-card"><div class="label">High-Conf (≥0.80)</div><div class="value {'green' if hc_acc > 0.5 else 'yellow'}">{hc_acc:.1%} ({len(hc)} trades)</div></div>
    <div class="stat-card"><div class="label">Combiner BUY</div><div class="value green">{len(buys)}</div></div>
    <div class="stat-card"><div class="label">Combiner SELL</div><div class="value red">{len(sells)}</div></div>
    <div class="stat-card"><div class="label">Strong Signals</div><div class="value orange">{len(strong)}</div></div>
  </div>

  <div class="panel">
    <h3>ML Ensemble Predictions (Last 50)</h3>
    <table>
      <thead><tr><th>Time</th><th>Close</th><th>Pred</th><th>Prob</th><th>Actual</th><th>Result</th><th>Pips</th></tr></thead>
      <tbody>{ml_html}</tbody>
    </table>
  </div>

  <div class="panel">
    <h3>Combiner Signals (Last 50)</h3>
    <table>
      <thead><tr><th>Time</th><th>Close</th><th>Direction</th><th>Score</th><th>Components</th></tr></thead>
      <tbody>{c_html}</tbody>
    </table>
  </div>
</div>
<div class="footer">SAVIOUR FUSION — Research/Paper Only · No Real-Money Orders · Honesty Charter Enforced</div>
</body>
</html>"""


def generate_indicators(candles, ind) -> str:
    last = len(candles) - 1

    indicators = [
        ("EMA(20)", ind["ema_20"][last], "Exponential Moving Average — trend smoothing"),
        ("EMA(50)", ind["ema_50"][last], "Longer-term trend smoothing"),
        ("RSI(14)", ind["rsi"][last], "Relative Strength Index — momentum oscillator"),
        ("ATR(14)", ind["atr"][last], "Average True Range — volatility"),
        ("ADX(14)", ind["adx"][last], "Average Directional Index — trend strength"),
        ("+DI", ind["di_plus"][last], "Plus Directional Indicator — bullish force"),
        ("-DI", ind["di_minus"][last], "Minus Directional Indicator — bearish force"),
        ("CCI(20)", ind["cci"][last], "Commodity Channel Index — cyclical momentum"),
        ("Bollinger Upper", ind["bb_upper"][last], "Upper Bollinger Band — volatility ceiling"),
        ("Bollinger Mid", ind["bb_middle"][last], "20-period SMA — trend baseline"),
        ("Bollinger Lower", ind["bb_lower"][last], "Lower Bollinger Band — volatility floor"),
        ("VWAP", ind["vwap"][last], "Volume Weighted Average Price — institutional benchmark"),
        ("Stochastic %K", ind["stoch_k"][last], "Stochastic oscillator — %K line"),
        ("Stochastic %D", ind["stoch_d"][last], "Stochastic oscillator — %D signal line"),
        ("Choppiness", ind["choppiness"][last], "Choppiness Index — trend vs range (0=trend, 100=choppy)"),
        ("Supertrend", "↑ BULL" if ind["supertrend_dir"][last] > 0 else "↓ BEAR" if ind["supertrend_dir"][last] < 0 else "→ FLAT", "Supertrend direction — trend following"),
        ("Keltner Squeeze", "ON" if ind["squeeze"][last] else "OFF", "Keltner Squeeze — volatility expansion potential"),
        ("Tenkan (Ichimoku)", ind["tenkan"][last], "Ichimoku Tenkan — short-term equilibrium (9-period)"),
        ("Kijun (Ichimoku)", ind["kijun"][last], "Ichimoku Kijun — medium-term equilibrium (26-period)"),
        ("Senkou A", ind["senkou_a"][last], "Ichimoku Senkou A — forward cloud boundary"),
        ("Senkou B", ind["senkou_b"][last], "Ichimoku Senkou B — forward cloud boundary"),
        ("Hurst Exponent", ind["hurst"][last], "Hurst — trend (>0.5) vs mean-reversion (<0.5)"),
        ("Aroon Up", ind["aroon_up"][last], "Aroon Up — % since highest high in 25 periods"),
        ("Aroon Down", ind["aroon_down"][last], "Aroon Down — % since lowest low in 25 periods"),
        ("Donchian Upper", ind["donch_upper"][last], "Donchian upper — 20-period high"),
        ("Donchian Lower", ind["donch_lower"][last], "Donchian lower — 20-period low"),
        ("Pivot Point", ind["pivot"][last], "Classic pivot point — support/resistance center"),
        ("R1 / R2", f"{ind['r1'][last]:.5f} / {ind['r2'][last]:.5f}", "Resistance levels 1 and 2"),
        ("S1 / S2", f"{ind['s1'][last]:.5f} / {ind['s2'][last]:.5f}", "Support levels 1 and 2"),
        ("FVG Bullish", "YES" if ind["fvg_bull"][last] > 0 else "NO", "Fair Value Gap — bullish imbalance zone"),
        ("FVG Bearish", "YES" if ind["fvg_bear"][last] > 0 else "NO", "Fair Value Gap — bearish imbalance zone"),
        ("OB Bullish", "YES" if ind["ob_bull"][last] > 0 else "NO", "Order Block — bullish supply zone"),
        ("OB Bearish", "YES" if ind["ob_bear"][last] > 0 else "NO", "Order Block — bearish demand zone"),
    ]

    rows = ""
    for name, value, desc in indicators:
        if isinstance(value, float):
            val_str = f"{value:.5f}"
        elif value is None:
            val_str = "N/A"
        else:
            val_str = str(value)
        rows += f"""
        <div class="indicator-row">
          <div class="indicator-name">{name}</div>
          <div class="indicator-value">{val_str}</div>
          <div class="indicator-desc">{desc}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAVIOUR — Indicator Library</title>
<style>{CSS}</style>
</head>
<body>
{nav_active("indicators")}
<div class="container">
  <h1>⛩️ INDICATOR LIBRARY</h1>
  <div class="subtitle">All {len(indicators)} indicators with current values · Last candle: {candles[-1].timestamp.strftime('%Y-%m-%d %H:%M UTC')} · Close: {candles[-1].close:.5f}</div>

  <div class="panel">
    <h3>Complete Indicator Reference</h3>
    {rows}
  </div>
</div>
<div class="footer">SAVIOUR FUSION — Research/Paper Only · No Real-Money Orders · Honesty Charter Enforced</div>
</body>
</html>"""


def generate_generator(candles, ind, models, predictions) -> str:
    last = len(candles) - 1
    last_close = candles[last].close
    last_ts = candles[last].timestamp.strftime("%Y-%m-%d %H:%M UTC")

    pred_html = ""
    for p in predictions:
        sig_class = f"signal-{p['direction'].lower()}"
        strength_class = f"strength-{p['strength'].lower()}" if p['strength'] != 'NONE' else ''
        score_pct = abs(p["score"]) * 100
        score_color = "#2da44e" if p["score"] > 0 else "#f85149"
        prob_pct = p["ml_prob"] * 100
        prob_color = "#2da44e" if p["ml_prob"] > 0.5 else "#f85149"
        pred_html += f"""
        <tr>
          <td>+{p['bars_ahead']} bar</td>
          <td class="{sig_class} {strength_class}">{p['direction']}</td>
          <td><div class="score-bar"><div class="score-fill" style="width:{score_pct}%;background:{score_color}"></div></div>{p['score']:+.3f}</td>
          <td><div class="score-bar"><div class="score-fill" style="width:{prob_pct}%;background:{prob_color}"></div></div>{prob_pct:.1f}%</td>
          <td>{p['est_price']:.5f}</td>
          <td class="{'signal-buy' if p['est_pips'] > 0 else 'signal-sell'}">{p['est_pips']:+.2f}</td>
          <td style="font-size:10px;color:#666">{p['components']}</td>
        </tr>"""

    # Current indicator state
    indicator_state = [
        ("Supertrend", "↑ BULL" if ind["supertrend_dir"][last] > 0 else "↓ BEAR" if ind["supertrend_dir"][last] < 0 else "→ FLAT"),
        ("RSI", f"{ind['rsi'][last]:.1f}" if ind['rsi'][last] else "N/A"),
        ("Stochastic %K/%D", f"{ind['stoch_k'][last]:.1f}/{ind['stoch_d'][last]:.1f}" if ind['stoch_k'][last] and ind['stoch_d'][last] else "N/A"),
        ("ATR", f"{ind['atr'][last]:.5f}" if ind['atr'][last] else "N/A"),
        ("VWAP", f"{ind['vwap'][last]:.5f}" if ind['vwap'][last] else "N/A"),
        ("Choppiness", f"{ind['choppiness'][last]:.1f}" if ind['choppiness'][last] else "N/A"),
        ("Ichimoku Tenkan/Kijun", f"{ind['tenkan'][last]:.5f}/{ind['kijun'][last]:.5f}" if ind['tenkan'][last] and ind['kijun'][last] else "N/A"),
        ("Hurst", f"{ind['hurst'][last]:.3f}" if ind['hurst'][last] else "N/A"),
        ("Aroon Up/Down", f"{ind['aroon_up'][last]:.0f}/{ind['aroon_down'][last]:.0f}" if ind['aroon_up'][last] and ind['aroon_down'][last] else "N/A"),
        ("Squeeze", "ON" if ind['squeeze'][last] else "OFF"),
        ("FVG Bull/Bear", f"{'Y' if ind['fvg_bull'][last]>0 else 'N'}/{'Y' if ind['fvg_bear'][last]>0 else 'N'}"),
        ("OB Bull/Bear", f"{'Y' if ind['ob_bull'][last]>0 else 'N'}/{'Y' if ind['ob_bear'][last]>0 else 'N'}"),
    ]

    state_html = ""
    for name, value in indicator_state:
        state_html += f"""
        <div class="indicator-row">
          <div class="indicator-name">{name}</div>
          <div class="indicator-value">{value}</div>
          <div class="indicator-desc"></div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAVIOUR — Signal Generator</title>
<style>{CSS}</style>
</head>
<body>
{nav_active("generator")}
<div class="container">
  <h1>⛩️ SIGNAL GENERATOR</h1>
  <div class="subtitle">Predicts FUTURE signals based on current indicator state · {last_ts} · Close: {last_close:.5f}</div>

  <div class="stats-grid">
    <div class="stat-card"><div class="label">Model Status</div><div class="value {'green' if models else 'red'}">{'ACTIVE' if models else 'NO MODEL'}</div></div>
    <div class="stat-card"><div class="label">Models</div><div class="value blue">{len(models)}</div></div>
    <div class="stat-card"><div class="label">Prediction Bars</div><div class="value orange">10</div></div>
    <div class="stat-card"><div class="label">Horizon</div><div class="value yellow">{HORIZON} bars</div></div>
  </div>

  <div class="grid-2">
    <div class="panel">
      <h3>Current Indicator State</h3>
      {state_html}
    </div>
    <div class="panel">
      <h3>How Future Signals Are Predicted</h3>
      <div class="pipeline-step">
        <div class="pipeline-num">1</div>
        <div class="pipeline-content">
          <h4>Snapshot Current State</h4>
          <p>Read the last candle's indicator values: RSI, ATR, Supertrend, Ichimoku, Hurst, Aroon, VWAP, Choppiness, Squeeze, SMC structure.</p>
        </div>
      </div>
      <div class="pipeline-step">
        <div class="pipeline-num">2</div>
        <div class="pipeline-content">
          <h4>ML Probability</h4>
          <p>Run the ensemble GBDT model on the last candle's features. Output: P(price goes up in next {HORIZON} bars).</p>
        </div>
      </div>
      <div class="pipeline-step">
        <div class="pipeline-num">3</div>
        <div class="pipeline-content">
          <h4>Combiner Scoring</h4>
          <p>Fuse ML probability + 12 indicator votes into one score (-1 to +1). Each indicator contributes a weighted vote based on its current value.</p>
        </div>
      </div>
      <div class="pipeline-step">
        <div class="pipeline-num">4</div>
        <div class="pipeline-content">
          <h4>Signal Tier</h4>
          <p>Score > 0.4 = STRONG BUY, > 0.2 = MODERATE BUY, < -0.4 = STRONG SELL, < -0.2 = MODERATE SELL. Estimated price = last_close + (P-0.5) × 2 × ATR.</p>
        </div>
      </div>
      <div class="pipeline-step">
        <div class="pipeline-num">5</div>
        <div class="pipeline-content">
          <h4>Display & Alert</h4>
          <p>Show predicted signals on dashboard. Send Telegram alerts for STRONG signals. Never place real orders — research/paper only.</p>
        </div>
      </div>
    </div>
  </div>

  <div class="panel">
    <h3>Predicted Signals — Next 10 Bars</h3>
    <table>
      <thead><tr><th>Bar</th><th>Signal</th><th>Score</th><th>ML Prob</th><th>Est Price</th><th>Est Pips</th><th>Components</th></tr></thead>
      <tbody>{pred_html}</tbody>
    </table>
  </div>
</div>
<div class="footer">SAVIOUR FUSION — Research/Paper Only · No Real-Money Orders · Honesty Charter Enforced</div>
</body>
</html>"""


# =====================================================================
# HTTP SERVER
# =====================================================================

class DashboardHandler(BaseHTTPRequestHandler):
    # Class-level cache
    _cache = None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if self._cache is None:
            print("Loading data...")
            candles = load_candles()
            models = load_ensemble()
            ind = compute_all_indicators(candles)
            ml_signals = compute_ml_signals(candles, models)
            combiner = compute_combiner_score(candles, ind, ml_signals)
            future_preds = predict_future_signals(candles, ind, models, n_future=10)
            stats = {
                "candles": len(candles),
                "ml_signals": len(ml_signals),
                "accuracy": sum(1 for s in ml_signals if s["correct"]) / len(ml_signals) if ml_signals else 0,
            }
            self._cache = (candles, models, ind, ml_signals, combiner, future_preds, stats)
            print(f"  Loaded {len(candles)} candles, {len(models)} models, {len(ml_signals)} ML signals, {len(combiner)} combiner signals, {len(future_preds)} future predictions")

        candles, models, ind, ml_signals, combiner, future_preds, stats = self._cache

        if path == "/" or path == "":
            html = generate_home(stats, candles, ml_signals, combiner, models)
        elif path == "/chart":
            html = generate_chart(candles, ind, ml_signals, combiner)
        elif path == "/signals":
            html = generate_signals(candles, ml_signals, combiner)
        elif path == "/indicators":
            html = generate_indicators(candles, ind)
        elif path == "/generator":
            html = generate_generator(candles, ind, models, future_preds)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Suppress logs


def main():
    print("SAVIOUR DASHBOARD — Multi-Page Trading Bot Interface")
    print(f"Starting server on http://localhost:{PORT}")
    print("Pages: / (Home) · /chart · /signals · /indicators · /generator")
    print("Press Ctrl+C to stop")

    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
