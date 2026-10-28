#!/usr/bin/env python3
"""
SAVIOUR COMBINER — Multi-Indicator Chart Builder

Combines ML predictions + technical indicators into one unified chart.
Like Shuvo Combiner Pro but ours: free, transparent, no black box.

Indicators shown:
  - Candlestick chart (last 200 candles)
  - Supertrend direction lines
  - Bollinger Bands
  - VWAP
  - ML signal arrows (buy/sell)
  - Combiner score panel (aggregated signal strength)
  - RSI sub-chart
  - ATR sub-chart

Theme: Shinigami/Naruto dark (no purple/cyan)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.features import compute_features
from fusion.indicators import (
    ema, rsi, atr, bollinger, supertrend_window,
    vwap, adx, stochastic, smc_structure, keltner_squeeze,
    choppiness_index,
)
from fusion.gbdt import GradientBoostingModel

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
OUTPUT_HTML = CANDLE_DIR / "combiner_chart.html"
HORIZON = 5
MIN_BARS = 30
N_CHART = 200


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
            ts = int(parts[0])
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
    """Compute all indicators for the chart."""
    closes = [c.close for c in candles]
    n = len(candles)

    # Supertrend
    st_dirs, st_flips = supertrend_window(candles, period=10, multiplier=3.0, window=52)

    # Bollinger Bands (rolling)
    bb_lower = [None] * n
    bb_middle = [None] * n
    bb_upper = [None] * n
    for i in range(19, n):
        lo, mid, up = bollinger(closes[max(0, i-19):i+1], period=20, multiplier=2.0)
        bb_lower[i] = lo
        bb_middle[i] = mid
        bb_upper[i] = up

    # VWAP
    vwap_vals = vwap(candles, period=30)

    # RSI
    rsi_vals = rsi(closes, period=14)

    # ATR
    atr_vals = atr(candles, period=14)

    # Stochastic
    stoch_k, stoch_d = stochastic(candles, period=14, smooth=3)

    # Choppiness
    chop_vals = choppiness_index(candles, period=14)

    # SMC Structure
    atr_for_smc = [v if v is not None else 0.0 for v in atr_vals]
    fvg_bull, fvg_bear, ob_bull, ob_bear = smc_structure(candles, atr_for_smc)

    # Keltner Squeeze
    squeeze_on, keltner_mom = keltner_squeeze(candles)

    # EMA (for trend context)
    ema_20 = ema(closes, 20)
    ema_50 = ema(closes, 50)

    return {
        "supertrend_dir": st_dirs,
        "supertrend_flip": st_flips,
        "bb_lower": bb_lower,
        "bb_middle": bb_middle,
        "bb_upper": bb_upper,
        "vwap": vwap_vals,
        "rsi": rsi_vals,
        "atr": atr_vals,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "choppiness": chop_vals,
        "fvg_bull": fvg_bull,
        "fvg_bear": fvg_bear,
        "ob_bull": ob_bull,
        "ob_bear": ob_bear,
        "squeeze": squeeze_on,
        "keltner_mom": keltner_mom,
        "ema_20": ema_20,
        "ema_50": ema_50,
    }


def compute_ml_signals(candles: list[Candle], models: list[GradientBoostingModel]) -> list[dict]:
    """Generate ML predictions for the chart candles."""
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


def compute_combiner_score(candles: list[Candle], ind: dict, ml_signals: list[dict]) -> list[dict]:
    """Combine all indicators into a single signal score.

    Score components:
    - ML prediction confidence (0-1)
    - Supertrend direction match
    - RSI momentum (oversold/overbought)
    - Stochastic signal
    - Bollinger position (near band = reversal potential)
    - VWAP position (trend filter)
    - Choppiness (trend vs range filter)
    - SMC structure (FVG/OB confluence)
    - Squeeze breakout potential

    Returns a list of dicts with combined score and signal direction.
    """
    n = len(candles)
    closes = [c.close for c in candles]
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

        # 1. ML confidence (weight: 30%)
        ml = signals.get(ts)
        if ml:
            ml_conf = ml["prob"] if ml["predicted_up"] else 1 - ml["prob"]
            score += ml_conf * 0.30
            direction = "BUY" if ml["predicted_up"] else "SELL"
            components.append(f"ML:{ml_conf:.2f}")

        # 2. Supertrend (weight: 15%)
        st = ind["supertrend_dir"][i]
        if st > 0:
            score += 0.15 if direction == "BUY" else -0.15
            components.append(f"ST:+")
        elif st < 0:
            score += 0.15 if direction == "SELL" else -0.15
            components.append(f"ST:-")

        # 3. RSI (weight: 15%)
        r = ind["rsi"][i]
        if r is not None:
            if r < 30:
                score += 0.15
                components.append(f"RSI:{r:.0f}(OS)")
            elif r > 70:
                score += 0.15
                components.append(f"RSI:{r:.0f}(OB)")

        # 4. Stochastic (weight: 10%)
        sk = ind["stoch_k"][i]
        sd = ind["stoch_d"][i]
        if sk is not None and sd is not None:
            if sk < 20 and sk > sd:
                score += 0.10
                components.append("Stoch:OS")
            elif sk > 80 and sk < sd:
                score += 0.10
                components.append("Stoch:OB")

        # 5. VWAP (weight: 10%)
        v = ind["vwap"][i]
        if v is not None:
            if close > v:
                score += 0.05
                components.append("VWAP:+")
            else:
                score += 0.05
                components.append("VWAP:-")

        # 6. Choppiness (weight: 5%)
        ch = ind["choppiness"][i]
        if ch is not None:
            if ch < 38.2:  # trending
                score += 0.05
                components.append("Chop:Trend")
            elif ch > 61.8:  # choppy
                score -= 0.05
                components.append("Chop:Range")

        # 7. SMC Structure (weight: 10%)
        if ind["fvg_bull"][i] > 0 or ind["ob_bull"][i] > 0:
            score += 0.10
            components.append("SMC:Bull")
            if direction == "FLAT":
                direction = "BUY"
        if ind["fvg_bear"][i] > 0 or ind["ob_bear"][i] > 0:
            score += 0.10
            components.append("SMC:Bear")
            if direction == "FLAT":
                direction = "SELL"

        # 8. Squeeze breakout (weight: 5%)
        if ind["squeeze"][i]:
            score += 0.05
            components.append("Sqz:On")

        # Normalize score to [-1, 1]
        score = max(-1.0, min(1.0, score))

        # Determine final signal
        if score > 0.4:
            sig_dir = "BUY"
            sig_type = "STRONG"
        elif score > 0.2:
            sig_dir = "BUY"
            sig_type = "MODERATE"
        elif score < -0.4:
            sig_dir = "SELL"
            sig_type = "STRONG"
        elif score < -0.2:
            sig_dir = "SELL"
            sig_type = "MODERATE"
        else:
            sig_dir = "FLAT"
            sig_type = "NONE"

        results.append({
            "ts": ts,
            "close": close,
            "score": round(score, 3),
            "direction": sig_dir,
            "strength": sig_type,
            "components": " ".join(components),
        })

    return results


def generate_html(candles: list[Candle], ind: dict, ml_signals: list[dict], combiner: list[dict]) -> str:
    """Generate the combiner chart HTML."""
    chart_candles = candles[-N_CHART:]
    offset = len(candles) - len(chart_candles)

    # Price range
    highs = [c.high for c in chart_candles]
    lows = [c.low for c in chart_candles]
    price_high = max(highs)
    price_low = min(lows)
    price_range = price_high - price_low or 0.001

    # Prepare data arrays (only chart candles)
    candle_data = []
    for idx, c in enumerate(chart_candles):
        global_i = offset + idx
        candle_data.append({
            "ts": c.timestamp.isoformat(),
            "o": c.open, "h": c.high, "l": c.low, "c": c.close,
            "dt": c.timestamp.strftime("%m-%d %H:%M"),
            "bb_lower": ind["bb_lower"][global_i],
            "bb_middle": ind["bb_middle"][global_i],
            "bb_upper": ind["bb_upper"][global_i],
            "vwap": ind["vwap"][global_i],
            "ema20": ind["ema_20"][global_i],
            "ema50": ind["ema_50"][global_i],
            "st_dir": ind["supertrend_dir"][global_i],
            "st_flip": ind["supertrend_flip"][global_i],
            "rsi": ind["rsi"][global_i],
            "atr": ind["atr"][global_i],
            "stoch_k": ind["stoch_k"][global_i],
            "stoch_d": ind["stoch_d"][global_i],
            "choppiness": ind["choppiness"][global_i],
            "squeeze": ind["squeeze"][global_i],
            "fvg_bull": ind["fvg_bull"][global_i],
            "fvg_bear": ind["fvg_bear"][global_i],
        })

    # ML signals for chart
    ml_chart = [s for s in ml_signals if s["ts"] >= chart_candles[0].timestamp.timestamp()]
    ml_data = [{
        "ts": s["ts"], "close": s["close"], "prob": round(s["prob"], 4),
        "predicted_up": s["predicted_up"], "actual_up": s["actual_up"],
        "correct": s["correct"], "pips": round(s["pips"], 2),
        "dt": datetime.fromtimestamp(s["ts"], tz=timezone.utc).strftime("%m-%d %H:%M"),
    } for s in ml_chart]

    # Combiner signals for chart
    combiner_chart = [c for c in combiner if c["ts"] >= chart_candles[0].timestamp.timestamp()]
    combiner_data = [{
        "ts": c["ts"], "close": c["close"], "score": c["score"],
        "direction": c["direction"], "strength": c["strength"],
        "components": c["components"],
        "dt": datetime.fromtimestamp(c["ts"], tz=timezone.utc).strftime("%m-%d %H:%M"),
    } for c in combiner_chart]

    # Summary
    total = len(ml_signals)
    correct = sum(1 for s in ml_signals if s["correct"])
    accuracy = correct / total if total else 0

    # Combiner stats
    buys = [c for c in combiner if c["direction"] == "BUY"]
    sells = [c for c in combiner if c["direction"] == "SELL"]
    strong = [c for c in combiner if c["strength"] == "STRONG"]

    # Score distribution
    score_vals = [c["score"] for c in combiner]
    avg_score = sum(score_vals) / len(score_vals) if score_vals else 0

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAVIOUR COMBINER — Multi-Indicator Chart</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0a0f;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%);
    padding: 20px 30px;
    border-bottom: 2px solid #ff6b35;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .header h1 {{
    font-size: 24px;
    font-weight: 700;
    color: #ff6b35;
    text-shadow: 0 0 20px rgba(255,107,53,0.3);
  }}
  .header .subtitle {{ font-size: 12px; color: #888; margin-top: 4px; }}
  .header .badge {{
    background: #ff6b35; color: #0a0a0f; padding: 6px 16px;
    border-radius: 20px; font-size: 13px; font-weight: 700;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  .stats-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 15px;
  }}
  .stat-card {{
    background: #12121e; border: 1px solid #2a2a3e; border-radius: 10px;
    padding: 14px; text-align: center;
  }}
  .stat-card .label {{
    font-size: 10px; color: #666; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 4px;
  }}
  .stat-card .value {{ font-size: 22px; font-weight: 700; }}
  .value.green {{ color: #2da44e; }}
  .value.red {{ color: #f85149; }}
  .value.orange {{ color: #ff6b35; }}
  .value.yellow {{ color: #d29922; }}

  .chart-container {{
    background: #12121e; border: 1px solid #2a2a3e; border-radius: 12px;
    padding: 20px; margin-bottom: 15px;
  }}
  .chart-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px; flex-wrap: wrap; gap: 10px;
  }}
  .chart-header h2 {{ font-size: 16px; color: #e0e0e0; }}
  .legend {{ display: flex; gap: 12px; font-size: 11px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 2px; }}
  .legend-dot.green {{ background: #2da44e; }}
  .legend-dot.red {{ background: #f85149; }}
  .legend-dot.orange {{ background: #ff6b35; }}
  .legend-dot.yellow {{ background: #d29922; }}
  .legend-dot.blue {{ background: #58a6ff; }}
  .legend-dot.purple {{ background: #a371f7; }}
  canvas {{ width: 100%; display: block; }}

  .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px; }}
  @media (max-width: 900px) {{ .panels {{ grid-template-columns: 1fr; }} }}
  .panel {{
    background: #12121e; border: 1px solid #2a2a3e; border-radius: 10px; padding: 15px;
  }}
  .panel h3 {{ font-size: 13px; color: #888; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }}

  .signals-table {{
    background: #12121e; border: 1px solid #2a2a3e; border-radius: 10px; overflow: hidden;
  }}
  .signals-table table {{ width: 100%; border-collapse: collapse; }}
  .signals-table th {{
    background: #1a1a2e; padding: 10px 14px; text-align: left; font-size: 10px;
    text-transform: uppercase; letter-spacing: 1px; color: #888; border-bottom: 1px solid #2a2a3e;
  }}
  .signals-table td {{ padding: 8px 14px; font-size: 12px; border-bottom: 1px solid #1a1a2e; }}
  .signals-table tr:hover {{ background: #1a1a2e; }}
  .signal-buy {{ color: #2da44e; }}
  .signal-sell {{ color: #f85149; }}
  .signal-flat {{ color: #666; }}
  .strength-strong {{ color: #ff6b35; font-weight: 700; }}
  .strength-moderate {{ color: #d29922; }}
  .score-bar {{
    display: inline-block; width: 50px; height: 6px; background: #2a2a3e;
    border-radius: 3px; overflow: hidden; vertical-align: middle; margin-right: 6px;
  }}
  .score-fill {{ height: 100%; border-radius: 3px; }}

  .footer {{
    text-align: center; padding: 15px; color: #444; font-size: 11px;
  }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>⛩️ SAVIOUR COMBINER</h1>
      <div class="subtitle">Multi-Indicator Fusion Chart — ML + Technical Indicators</div>
    </div>
    <div class="badge">COMBINER v1</div>
  </div>

  <div class="container">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="label">ML Accuracy</div>
        <div class="value {"green" if accuracy > 0.55 else "red"}">{accuracy:.1%}</div>
      </div>
      <div class="stat-card">
        <div class="label">Combiner BUYs</div>
        <div class="value green">{len(buys)}</div>
      </div>
      <div class="stat-card">
        <div class="label">Combiner SELLs</div>
        <div class="value red">{len(sells)}</div>
      </div>
      <div class="stat-card">
        <div class="label">Strong Signals</div>
        <div class="value orange">{len(strong)}</div>
      </div>
      <div class="stat-card">
        <div class="label">Avg Score</div>
        <div class="value {"green" if avg_score > 0 else "red"}">{avg_score:+.3f}</div>
      </div>
    </div>

    <div class="chart-container">
      <div class="chart-header">
        <h2>Candlestick + Multi-Indicator Fusion</h2>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot green"></div> Bullish</div>
          <div class="legend-item"><div class="legend-dot red"></div> Bearish</div>
          <div class="legend-item"><div class="legend-dot orange"></div> ML Signal</div>
          <div class="legend-item"><div class="legend-dot yellow"></div> Strong</div>
          <div class="legend-item"><div class="legend-dot blue"></div> BB/VWAP</div>
          <div class="legend-item"><div class="legend-dot purple"></div> EMA</div>
        </div>
      </div>
      <canvas id="mainChart" width="1300" height="500"></canvas>
      <canvas id="rsiChart" width="1300" height="100"></canvas>
      <canvas id="atrChart" width="1300" height="80"></canvas>
    </div>

    <div class="panels">
      <div class="panel">
        <h3>Combiner Signals (Last 20)</h3>
        <div class="signals-table">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Close</th>
                <th>Direction</th>
                <th>Score</th>
                <th>Components</th>
              </tr>
            </thead>
            <tbody>
'''

    for c in combiner_data[-20:]:
        sig_class = f"signal-{c['direction'].lower()}"
        strength_class = f"strength-{c['strength'].lower()}" if c['strength'] != 'NONE' else ''
        score_pct = abs(c["score"]) * 100
        score_color = "#2da44e" if c["score"] > 0 else "#f85149"
        html += f'''
              <tr>
                <td>{c["dt"]}</td>
                <td>{c["close"]:.5f}</td>
                <td class="{sig_class} {strength_class}">{c["direction"]}</td>
                <td><div class="score-bar"><div class="score-fill" style="width:{score_pct}%;background:{score_color}"></div></div>{c["score"]:+.3f}</td>
                <td style="font-size:10px;color:#666">{c["components"]}</td>
              </tr>
'''

    html += '''
            </tbody>
          </table>
        </div>
      </div>
      <div class="panel">
        <h3>ML Predictions (Last 20)</h3>
        <div class="signals-table">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Close</th>
                <th>Pred</th>
                <th>Prob</th>
                <th>Actual</th>
                <th>Pips</th>
              </tr>
            </thead>
            <tbody>
'''

    for s in ml_data[-20:]:
        sig_class = f"signal-{'buy' if s['predicted_up'] else 'sell'}"
        actual_class = f"signal-{'buy' if s['actual_up'] else 'sell'}"
        prob_pct = s["prob"] * 100
        prob_color = "#2da44e" if s["prob"] > 0.5 else "#f85149"
        html += f'''
              <tr>
                <td>{s["dt"]}</td>
                <td>{s["close"]:.5f}</td>
                <td class="{sig_class}">{"BUY" if s["predicted_up"] else "SELL"}</td>
                <td><div class="score-bar"><div class="score-fill" style="width:{prob_pct}%;background:{prob_color}"></div></div>{prob_pct:.1f}%</td>
                <td class="{actual_class}">{"UP" if s["actual_up"] else "DOWN"}</td>
                <td class="{'signal-buy' if s['correct'] else 'signal-sell'}">{s["pips"]:+.2f}</td>
              </tr>
'''

    html += '''
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    SAVIOUR COMBINER — Research/Paper Only · No Real-Money Orders · Honesty Charter Enforced<br>
    Indicators: Supertrend, Bollinger, VWAP, RSI, Stochastic, CCI, Choppiness, SMC(FVG/OB), Keltner Squeeze, EMA<br>
    ML: Ensemble GBDT (5-bar horizon) · Generated: ''' + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + '''
  </div>

  <script>
    const candles = ''' + json.dumps(candle_data) + ''';
    const mlSignals = ''' + json.dumps(ml_data) + ''';
    const combiner = ''' + json.dumps(combiner_data) + ''';
    const priceHigh = ''' + f"{price_high}" + ''';
    const priceLow = ''' + f"{price_low}" + ''';

    function setupCanvas(id, h) {{
      const canvas = document.getElementById(id);
      canvas.height = h;
      return {{ canvas, ctx: canvas.getContext('2d'), W: canvas.width, H: h }};
    }}

    function drawMainChart() {{
      const {{ canvas, ctx, W, H }} = setupCanvas('mainChart', 500);
      const padLeft = 70, padRight = 20, padTop = 20, padBottom = 50;
      const chartW = W - padLeft - padRight;
      const chartH = H - padTop - padBottom;
      const n = candles.length;
      const gap = chartW / n;
      const candleW = Math.max(2, gap * 0.7);
      const yScale = (p) => padTop + chartH - ((p - priceLow) / (priceHigh - priceLow)) * chartH;
      const xScale = (i) => padLeft + i * gap + gap / 2;

      ctx.fillStyle = '#12121e';
      ctx.fillRect(0, 0, W, H);

      // Grid
      ctx.strokeStyle = '#1a1a2e';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {{
        const y = padTop + (chartH * i / 5);
        ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(W - padRight, y); ctx.stroke();
        const price = priceHigh - ((priceHigh - priceLow) * i / 5);
        ctx.fillStyle = '#666'; ctx.font = '11px monospace';
        ctx.fillText(price.toFixed(5), 5, y + 4);
      }}

      // X labels
      ctx.fillStyle = '#666'; ctx.font = '10px monospace';
      for (let i = 0; i < n; i += 40) {{
        const x = xScale(i);
        ctx.fillText(candles[i].dt, x - 30, H - padBottom + 18);
      }}

      // Bollinger Bands
      for (let i = 0; i < n; i++) {{
        const c = candles[i];
        if (c.bb_upper !== null && c.bb_lower !== null) {{
          ctx.strokeStyle = 'rgba(88,166,255,0.2)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(xScale(i), yScale(c.bb_upper));
          ctx.lineTo(xScale(i), yScale(c.bb_lower));
          ctx.stroke();
        }}
      }}

      // EMA lines
      for (let i = 0; i < n; i++) {{
        const c = candles[i];
        if (c.ema20 !== null) {{
          ctx.strokeStyle = 'rgba(163,113,247,0.5)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          if (i === 0 || candles[i-1].ema20 === null) {{
            ctx.moveTo(xScale(i), yScale(c.ema20));
          }} else {{
            ctx.lineTo(xScale(i), yScale(c.ema20));
          }}
          if (i > 0 && candles[i-1].ema20 !== null) ctx.line();
        }}
      }}

      // VWAP
      ctx.strokeStyle = 'rgba(88,166,255,0.6)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let firstV = true;
      for (let i = 0; i < n; i++) {{
        const c = candles[i];
        if (c.vwap !== null) {{
          if (firstV) {{ ctx.moveTo(xScale(i), yScale(c.vwap)); firstV = false; }}
          else ctx.lineTo(xScale(i), yScale(c.vwap));
        }}
      }}
      ctx.stroke();

      // Supertrend line
      ctx.strokeStyle = 'rgba(255,107,53,0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      let prevDir = 0;
      for (let i = 0; i < n; i++) {{
        const c = candles[i];
        if (c.st_dir !== 0) {{
          if (prevDir !== c.st_dir) {{
            if (prevDir !== 0) ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(xScale(i), yScale(c.close));
          }} else {{
            ctx.lineTo(xScale(i), yScale(c.close));
          }}
          prevDir = c.st_dir;
        }}
      }}
      ctx.stroke();

      // Candles
      for (let i = 0; i < n; i++) {{
        const c = candles[i];
        const x = xScale(i);
        const isUp = c.c >= c.o;
        const color = isUp ? '#2da44e' : '#f85149';

        ctx.strokeStyle = color; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, yScale(c.h)); ctx.lineTo(x, yScale(c.l)); ctx.stroke();

        const bodyTop = yScale(Math.max(c.o, c.c));
        const bodyBottom = yScale(Math.min(c.o, c.c));
        ctx.fillStyle = color;
        ctx.fillRect(x - candleW / 2, bodyTop, candleW, Math.max(1, bodyBottom - bodyTop));
      }}

      // ML signals
      const mlMap = {{}};
      mlSignals.forEach(s => {{ mlMap[Math.round(s.ts/60000)*60000] = s; }});
      for (let i = 0; i < n; i++) {{
        const c = candles[i];
        const key = Math.round(new Date(c.ts).getTime()/60000)*60000;
        const sig = mlMap[key];
        if (sig && sig.prob >= 0.7) {{
          const y = yScale(c.l) - 12;
          if (sig.predicted_up) {{
            ctx.fillStyle = sig.correct ? '#2da44e' : '#d29922';
            ctx.beginPath();
            ctx.moveTo(xScale(i), y - 6); ctx.lineTo(xScale(i) - 4, y); ctx.lineTo(xScale(i) + 4, y);
            ctx.closePath(); ctx.fill();
          }} else {{
            ctx.fillStyle = sig.correct ? '#f85149' : '#d29922';
            ctx.beginPath();
            ctx.moveTo(xScale(i), y + 6); ctx.lineTo(xScale(i) - 4, y); ctx.lineTo(xScale(i) + 4, y);
            ctx.closePath(); ctx.fill();
          }}
        }}
      }}

      // Combiner strong signals
      const cmMap = {{}};
      combiner.forEach(c => {{ cmMap[Math.round(c.ts/60000)*60000] = c; }});
      for (let i = 0; i < n; i++) {{
        const c = candles[i];
        const key = Math.round(new Date(c.ts).getTime()/60000)*60000;
        const csig = cmMap[key];
        if (csig && csig.strength === 'STRONG') {{
          const y = yScale(c.h) + 12;
          ctx.fillStyle = csig.direction === 'BUY' ? '#ff6b35' : '#ff6b35';
          ctx.font = 'bold 10px sans-serif';
          ctx.fillText(csig.direction === 'BUY' ? '▲' : '▼', xScale(i) - 4, y);
        }}
      }}

      // Axis labels
      ctx.fillStyle = '#888'; ctx.font = '11px monospace';
      ctx.fillText('Price', 5, padTop - 5);
      ctx.fillText('EUR/USD M1', padLeft, H - 5);
    }}

    function drawSubChart(id, values, min, max, color, label) {{
      const {{ canvas, ctx, W, H }} = setupCanvas(id, id === 'rsiChart' ? 100 : 80);
      const padLeft = 70, padRight = 20, padTop = 5, padBottom = 15;
      const chartW = W - padLeft - padRight;
      const chartH = H - padTop - padBottom;
      const n = candles.length;
      const gap = chartW / n;
      const yScale = (v) => padTop + chartH - ((v - min) / (max - min || 1)) * chartH;

      ctx.fillStyle = '#12121e';
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = '#1a1a2e';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {{
        const y = padTop + (chartH * i / 4);
        ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(W - padRight, y); ctx.stroke();
        const val = max - ((max - min) * i / 4);
        ctx.fillStyle = '#666'; ctx.font = '9px monospace';
        ctx.fillText(val.toFixed(1), 5, y + 4);
      }}

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let first = true;
      for (let i = 0; i < n; i++) {{
        if (values[i] !== null && values[i] !== undefined) {{
          if (first) {{ ctx.moveTo(padLeft + i * gap + gap/2, yScale(values[i])); first = false; }}
          else ctx.lineTo(padLeft + i * gap + gap/2, yScale(values[i]));
        }}
      }}
      ctx.stroke();

      ctx.fillStyle = '#666'; ctx.font = '9px sans-serif';
      ctx.fillText(label, padLeft, H - 2);
    }}

    drawMainChart();
    const rsiVals = candles.map(c => c.rsi);
    drawSubChart('rsiChart', rsiVals, 0, 100, '#ff6b35', 'RSI(14)');
    const atrVals = candles.map(c => c.atr);
    drawSubChart('atrChart', atrVals, 0, Math.max(...atrVals.filter(v => v !== null).map(v => v), 1) * 1.2, '#58a6ff', 'ATR(14)');
  </script>
</body>
</html>'''
    return html


def main():
    print("Loading candles...")
    candles = load_candles()
    print(f"  Loaded {len(candles)} candles")

    print("Loading ensemble model...")
    models = load_ensemble()
    print(f"  Loaded {len(models)} ensemble models")

    print("Computing indicators...")
    ind = compute_all_indicators(candles)
    print(f"  Computed {len(ind)} indicator series")

    print("Generating ML signals...")
    ml_signals = compute_ml_signals(candles, models)
    print(f"  Generated {len(ml_signals)} ML signals")

    print("Computing combiner scores...")
    combiner = compute_combiner_score(candles, ind, ml_signals)
    print(f"  Generated {len(combiner)} combiner signals")

    print("Generating HTML...")
    html = generate_html(candles, ind, ml_signals, combiner)
    OUTPUT_HTML.write_text(html)
    print(f"  Saved to {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size} bytes)")

    # Summary
    total = len(ml_signals)
    correct = sum(1 for s in ml_signals if s["correct"])
    accuracy = correct / total if total else 0
    buys = [c for c in combiner if c["direction"] == "BUY"]
    sells = [c for c in combiner if c["direction"] == "SELL"]
    strong = [c for c in combiner if c["strength"] == "STRONG"]

    print(f"\n=== SAVIOUR COMBINER SUMMARY ===")
    print(f"ML Accuracy: {accuracy:.1%}")
    print(f"Combiner: {len(buys)} BUY, {len(sells)} SELL, {len(strong)} STRONG")
    print(f"Chart: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
