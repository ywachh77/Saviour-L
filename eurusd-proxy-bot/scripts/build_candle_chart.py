#!/usr/bin/env python3
"""
Generate a candlestick chart HTML page with signal prediction markers.

- Loads real EUR/USD candles from var/real_candles/
- Runs the ensemble GBDT model to generate predictions
- Generates a self-contained HTML file with:
  - Canvas-based candlestick chart (last 200 candles)
  - Volume bars
  - Signal markers (buy/sell arrows)
  - Outcome verification (correct/wrong signals)
  - Summary panel with accuracy metrics
  - Shinigami/Naruto dark theme (no purple/cyan)
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
from fusion.features import compute_features, FEATURE_NAMES, FEATURE_LABELS
from fusion.gbdt import GradientBoostingModel

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
OUTPUT_HTML = CANDLE_DIR / "candle_chart.html"
HORIZON = 5
MIN_BARS = 30
N_CHART = 200  # Number of candles to display


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
                volume=0.0,
            ))
    candles.sort(key=lambda c: c.timestamp)
    return candles


def load_ensemble() -> list[GradientBoostingModel]:
    """Load ensemble models from JSON."""
    ensemble_path = CANDLE_DIR / "ensemble_model.json"
    if not ensemble_path.exists():
        print("No ensemble model found. Running training...")
        return []

    data = json.loads(ensemble_path.read_text())
    models = []
    for entry in data.get("ensemble", []):
        model = GradientBoostingModel.from_dict(entry["model"])
        if model:
            models.append(model)
    return models


def generate_signals(candles: list[Candle], models: list[GradientBoostingModel]) -> list[dict]:
    """Generate predictions for the last N candles."""
    if not models:
        return []

    feature_dicts = compute_features(candles)

    signals = []
    for i in range(len(candles) - N_CHART - HORIZON):
        if i < MIN_BARS:
            continue
        feat = feature_dicts[i]
        if feat is None:
            continue

        # Ensemble prediction
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


def generate_html(candles: list[Candle], signals: list[dict]) -> str:
    """Generate the HTML page with candlestick chart."""
    chart_candles = candles[-N_CHART:]
    chart_signals = [s for s in signals if s["ts"] >= chart_candles[0].timestamp.timestamp()]

    # Calculate price range
    highs = [c.high for c in chart_candles]
    lows = [c.low for c in chart_candles]
    price_high = max(highs)
    price_low = min(lows)
    price_range = price_high - price_low
    if price_range == 0:
        price_range = 0.001

    # Convert candles to JSON
    candle_data = [{
        "ts": c.timestamp.isoformat(),
        "o": c.open, "h": c.high, "l": c.low, "c": c.close,
        "dt": c.timestamp.strftime("%m-%d %H:%M"),
    } for c in chart_candles]

    signal_data = [{
        "ts": s["ts"],
        "close": s["close"],
        "prob": round(s["prob"], 4),
        "predicted_up": s["predicted_up"],
        "actual_up": s["actual_up"],
        "correct": s["correct"],
        "pips": round(s["pips"], 2),
        "dt": datetime.fromtimestamp(s["ts"], tz=timezone.utc).strftime("%m-%d %H:%M"),
    } for s in chart_signals]

    # Summary stats
    total = len(signals)
    correct = sum(1 for s in signals if s["correct"])
    accuracy = correct / total if total else 0
    wins = sum(1 for s in signals if s["correct"] and s["predicted_up"])
    losses = sum(1 for s in signals if not s["correct"] and s["predicted_up"])
    net_pips = sum(s["pips"] if s["correct"] else -abs(s["pips"]) for s in signals)

    # High-confidence signals
    hc = [s for s in signals if s["prob"] >= 0.80 or s["prob"] <= 0.20]
    hc_correct = sum(1 for s in hc if s["correct"])
    hc_accuracy = hc_correct / len(hc) if hc else 0

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAVIOUR FUSION — Live Market Candle Chart</title>
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
  .header .subtitle {{
    font-size: 12px;
    color: #888;
    margin-top: 4px;
  }}
  .header .badge {{
    background: #ff6b35;
    color: #0a0a0f;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
  }}
  .container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
  }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin-bottom: 20px;
  }}
  .stat-card {{
    background: #12121e;
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
  }}
  .stat-card .label {{
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
  }}
  .stat-card .value {{
    font-size: 28px;
    font-weight: 700;
  }}
  .stat-card .value.green {{ color: #2da44e; }}
  .stat-card .value.red {{ color: #f85149; }}
  .stat-card .value.orange {{ color: #ff6b35; }}
  .stat-card .value.yellow {{ color: #d29922; }}

  .chart-container {{
    background: #12121e;
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
  }}
  .chart-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
  }}
  .chart-header h2 {{
    font-size: 16px;
    color: #e0e0e0;
  }}
  .chart-header .legend {{
    display: flex;
    gap: 15px;
    font-size: 12px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
  .legend-dot.green {{ background: #2da44e; }}
  .legend-dot.red {{ background: #f85149; }}
  .legend-dot.orange {{ background: #ff6b35; }}
  .legend-dot.yellow {{ background: #d29922; }}
  canvas {{ width: 100%; display: block; }}

  .signals-table {{
    background: #12121e;
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    overflow: hidden;
  }}
  .signals-table table {{
    width: 100%;
    border-collapse: collapse;
  }}
  .signals-table th {{
    background: #1a1a2e;
    padding: 12px 16px;
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    border-bottom: 1px solid #2a2a3e;
  }}
  .signals-table td {{
    padding: 10px 16px;
    font-size: 13px;
    border-bottom: 1px solid #1a1a2e;
  }}
  .signals-table tr:hover {{ background: #1a1a2e; }}
  .signal-correct {{ color: #2da44e; font-weight: 600; }}
  .signal-wrong {{ color: #f85149; font-weight: 600; }}
  .signal-buy {{ color: #2da44e; }}
  .signal-sell {{ color: #f85149; }}
  .prob-bar {{
    display: inline-block;
    width: 60px;
    height: 6px;
    background: #2a2a3e;
    border-radius: 3px;
    overflow: hidden;
    vertical-align: middle;
    margin-right: 8px;
  }}
  .prob-fill {{ height: 100%; border-radius: 3px; }}

  .footer {{
    text-align: center;
    padding: 20px;
    color: #444;
    font-size: 12px;
  }}
  .footer a {{ color: #ff6b35; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>⛩️ SAVIOUR FUSION</h1>
      <div class="subtitle">EUR/USD M1 Candlestick Chart — Real Market Data from Supabase</div>
    </div>
    <div class="badge">ENSEMBLE GBDT v2</div>
  </div>

  <div class="container">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="label">Total Signals</div>
        <div class="value orange">{total}</div>
      </div>
      <div class="stat-card">
        <div class="label">Accuracy</div>
        <div class="value {"green" if accuracy > 0.55 else "red"}">{accuracy:.1%}</div>
      </div>
      <div class="stat-card">
        <div class="label">Net Pips (5-bar)</div>
        <div class="value {"green" if net_pips > 0 else "red"}">{net_pips:.2f}</div>
      </div>
      <div class="stat-card">
        <div class="label">High-Conf Acc (≥0.80)</div>
        <div class="value {"green" if hc_accuracy > 0.55 else "yellow"}">{hc_accuracy:.1%} ({len(hc)} trades)</div>
      </div>
    </div>

    <div class="chart-container">
      <div class="chart-header">
        <h2>Candlestick Chart — Last {len(chart_candles)} Candles</h2>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot green"></div> Correct BUY</div>
          <div class="legend-item"><div class="legend-dot red"></div> Correct SELL</div>
          <div class="legend-item"><div class="legend-dot orange"></div> Signal</div>
        </div>
      </div>
      <canvas id="chart" width="1300" height="600"></canvas>
    </div>

    <div class="signals-table">
      <table>
        <thead>
          <tr>
            <th>Time (UTC)</th>
            <th>Close</th>
            <th>Prediction</th>
            <th>Confidence</th>
            <th>Actual</th>
            <th>Result</th>
            <th>Pips</th>
          </tr>
        </thead>
        <tbody>
'''

    # Add signal rows (last 30)
    for s in signal_data[-30:]:
        result_class = "signal-correct" if s["correct"] else "signal-wrong"
        signal_class = "signal-buy" if s["predicted_up"] else "signal-sell"
        actual_class = "signal-buy" if s["actual_up"] else "signal-sell"
        actual_str = "UP" if s["actual_up"] else "DOWN"
        signal_str = "BUY" if s["predicted_up"] else "SELL"
        prob_pct = s["prob"] * 100
        prob_color = "#2da44e" if s["prob"] > 0.5 else "#f85149"
        pips_class = "signal-correct" if s["correct"] else "signal-wrong"

        html += f'''
          <tr>
            <td>{s["dt"]}</td>
            <td>{s["close"]:.5f}</td>
            <td class="{signal_class}">{signal_str}</td>
            <td><div class="prob-bar"><div class="prob-fill" style="width:{prob_pct}%;background:{prob_color}"></div></div>{prob_pct:.1f}%</td>
            <td class="{actual_class}">{actual_str}</td>
            <td class="{result_class}">{"✓ CORRECT" if s["correct"] else "✗ WRONG"}</td>
            <td class="{pips_class}">{s["pips"]:+.2f}</td>
          </tr>
'''

    html += '''
        </tbody>
      </table>
    </div>
  </div>

  <div class="footer">
    SAVIOUR FUSION BOT — Research/Paper Only · No Real-Money Orders · Honesty Charter Enforced<br>
    Data: Supabase candles_m1 table · Model: Ensemble GBDT (5-bar horizon) · Generated: ''' + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + '''
  </div>

  <script>
    const candles = ''' + json.dumps(candle_data) + ''';
    const signals = ''' + json.dumps(signal_data) + ''';
    const priceHigh = ''' + f"{price_high}" + ''';
    const priceLow = ''' + f"{price_low}" + ''';

    function drawChart() {
      const canvas = document.getElementById('chart');
      const ctx = canvas.getContext('2d');
      const W = canvas.width;
      const H = canvas.height;
      const padLeft = 70, padRight = 20, padTop = 20, padBottom = 60;
      const chartW = W - padLeft - padRight;
      const chartH = H - padTop - padBottom;
      const n = candles.length;
      const candleW = Math.max(2, chartW / n * 0.7);
      const gap = chartW / n;

      // Price range
      const priceRange = priceHigh - priceLow;
      const yScale = (price) => padTop + chartH - ((price - priceLow) / priceRange) * chartH;
      const xScale = (i) => padLeft + i * gap + gap / 2;

      // Clear
      ctx.fillStyle = '#12121e';
      ctx.fillRect(0, 0, W, H);

      // Grid lines
      ctx.strokeStyle = '#1a1a2e';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const y = padTop + (chartH * i / 5);
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(W - padRight, y);
        ctx.stroke();
        const price = priceHigh - (priceRange * i / 5);
        ctx.fillStyle = '#666';
        ctx.font = '11px monospace';
        ctx.fillText(price.toFixed(5), 5, y + 4);
      }

      // X-axis labels (every 30 candles)
      ctx.fillStyle = '#666';
      ctx.font = '11px monospace';
      for (let i = 0; i < n; i += 30) {
        const x = xScale(i);
        ctx.fillText(candles[i].dt, x - 30, H - padBottom + 20);
      }

      // Signal map for quick lookup
      const signalMap = {};
      signals.forEach(s => {
        const candleTs = new Date(s.ts).getTime();
        // Find the candle with matching timestamp
        for (let i = 0; i < candles.length; i++) {
          const cTs = new Date(candles[i].ts).getTime();
          if (Math.abs(cTs - candleTs) < 60000) {
            signalMap[i] = s;
            break;
          }
        }
      });

      // Draw candles
      for (let i = 0; i < n; i++) {
        const c = candles[i];
        const x = xScale(i);
        const isUp = c.c >= c.o;
        const color = isUp ? '#2da44e' : '#f85149';

        // Wick
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, yScale(c.h));
        ctx.lineTo(x, yScale(c.l));
        ctx.stroke();

        // Body
        const bodyTop = yScale(Math.max(c.o, c.c));
        const bodyBottom = yScale(Math.min(c.o, c.c));
        const bodyH = Math.max(1, bodyBottom - bodyTop);
        ctx.fillStyle = color;
        ctx.fillRect(x - candleW / 2, bodyTop, candleW, bodyH);

        // Signal markers
        if (signalMap[i]) {
          const sig = signalMap[i];
          if (sig.prob >= 0.80) {
            const markerY = yScale(c.l) - 15;
            if (sig.predicted_up) {
              // Buy arrow
              ctx.fillStyle = sig.correct ? '#2da44e' : '#d29922';
              ctx.beginPath();
              ctx.moveTo(x, markerY - 8);
              ctx.lineTo(x - 5, markerY);
              ctx.lineTo(x + 5, markerY);
              ctx.closePath();
              ctx.fill();
            } else {
              // Sell arrow
              ctx.fillStyle = sig.correct ? '#f85149' : '#d29922';
              ctx.beginPath();
              ctx.moveTo(x, markerY + 8);
              ctx.lineTo(x - 5, markerY);
              ctx.lineTo(x + 5, markerY);
              ctx.closePath();
              ctx.fill();
            }
          }
        }
      }

      // Draw axis labels
      ctx.fillStyle = '#888';
      ctx.font = '12px monospace';
      ctx.fillText('Price', 5, padTop - 5);
      ctx.fillText('EUR/USD M1', padLeft, H - 5);

      // Draw summary
      ctx.fillStyle = '#ff6b35';
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(`Last Close: ${candles[n-1].c.toFixed(5)}`, padLeft, padTop - 5);
    }

    drawChart();
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

    if not models:
        print("  ERROR: No ensemble model found. Generating chart without signals.")
        models = []

    print("Generating signals...")
    signals = generate_signals(candles, models)
    print(f"  Generated {len(signals)} signals")

    print("Generating HTML chart...")
    html = generate_html(candles, signals)
    OUTPUT_HTML.write_text(html)
    print(f"  Saved to {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size} bytes)")

    # Print summary
    total = len(signals)
    correct = sum(1 for s in signals if s["correct"])
    accuracy = correct / total if total else 0
    net_pips = sum(s["pips"] if s["correct"] else -abs(s["pips"]) for s in signals)
    hc = [s for s in signals if s["prob"] >= 0.80 or s["prob"] <= 0.20]
    hc_correct = sum(1 for s in hc if s["correct"])
    hc_accuracy = hc_correct / len(hc) if hc else 0

    print(f"\nSignal Accuracy Summary:")
    print(f"  Total signals: {total}")
    print(f"  Correct: {correct} ({accuracy:.1%})")
    print(f"  Net pips (5-bar): {net_pips:.2f}")
    print(f"  High-confidence (≥0.80): {len(hc)} trades, {hc_accuracy:.1%} accuracy")


if __name__ == "__main__":
    main()
