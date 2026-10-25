# Saviour-L

A research-only trading-signal engine for one-minute bars across **EUR/USD**,
**BTC/USD**, and **ETH/USD**. Every prediction is sealed before its entry bar
opens, every trade pays its full round-turn spread, and zero broker orders are
ever placed.

**[Shinigami Gate & Operator Console](docs/index.html)** — Pure ceremonial gate
with Eight-Trigrams (*Fuinjutsu*) release into a 6-page mobile + desktop bot
control panel (`Signals`, `Telegram`, `Health`, `Strategies`, `Credentials`,
`Risk & Charter`) with standalone web signal generation.

## Capabilities

- **Multi-market unit accuracy (`fusion/instruments.py`)** — EUR/USD (`0.0001` pip,
  24/5 FX sessions) alongside 24/7 spot BTC/USD (`$1.00` pip) and ETH/USD
  (`$0.10` pip), where spread-to-movement is 4–14x lower than one-minute FX.
- **Real-data-validated signal improvements (`fusion/mtf.py`, `fusion/scalping.py`)**
  — 15-minute locally resampled higher-timeframe confirmation (lifted Oct 2026
  out-of-sample EUR/USD win rate to **58.51%** and net expectancy from `-0.116`
  to **+0.338 pips/trade** after 1.20-pip spread across 101,730 real M1 bars)
  plus the `scalp_expansion` prototype (`+0.378` Train / `+0.382` OOS net pips).
- **Automated look-ahead & recursive warmup guard (`fusion/lookahead.py`)** —
  Freqtrade-style adversarial causality audit across features, MTF resamplers,
  all 13 council strategies, and live sealing.
- **Macro blackout & paper risk engine (`fusion/news.py`, `fusion/risk.py`)** —
  DST-exact US Non-Farm Payrolls + configurable release blackouts, hard ATR
  invalidation stops, and 1–2% capped paper position sizing.
- **PostgREST full-history paging (`fusion/store.py`)** — pages past Supabase's
  1,000-row response cap and reads exact `Content-Range` counts so nightly
  walk-forward trains on the full 101,730+ bar history.

## Verification

```bash
cd eurusd-proxy-bot
python3 -m unittest discover -s tests -t .     # 649 tests
python3 scripts/lookahead_check.py             # 5/5 causality & warmup audits
python3 scripts/smoke.py 420                   # end-to-end honesty checks
python3 scripts/web_dashboard.py --port 8080   # Operator Console + Control API
```

## Boundaries

- Research and paper settlement only — no broker orders, no synthetic OTC feeds,
  no martingale sizing.
- GitHub Actions is the sole scheduled runner; secrets live only in local `.env`
  and GitHub Secrets.
- The 19-of-20 (95%) win-rate request remains explicitly refused: at 1-minute
  horizons a 95% hit rate is manufactured only by tiny targets without stops and
  is net negative once tail losses settle. Net expectancy after spread governs.
