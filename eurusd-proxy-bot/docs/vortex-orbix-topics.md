# Vortex-x1 + orbix — topics, ideas and learning path

A companion to the bot, not part of it. This is the research notebook from the
two public repos that FUSION BOT was built from: what they are good at, what to
learn from them, and what to build next.

Repos: `Aizennn7/Vortex-x1` (VORTEX SI, TypeScript signal engine) and
`Aizennn7/orbix` (Python signal dashboard).

---

## 1. What each repo actually teaches

### Vortex-x1 — production discipline

The valuable part of VORTEX SI is not a strategy; it is the operational
plumbing most hobby signal bots never get around to.

- **Key pool with pre-reservation.** Quota is reserved *before* the HTTP call so
  concurrent workers cannot overspend a plan. This is the single best idea in
  either repo.
- **Job slots.** A `(job, bucket)` claim makes scheduled work idempotent, which
  is what makes a phone-plus-Actions hybrid safe.
- **Role separation.** Worker vs relay vs operator, with a scheduler and a
  runner that can be reasoned about separately.
- **Honest degradation.** When the budget is gone the pool says so instead of
  looping.

### orbix — research discipline

- **The 6-check confluence gate.** Transparent, countable, and explainable in
  one sentence. A rule score is presented as a count, never as a probability.
- **Isotonic calibration.** A logistic score is not a probability. PAVA fixes
  that without assuming a shape — and a flat map is *labelled*, not trusted.
- **Explainable contributions.** Every signal can show which features moved it.
- **Abstention as a first-class outcome.** Most bars produce nothing, and the
  system says so clearly.

---

## 2. Repo topics (for GitHub)

Vortex-x1:
`typescript` `trading-bot` `signal-engine` `forex` `twelvedata` `supabase`
`telegram-bot` `worker` `scheduler` `rate-limiting` `key-rotation` `paper-trading`

orbix:
`python` `trading` `signals` `machine-learning` `logistic-regression`
`isotonic-regression` `calibration` `technical-analysis` `rsi` `adx` `atr`
`backtesting` `streamlit` `research`

FUSION BOT:
`python` `eurusd` `forex` `m1` `signal-bot` `telegram` `supabase` `github-actions`
`github-actions` `online-learning` `walk-forward` `calibration` `paper-trading` `zeroconfig`

---

## 3. Learning topics, in the order they pay off

1. **Probability calibration.** Brier score, reliability diagrams, ECE, isotonic
   regression vs Platt scaling. This is the difference between "0.6" meaning
   something and meaning nothing.
2. **Prequential evaluation.** Test-before-learn, expanding-window walk-forward,
   and *purging* (a label that looks N bars ahead must not be trained on up to
   the test boundary).
3. **Cost-aware expectancy.** Gross direction accuracy and net expectancy are
   different populations. A 55% hit rate can be a guaranteed loss.
4. **Multiple-comparisons.** If you try 30 rule sets on one history, the best
   one is lucky. Walk-forward gates are the cheap defence.
5. **Class imbalance and base rates.** On M1, "up" is near 50%. Anything that
   looks impressive is probably measuring the base rate.
6. **Drift.** A model fitted on January does not describe June. Recency weighting
   and demotion-on-decay are the practical answers.
7. **Idempotency and exactly-once-ish scheduling.** Unglamorous, and the reason
   a bot survives a restart at 3am.

---

## 4. Project ideas, smallest first

| # | Idea | Why it is worth doing |
|---|---|---|
| 1 | Spread monitor | Log the provider's real spread by hour. The modelled spread is a guess; the real one has a shape. |
| 2 | Session filter | Tag each bar London / New York / Asia / off-hours and report expectancy per session. Cheap, often the single biggest real edge. |
| 3 | Abstention dashboard | Plot what the gate abstained on. The bars you skip are half the model. |
| 4 | Calibration drift chart | Reliability diagram per week. Shows drift before the P&L does. |
| 5 | Recency-weighted refit | Weight training samples by `0.5^(age/half-life)`. Standard for drifting series; must be validated, not assumed. |
| 6 | Multi-horizon read | Predict 3, 5 and 15 minutes and require agreement. More data per bar, more ways to overfit. |
| 7 | Cost sweep | Re-run the replay at several spread assumptions. If the edge only exists at 0.5 pips, it is not an edge. |
| 8 | Purge-and-embargo backtester | Add an embargo gap after each test window on top of the purge. |
| 9 | Regime-conditional models | One model per regime. Attractive, and triples the data you need. |
| 10 | Monte Carlo over replayed signals | Resample the trade sequence to get a distribution, not a point. |

---

## 5. Content ideas (if you ever write this up)

- "My bot refuses to trade 80% of the time, and that is the feature."
- "A 64.9% win rate that loses money" — the cost-charged settlement walkthrough.
- "Why your calibrated model is lying to you" — in-sample vs out-of-sample
  isotonic fitting.
- "Purging: the backtest leak nobody mentions on M1."
- "Running a 24/7 trading bot on a phone, and the two-line fallback that saves
  it."
- "What I learned merging two signal projects into one."

---

## 6. Standing warnings

- Past replay is not forward performance. One path through one history.
- A win rate without a confidence interval is marketing.
- Any strategy that only works on synthetic data works on nothing.
- If a rule set needs a parameter sweep to look good, it is fitting noise.
- Research and paper settlement only. No broker orders, no OTC binaries, no
  Martingale, no guaranteed accuracy.
