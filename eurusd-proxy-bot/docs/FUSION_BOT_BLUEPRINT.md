# FUSION BOT — architecture blueprint

The design record for the unified EUR/USD M1 research signal bot. It explains
why each piece is shaped the way it is, so a change can be judged against the
reason it exists rather than against taste.

> Supersedes the earlier 20-command plan. The runtime is **zero-touch**: five
> Telegram commands exist (`/start /help /status /pause /resume`) and nothing
> else needs an operator.

---

## 1. What was merged, and what each source contributed

| Source | Contribution |
|---|---|
| **Aizennn7/Vortex-x1** (VORTEX SI, TypeScript) | key-pool quota discipline (reserve before the call, refund, rotate, cooldown), job-slot idempotency, heartbeat-only liveness, Actions fallback |
| **Aizennn7/orbix** (Python dashboard) | the 6-check confluence gate, the 24-feature set, isotonic (PAVA) calibration, prequential online learning, explainable contributions |
| EUR/USD proxy bot | delivery honesty: sealed signals, `SENT` only on a confirmed message id, cost-charged paper settlement, UNRESOLVED never fabricated |

The result is one Python process, standard library only, that fits on a phone.

---

## 2. The per-minute pipeline

```
ingest ──▶ data gate ──▶ features ──▶ regime ──▶ council ──▶ meta-model
                                                                 │
                                                                 ▼
                                                     isotonic calibration
                                                                 │
                                                                 ▼
                                                          sealed signal
                                                  entry = open of the forming candle
```

| Stage | Refuses when | Why |
|---|---|---|
| **Data gate** | candle older than 180 s, spread above 2.5 pips, fewer than 60 bars | a signal built on bad data looks identical to a good one |
| **Features** | fewer than 30 completed bars | indicators would be undefined, not uncertain |
| **Regime** | never refuses; classifies TREND / RANGE / VOLATILE | it only decides which prototype families may speak |
| **Council** | core gate below 5-of-6, or mandatory trend/ADX checks fail, or learned prototypes oppose with net mass > 0.35 | the core decides; prototypes may veto, never create |
| **Meta-model** | score on the wrong side of 0.5 by ≥ 0.03 (VETO mode) | by default the model can only block |
| **Calibration** | the isotonic map collapsed to one level | a constant probability is a base rate, not a ranking |
| **Seal** | sealed more than 10 s after the minute opened | a backdated entry is a lie |

**Entry convention.** A candle's timestamp is its OPEN time. At minute `T` the
candle stamped `T-1` has just closed; the pipeline uses it, and the signal
names the open of candle `T` as the entry. The entry price is filled from that
candle's open once it exists — never invented before it does.

**Settlement.** Exit is the close of the candle stamped `T + horizon - 1`. Every
settled trade pays the spread. Direction correctness and profitability are
reported separately, because they are different things.

---

## 3. Learning

### Online (every confirmed real outcome)

1. **Real data only.** Synthetic and replay outcomes return early; the charter
   is not configurable.
2. **Test before learn.** The probability is recorded *before* the gradient
   step, so the logged error is genuinely out-of-sample.
3. **One update per outcome.** Ties and unresolved bars are skipped.
4. Calibration is refit from the accumulated prequential points.

### Nightly (23:55 UTC)

1. Rebuild labelled samples from stored candles. A sample at bar `j` is labelled
   by the move from `open(j+1)` to `close(j+horizon)` — the same path a live
   signal takes.
2. **Purged walk-forward** over expanding folds. Any training sample whose label
   reaches into the test window is dropped, because a one-minute label looks
   `horizon` bars ahead.
3. **System replay** (`fusion/backtest.py`): drive the *live decision path* over
   each test fold, settle every gated signal with costs.
4. **Gates** — all must pass to promote:
   - directional accuracy ≥ 0.505
   - Brier skill > 0 (better than a 0.50 base rate)
   - walk-forward expectancy > 0 after costs
   - Brier not worse than the champion
   - **system replay net pips > 0** (only when ≥ 200 signals, otherwise abstain)
5. **Demotion, not just promotion.** If the replay is negative on enough
   signals, an already-promoted model drops back to VETO.

### Calibration ladder

| State | Condition | Behaviour |
|---|---|---|
| `RAW` | fewer than 60 real outcomes | raw score reported and labelled as raw |
| `CALIBRATED` | isotonic map fitted with ≥ 2 levels | calibrated probability used |
| `FLAT` | map collapsed to one level | **abstain** — a constant probability erases the ranking |

Calibration is fitted **only on out-of-sample scores**. Fitting an isotonic map
on the bars a model trained on produces a sharper-looking, wrong map.

---

## 4. Council power balance

```
core gate (weight 1.0, always heard)
  + prototypes (weight 0 until learned)
        · each needs 30 real settled outcomes before it is heard at all
        · per-prototype cap 0.30
        · total prototype mass cap 0.60
        · veto threshold 0.35  →  no single prototype can veto alone
```

A prototype's weight is `max(0, net / (seen + 40))`, capped and scaled. A
prototype that loses money stays silent. Prototypes can block the core; they
can never replace it.

---

## 5. Storage (Supabase, 13 tables, RLS on all)

`candles_m1`, `signals`, `settlements`, `model_state`, `model_versions`,
`council_weights`, `learning_events`, `job_runs`, `heartbeats`, `api_keys`,
`key_usage`, `outbox`, `alerts`.

Row-level security is enabled on every table with **no policies** for `anon` or
`authenticated`. The publishable key can read and write nothing; only the
`service_role` key, which never leaves `.env` and GitHub secrets, can touch the
data.

A local JSON-lines mirror is always written, so a Supabase outage degrades to
"stored locally" rather than "lost".

---

## 6. Automation

| Trigger | Action |
|---|---|
| every minute boundary | ingest → evaluate → seal/send or abstain |
| continuously | heartbeat, Telegram polling, outbox retry, settlement |
| after each confirmed real outcome | online update + calibration refit |
| 23:55 UTC | walk-forward + system replay + promote/demote |
| 00:05 UTC | daily digest |
| human must act | alert: stale feed, unknown delivery, key pool, negative expectancy, rollback |

**Idempotency.** Every scheduled step claims a `(job, bucket)` slot first. If
Termux and Actions overlap, the second is skipped — a minute is never processed
twice.

**Fallback.** `scripts/fallback.py` runs from Actions every 5 minutes and does
nothing unless the Termux heartbeat is older than 90 seconds. Its rows are
tagged `runner='actions-fallback'`.

**Key pool.** Quota is reserved *before* each HTTP call and refunded on network
failure. HTTP 429 → cooldown with exponential backoff; 401/403 → disable the
key; 5xx → short cooldown. Slots exist for legitimately provisioned keys only.

---

## 7. Honesty charter

| Rule | Enforced in |
|---|---|
| UNRESOLVED is never a win, loss or tie | `settle.py`, `test_settle.py`, `test_properties.py` |
| `SENT` only on a confirmed message id | `telegram.py`, `test_telegram.py` |
| late signals (>10 s) discarded, never backdated | `engine.py`, `test_engine.py` |
| spread charged on every settled trade | `charter.net_pips`, `test_charter.py` |
| synthetic data trains nothing | `learning.online_step`, `test_runner.py` |
| heartbeat-only liveness | `heartbeat.py`, `test_heartbeat.py` |
| RLS on all 13 tables | `supabase/migrations/` |
| bot payloads and the SQL schema cannot drift apart | `tests/test_schema_contract.py` |
| every win rate carries a Wilson interval | `digest.py`, `test_digest.py` |
| backtest folds are purged | `backtest.py`, `test_backtest.py` |
| calibration fitted out-of-sample only | `learning.run_nightly`, `test_backtest.py` |

---

## 8. Known limits (stated, not hidden)

- **Nothing here is validated against live market data.** Every number in this
  repository comes from synthetic series. The first honest performance
  measurement will come from `scripts/backtest.py` once real candles are stored.
- The volatility band in the core gate (`atr_pct ≤ 0.004`) and the regime's
  volatile threshold (`0.0012`) were inherited from the reference projects. On
  M1 EUR/USD they rarely trigger; they are scale-free enough to be harmless and
  are exposed as constants, not magic numbers buried in a rule.
- The backtest assumes the feed was healthy at the time — freshness cannot be
  checked historically.
- A replay measures one path through one history. It is not a Monte Carlo.

## 9. Boundaries

Research and paper settlement only. No broker orders, no Quotex / Pocket Option
/ OTC binaries, no Martingale, no accuracy guarantees. If a number in the output
reads like a promise, that is a bug.
