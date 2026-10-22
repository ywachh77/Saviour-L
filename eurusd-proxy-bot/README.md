# FUSION BOT

**A unified EUR/USD M1 research signal bot.** One Python process, one Telegram
chat, zero commands to remember.

It merges three lines of work into a single engine:

| Source | What it contributed |
|---|---|
| **Vortex-x1** (VORTEX SI) | key-pool quota discipline, job-slot idempotency, heartbeat liveness |
| **orbix** | the 6-check confluence gate, feature set, isotonic calibration, prequential learning |
| EUR/USD proxy bot | delivery honesty: sealed signals, confirmed message ids, cost-charged settlement |

Standard library only. No pip install, no numpy, no pandas — it runs on Termux
as happily as on GitHub Actions.

---

## What this is not

- It does **not** place broker orders and never will. Signals are paper
  settlements with the spread charged on every trade.
- It makes **no** accuracy promise. A probability shown next to a signal is a
  calibrated estimate on past data, not a prediction about the next candle.
- It is **not** wired to Quotex, Pocket Option, OTC binaries, or anything
  similar, and it will not implement Martingale.

If a number in this repository looks like a guarantee, that is a bug.

---

## Five commands, and that is all

```
/start    begin receiving signals
/help     what the bot does
/status   health, ledger, learning state
/pause    stop sending (settlement and learning continue)
/resume   start sending again
```

Everything else is automatic: ingest, signals, settlement, online learning,
nightly walk-forward, daily digest, key rotation, fallback.

---

## How one minute works

```
ingest ──▶ data gate ──▶ features ──▶ regime ──▶ strategy council
                                                        │
                                                        ▼
                                              meta-model (veto, or
                                              advisory once promoted)
                                                        │
                                                        ▼
                                          isotonic calibration
                                          (RAW ▸ CALIBRATED ▸ FLAT abstains)
                                                        │
                                                        ▼
                                             sealed signal
                                     entry = open of the forming candle
```

1. **Data gate** — refuse to reason about data that is stale (>180 s), gappy, or
   too wide. A blocked bar is reported, not hidden.
2. **Features** — EMA, RSI, ADX/±DI, ATR, momentum, candle quality, breakout
   distance. Computed from completed candles only; tests prove there is no
   look-ahead.
3. **Regime** — TREND / RANGE / VOLATILE, which decides which prototype
   families are allowed to speak.
4. **Strategy council** — the core 6-check gate decides whether there is a
   signal. Prototypes start silent (weight 0) and only earn a voice after 30
   real settled outcomes. Learned prototypes can **veto** the core, never
   outvote it.
5. **Meta-model** — a prequential online logistic model. In `VETO` mode (the
   default) it can only block. Nightly walk-forward gates can promote it to
   `ADVISORY`, where agreement is also required. It never creates a signal.
6. **Calibration** — below 60 confirmed real outcomes the raw score is shown and
   labelled `RAW`. At 60+ an isotonic (PAVA) map is fitted. If that map collapses
   to one level the gate **abstains** rather than trading a constant probability.
7. **Seal** — entry is the open of the candle that is forming right now. If the
   pipeline cannot seal and send within 10 seconds of the minute opening, the
   signal is **discarded, never backdated**.

---

## The honesty charter (enforced by tests)

| Rule | Where it is enforced |
|---|---|
| UNRESOLVED is never turned into a win, loss or tie | `settle.py`, `tests/test_settle.py` |
| `SENT` is recorded only on a confirmed Telegram message id | `telegram.py`, `tests/test_telegram.py` |
| Late signals (>10 s) are dropped, never backdated | `engine.py`, `tests/test_engine.py` |
| The spread is charged on every settled trade | `charter.net_pips`, `tests/test_charter.py` |
| Synthetic / replay data trains nothing | `learning.online_step`, `tests/test_learning.py` |
| Liveness is heartbeats only; Actions stands in after 90 s | `heartbeat.py`, `tests/test_heartbeat.py` |
| RLS on all 13 Supabase tables | `supabase/migrations/0001_init.sql` |
| A win rate always travels with its Wilson interval | `digest.py`, `tests/test_digest.py` |

Run `python -m unittest discover -s tests -t .` — 191 tests. If one fails, the
charter is broken; do not ship it.

---

## Setup, one step at a time

Do these in order. Each one is small on purpose.

### Step 1 — see the pipeline fire (no keys needed)

```bash
cd eurusd-proxy-bot
python scripts/smoke.py 420
```

This replays 420 synthetic one-minute candles through the whole pipeline and
prints abstentions, signals, cost-charged settlements and the charter checks.
It trains **nothing** — watch the net-pips line, which is often negative even
when the win rate looks good. That is the point: gross direction is not profit.

### Step 2 — Supabase (free tier is fine)

1. Create a project at [supabase.com](https://supabase.com).
2. Open **SQL Editor → New query**, paste the whole of
   `supabase/migrations/0001_init.sql`, and run it. It creates 13 tables with
   row-level security enabled and no public policies.
3. Go to **Project Settings → API** and copy:
   - **Project URL** → `SUPABASE_URL`
   - **service_role key** → `SUPABASE_SERVICE_ROLE_KEY`

   The service_role key bypasses RLS. It stays in `.env` and in GitHub secrets.
   Never in chat, never in a commit.

4. Check it: `python scripts/doctor.py`. It queries all 13 tables and will tell
   you which ones are missing if the migration did not finish.

### Step 3 — GitHub (private repo)

```bash
git init
git add .
git commit -m "FUSION BOT"
git branch -M main
git remote add origin git@github.com:YOUR-USERNAME/eurusd-proxy-bot.git
git push -u origin main
```

Keep the repository **private** — it holds your `.gitignore`d secrets on the
Actions side, and private costs nothing.

Then add the same values under **Settings → Secrets and variables → Actions →
Secrets**: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`,
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_OWNER_CHAT_ID`, and at least one
`TWELVE_DATA_API_KEY`.

### Step 4 — Telegram

1. Message [@BotFather](https://t.me/BotFather), `/newbot`, follow the prompts,
   copy the token → `TELEGRAM_BOT_TOKEN`.
2. Message [@userinfobot](https://t.me/userinfobot), copy your **numeric** id →
   `TELEGRAM_OWNER_CHAT_ID`.
3. Send `/start` to your bot. Any other chat id is ignored silently — this bot
   is owner-only and does not talk to strangers. A bot cannot message you first,
   so this step is not optional.

4. Check it: `python scripts/doctor.py --send`. It puts one test message in your
   chat and confirms the message id came back, which is the same proof the bot
   requires before it ever records a signal as `SENT`.

### Step 5 — Termux (the 24/7 primary)

```bash
pkg update && pkg install python git
git clone git@github.com:YOUR-USERNAME/eurusd-proxy-bot.git
cd eurusd-proxy-bot/eurusd-proxy-bot
cp .env.example .env
nano .env          # fill in your four values
python scripts/run_bot.py
```

That is the only command you need. The bot then owns its own schedule.

If something is wrong, `python scripts/doctor.py` checks all four services for
real (not just "is the variable set?") and prints one next action per problem.

Recommended, so the phone does not kill it:

```bash
termux-wake-lock
```
and disable battery optimisation for Termux in Android settings.

---

## Key slots

`TWELVE_DATA_KEY_1` … `TWELVE_DATA_KEY_30` exist so the pool can rotate across
keys **you have legitimately provisioned**. The pool reserves quota before each
HTTP call, refunds on network failure, cools a key down on HTTP 429, disables it
on 401/403, and reports exhaustion honestly instead of looping.

Deliberately **not** supported: registering multiple free accounts to work
around a provider's rate limit. That breaks the provider's terms and gets keys
banned. One free key gives 8 credits/minute, which is far more than one pair on
M1 needs.

---

## Automation you do not have to think about

| When | What happens |
|---|---|
| every minute boundary | ingest → evaluate → seal/send or abstain |
| continuously | heartbeat, Telegram commands, outbox retry, settlement |
| after each confirmed real outcome | test-before-learn online update, calibration refit |
| 23:55 UTC | walk-forward over all labelled bars; promote, keep, or roll back |
| 00:05 UTC | daily digest: signals, ledger with Wilson interval, learning, key pool |
| only when a human must act | alert: stale feed, unknown delivery, key pool, negative expectancy, rollback |

**Idempotency.** Every scheduled step claims a `(job, bucket)` slot first. If
Termux and Actions ever overlap, the second one is skipped — a minute is never
processed twice.

**Fallback.** `scripts/fallback.py` runs from Actions every 5 minutes and does
nothing at all unless the Termux heartbeat is older than 90 seconds. When it
does act, its rows are tagged `actions-fallback`.

---

## Layout

```
eurusd-proxy-bot/
├── fusion/                  the engine (standard library only)
│   ├── charter.py           the rules, as executable constants
│   ├── engine.py            the per-minute pipeline
│   ├── council.py           core gate + learned prototypes
│   ├── model.py             prequential logistic + bootstrap ensemble
│   ├── calibration.py       isotonic (PAVA) calibration
│   ├── learning.py          online updates, walk-forward, promotion gates
│   ├── keypool.py           reserve / refund / rotate / cooldown
│   ├── provider.py          Twelve Data over urllib
│   ├── settle.py            cost-charged paper settlement
│   ├── store.py             Supabase REST + local JSON-lines fallback
│   └── doctor.py            setup self-check (which step is broken, and why)
│   ├── telegram.py          owner-only delivery with honest states
│   ├── runner.py            the zero-touch loop
│   └── simulate.py          deterministic offline replay
├── scripts/
│   ├── run_bot.py           the one command you run
│   ├── doctor.py            check your setup: python scripts/doctor.py
│   ├── smoke.py             offline pipeline demo  (python scripts/smoke.py 420)
│   ├── nightly.py           walk-forward + gates
│   ├── fallback.py          Actions stand-in, heartbeat-gated
│   └── backfill.py          history import (BACKFILL=1, env-flagged on purpose)
├── supabase/migrations/     0001_init.sql — 13 tables, RLS on all
├── .github/workflows/       tests.yml, nightly.yml, fallback.yml
└── tests/                   177 tests, many of them charter guards
```

## Useful extras

```bash
python -m unittest discover -s tests -t .   # the charter, asserted
python scripts/smoke.py 420                 # watch it work offline
python scripts/doctor.py                    # is my setup actually correct?
python scripts/doctor.py --send             # ...and prove delivery works
BACKFILL=1 BACKFILL_MINUTES=20000 python scripts/backfill.py
```

The backfill spends provider credits in 5,000-candle pages, so it is
environment-flagged rather than a Telegram command on purpose.

---

## Boundaries, stated once more

Research and paper settlement only. Every signal says what it costs and that no
broker order is placed. Every win rate carries its confidence interval. Every
unresolved outcome stays unresolved. If you want to change any of that, change
the tests first.
