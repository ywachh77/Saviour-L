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
   distance, plus scalping microstructure (5-bar range, range compression, body
   ratio, signed impulse, stochastic, CCI, distance from VWAP). Computed from
   completed candles only; tests prove there is no look-ahead.
3. **Regime** — TREND / RANGE / VOLATILE, which decides which prototype
   families are allowed to speak. Continuation scalps sit out a range;
   reversion scalps sit out a trend or a volatility spike.
4. **Strategy council** — the core 6-check gate decides whether there is a
   signal. Prototypes start silent (weight 0) and only earn a voice after 30
   real settled outcomes. Learned prototypes can **veto** the core, never
   outvote it. The scalping members are prototypes too, so they can never start
   trading before live data proves them.
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
| RLS on all 13 Supabase tables | `supabase/migrations/` |
| A win rate always travels with its Wilson interval | `digest.py`, `tests/test_digest.py` |
| Bot payloads and the SQL schema cannot drift apart | `tests/test_schema_contract.py` |
| A scalp that cannot pay its own spread stays quiet | `charter.scalp_cost_ok`, `tests/test_scalping.py` |
| The replay learns the council as the live bot does | `backtest.run_backtest`, `tests/test_backtest.py` |

Run `python -m unittest discover -s tests -t .` — 322 tests. If one fails, the
charter is broken; do not ship it.

---

## Scalping

Six scalping methods are wired in, and they come in through **both** doors.

**Into the prediction engine.** The model used 24 inputs. It now uses 32. The
eight new ones describe things a scalp actually lives on: 5-bar range, range
compression, body ratio, signed impulse, stochastic %K/%D, CCI, and distance
from VWAP. Bumping `FEATURE_VERSION` to 4 invalidates any stored model rather
than letting it be silently reinterpreted, so the next nightly refits from
scratch.

**Into the strategy council**, as six new prototypes:

| Member | Idea |
|---|---|
| `scalp_impulse` | Body covers half an ATR and closes on its extreme — continuation |
| `scalp_reversion` | CCI and stochastic both stretched, and %K has turned back |
| `scalp_micro_breakout` | Close escapes a *tight* 5-bar box |
| `scalp_stoch_cross` | Fresh stochastic cross that agrees with the EMA trend |
| `scalp_vwap_reversion` | Price stretched from VWAP and this bar is already closing back |
| `scalp_pivot_fade` | 5-bar pivot swept, then rejected back inside |

Two mechanisms keep them honest.

**A cost floor.** Scalping is where the spread does the most damage — a 4-pip
scalp keeps 2.8 pips, and the same 4 pips is noise on a swing trade. So a
scalping member refuses to speak unless the bar's ATR projects a gross move of
at least `SCALP_MIN_REWARD_TO_COST` (1.5×) the spread. On a normal 1.2-pip
spread that means roughly 3.6 pips of ATR. Widening the spread to 4 pips
silences the entire family.

**They are prototypes, not core.** A scalping rule cannot create a signal. It
sits at weight 0 until learning has seen 30 real, cost-charged outcomes for it,
is capped at 0.30, and can only veto the core when the opposition mass passes
0.35. An unproven scalp therefore cannot start trading on its own no matter how
good it looks on paper.

On the synthetic replay they behaved exactly that way: `scalp_impulse` became
the highest-weighted learned rule at 0.169, and the family's combined mass never
reached the veto threshold, so it changed no signal. That is the design working,
not a shortfall — but it also means **no scalping rule here has been shown to
make money**, and on this synthetic history nothing does.

### Measuring whether it actually works

```bash
python scripts/backtest.py            # purged walk-forward replay of the full system
python scripts/backtest.py 5000       # most recent 5000 bars
```

This is **not** the model's score on every bar. It replays the bars that survive
the data gate, the regime filter, the council and the meta-model veto, and
charges the spread on every one. Folds are purged, so a label that looks ahead
is never trained on. If it reports fewer than 200 signals, it says so and
declines to call the number evidence.

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
2. Open **SQL Editor → New query** and run the migrations in order —
   `supabase/migrations/0001_init.sql` first, then
   `0002_add_missing_columns.sql`. Together they create 13 tables with
   row-level security enabled and no public policies. Both are idempotent, so
   re-running them is safe.
3. Go to **Project Settings → API** and copy:
   - **Project URL** → `SUPABASE_URL`
   - **service_role key** → `SUPABASE_SERVICE_ROLE_KEY`

   The service_role key bypasses RLS. It stays in `.env` and in GitHub secrets.
   Never in chat, never in a commit.

4. Check it: `python scripts/doctor.py`. It queries all 13 tables and will tell
   you which ones are missing if the migration did not finish.

### Step 3 — GitHub (private repo)

One command, run from `eurusd-proxy-bot/`:

```bash
./scripts/publish_repo.sh
```

It creates a **private** `eurusd-proxy-bot` repo in your account and pushes this
folder as its root. Pass a different name if you prefer:
`./scripts/publish_repo.sh my-bot-name`.

Prefer to do it by hand? Then:

```bash
git init && git add . && git commit -m "FUSION BOT"
git remote add origin git@github.com:YOUR-USERNAME/eurusd-proxy-bot.git
git branch -M main && git push -u origin main
```

Keep the repository **private** — private costs nothing, and it keeps your bot's
code and schedule out of public view. (`.env` is git-ignored either way, so no
secret can ever be pushed.)

### Step 4 — Telegram

1. Message [@BotFather](https://t.me/BotFather), `/newbot`, follow the prompts,
   copy the token — it looks like `123456789:AAF…`.
2. Message [@userinfobot](https://t.me/userinfobot), copy your **numeric** id.
3. Put both in `.env` with the helper, which checks the shape of each before it
   stores them and never prints the token back in full:

   ```bash
   python scripts/set_telegram.py                      # asks for both
   python scripts/set_telegram.py --token T --chat C   # non-interactive
   python scripts/set_telegram.py --show               # what is set, masked
   ```

4. Send `/start` to your bot. Any other chat id is ignored silently — this bot
   is owner-only and does not talk to strangers. A bot cannot message you first,
   so this step is not optional.

5. Check it: `python scripts/doctor.py --send`. It puts one test message in your
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

### Adding keys

Do not hand-edit thirty lines — use the helper:

```bash
python scripts/set_keys.py              # paste keys, one per line, blank line to finish
python scripts/set_keys.py keys.txt     # or point it at a file
python scripts/set_keys.py --show       # what is configured, masked
python scripts/set_keys.py --clear      # remove every key
```

It de-duplicates, caps at 30, rewrites only the Twelve Data block (your Supabase
and Telegram lines are untouched), and chmods `.env` to owner-only. It never
prints a key in full — `--show` and the summary both use masked labels. Running
it repeatedly is safe; the block is replaced, not appended.

With 30 keys at the free tier's 8 credits/minute the pool has ~240 req/min of
headroom. That is capacity, not accuracy: more keys raise your rate limit, they
do not make the strategy profitable.

If you rotate keys, push them to Actions afterwards with `./scripts/push_secrets.sh`.

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

### Step 6 — GitHub Actions secrets

One command, from `eurusd-proxy-bot/`:

```bash
./scripts/push_secrets.sh
```

It reads your `.env`, sends **only** the genuinely secret values to GitHub
Actions, and prints their lengths — never the values themselves. Use
`./scripts/push_secrets.sh owner/repo` if the bot repo is not the one you are
standing in, and `--dry-run` first if you want to see the list.

Nothing here ever asks you to paste a key into a chat window.

## Layout

```
eurusd-proxy-bot/
├── fusion/                  the engine (standard library only)
│   ├── charter.py           the rules, as executable constants
│   ├── engine.py            the per-minute pipeline
│   ├── council.py           core gate + learned prototypes
│   ├── strategies.py        the rule sets, including the core 6-check gate
│   ├── scalping.py          six scalping members + the spread cost floor
│   ├── features.py          32 model inputs + strategy-facing indicator levels
│   ├── indicators.py        EMA/RSI/ATR/ADX/stochastic/CCI/VWAP, no look-ahead
│   ├── regime.py            TREND / RANGE / VOLATILE gating
│   ├── model.py             prequential logistic + bootstrap ensemble
│   ├── backtest.py          purged walk-forward replay of the gated system
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
│   ├── publish_repo.sh      Step 3 in one command (private repo + push)
│   └── push_secrets.sh      Step 6 in one command (.env → Actions secrets)
│   ├── smoke.py             offline pipeline demo  (python scripts/smoke.py 420)
│   ├── nightly.py           walk-forward + gates
│   ├── fallback.py          Actions stand-in, heartbeat-gated
│   ├── backtest.py          purged walk-forward replay of the full system
│   ├── backfill.py          history import (BACKFILL=1, env-flagged on purpose)
│   ├── set_keys.py          add/replace the Twelve Data keys in .env
│   ├── set_telegram.py      put the bot token and owner chat id in .env
│   └── termux_boot.sh       optional auto-start + restart-on-crash for Termux
├── docs/
│   ├── FUSION_BOT_BLUEPRINT.md   architecture record: why each piece exists
│   └── vortex-orbix-topics.md    research notebook from the two source repos
├── supabase/migrations/     0001_init.sql — 13 tables, RLS on all
│                            0002_add_missing_columns.sql — runtime columns
├── .github/workflows/       tests.yml, nightly.yml, fallback.yml
└── tests/                   322 tests, many of them charter guards
```

## Useful extras

```bash
python -m unittest discover -s tests -t .   # the charter, asserted
python scripts/smoke.py 420                 # watch it work offline
python scripts/doctor.py                    # is my setup actually correct?
python scripts/doctor.py --send             # ...and prove delivery works
python scripts/backtest.py                  # does the gated system pay its costs?
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
