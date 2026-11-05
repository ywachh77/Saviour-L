# Scalping: what the research actually found

This note records every measurement made on the scalping family, what it
implies, and — the part that matters — what it means for a 95% win-rate target.
It is written to be re-run, not taken on trust. Both scripts print the numbers
that decide the question.

    python3 scripts/scalp_research.py --paths 5 --bars 4000 --volatility 0.00024
    python3 scripts/scalp_edge_search.py --bars 3000 --paths 4

---

## 1. The arithmetic that governs every scalp

A scalp enters at the open of the next bar and exits at the close of a bar a
few minutes later. It pays the round-turn spread on every trade, win or lose.
If the average gross move it captures is `m` and the spread is `s`, then the net
per trade at hit rate `p` is:

    net = p·(m − s) − (1 − p)·(m + s) = m·(2p − 1) − s

so it breaks even at:

    p = 0.5 + s / (2m)

Two numbers decide everything: the mean gross move, and the spread.

| mean gross move `m` | spread `s` = 1.2 | hit rate needed |
|---|---|---|
| 1.4 pips | 1.2 pips | **92.9%** |
| 1.8 pips | 1.2 pips | 83.3% |
| 2.4 pips | 1.2 pips | 75.0% |
| 4.0 pips | 1.2 pips | 65.0% |
| 6.0 pips | 1.2 pips | 60.0% |

Read the first row again. A scalp that captures **1.4 pips** — which is what
was measured — needs a **92.9% hit rate** merely to break even. Not 95% net, not
95% after costs: 92.9% to make exactly zero.

## 2. What the 26 members actually do

`scalp_research.py`, 12,000 one-minute bars across 3 independent paths, at a
volatility that puts the M1 ATR in the 2–4 pip band the cost guard requires
(otherwise no member speaks at all, which is itself the finding):

| member | spoke (% of bars) | gross win % | needed % | mean gross | mean **net** |
|---|---|---|---|---|---|
| scalp_impulse | 20.0% | 50.5 | 77.8 | −0.01 | **−1.21** |
| scalp_vwap_reversion | 30.8% | 50.6 | 77.9 | +0.01 | **−1.19** |
| scalp_reversion | 10.5% | 50.5 | 77.4 | +0.06 | **−1.14** |
| scalp_stoch_cross | 10.7% | 50.1 | 78.0 | +0.02 | **−1.18** |
| scalp_micro_breakout | 6.4% | 50.3 | 77.9 | −0.02 | **−1.22** |
| scalp_pivot_fade | 10.8% | 50.7 | 78.4 | +0.05 | **−1.15** |
| scalp_expansion | 4.8% | 48.3 | 75.4 | −0.02 | **−1.22** |
| scalp_wick_reversal | 0.5% | 40.7 | 76.8 | −0.24 | **−1.44** |

Every single member, at every hold length from one to five minutes:

* **gross hit rate 49–52%.** That is a coin flip. The patterns do not predict
  direction.
* **mean gross move ≈ 1.4 pips**, because the cost guard admits bars whose ATR
  is barely above the move they need, and the realised move is smaller than the
  ATR (the wick is range no position captures).
* **net −1.1 to −1.4 pips per trade.** Which is what the arithmetic in §1 says
  it must be.

This is the important part: the loss is **not** a bug, a bad parameter, or an
unlucky sample. It is the spread. A random direction on these bars nets the same
number.

## 3. Did filters rescue it? No.

The obvious hope is that a *condition* — a trending regime, the London session,
high volatility, higher-timeframe agreement — lifts the hit rate above the line
in §1. `scalp_edge_search.py` tests exactly that, and is built to not fool
itself:

* candidates are fixed conjunctions of conditions specified before any numbers
  were seen (regime, session, volatility band, EMA alignment, MTF tilt);
* scored on a **train** split, then re-scored on **independent held-out paths**;
* a two-sided-significance `p` is printed beside each, so 57% over 300 trades
  is visibly not 57% over 30.

Train split, best candidates:

| member | horizon | session | filter | n | gross win % | needed % | p |
|---|---|---|---|---|---|---|---|
| scalp_vwap_reversion | 3 | ASIA/off | close_middle | 313 | **57.5** | 68.2 | 0.004 |
| scalp_candle_cluster | 3 | any | mtf15_flat | 324 | 52.8 | 67.0 | 0.16 |
| scalp_trend_pullback | 3 | London | adx≥20 + MTF | 218 | 54.1 | 67.5 | 0.11 |

The 57.5% / p=0.004 candidate looks like a discovery. On the held-out paths:

| member | horizon | session | filter | n | gross win % | needed % | p |
|---|---|---|---|---|---|---|---|
| scalp_vwap_reversion | 3 | ASIA/off | close_middle | 326 | **46.6** | 67.6 | 0.89 |
| scalp_candle_cluster | 3 | any | mtf15_flat | 318 | 50.0 | 68.4 | 0.50 |
| scalp_trend_pullback | 3 | London | adx≥20 + MTF | 231 | 45.0 | 67.2 | 0.94 |

**None of the train split's best twenty stayed positive out of sample.** The
best in-sample candidate fell to 46.6%; every other candidate landed back on
50%. Twenty candidates tested at p≈0.05 will throw a false positive about every
time, which is precisely what the in-sample column showed.

## 4. What this means for "95% accuracy, one loss in twenty"

Assuming 20 trades, 95% is **19 correct and 1 wrong**. Given §1, that requires:

* a **≥92.9%** true hit rate on the payoff actually available (which is not
  observed, and not achievable by any filter tested), **or**
* a **larger payoff per winner** — hold out for a big move and cut the rest.
  But that raises the cost per trade, and a 1.2 pip spread becomes a much
  larger fraction of the average loser, so the required hit rate rises rather
  than falls.

There is no third route. Under a fixed 1.2 pip spread, a 1.4 pip mean move, and
a 50% base rate, 95% is not a target you tune toward. A run of 20 that happens
to lose once is **not** evidence of a 95% system either: at a 60% true rate, one
loss-or-fewer in twenty happens about 0.02% of the time, so a 19/20 sample says
much more about the sample than about the system. That is why this project
reports expectancy, the required hit rate, and a Wilson interval next to every
win rate, and why the 95% target has been refused in writing rather than
implemented.

The honest reframing, which the code already uses:

* **the metric is net expectancy per trade**, not win rate;
* **quality is measured as "better than the rate the spread demands"**, not
  "better than 50%";
* **a 20-trade sample is a description of 20 trades**, and the interval printed
  beside it will nearly always include zero.

## 5. What was changed as a result

* `fusion/scalping.py` — module docstring now carries this measurement and the
  three conclusions, so the next person to read the rules knows what they
  measured rather than what they hoped.
* `scripts/scalp_research.py` — per-member measurement, gross and net, with the
  required hit rate beside the achieved one.
* `scripts/scalp_edge_search.py` — the conjunction search, with a held-out split
  and a significance column.
* `fusion/costs.py` — `break_even_win_rate_from_move()`, the arithmetic in §1 as
  a function, so a measured move can be turned into a required rate directly.
* `tests/test_scalping.py::ScalpEdgeIsMeasuredNotAssumed` — pins the arithmetic,
  pins that a 95% rate *would* be recognised as profitable if it existed, and
  pins the safety property that a scalping prototype can never originate a
  signal.

## 6. What would change the conclusion

Not a better optimiser. One of these:

1. **A lower spread.** At 0.2 pips the required rate on a 1.4 pip move falls to
   57.1%; at 0.1 pips, to 53.6%. Spread is the binding constraint, and a broker
   with a genuinely tighter book changes the arithmetic more than any filter.
2. **A larger mean move per trade**, with the hit rate held. That means holding
   longer — which stops being scalping. This is why the adaptive-horizon work
   (stretching five minutes to ~30) is the part of this repository that has a
   measured positive effect, small and noisy as it is.
3. **A directional forecast with real skill**, measured out of sample, on the
   bars in question. A model at 58% on a 4 pip move would clear the line. The
   walk-forward machinery in `fusion/learning.py` is where that has to be shown,
   and it has to be shown out of sample, with the interval excluding zero.

Until one of those is demonstrated, the correct behaviour for this family is
what it already does: speak only as a prototype, never originate a signal, and
be measured rather than assumed.

---

## 7. "So how do we achieve it?"

Answered by `scripts/scalp_geometry_study.py`, which measures every lever rather
than arguing about it. The short version: **you do not achieve 95% by forecasting
better. You achieve it by changing the payoff** — and then the number stops
being 95%.

### The payoff grows with the square root of the hold

The system settles at a fixed expiry (`fusion.settle`), so a trade's payoff is
the mean absolute move over its hold. Nothing about the forecast changes that:

| hold | mean abs move | spread as share of move | required hit rate @ 1.2 pip |
|---|---|---|---|
| 1 min | 1.92 pips | 62.4% | **81.2%** |
| 2 min | 2.72 | 44.1% | 72.0% |
| 5 min | 4.29 | 28.0% | 64.0% |
| 15 min | 7.28 | 16.5% | 58.2% |
| 30 min | 10.30 | 11.6% | 55.8% |
| 60 min | 14.97 | 8.0% | **54.0%** |

Holding from one minute to sixty drops the required hit rate from 92.9% to 54%.
That is free — it needs no new model — and it is why the adaptive-horizon work is
the only part of this repository whose effect has been measured as positive. It
is also, of course, no longer scalping.

### Spread moves it just as hard

Required rate at a 1-minute hold: 81.2% at 1.2 pips, 68.2% at 0.7, 57.8% at
0.3, 52.6% at 0.1. A better book does more for this family than any filter in
the search. At 0.1 pips, a coin flip is nearly good enough.

### The trap that produces "95%" and loses money

The tempting route — small target, wide stop — really does manufacture a high
hit rate, **with no forecast at all**. Measured on the same paths, entries taken
by coin flip:

| target (ATR) | stop (ATR) | measured hit rate | theory 1/(1+k) | net per trade |
|---|---|---|---|---|
| 2.00 | 0.50 | 20.4% | 20.0% | +0.02 pips |
| 1.50 | 0.50 | 23.5% | 25.0% | −0.07 |
| 1.00 | 1.00 | 35.3% | 50.0% | −0.70 |
| 0.50 | 2.00 | 56.3% | 80.0% | −1.42 |
| 0.25 | 5.00 | 79.1% | 95.2% | **−2.04** |

The hit rate follows 1/(1+k), the reward-to-risk ratio, and nothing else. This
is the crucial point: **a 95% win rate is a 0.053:1 payoff.** You get it by
being right nineteen times for a penny and wrong once for a pound. Every one of
those twenty trades pays the spread, so the geometry that maximises the win rate
minimises the money. In the table the highest hit rate is the most expensive
row.

### Why "tune until it's 19 of 20" is not a route

The same script runs that search honestly on a random walk with **no edge**,
counting only trades actually taken:

```
13,734 threshold/window configurations tested in-sample
6 of them produced 19 or 20 wins out of 20     (best in-sample: 20/20)

Those 'discoveries' re-scored on unscored bars:
0 still produced 19/20
mean block out of sample: 9.8/20
```

A search *manufactures* 19/20 blocks on pure noise. They revert to 9.8/20 the
moment they meet bars they were not tuned on. Any procedure that can be tuned
until a 20-trade block contains one loss is a procedure that will, and the block
it finds means nothing. **The 19/20 block is what a search produces; it is not
what a system is.**

### The three routes that do work

1. **Hold longer** — free, already implemented, measured positive. 54% needed at
   60 minutes instead of 81% at one.
2. **Pay less spread** — free, and larger in effect than any filter tested. 57.8%
   needed at 0.3 pips instead of 81.2%.
3. **A forecast with demonstrated out-of-sample skill at a hold long enough to
   clear the table above.** This is the only route that scales, and the
   walk-forward machinery in `fusion/learning.py` is where it must be proven —
   on the held-out split, with an interval that excludes zero.

What is *not* a route: tuning thresholds until a 20-trade sample looks good.
Section 5 above measures exactly that and it is noise.

### What the honest target looks like

Not "19 of every 20 win". Instead, on a rolling block of twenty trades:

- **is net expectancy per trade positive**, after the spread, with an interval
  that excludes zero?
- **is the hit rate above the required rate for the payoff actually achieved**
  (54% at a 60-minute hold, 81% at one minute) — measured, not assumed?
- **does the block hold out of sample**, on trades the threshold never saw?

Those three are answerable, falsifiable, and they are what the ledger, the
Wilson interval and the walk-forward replay already compute.

---

## 8. One engine instead of 132, and the blind-long defect

### The defect, measured

`scripts/direction_audit.py` evaluates every member on identical bars and counts
which way it points. Two findings, both defects:

**One member was structurally blind.** `smc_ascending_triangle` produced **259
BUYs and 0 SELLs** — 87% of them aligned with the current trend. Its two "pattern"
conditions were:

| condition | what it actually read | true on |
|---|---|---|
| flat resistance | `triangle_flat_high` = the 20-bar **high** | **98.5%** of bars |
| rising lows | `triangle_rising_low` = the 10-bar **minimum low** | **98.5%** of bars |

So "ascending triangle breakout" meant "close above the recent high" — a trend
confirmation with no opposite. The bearish mirror had the same class of bug:
`descending_flat_low` accepted a single touch, and the minimum of a window is
always touched once, so it returned a level on **60.8%** of bars and a breakdown
on **0%**. It has never fired, in either direction, on any data.

Fixed: a flat high now requires a level tested at least twice with flat highs and
a genuine higher-low sequence (7.0% and 21.4% of bars); support requires a second
touch from a different candle (8.7%).

**Four members were one rule.** `breakout`, `quant_donchian_breakout`,
`scalp_breakout_retest` and `smc_ascending_wedge` emitted *identical* signals bar
for bar, in both directions. The wedge's `wedge_breakup`/`wedge_breakdown` were
just `close > prior_high` / `close < prior_low`, the same as a plain breakout.
Four votes, one opinion. The wedge now requires the boundary lines to actually
converge; the remaining three are declared as a duplicate group and cast one vote.

### Was it "blindly following the trend"? Largely, yes

For every member: `trend%` is how often its signal agrees with the *current*
trend, `next%` how often it agrees with the *next* bar. The gap is the amount by
which it is describing the past rather than forecasting the future.

| member | trend% | next% | gap |
|---|---|---|---|
| `smc_ascending_triangle` | 86.9% | 55.2% | +31.7 |
| `ema_ribbon_pullback` | 98.5% | 43.9% | **+54.6** |
| `quant_adx_threshold` | 90.2% | 51.2% | +39.0 |
| `viral_rsi_trend` | 90.1% | 49.8% | +40.3 |
| **median member** | **77%** | **50.5%** | **+26** |

The council's median member is trend-aligned 77% of the time and right about the
*next* move 50.5% of the time. A majority vote of that is a trend echo — which is
both why it loses to the spread and why it looked BUY-heavy in an uptrend.

### The single engine

`fusion/oracle.py` keeps all 132 members as **evidence, not voters**:

```
94 features (incl. microstructure)  +  12 panel inputs (per-family stances,
                                        net, agreement, participation,
                                        dispersion, trend gap)
                    -> one logistic model -> one probability -> one decision
```

Members are collapsed three ways so an opinion cannot be counted twice: identical
rules fold into one (`DUPLICATE_GROUPS`), members aggregate to their family, and
an abstention is not a neutral vote. Wired into `FusionEngine.decide(..., oracle=)`,
the engine's read becomes the verdict — the council is consulted, never obeyed.

Measured on 11,382 bars, held out on paths the model never trained on:

| candidate | accuracy | 95% interval | net/trade | BUY share |
|---|---|---|---|---|
| always long | 51.1% | 49.8–52.4 | −0.88 pips | 100% |
| **council majority vote** | 48.7% | 47.4–50.0 | −1.38 | 50% |
| **one engine, raw call** | **52.7%** | 51.4–54.0 | −0.61 | 49% |
| one engine, at cost floor | 53.6% | 52.0–55.2 | −0.46 | 48% |

Three things to read carefully:

1. The engine beats the council by **4 points of accuracy and 0.8 pips a trade**.
   That is a real improvement in ranking, and it is the honest justification for
   replacing the vote.
2. It still loses to the spread, because 53.6% is short of the **59.9%** a 7.35 pip
   payoff at 1.2 pips demands. One engine is a better engine; it is not a licence
   to claim a profit. §7 explains why no filter closes that gap and what would.
3. A **label-shuffle control** is part of the test suite: scramble the training
   labels and the engine's accuracy must fall to ~50%. It does (50.8%). Without
   that control, the 52.7% figure above would not mean anything.

Two bugs in the engine itself were caught by tests while writing this, both of
the same kind — an asymmetry that manufactures a directional bias:

* direction was first taken from the *calibrated* probability. Isotonic
  calibration fitted on a tail whose base rate is 46–52% tops out below 0.5, so
  "calibrated ≥ 0.5" could never fire: on a sample rising 48.6% of the time the
  engine called BUY on 8.4% of bars.
* taking it from the *raw* score was no better: the logistic intercept settles
  wherever the data says, and drifted to bias −0.374, putting the entire score
  distribution below 0.5.

Direction is now anchored to the **median raw score**, which is two-sided by
construction — the engine expresses a *relative* view, which is all an
uncalibrated score can honestly express, and the cost floor decides whether that
view is worth money. Tested by flipping the market's own drift: 15% BUY on a
falling sample, 61% BUY on a rising one.

---

## 9. Why there are consecutive losses

Measured by `scripts/loss_streak_study.py`. Three causes; only one is a defect.

### 1. At a coin-flip hit rate, streaks are arithmetic (not a defect)

The measured gross hit rate is ~50%. At p = 0.50, probability of at least one run
of k losses in n trades:

| trades | P(2+ run) | P(3+ run) | P(5+ run) | P(8+ run) |
|---|---|---|---|---|
| **20** | **98.3%** | **78.7%** | 25.0% | 2.7% |
| 50 | 100% | 98.3% | 55.2% | 8.4% |
| 100 | 100% | 100% | 81.0% | 17.0% |
| 500 | 100% | 100% | 100% | 62.5% |

**In a 20-trade block, two consecutive losses is the normal case (98%), and three
is the likely case (79%).** Twenty trades with *no* two-loss run is the unusual
outcome. This is what a 50% hit rate looks like — it is not a bug, and no counter
or guard removes it. Only a higher hit rate does, and §7 shows why filtering
cannot produce one.

### 2. The stop-loss guard cannot prevent a streak it observes (structural)

A loss is only visible once **settled**, and settlement happens at entry + hold.
So the sequence is always:

```
trade 1 sealed  ->  ...  ->  trade 1 settles LOSS
                            trade 2, 3, ... already sealed, settle later, also LOSS
```

The guard then blocks for the cooldown — but the losses it reacts to have already
been taken. **It limits the next sequence, not this one.** Consecutive losses are
not a failure of the guard; they are what the guard is downstream of.

Note also `loss_guard_streak = 1` in config: it blocks after *any single* loss,
then releases after 60 minutes, and with `loss_guard_shadow_recovery` on it holds
up to a further 720 minutes and releases **on a timeout regardless of whether
anything improved**. The blocking itself is wired correctly (verified through
`Runner._check_runtime_protections`); it simply cannot act retroactively.

### 3. The loss-learning loop fired every night and fitted the recent past (a defect, fixed)

`loss_patterns.should_trigger_loss_learning` compared the win rate against a
**hardcoded 0.95**. A live win rate is ~0.50, so that condition was true on
every single run. It was not a loss detector — it was "every night". It then:

* duplicated the most recent **losing** samples (1–3×, scaled by loss size),
* refit, and
* installed the result as the champion **before** walk-forward.

Fitting harder to the bars that just went against you is recency-fitting. What it
actually costs, measured on held-out bars over 6 paths:

| clean fit | recency-weighted refit | change |
|---|---|---|
| 58.3% | 59.4% | +1.1% |
| 49.2% | 54.6% | +5.4% |
| 53.0% | 52.8% | −0.3% |
| 50.3% | 48.8% | −1.5% |
| 47.6% | 50.6% | +2.9% |
| 48.6% | 58.0% | +9.4% |

**Mean +2.84%, standard error ±1.51%.** Indistinguishable from zero, with four of
six paths inside the noise band. The honest conclusion: upweighting recent losses
neither fixes the losses nor measurably deepens them — it is **noise dressed as a
correction, run every single night**, which is a nightly opportunity to overfit
for no measured benefit.

### Fixed

`should_trigger_loss_learning` now derives its target from the trades themselves:

```
break-even = 0.5 + spread / (2 × mean |gross pips|)
```

and requires the shortfall to be **statistically real**: the Wilson *upper* bound
on the observed win rate must sit below that line. Consequences:

| sample | mean move | required | triggered? |
|---|---|---|---|
| 9-of-20 | 12.0 pips | 55% | **no** — 65.8% upper bound, this is noise |
| 9-of-20 | 3.0 pips | 70% | **yes** — 65.8% upper bound is genuinely short |
| 60-of-100 | 4.0 pips | 65% | no — healthy |
| 60-of-100 | 2.6 pips | 73% | **yes** — same win rate, thinner payoff |

The same 9-of-20 is judged differently depending on the payoff, which a fixed
0.95 could never do. And when the mean gross move does not clear the spread at
all, the retrain is **refused outright** — no win rate rescues arithmetic.

`calculate_loss_learning_rate` is anchored to a fair coin (0.50) instead of 0.95,
so it no longer sits permanently at its cap. `config.target_win_rate` was dead
config — nothing read it — and is now documented as such.

Pinned by `tests/test_loss_streaks.py` (15 tests), including the streak
probabilities above, so that nobody "fixes" consecutive losses by suppressing the
counter instead of raising the hit rate.

---

## 11. The uploaded strategy, measured

`reference/eurusd_strategy.py` (the project owner's file, kept **unmodified**),
analysed by `scripts/the_file_edge_study.py`. Seven samples: the 740 real bars
plus 6 zero-drift random walks of 4,000 bars each.

The file has **no ADX and no cost model**. ADX appears below only as a
*conditioning* variable — when its signals work, not how they are formed.

### It loses on both kinds of data

| threshold | expiry | trades | gross win% | net win% | net pips |
|---|---|---|---|---|---|
| 65 | 1m | 330 | 50.0% | 30.6% | −437 |
| 65 | 5m | 330 | 47.9% | 38.2% | −481 |
| 65 | 15m | 330 | 41.5% | 34.5% | −855 |
| **80** | any | **1** | — | — | −4 |

Pooled over all seven samples: **2,205 trades, 40.6% net win rate, −1.29 pips
per trade.** The gross rate is 41–50%; the spread removes the rest. The default
threshold of 80 produces **one trade in 20,000 bars**, which is the score-cliff
problem from §10 of the file review.

On the real bars alone, expiry 15m is the one positive cell (+67 pips over 116
trades) — one cell out of seven, on one day. It is not a finding.

### Where its signals win and lose: nothing survives

| condition | best bucket | net win% | vs base | p |
|---|---|---|---|---|
| RSI | 30–50 | 41.6% | +1.0% | 0.55 |
| ADX | 25–35 | 43.5% | +2.9% | 0.18 |
| ATR | mid | 43.0% | +2.4% | 0.29 |
| EMA slope | falling | 44.8% | +3.6% | 0.23 |
| Close in range | low | 44.0% | +2.8% | 0.32 |

**Survives multiple-comparison correction: none.** The base rate is 40.6%, so
every bucket sits within a few points of it and none is distinguishable from the
others. Notably, the earlier "ADX < 20 is strongly bullish" result from §10
**does not appear here** (ADX < 20 is −1.4 points) — further evidence that the §10
result was one trending session, not a regime effect.

Direction balance is fine: 200 BUY / 130 SELL. It is not blind-long.

### The 19/20 question, answered

Every expiry was searched in-sample across all samples (chronological half for
tuning, untouched half for testing):

| expiry | trades | blocks of 20 | best block | 19+ blocks | net/trade |
|---|---|---|---|---|---|
| 1m | 1,130 | 997 | 13 | 0 | −1.15 |
| 3m | 1,126 | 993 | 14 | 0 | −1.34 |
| 5m | 1,123 | 990 | 14 | 0 | −1.41 |
| 10m | 1,117 | 984 | 16 | 0 | −1.47 |
| **15m** | 1,115 | 982 | **19** | **1** | −1.51 |

A single 19/20 block appears — at 15-minute expiry, out of **982 overlapping
windows**, on data where the strategy loses 1.51 pips a trade. On the held-out
half that same parameter gives: **best 17/20, zero 19+ blocks, −1.97 pips a
trade.**

So the answer to "I need 19 wins under 20" is:

* a 19/20 block **can be found by searching** — one exists here, in ~6,000 windows
  of overlapping data, on a strategy that loses money;
* it **does not survive the search being over.** Held out: 17/20 and losing
  twice as much per trade;
* on a fair coin, 19-of-20 occurs with probability **21/2²⁰ ≈ 2.0e-05 per block** —
  across 982 overlapping windows you would expect it to appear now and then
  without any edge at all. That is exactly what happened.

The block is a property of having looked at 982 windows, not a property of the
strategy.

---

## 12. `reference/eurusd_m1.py` — a correct implementation of a losing strategy

The uploaded engine is **the honest version of the 19/20 experiment**, and it is
noticeably better built than most. Verified by `scripts/verify_eurusd_m1.py`:

| check | result |
|---|---|
| Causality (truncation test) | **PASS** — 0 mismatches |
| `net_pips == gross − spread` | **PASS** — 0 wrong of 229 |
| Break-even formula | **PASS** — +0.050 / 117.20 → 99.9574% |
| Intrabar ambiguity | 0 of 229 touched both barriers; resolved **stop-first** (conservative) |

**What it does right**, and it is a genuine list:

* Charges the spread on every trade and reports **net** pips, not gross.
* Chronological train / held-out split, with boundary trades excluded.
* **Wilson interval** on the win rate and a **bootstrap CI** on expectancy.
* Reports worst trade, max drawdown, longest losing run.
* Explicit `safety` block: no martingale, no averaging down, no adding to a
  loser, no same-candle re-entry, no live execution.
* **States its own limitations** — including the one that matters: OHLC cannot
  reveal intrabar order, so it assumes the stop came first.
* Warns if `tp <= spread`.

That is the discipline this repository has been asking for. The problem is not
the code. The problem is the geometry it was asked to test.

### The result, and why every number in it is negative

| mode | split | n | win rate | **needed** | margin | net/trade |
|---|---|---|---|---|---|---|
| coinflip | train | 1,031 | 99.81% | 99.96% | −0.15pp | −0.177 |
| coinflip | held out | 902 | 99.67% | 99.96% | −0.29pp | −0.340 |
| ema | train | 516 | 99.03% | 99.96% | −0.93pp | −1.086 |
| ema | held out | 740 | 99.59% | 99.96% | −0.36pp | −0.425 |
| momentum | train | 731 | 99.59% | 99.96% | −0.37pp | −0.431 |
| momentum | held out | 1,115 | 99.82% | 99.96% | −0.14pp | −0.160 |

The "margin" column is the whole story. The win rate is 99%+ and **it is never
enough**, because a 1.25 pip target against a 1.20 pip spread leaves **0.05 pips
of profit per win** — 4% of the spread. A single 116 pip stop-out erases **2,344
winning trades**. Break-even requires 99.9574%.

### The control that makes the finding airtight

The same engine, three direction rules. One of them is a **coin flip** — a
deliberate admission that the signal cannot matter at this geometry.

| mode | trades (train) | avg bars held |
|---|---|---|
| coinflip | 1,031 | 8.7 |
| ema | 516 | 18.0 |
| momentum | 731 | 11.5 |

The EMA rule is stopped out **twice as often** and holds twice as long, yet
earns nearly the same negative expectancy. **Direction selection changes how
often you get stopped, not whether you make money.** That is the cleanest
demonstration in this repository that at this payoff the win rate is a property
of the barriers, and the barriers do not touch the spread.

### What would make this engine profitable

Not a better signal. Either:

* **`--tp` well above the spread.** At 1.25 pips against 1.20, profit per win is
  0.05; at 4 pips it is 2.80, and the break-even win rate falls to 50.6% (the
  expectation is *unchanged*, but the variance and the tail collapse).
* **`--spread 0.3`.** Profit per win becomes 0.95 pips, break-even 99.2%, and
  the same trades turn positive.

Run either and re-read the table; the code does not need to change, which is a
compliment to the code.

---

## 13. "At least 18 of 20, with no consecutive loss" — measured

Relaxing 19 to 18 changes the *shape* requirement, not the arithmetic.

**Geometry.** On a driftless walk the win rate is ≈ `sl/(tp+sl)`, so 90%
requires the stop to be **9× the target**. At `--tp 2.5 --sl 22.5 --spread 1.2`
(coin-flip direction, 20,000 bars):

| split | trades | win rate | blocks ≥18/20 | ≥18 **and** no back-to-back loss | longest losing run | net/trade |
|---|---|---|---|---|---|---|
| train | 1,340 | 93.88% | 1,165 | 1,150 | **3** | −0.230p |
| held out | 1,354 | 93.72% | 1,184 | 1,155 | **3** | −0.269p |

So ≥18/20 blocks are everywhere — and the run still had a **3-loss streak in
both halves** and 5–7 back-to-back loss pairs. Average win +1.30p, average loss
−23.70p, break-even win rate **94.80%** against 93.9% achieved: relaxing the
target moved the shortfall, it did not close it.

**"No consecutive loss" cannot be a property of a strategy that keeps
trading.** At a 10% per-trade loss probability, the chance of *never* seeing
two losses in a row is 83.9% over 20 trades, 40.2% over 100, and **0.01% over
1,000**. At the measured 6.2% loss rate it is still only 2.6% over 1,000
trades, with the first back-to-back loss expected around trade ~260. It is a
statement about a short sample, not a designable guarantee.

The only mechanisms that can *enforce* it are: stop trading permanently after
the first loss (one loss ever, no strategy), or increase size / hold after a
loss until it recovers — martingale and averaging down, both forbidden by the
project charter, and both convert a losing streak into a single catastrophic
loss rather than removing it.

**What is honestly available:** a 90%-shaped payoff (1:9 barriers) whose
18-of-20 blocks are common, reported *alongside* −0.23 to −0.27 pips/trade,
a 94.8% break-even rate, and a longest losing run of 3. The number 18 is a
dial setting, exactly as 19 was; turning it down buys a lower break-even bar
(99.96% → 94.80%) but not a positive one.

---

## 14. "How can we achieve 17 wins in 20 trades?" — measured

17/20 is an 85% win rate. On a driftless walk the win rate is set by the
barriers alone, `win% ≈ sl/(tp+sl)`, so 85% requires the stop to be
**5.67× the target**. Nothing about the signal is involved.

Measured with the uploaded engine (`--tp 3.0 --sl 17.0 --spread 1.2`, 20,000
bars):

| mode | split | trades | win rate | **break-even** | blocks ≥17/20 | net/trade |
|---|---|---|---|---|---|---|
| coinflip | train | 1,360 | 88.53% | 91.00% | 1,099 | −0.494p |
| coinflip | held out | 1,453 | 89.47% | 91.00% | 1,211 | −0.306p |
| ema | train | 1,341 | 87.62% | 91.00% | 1,020 | −0.676p |
| ema | held out | 1,412 | 88.88% | 91.00% | 1,134 | −0.424p |
| momentum | train | 1,363 | 88.55% | 91.00% | 1,091 | −0.489p |
| momentum | held out | 1,460 | 90.41% | 91.00% | 1,231 | −0.118p |

So 17-of-20 blocks are not merely achievable, they are the *typical* outcome
(>80% of rolling windows), for a coin flip as readily as for a signal. The
17/20 headline is satisfied by construction. The cost column is unchanged in
sign: average win +1.80p, average loss −18.20p, one loss erases 10.1 wins,
worst trade −18.2p, max drawdown 234–915 pips, longest losing run 3–5.

### The one real lever at a fixed win-rate target

Break-even is `(sl + spread)/(tp + sl)`, so the gap between the driftless win
rate and break-even is exactly **`spread/(tp+sl)`**. The ratio fixes the win
rate; the *absolute size* fixes how much genuine edge you need:

| tp | sl | ratio | driftless | break-even | **edge required** |
|---|---|---|---|---|---|
| 3 | 17.0 | 1:5.7 | 85.0% | 91.0% | **+6.0pp** |
| 4 | 22.7 | 1:5.7 | 85.0% | 89.5% | +4.5pp |
| 5 | 28.3 | 1:5.7 | 85.0% | 88.6% | +3.6pp |
| 10 | 56.7 | 1:5.7 | 85.0% | 86.8% | **+1.8pp** |

Same 17/20 shape, same 85%, but at `tp 10 / sl 56.7` the strategy needs only
**1.8 percentage points** of real directional edge above a coin flip to break
even, instead of 6.0. That is the honest version of "how to achieve it":
keep the shape, enlarge the barriers until the spread is a small fraction of
the trade, and then the remaining gap is small enough that a real edge —
if one is ever found — could plausibly close it. No edge of even 1.8pp has
been measured in this repository yet (§10, §11); that, not the barrier ratio,
is the open problem.

---

## 15. Target locked: 17 of 20 is the floor

Recorded as a project constraint: **no configuration below 17 wins per 20
trades (85%) will be considered.** Interpreted precisely, because the loose
reading is unachievable: the floor applies to the overall win rate over a
stated sample (with its Wilson lower bound reported), **not** to every rolling
block of 20 — at an 85–90% win rate some blocks always fall below 17, and a
per-block guarantee would require the martingale class of mechanisms the
charter forbids.

All three 1:5.7 geometries measured (20,000 bars, spread 1.2, three direction
modes, train + held out = six cells each):

| geometry | trades/split | win rates across 6 cells | cells ≥85% | break-even | edge required |
|---|---|---|---|---|---|
| tp 3 / sl 17 | ~1,400 | 87.6–90.4% | **6 of 6** | 91.00% | +6.0pp |
| tp 5 / sl 28.3 | ~490 | 84.2–89.4% | 5 of 6 | 88.59% | +3.6pp |
| tp 10 / sl 56.7 | ~100 | 84.8–89.7% | 5 of 6 | 86.81% | +1.8pp |

Reading: **the small geometry holds the floor most reliably** (more trades,
tighter intervals — coin-flip Wilson lower bounds 86.5% / 85.8% at tp 5, and
tp 3 never came close to 85), **but it is the furthest from break-even**. The
large geometry needs only +1.8pp of genuine edge, yet with only ~100 trades
per split its win rate is noise-dominated (84.8% in one cell) and cannot
certify an 85% floor.

So the locked target creates an explicit two-gate acceptance rule going
forward:

1. **Floor gate:** win rate ≥ 85.0% in *both* train and held-out, n ≥ 400
   trades per split, Wilson lower bound reported beside the point estimate.
2. **Survival gate (unchanged):** net expectancy, break-even win rate, worst
   trade, and max drawdown reported beside any win rate; a configuration
   that passes gate 1 while losing net pips is recorded as *floor met,
   expectancy negative* — never as a success.

On current measurements `tp 3 / sl 17` passes gate 1 in all six cells and
fails gate 2 in all six (−0.12 to −0.68 pips/trade). Closing gate 2 requires
+6.0pp of directional edge at that size, or accepting fewer, larger trades
at tp 10 where +1.8pp would suffice but the floor cannot yet be certified.
That tension — floor reliability versus edge required — is now the stated
shape of the problem.

---

## 16. Supervisor fine-tune: plan, and why the data comes first

Agreed direction: fine-tune the uploaded LLM as a **supervisor** — it sits
above the barrier engine and answers APPROVE / VETO on each candidate signal,
seeing only signal-bar information. `scripts/build_supervisor_dataset.py`
builds the training data with the correct labels: a candidate is APPROVE iff
taking it would have won **net of spread** under the engine's own rules
(next-bar-open entry, stop-first, non-overlapping candidates).

Measured on the repo's real candles (740 bars, spread 1.2):

| geometry | labelled trades | train / held out | approve-all net |
|---|---|---|---|
| tp 3 / sl 17 | **33** | 16 / 17 | −11.2p / +10.6p |
| tp 10 / sl 56.7 | **4** | 2 / 2 | +17.6p / +17.6p |

This settles the sequencing: **33 examples cannot fine-tune anything.** A
model trained on them would memorise 33 trades and report whatever the
memorisation implies. The dataset is, however, exactly the right shape to
*e*valuate a supervisor later (does it beat approve-all on held-out trades?).

Blocking requirements before any fine-tune runs:

1. **The model upload.** Not yet received. Needed: full weights
   (safetensors), `config.json`, and tokenizer files. A GGUF file cannot be
   fine-tuned (inference-only format). This sandbox has **no GPU**, so CPU
   LoRA is realistic only for small models (≤ ~1B parameters); a larger model
   means either a hosted fine-tuning API or accepting evaluation-only here.
2. **Real training volume.** Months of EUR/USD M1 (≥ ~50k bars → ~2,000+
   labelled trades at tp 3), split chronologically, never shuffled.
3. **The two-gate rule still governs the result** (§15): the supervisor is
   only a success if its approved set holds ≥85% in held-out data *and* has
   positive net expectancy against the 91.0% break-even (tp 3) — a veto
   model that merely skips trades has not added edge until both are shown.

---

## 17. NVIDIA supervisor service (24/7) + one-time 19/20 test

Built: `eurusd-proxy-bot/scripts/nvidia_supervisor_service.py`, using the
repo's existing AI gateway (`fusion/ai_gateway`, NVIDIA NIM provider at
`integrate.api.nvidia.com/v1`). It does not add a second engine: the
supervisor only APPROVEs/VETOes candidates from the same barrier engine.

* **24/7 mode** (`--daemon`): every cycle it auto-loads stored credentials
  (a key connected later takes effect live, no restart), health-checks the
  gateway, writes `var/nvidia_supervisor_heartbeat.json`, and re-runs the
  test every N cycles. Verified locally: cycles run, heartbeat written,
  clean SIGTERM stop. Deploy unit:
  `eurusd-proxy-bot/deploy/nvidia-supervisor.service`.
* **One-time 19/20 test** (`--once`, tp 1.25 / sl 116 / spread 1.2), run
  2026-11-05 in this sandbox. No NVIDIA credential is stored here and
  NVIDIA hosts are unreachable from the sandbox, so the supervisor ran in
  the explicitly labelled `NO_NVIDIA_CREDENTIAL` approve-all baseline mode
  — the number the AI must beat, not an AI result:

| data | held-out n | win rate | break-even | net/trade | worst | max DD | 19+ windows |
|---|---|---|---|---|---|---|---|
| synthetic 20k, zero drift | 784 | 99.62% | 99.96% | −0.399p | −117.2p | 319.9p | 761 |
| real 740 bars | 14 | 100.0% | 99.96% | +0.05p | +0.05p | 0.0p | 0 (n<20, inconclusive) |

Gates on the 20k test: floor-17 **passed**, expectancy **failed** — the same
shape as every previous 19/20 measurement (§11, §12). When a real NVIDIA
key is stored on the deployment host, the identical command re-runs with
the Qwen supervisor voting, and the two rows (baseline vs approved) become
the comparison that decides whether the AI adds anything.

**Secret wiring (same day):** the gateway originally read only its
credential file, so an `NVIDIA_API_KEY` added in Secrets would have been
invisible. The service now checks the environment secret **every cycle**
and connects it through the gateway's validate-and-store path; the key
value is never printed or logged. Verified with a throwaway fake key: the
connection attempt is detected and fails gracefully in this sandbox
(NVIDIA hosts unreachable here), the fake key is stored nowhere, and the
service continues in baseline mode. On a host with NVIDIA reachability,
adding the secret named exactly **`NVIDIA_API_KEY`** is the only step
needed — the already-running 24/7 service connects on its next cycle.
