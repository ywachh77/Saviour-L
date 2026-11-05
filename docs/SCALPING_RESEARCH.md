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
