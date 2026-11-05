
---

## 10. Where signals win, where they fail: RSI, ADX, and the rest

`scripts/condition_edge_study.py` takes one bar at a time, records the market's
state (RSI, ADX, ATR, EMA slope, where the bar closed in its range, session,
volatility ratio), and then asks what price actually did over the next hold. It
reports, per condition bucket: **n, how often price rose, how far that is from
the sample's own base rate, net pips after spread, and a p-value**.

Two disciplines make the numbers mean something:

* **the null is the base rate, not 50%.** If price rises 47.6% of the time in
  this sample, a "52.5% winner" has earned 4.9 points, not 2.5.
* **every bucket is corrected for being one of many.** Benjamini-Hochberg at 5%
  across all buckets, printed at the bottom of every run.

### The control run matters more than the result

On a **zero-drift random walk** the corrected list is **empty** — no condition is
distinguishable from the base rate. That is the test that the method is sound.

It did not pass at first. An earlier version sampled every bar, so consecutive
15-minute windows shared 14 of their 15 minutes; n was ~15× too high and the
random walk produced **five confident "findings"** (RSI < 30 at 56.8%, ADX < 20 at
54.1%, and three more). Nothing was wrong with the market — the p-values were
lying because the samples were not independent. The study now samples **every
`hold` bars** and the control is clean. Any p-value in this project that came from
overlapping windows should be treated as void.

### The real data, and why it cannot answer the question yet

The repository ships **740 real EUR/USD M1 bars** (30 Oct 2026, 04:11–16:30Z,
mean bar range 1.91 pips). At a 15-minute hold that is **42 non-overlapping
windows**. The corrected list is empty, and it must be.

A raw 15-minute move on real data averages **4.48 pips** against a 1.2 pip
spread, so break-even needs **63.4%** — and no bucket in 42 samples can establish
63%. The honest statement is: *on the data in this repository, no conditioning
variable is demonstrably profitable.*

### What the over-fitted version said — and why to distrust it

For completeness, because you will see tables like this everywhere. Sampling every
bar (635 overlapping rows) produced:

| condition | bucket | up% | vs base | net/trade | naive p |
|---|---|---|---|---|---|
| **ADX** | ADX < 20 | 71.8% | +24.2% | **+1.08 pips** | 0.000 |
| **ADX** | ADX 25-35 | 30.4% | −17.1% | −3.04 pips | 0.000 |
| **RSI** | RSI < 30 | 29.5% | −18.0% | −4.04 pips | 0.017 |
| ATR | ATR high | 55.9% | +8.4% | +0.14 pips | 0.015 |
| Session | ASIA 00–07 | 66.7% | +19.1% | +0.21 pips | 0.000 |
| Session | LONDON 08–12 | 40.0% | −7.6% | −2.40 pips | 0.009 |

Three of those survived the correction — and they are still not evidence, because
the windows overlapped. Notice also that the two ADX findings **contradict each
other**: ADX below 20 is strongly bullish and ADX 25–35 strongly bearish, in a
single 12-hour session that happened to trend one way. That is one day being
described six ways, not six effects.

### What can be said without overreaching

* **Nothing here supports "RSI below 30 buys the dip."** The direction on this
  sample is the opposite (−18 points), with 44 observations.
* **ADX as a directional filter is not supported.** Its apparent power is the
  sample's trend leaking into a regime variable.
* **The volatility result is the one worth watching**, because it agrees with
  independent arithmetic: higher ATR widens the payoff, and §7 shows the required
  hit rate falls as the payoff grows. ATR-high was the only bucket with a
  non-negative net (+0.14 pips). It is one bucket, uncorrected, on 211 overlapping
  rows.
* **Every row in every table is negative or barely positive after spread except
  those.** The spread is the single largest term in this analysis.

### What it would take to answer this properly

Not a better filter — more days. At the measured effect sizes, separating a 63%
break-even rate from a 55% bucket at 80% power needs roughly **500 non-overlapping
15-minute windows ≈ 125 hours of M1 data ≈ 5 trading days**, and the repository
holds 12 hours. Ingest a month of real candles and re-run:

```
python3 scripts/condition_edge_study.py --real
```

The script is already corrected for multiplicity, so the day the list stops being
empty is the day there is something to talk about.
