# The prompt that produces 19 wins in 20 — and what it costs

## Short answer

The win rate is not a capability of the strategy. **It is a dial**, and the dial
is the reward-to-risk ratio. On a driftless random walk, with a 1.2 pip spread:

| take profit | stop loss | measured win rate | net per trade |
|---|---|---|---|
| 10.0 pips | 10.0 pips | 49.7% | −1.26 pips |
| 5.0 | 20.0 | 78.7% | −1.52 |
| 2.0 | 38.0 | 93.7% | −1.72 |
| 1.25 | 47.5 | **96.2%** | −1.82 |
| **1.22** | **116.0** | **98.5%** | **−1.69** |

The win-rate column moves from 50% to 98.5%. **The net column does not move at
all** — it sits at minus the spread, in every row, because that is what a
driftless walk pays after a round turn. Setting the dial to 19/20 is
arithmetic, not research.

## What 19/20 actually demands

Same barrier geometry, this time giving the entries genuine forecast edge, and
bisecting on the edge needed to break even:

| setup | edge per trade needed to break even |
|---|---|
| plain 1:1 (10 pip / 10 pip) | **1.15 pips** |
| 19/20 (1.22 pip / 116 pip) | **2.84 pips** |

A 19-in-20 system needs **2.5× more forecast skill** to break even than an
ordinary 1:1 setup — and it pays for the 98.5% win rate with a 116-pip tail.
A 19/20 system and a 1:1 system with the *same* edge are not equally good; the
1:1 one is far better, because the 19/20 one spends its edge buying a number.

---

## The prompt

Attached warning is part of the prompt on purpose: if it travels without it,
someone will read the 98% and believe it.

```
Generate a EUR/USD M1 strategy whose backtest shows at least 19 wins in 20
consecutive trades, and prove it on a held-out split.

KNOWN OUTCOME, STATE IT FIRST IN YOUR ANSWER:
this will report a win rate of 98-99% and a NEGATIVE net expectancy per trade,
because that is what this geometry does. Do not present the win rate as an
achievement without the expectancy beside it.

Rules:
1. Take profit 1.2-1.3 pips, stop loss 100-120 pips. This alone produces the
   win rate; do not add filters that pretend otherwise.
2. Direction may be any rule, including a coin flip. State plainly that the
   direction rule does not matter at this geometry -- the win rate comes from
   the barrier distance, not the signal.
3. Charge the spread on every trade. Report net pips, never gross.
4. Split the data chronologically 50/50. Tune only on the first half. Report
   the 19/20 block found on the first half AND the block count on the second.
5. Report, for both halves: total trades, net win rate, net pips per trade,
   worst single-trade loss, longest losing run, and maximum drawdown in pips.
6. Enforce these, and fail loudly if violated:
   - no martingale or any sizing that increases after a loss
   - no averaging down, no adding to a losing position
   - no re-entry within the same candle
7. Print this line at the end, filled in with your numbers:
   "Win rate: X%. Net per trade: Y pips. To break even at this payoff the win
    rate must be Z%. Required forecast edge: W pips per trade."
```

## What it will produce, and how to check it

You will get a 19/20 block, and probably several — at 98.5% a 20-trade window
misses one trade about once in 40 windows. That is not the strategy working; it
is the geometry working, which is why the prompt forbids claiming otherwise.

Check these three numbers before believing anything:

1. **Net pips per trade** — it will be negative, near −1.7. That is the spread.
2. **Worst single loss** — about −117 pips for every +1.2 you win. The equity
   curve is a slow staircase up followed by a cliff.
3. **Break-even win rate required** — 98.5% is what you have and what you need;
   there is no margin, so one extra loss in twenty flips the whole record red.

## The prompt worth more

If the goal behind 19/20 is a record that holds up, ask for this instead:

```
Build a EUR/USD M1 strategy and report, on a held-out chronological split:
net pips per trade after a 1.2 pip spread, the Wilson 95% interval on that
expectancy, the number of non-overlapping trades, and the break-even win rate
implied by the realised average move. Reject any variant whose interval
includes zero. Do not report a win rate without the required rate next to it.
```

That prompt cannot be satisfied by widening a stop. It can only be satisfied by
an edge, which is the only thing that survives contact with a live account.

See `docs/SCALPING_RESEARCH.md` §7 for why hold length and spread — not filters —
are the levers that move the required rate, and §11 for the measured attempt on
the uploaded strategy, where one 19/20 block did appear in 982 windows and
reverted to 17/20 out of sample.
