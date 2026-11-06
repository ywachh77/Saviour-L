# EUR/USD: market, drivers, risks, and what our evidence actually says

**Research snapshot:** 6 November 2026. This is a research briefing, not a forecast or investment recommendation. FX is too large, adaptive, and partly opaque for any report to claim it understands every participant, flow, or future regime. This document separates established market structure, current dated facts, hypotheses about price drivers, and what this repository has actually tested.

## Executive summary

EUR/USD is the number of US dollars per euro. A rise means the euro appreciates against the dollar (or the dollar depreciates against the euro); a fall means the reverse. It is the most actively traded currency pair in the BIS 2025 survey. The FX market is nevertheless OTC and fragmented, not a single centralized order book with one authoritative spot price. Quoted prices, spreads, financing, execution and even the exact instrument differ by venue and provider.

The durable framework is **relative**, not one-sided: the pair reprices the expected path of ECB policy relative to Federal Reserve policy, relative growth/inflation and risk, with energy and geopolitical shocks especially relevant to Europe. Those are explanatory inputs, not standalone buy/sell signals. Surprise relative to expectations generally matters more at a release than whether a number is simply “good” or “bad.”

The repository’s 2021 tick-derived M1 research tested five simple, predeclared directions at three high-win-rate barrier geometries, on chronological held-out data. **No rule met the locked genuine-95% criterion.** Some configurations printed win rates above 95%, but failed uncertainty, sample-size, or net-expectancy requirements; one random control also appeared profitable in a small sample. This is not evidence that a profitable 95% system exists. No trading edge is established by this report.

## 1. Instrument and quote mechanics

- **Pair:** EUR/USD. EUR is the base currency; USD is the quote/terms currency. A quote of 1.10 means one euro costs 1.10 US dollars.
- **Direction:** long EUR/USD benefits from a rise; short EUR/USD benefits from a fall. A long is economically long EUR and short USD. The inverse is true for a short.
- **Pip convention:** usually 0.0001 USD per EUR; fractional pipettes are common. A 1-pip move at 100,000 EUR notional is approximately $10 before financing, conversion, and broker-specific contract details. At 10,000 EUR it is approximately $1/pip. This is linear arithmetic, not a statement about a recommended position size.
- **P&L:** for a EUR-notional position, approximate USD P&L is `EUR notional × change in EUR/USD`. Convert to account currency if needed. Actual P&L also depends on contract conventions, spread, commission, slippage, swaps/rollover, and execution.
- **Bid/ask:** a buy normally executes at the ask and a sell at the bid. “Mid-price” candles omit the executable spread. A backtest that enters at mid and forgets both sides’ costs overstates performance. Costs are state-dependent: spreads can widen at roll, holidays, illiquid periods, and news.
- **Spot versus derivatives:** “EUR/USD” can mean deliverable spot, rolling spot/CFD, futures, forwards, or options. Their settlement, funding, margin, price formation, and protections differ. Do not transfer a result from one product to another without modeling those differences.
- **Carry/roll:** holding a position can earn or pay financing. The rough economic driver is the interest-rate differential, but retail rollover is set by the provider and may include markups, day-count/settlement conventions, and triple-roll days. Interest-rate parity is a useful pricing relation, not a reliable short-horizon spot prediction.

## 2. Market structure and liquidity

The BIS 2025 Triennial Survey reported average global OTC FX turnover of about **$9.6 trillion per day** in April 2025; USD was on one side of **89.2%** of trades and EUR on one side of **28.9%**. These are currency-side shares and overlap, not percentages that sum to 100. The BIS describes the market as decentralized and fragmented: trading is split across dealer relationships, single- and multi-dealer platforms, electronic venues and internalized dealer flow. The survey is a comprehensive snapshot, not a live consolidated tape. [BIS 2025 turnover](https://www.bis.org/statistics/rpfx25_fx.htm) [BIS 2025 execution landscape](https://www.bis.org/publ/qtrpdf/r_qt2512v.htm)

Practical consequences:

1. There is no single public consolidated EUR/USD order book or universal “true” tick. Data vendors can differ in timestamps, quote filtering, bid/ask, and candle construction.
2. High average liquidity does not guarantee a tight spread or fill at a chosen price at every instant. Volatility shocks can change both spread and available depth.
3. Retail broker prices and execution are not identical to institutional interbank execution. A price-direction result without broker-specific costs is incomplete.
4. A 24-hour weekday market is not uniformly liquid. The broad hand-off from Asia to Europe and the London/New York overlap are active periods, but session labels are only a prior; they do not by themselves create an edge. Daylight-saving changes mean UTC offsets shift seasonally.
5. Sunday opening, Friday close, holidays, month/quarter-end flows, and scheduled data can create gaps or unstable execution. “Stop at price X” is not necessarily a guaranteed fill at X.

The ECB’s euro reference rate is a **once-daily informational benchmark**, not a dealing price; the ECB explicitly discourages transaction use. It targets market conditions around 14:15 CET, is generally published around 16:00 CET, and is not a live rate. [ECB reference-rate methodology](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html)

## 3. What moves EUR/USD

Think in terms of repricing of the *relative* outlook and of positioning/liquidity. A narrative can be directionally plausible but still fail as a trade if it was already priced in, arrives at the wrong horizon, or costs too much to express.

### Monetary policy and yields

ECB and Fed policy rates, expected future paths, real yields, and the short-end yield differential influence relative returns and hedging. The exchange rate reacts to the change in expected path, not just the current policy-rate level. Forward guidance, votes, projections, inflation persistence, labour data, and central-bank balance-sheet policy all matter. Policy divergence can support one currency only conditionally; risk premia and growth can overwhelm it.

**Dated snapshot:** the ECB’s 29 October 2026 decision kept its deposit facility rate at 2.50% (main refinancing 2.65%, marginal lending 2.90%) and emphasized data dependence and no pre-commitment to a path. [ECB monetary policy decisions archive](https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html) (decision dated 29 Oct 2026) The Fed’s 28 October 2026 FOMC statement maintained a 3.75%–4.00% target range; it described activity as strong and inflation as elevated, with two dissenters preferring a 25bp hike. [Fed 28 Oct 2026](https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a.htm)

The mechanical gap between the Fed target range and ECB deposit rate is currently around 1.25–1.50 percentage points, but that is **not** a EUR/USD forecast or a retail carry quote. The markets price the expected path, and the ECB deposit rate and Fed target range are not perfectly like-for-like instruments.

### Inflation, employment, and growth

US CPI/PCE inflation, payrolls, wages, unemployment, retail sales, GDP and surveys affect expectations for the Fed’s reaction function. Euro-area HICP, wages, PMIs, GDP, credit conditions, and national fiscal developments inform the ECB outlook. Compare surprises against consensus and revisions—not just the headline. Components and revisions often change the policy interpretation.

### Risk, energy, and geopolitics

Broad USD demand can rise during global stress because of liquidity, funding, and safe-haven dynamics, although it is not an invariant rule. Euro-area sensitivity to imported energy and terms-of-trade shocks can matter. War, sanctions, tariffs, fiscal risk, banking stress, and political fragmentation can alter both growth and risk premia rapidly. These are scenario risks, not stable directional signals.

### Positioning, flows, and technical levels

Hedging, real-money rebalancing, corporate conversion, options hedging, leveraged positioning and stop-triggered flows can affect short-horizon price action. Public positioning data (where available) are delayed and partial. Support/resistance, trend, momentum, volatility and mean reversion are descriptions or testable hypotheses—not causal guarantees. A level’s relevance can vanish after regime changes or data-mining.

### Scheduled catalysts

ECB/Fed decisions and press conferences; US employment and inflation releases; euro-area inflation and activity data; US Treasury/Fed supply and fiscal developments; and unexpected geopolitical headlines can dominate intraday movement. For research, store the release calendar and actual publication timestamps, use only information available at decision time, and model slippage/spread expansion. Do not enter a backtest using revised values as if they were known contemporaneously.

## 4. Volatility and horizon are different questions

A wider movement range creates more opportunity and more risk; it does not identify direction. Realized volatility is time-varying and clusters. A strategy may appear profitable in one volatility or news regime and fail in another. A sensible evaluation reports results by predeclared regime, while preserving a final untouched time period; it must not repeatedly tune to that period.

At very short horizons, spread, latency, quote quality and fill assumptions can dominate a small signal. At longer horizons, carry, macro path, gap risk and regime shifts matter more. A model’s historical win rate is not portable across brokers, spreads, leverage, timeframes, or market regimes without new validation.

## 5. What this repository has measured (not a market-wide truth)

The research record in [`SCALPING_RESEARCH.md`](SCALPING_RESEARCH.md) and [`eurusd-proxy-bot/docs/SCALPING_RESEARCH.md`](../eurusd-proxy-bot/docs/SCALPING_RESEARCH.md) contains the detailed protocols and tables. Highlights:

- **2021 tick-derived data:** 179,843 M1 bars in H1 and 184,969 in H2. Mean observed bid/ask spread was about 0.35 pips; the locked hunt also charged 1.2 pips and checked 0.4 pips. Historical spread averages do not represent all current retail execution.
- **Genuine-95 hunt:** coin-flip control, EMA 9/21, five-bar momentum, RSI mean-reversion, and EMA+momentum; 2/38, 5/95 and 10/190 pip target/stop geometries; chronological held-out and H2 replication. No candidate passed all gates: ≥95% win rate, Wilson lower bound above break-even, positive net expectancy, and n≥100.
- **Important false-positive lesson:** at 10/190, the small H1/H2 samples included apparently attractive positive cells that flipped sign across halves, and the coin-flip control could also look positive. That is exactly why a point estimate is not an edge.
- **Break-even math:** with target `T`, stop `S`, and total per-trade cost `c` measured in pips, ignoring partial fills and non-binary exits, break-even win probability is `(S+c)/(T+S)`. For 5/95 and c=1.2, this is 96.20%; a 95% win rate necessarily loses in that simplified payoff model.
- **Other barriers:** the repository’s 3/17 geometry produced 87.6–90.4% wins in measured cells but negative net expectancy (−0.12 to −0.68 pips/trade) against a 91.0% break-even. Raising win rate by making the stop much wider is not by itself an improvement.

These are outcomes of the documented sample and simulator, not proof about every EUR/USD strategy. The 2021 period is one historical regime, M1 bars hide within-minute paths, stop-first handling is conservative when both barriers are touched, and historical spreads/slippage may differ from live fills. Future research should independently reproduce the source data and every conversion before relying on these results.

## 6. A defensible research protocol for this pair

1. **Specify the object:** spot-like product or future; timeframe; broker/feed; timezone; executable bid/ask; entry/exit and financing rules.
2. **Lock the hypothesis before inspecting the test set:** features, thresholds, direction, holding/exit rule, costs, exclusions, sample size and pass/fail criteria.
3. **Use point-in-time data:** no look-ahead, revised macro values, future candle highs/lows at entry, or overlapping forward observations treated as independent.
4. **Use chronological train/validation/test splits** and keep a final holdout untouched. If multiple rules are tried, correct for selection/multiple testing or reserve a fresh replication period.
5. **Model realistic frictions:** bid/ask side, commissions, financing, slippage, rejected/partial fills, gaps, spread widening and rollovers. Stress costs; do not select a spread because it makes the strategy pass.
6. **Report more than accuracy:** number of trades, wins/losses, win rate and Wilson interval, break-even win rate beside it, net expectancy, profit factor, worst trade, drawdown, exposure, and results by year/regime. Show the benchmark and losing cells too.
7. **Paper test before any deployment.** This repository’s charter is research and paper settlement only. No backtest justifies a guaranteed accuracy or a broker-ordering claim.

### Current research conclusion

EUR/USD is unusually liquid by global standards, but liquidity is not predictability. Macro differentials explain many large moves; they do not make short-term direction certain. On the evidence actually tested here, **there is no established profitable 95% system, and no measured live forecasting accuracy to promise**. The useful next step is not to claim full understanding; it is to acquire diverse, timestamped bid/ask data across more years and regimes, specify a single causal hypothesis, and try to falsify it on untouched data.

## 7. One candidate pattern to research (not yet validated)

**Name:** EUR/USD M1 trend-pullback reclaim. This is a single, falsifiable research hypothesis—not a live signal, not a claim of profitability, and not a promise of 17/20 or zero consecutive losses. It combines trend direction with a pullback entry rather than chasing an extended candle. Because the existing tests already studied simple EMA/momentum/RSI rules, this more specific entry structure must be treated as a new post-hoc hypothesis and evaluated only on a fresh untouched dataset or period.

### Locked candidate specification

- **Bars/features:** completed EUR/USD M1 candles only; EMA(20), EMA(50), ATR(14), RSI(14), all calculated causally from data available at the signal close. No intra-bar signal look-ahead.
- **Long setup:** EMA20 > EMA50; EMA20 is higher than it was 5 completed bars ago; signal candle low reaches or passes EMA20; it closes above EMA20 and closes above its open; RSI14 is 45–65.
- **Short setup:** exact mirror: EMA20 < EMA50; EMA20 is lower than 5 bars ago; signal candle high reaches or passes EMA20; it closes below EMA20 and below its open; RSI14 is 35–55.
- **Cost/quality veto:** do not signal when ATR14 < 2.4 pips (at the charter’s 1.2-pip cost assumption, this avoids the quietest conditions but does not guarantee a positive edge), or when the observed executable spread exceeds the project’s configured maximum. If actual spread/commission/slippage is unavailable, record the result as cost-incomplete, not a pass.
- **Entry:** next M1 bar open, with execution at the appropriate bid/ask. If the next bar is missing or the signal is stale, skip it. Never enter on the same candle that created the signal.
- **Exit:** initial stop distance = 1.0 × ATR14 measured at signal close; target distance = 1.5 × that ATR. No trailing-stop optimization in this first test. If stop and target are both touched in one M1 bar, resolve stop-first. Exit at the first barrier; model spread, commissions, slippage and financing explicitly.
- **Trade discipline:** one position at a time; no scaling, martingale, averaging down, adding to losers, or same-candle re-entry. After any exit, require a new, independently completed setup. No forced “make back” trade.

### What counts as a result

This pattern is **not implemented as a promoted strategy** until an independent chronological test passes the project’s locked gates: at least 85% overall win rate in both train and untouched test (the 17/20 floor, with Wilson interval reported), adequate sample (at least 400 trades per split for the project gate), positive net expectancy after realistic costs, and complete reporting of break-even rate, worst trade and maximum drawdown beside win rate. In addition, test the rate of adjacent losses, but do not make “no consecutive losses” a pass condition that can be gamed by suppressing or relabeling trades. A finite sample with no adjacent losses is an observation, never a guarantee.

**Implementation status:** the hypothesis is implemented as the isolated, paper-only function `eurusd-proxy-bot/fusion/research_pattern.py::evaluate_trend_pullback`, with six unit tests. It is intentionally not registered in or wired to the live strategy council. Tests verify mechanics (including long/short symmetry, abstention gates, and cost-adjusted break-even calculation); they do **not** verify predictive value.

**Performance status: tested and failed the gates.** The first fixed-rule evaluation on the 2022 tick-derived M1 data is reported below. It does not establish that all possible EUR/USD patterns fail; it does establish that this exact candidate should not be promoted or tuned on that holdout. Do not use it for real-money decisions.

### Additional 19/20 research cycle: 2020 independent historical replication

To continue the research without tuning the failed trend-pullback pattern on its holdout, I ran the existing fixed, predeclared genuine-19/20 test on the separate public `EURUSD-2020` tick-data branch. The source contains **373,474 M1 bars**, from 2020-01-01 to 2020-12-31; the script split chronologically at 2020-07-01 19:50 UTC. It tested the same five declared direction rules (coin-flip control, EMA 9/21, five-bar momentum, RSI mean-reversion, and EMA+momentum) at 2/38, 5/95 and 10/190 pip target/stop geometries. No parameter was fitted to the held-out half.

**Result: 0 of 15 cells passed** the full locked test (held-out win rate ≥95%, Wilson lower bound above break-even, positive net pips/trade, n≥100). The closest cells illustrate why all gates matter:

| Held-out rule / geometry | Cost | n | Win (Wilson lower) | Break-even | Net/trade | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Mean-reversion, 10/190 | 1.2p | 181 | 97.24% (93.70%) | 95.60% | +3.275p | Fails Wilson bound |
| Mean-reversion, 10/190 | 0.4p | 181 | 97.24% (93.70%) | 95.20% | +4.075p | Fails Wilson bound |
| Momentum, 5/95 | 0.4p | 672 | 95.39% (93.53%) | 95.40% | −0.013p | Fails Wilson and expectancy |
| Coin-flip control, 10/190 | 0.4p | 151 | 95.36% (90.74%) | 95.20% | +0.328p | Control also looks positive; fails Wilson |

At the largest geometry, the RSI rule’s positive point estimate still cannot establish that its true hit rate clears break-even. The coin-flip control’s positive point estimate at 0.4p shows why small positive samples are not a deployable edge. The complete terminal output can be reproduced with `python3 scripts/genuine_19_hunt.py --csv /path/to/eurusd2020_m1.csv --spread 1.2` and `--spread 0.4`.

### First evaluation of the candidate pattern

A fixed-rule evaluation was run on the public `FX-Data/FX-Data-EURUSD-DS` branch `EURUSD-2022`, using bid/ask ticks aggregated into mid-price M1 bars. The available branch data covered **2022-01-02 through 2022-07-08** (189,259 M1 bars), not the full calendar year. The exact rule and thresholds above were held fixed; there was no grid search or threshold selection. The first chronological half was reported separately from the second-half holdout. Execution assumptions: entry at next bar open; 1 ATR stop/1.5 ATR target; stop-first on an ambiguous bar; maximum 120-minute hold then close; one trade at a time; no same-candle re-entry. The measured signal-bar spread was charged once per trade in the observed-cost run; a separate run charged a fixed 1.2 pips. This is a bar-level proxy, not a broker fill simulation.

| Cost case | Split | Trades | Win rate (Wilson 95%) | Mean barrier break-even* | Net pips/trade | Worst trade | Max drawdown | Longest net-loss run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Observed spread (mean 0.454p) | First half | 2,828 | 39.14% (37.36–40.96%) | 48.22% | −0.334 | −10.85p | 950.8p | 18 |
| Observed spread (mean 0.454p) | Second-half holdout | 2,826 | 39.17% (37.39–40.99%) | 48.17% | −0.416 | −13.22p | 1,207.4p | 13 |
| Fixed 1.2p spread | First half | 813 | 39.61% (36.30–43.01%) | 55.45% | −1.141 | −11.72p | 927.9p | 10 |
| Fixed 1.2p spread | Second-half holdout | 1,047 | 38.49% (35.59–41.48%) | 55.30% | −1.271 | −14.01p | 1,341.1p | 9 |

\*The mean break-even figure is the per-trade binary target/stop break-even `(stop distance + cost)/(target distance + stop distance)`, averaged over trades. It is a reference rather than a complete expectancy estimator because some trades time out; net P&L includes those timeout exits and is the primary result. Win rates count net-positive trades.

**Verdict: reject this candidate as a trading edge on this test.** Both halves were well below the project’s 85% minimum floor, the Wilson upper bounds were below even the simplified break-even figures, and net expectancy was negative with either cost assumption. The longest observed consecutive net-loss runs were 18 and 13 in the two halves—so the pattern does not meet a no-consecutive-loss goal either. This is an empirical failure, not a reason to optimize thresholds on the same data.

A cross-period robustness check on the separate 2020 ticks also failed: second-half observed-spread results were 38.93% wins (Wilson 37.66–40.21%), 48.64% mean barrier break-even, −0.375 pips/trade, and a 12-loss run; at fixed 1.2p cost they were 37.18% (34.48–39.95%), 55.98% break-even, −1.394 pips/trade, and an 18-loss run. This is a historical robustness check, not a forward holdout, because the candidate was formed later. Keep the pattern isolated as a failed research candidate; do not promote it or tune it on these samples. Reproduce with:

```bash
cd eurusd-proxy-bot
python3 scripts/evaluate_research_pattern.py --csv /path/to/eurusd2022_m1.csv
python3 scripts/evaluate_research_pattern.py --csv /path/to/eurusd2022_m1.csv --fixed-cost 1.2
```

### Checking an existing scalper instead of trusting its comment

The checked-in `scalp_expansion` rule claims +0.378 net pips/trade in training and +0.382 out of sample on 71,939 real M1 bars. The repository’s scalping research note also reports negative performance for expansion under a different synthetic experiment. Rather than treating either statement as proof, I independently evaluated the exact checked-in callable on 2020 tick-derived EUR/USD bars (373,474 M1 bars), split chronologically, with next-bar-open entry and 1/2/3/5-minute close exits. It charges one spread per trade. The limited features used were checked against `compute_features`: **zero numerical difference** for all eight strategy inputs on 90 verification bars. This evaluation is a historical replication, not a live-fill test.

| Horizon / cost | Split | Trades | Net win rate (Wilson 95%) | Break-even | Net pips/trade | Worst trade | Max DD | Longest loss run |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3 min / observed mean spread | First half | 6,225 | 40.92% (39.70–42.14%) | 56.70% | −0.412 | −32.45p | 2,591.3p | 12 |
| 3 min / observed mean spread | Second-half holdout | 6,860 | 40.26% (39.11–41.43%) | 57.25% | −0.390 | −30.53p | 2,681.8p | 15 |
| 3 min / fixed 1.2p | First half | 2,069 | 34.27% (32.25–36.34%) | 62.54% | −1.341 | −33.25p | 2,782.6p | 18 |
| 3 min / fixed 1.2p | Second-half holdout | 1,467 | 34.63% (32.24–37.10%) | 67.81% | −1.260 | −31.35p | 1,850.8p | 12 |

The same setup had negative net expectancy at **all four tested holds** (1, 2, 3, 5 minutes) in both halves under both cost assumptions. Its measured function does what its name says—waits for a close beyond a Bollinger band and the previous 20-bar extreme, aligned with EMA direction—but on this real-data sample the breakout condition did not offset reversal/false-break and transaction costs. This supports **not promoting or treating this scalper as an independent edge**. The earlier +0.38 claim is not reproduced on this independent period; its original data, exact hold, and execution assumptions need provenance before that number can be relied on.

Reproduce with `python3 scripts/verify_scalp_expansion_real.py --csv /path/to/eurusd2020_m1.csv`. The script reports all horizons, both chronological halves, costs, Wilson interval, break-even, expectancy, worst trade, drawdown, and loss run.

## Locked target requested: 98.15% (research criterion, not an achievement)

The 98.15% figure is the highest observed point estimate so far (53/54 wins in one RSI 10/190 cell), not a validated system accuracy. It is now treated as a **research challenge threshold only**: any candidate considered for promotion must achieve at least 98.15% net-positive trades in both chronological development and untouched confirmation samples, with at least 400 trades per split, the 95% Wilson lower bound above the per-trade break-even rate, positive net expectancy after costs, and worst trade/max drawdown reported. A failure on any gate means no promotion. A point estimate alone never qualifies.

**No current strategy is upgraded or advertised at 98.15%.** The highest cell’s 54-trade Wilson lower bound (90.23%) is far below its 95.60% break-even, and tested current candidates have negative net results. I have not changed the runtime `TARGET_WIN_RATE` setting: feeding an unsupported target into loss-focused retraining could encourage overfit and is not evidence-based. The existing strategies remain research/paper candidates only until a fresh, properly powered confirmation passes.

## Seven net wins in one hour, zero losses: historical paper test

I tested the literal goal on the same 2020 tick-derived EUR/USD M1 sample (373,474 bars), separated into two chronological halves. “One hour” means a non-overlapping UTC clock-hour with observed quotes. A qualifying hour must have **at least seven trades entered and settled inside that hour, with every trade entered in that hour net-positive**. I also counted non-overlapping runs of seven consecutive net-winning trades whose first entry to seventh exit lasted no more than 60 minutes. The two tested candidates were the trend-pullback rule (barrier exits, 120-minute timeout) and the checked-in `scalp_expansion` at a fixed 3-minute exit. One position at a time; next-bar-open entry; spread charged once; no martingale or same-candle re-entry.

| Candidate / cost | Split | Trades | Net win rate (Wilson 95%) | Break-even | Net/trade | Worst loss | Max DD | Loss run | Hourly qualifying blocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trend-pullback / observed (0.402p) | First | 4,956 | 41.12% (39.76–42.50%) | 47.48% | −0.362p | −25.58p | 1,818.6p | 18 | 0/3,119 |
| Trend-pullback / observed (0.377p) | Second | 5,625 | 38.74% (37.47–40.02%) | 48.40% | −0.444p | −9.61p | 2,519.3p | 14 | 0/3,132 |
| Trend-pullback / 1.2p | First | 1,678 | 39.87% (37.55–42.23%) | 52.84% | −1.249p | −25.61p | 2,112.5p | 18 | 0/3,119 |
| Trend-pullback / 1.2p | Second | 1,197 | 37.18% (34.48–39.95%) | 54.71% | −1.394p | −10.34p | 1,691.1p | 18 | 0/3,132 |
| Expansion 3m / observed (0.402p) | First | 6,225 | 40.92% (39.70–42.14%) | 56.70% | −0.412p | −32.45p | 2,591.3p | 12 | 0/3,119 |
| Expansion 3m / observed (0.377p) | Second | 6,859 | 40.27% (39.11–41.43%) | 57.25% | −0.390p | −30.53p | 2,680.0p | 15 | 0/3,132 |
| Expansion 3m / 1.2p | First | 2,069 | 34.27% (32.25–36.34%) | 62.54% | −1.341p | −33.25p | 2,782.6p | 18 | 0/3,119 |
| Expansion 3m / 1.2p | Second | 1,467 | 34.63% (32.24–37.10%) | 67.81% | −1.260p | −31.35p | 1,850.8p | 12 | 0/3,132 |

**Observed result:** zero qualifying UTC hours in 6,251 observed hours, and zero seven-consecutive-win sequences completed within 60 minutes, for both candidates and both cost cases. Hours with at least seven trade entries that settled within the hour were rare (0–4 per half, depending on candidate/cost), and none had every trade win. This does **not** prove that no other strategy could ever do it; it says these two tested candidates did not, and both had negative expectancy. The study is historical, one data source/year, and midpoint OHLC with simplified spread costs—not a guarantee about future odds or executable broker fills.

Reproduce with `python3 scripts/research_seven_wins_hour.py --csv /path/to/eurusd2020_m1.csv`.

## Sources

- BIS, [OTC foreign exchange turnover in April 2025](https://www.bis.org/statistics/rpfx25_fx.htm).
- BIS, [The FX trade execution landscape through the prism of the 2025 BIS Triennial Survey](https://www.bis.org/publ/qtrpdf/r_qt2512v.htm).
- ECB, [Euro foreign exchange reference rates and methodology](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html).
- ECB, [Monetary policy statement, 29 October 2026](https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/html/index.en.html).
- Federal Reserve, [FOMC statement, 28 October 2026](https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a.htm).
- CFTC, [Eight Things You Should Know Before Trading Forex](https://www.cftc.gov/PressRoom/PressReleases/8566-22) — OTC counterparty, leverage, dealer, cost and fraud risks.
