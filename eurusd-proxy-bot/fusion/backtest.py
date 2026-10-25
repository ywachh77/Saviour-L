"""Purged walk-forward backtest of the system the bot actually runs.

Why this exists: judging the bot on the *model's* numbers is misleading. The
model scores every bar; the bot trades only the bars that survive the data gate,
the regime filter, the six-check council and the meta-model veto. Those are
different populations, and only the second one pays the spread.

So this replays the real thing:

* features are computed once over the whole series (no per-bar recomputation, so
  a month of M1 bars stays linear);
* folds are **purged** — a training sample whose label reaches into the test
  window is dropped, because a one-minute label looks `horizon` bars ahead;
* each fold trains on the past only, then drives the live decision path
  (`FusionEngine.evaluate_prepared`) bar by bar, so the backtest cannot drift
  from production;
* every replayed signal pays the spread and is settled from real candle opens
  and closes.

Two honesty notes that travel with every report:

1. Fewer than ~200 replayed signals means the confidence interval is wide enough
   that the number is entertainment, not evidence. The report says so.
2. This measures one path through one random-looking history. It is not a
   Monte Carlo and it is not a promise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Sequence

from . import charter, costs, datagate, instruments
from .config import Config
from .council import CouncilWeights, spoken_votes
from .domain import Candle
from .engine import FusionEngine
from .features import compute_features
from .indicators import atr, mean
from .model import OnlineLogisticModel
from .session import session_of
from .mistakes import MistakeMemory
from .regime import classify
from .stats import EdgeVerdict, MultiPathSummary, edge_verdict, summarise_paths

MIN_TRAIN_PER_FOLD = 150
MIN_BARS_FOR_BACKTEST = 600
MIN_SIGNALS_FOR_A_VIEW = 200
MAX_FOLD_TRAIN_SAMPLES = 12_000


@dataclass
class BacktestReport:
    bars: int = 0
    folds: int = 0
    signals: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    unresolved: int = 0
    gross_pips: float = 0.0
    net_pips: float = 0.0
    win_rate: float | None = None
    wilson_low: float = 0.0
    wilson_high: float = 0.0
    # Per-trade net outcomes, kept so expectancy can be given an interval
    # instead of a bare average. A mean without a spread is a number that
    # invites a conclusion the sample cannot support.
    trade_net_pips: list[float] = field(default_factory=list)
    # Hold lengths actually used, one per signal. Visible because with adaptive
    # horizon on, the hold length is a decision the system made, and a decision
    # that never appears in the output is one nobody can check.
    chosen_horizons: list[int] = field(default_factory=list)
    # The hit rate this horizon and cost DEMAND, next to the one achieved. A
    # 56% win rate reads as skill until the arithmetic says 67% was needed.
    required_win_rate: float | None = None
    expectancy_low: float = 0.0
    expectancy_high: float = 0.0
    edge_status: str = ""
    samples_needed: int = -1
    edge: EdgeVerdict | None = None
    coverage: float = 0.0
    model_accuracy: float | None = None
    model_points: int = 0
    learned_weights: dict[str, float] = field(default_factory=dict)
    per_fold: list[dict] = field(default_factory=list)
    note: str = ""

    @property
    def enough_for_a_view(self) -> bool:
        return self.signals >= MIN_SIGNALS_FOR_A_VIEW

    @property
    def mean_horizon(self) -> float:
        """Mean hold length actually used, in minutes."""
        if not self.chosen_horizons:
            return 0.0
        return sum(self.chosen_horizons) / len(self.chosen_horizons)

    @property
    def expectancy_pips(self) -> float | None:
        """Average net pips per replayed signal."""
        decided = self.wins + self.losses
        return (self.net_pips / decided) if decided else None

    def to_dict(self) -> dict:
        return {
            "bars": self.bars,
            "folds": self.folds,
            "signals": self.signals,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "unresolved": self.unresolved,
            "gross_pips": round(self.gross_pips, 2),
            "net_pips": round(self.net_pips, 2),
            "expectancy_pips": (None if self.expectancy_pips is None else round(self.expectancy_pips, 4)),
            "win_rate": self.win_rate,
            "wilson": [round(self.wilson_low, 4), round(self.wilson_high, 4)],
            "expectancy_low": round(self.expectancy_low, 4),
            "expectancy_high": round(self.expectancy_high, 4),
            "edge_status": self.edge_status,
            "samples_needed": self.samples_needed,
            "mean_horizon": round(self.mean_horizon, 2),
            "max_horizon": (max(self.chosen_horizons) if self.chosen_horizons else 0),
            "required_win_rate": self.required_win_rate,
            "min_horizon": (min(self.chosen_horizons) if self.chosen_horizons else 0),
            "coverage": round(self.coverage, 5),
            "model_accuracy": self.model_accuracy,
            "model_points": self.model_points,
            "learned_weights": {k: round(v, 4) for k, v in self.learned_weights.items()},
            "enough_for_a_view": self.enough_for_a_view,
            "per_fold": self.per_fold,
            "note": self.note,
        }


def _purged_splits(
    indices: Sequence[int],
    folds: int,
    horizon: int,
    bar_horizons: Sequence[int | None] | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Expanding-window folds with a purge before each test window.

    A sample at bar j is labelled by the close of bar j + h_j, so training on
    any j whose label lands inside the test window is leakage. Those are dropped.
    """
    total = len(indices)
    folds = max(2, min(int(folds), 6))
    if total < folds * 60:
        return []
    bounds = [int(round(total * step / (folds + 1))) for step in range(1, folds + 2)]

    splits: list[tuple[list[int], list[int]]] = []
    for fold in range(folds):
        train_end = bounds[fold]
        test_start, test_end = bounds[fold], bounds[fold + 1]
        first_test_bar = indices[test_start]
        train = [
            j
            for j in indices[:train_end]
            if j + (bar_horizons[j] if (bar_horizons and bar_horizons[j]) else horizon) < first_test_bar
        ]
        test = indices[test_start:test_end]
        if len(train) >= MIN_TRAIN_PER_FOLD and test:
            splits.append((train, test))
    return splits


def _label(candles: Sequence[Candle], index: int, horizon: int) -> int | None:
    entry_index = index + 1
    exit_index = index + horizon
    if exit_index >= len(candles):
        return None
    if (candles[exit_index].timestamp - candles[entry_index].timestamp) > timedelta(minutes=horizon + 5):
        return None  # weekend or feed outage gap: never settle across missing hours
    move = candles[exit_index].close - candles[entry_index].open
    if math.isclose(move, 0.0, abs_tol=1e-12):
        return None  # a tie carries no direction; dropped, not guessed
    return 1 if move > 0 else 0


def run_backtest(
    candles: Sequence[Candle],
    config: Config,
    *,
    folds: int = 4,
    horizon: int | None = None,
    spread_pips: float | None = None,
    weights: CouncilWeights | None = None,
    min_bars: int = MIN_BARS_FOR_BACKTEST,
    mistakes: MistakeMemory | None = None,
) -> BacktestReport:
    """Replay the full pipeline over history and settle every signal with costs."""
    candles = sorted(candles, key=lambda candle: candle.timestamp)
    horizon = int(horizon if horizon is not None else config.horizon_minutes)
    spread = float(spread_pips if spread_pips is not None else config.spread_pips)
    report = BacktestReport(bars=len(candles))

    if len(candles) < min_bars:
        report.note = (
            f"Only {len(candles)} bars stored; {min_bars} are needed before a replay means anything."
        )
        return report

    # The pip is the one unit that is not dimensionless, so the replay has to
    # know which market it is holding. Everything else -- the cost guard, the
    # reward-to-cost test, the break-even rate -- is a ratio and transfers
    # unchanged.
    instrument = instruments.for_symbol(getattr(config, "pair", None))
    feature_rows = compute_features(candles, instrument=instrument)
    atr_values = atr(candles, 14)
    adaptive = bool(getattr(config, "adaptive_horizon", False))
    bar_horizons: list[int | None] = [horizon] * len(candles)
    if adaptive:
        r2c = float(getattr(config, "reward_to_cost", 3.0))
        max_h = int(getattr(config, "max_horizon_minutes", 120))
        for idx, feat in enumerate(feature_rows):
            if feat is None:
                bar_horizons[idx] = None
            else:
                atr_p = abs(float(feat.get("atr", 0.0))) / instrument.pip
                bar_horizons[idx] = costs.resolve_horizon(
                    atr_p, spread, horizon, adaptive=True, reward_to_cost=r2c, max_horizon=max_h
                )

    usable = [
        index
        for index in range(len(candles))
        if index > 0 and feature_rows[index] is not None and index + horizon < len(candles)
    ]
    if len(usable) < min_bars // 2:
        report.note = "Not enough bars with complete indicators to build folds."
        return report

    splits = _purged_splits(usable, folds, horizon, bar_horizons if adaptive else None)
    if not splits:
        report.note = "Not enough data to build purged walk-forward folds."
        return report

    engine = FusionEngine(config)
    weights = weights or CouncilWeights()
    model_points: list[tuple[float, int]] = []
    cooldown_mins = max(0, int(getattr(config, "signal_cooldown_minutes", 0) or 0))
    loss_streak_cap = max(0, int(getattr(config, "loss_guard_streak", 0) or 0))
    loss_lookback_m = max(1, int(getattr(config, "loss_guard_lookback_minutes", 180) or 180))
    loss_cd_m = max(1, int(getattr(config, "loss_guard_cooldown_minutes", 60) or 60))
    shadow_rec = bool(getattr(config, "loss_guard_shadow_recovery", False))

    for fold_index, (train_idx, test_idx) in enumerate(splits, start=1):
        pairs: list[tuple[dict, int]] = []
        for index in train_idx:
            train_h = bar_horizons[index] if adaptive else horizon
            if train_h is None:
                continue
            label = _label(candles, index, train_h)
            if label is not None:
                pairs.append((feature_rows[index], label))
        if len(pairs) < MIN_TRAIN_PER_FOLD:
            continue
        if len(pairs) > MAX_FOLD_TRAIN_SAMPLES:
            pairs = pairs[-MAX_FOLD_TRAIN_SAMPLES:]
        try:
            model = OnlineLogisticModel().fit(pairs, epochs=90, validation_fraction=0.0)
        except ValueError:
            continue

        fold_signals = fold_wins = fold_losses = 0
        fold_net = 0.0
        next_allowed_ts = candles[0].timestamp - timedelta(minutes=1)
        loss_pause_until_ts = candles[0].timestamp - timedelta(minutes=1)
        settled_outcomes: list[tuple] = []
        shadow_trade: tuple[datetime, bool] | None = None
        shadow_cleared_ts: datetime | None = None

        for index in test_idx:
            features = feature_rows[index]
            candle = candles[index]
            entry_stamp = candle.timestamp + timedelta(minutes=1)
            if shadow_trade is not None and shadow_trade[0] <= entry_stamp:
                sh_ts, sh_win = shadow_trade
                shadow_trade = None
                if sh_win:
                    shadow_cleared_ts = sh_ts
            recent = [value for value in atr_values[max(0, index - 29) : index + 1] if value is not None]
            regime = classify(
                atr_value=float(features["atr"]),
                atr_pct=float(features["atr_pct"]),
                atr_baseline=mean(recent) if recent else 0.0,
                adx_value=float(features["adx"]),
                close=candle.close,
                ema_fast=float(features["ema_fast"]),
                ema_slow=float(features["ema_slow"]),
            )
            # Historical bars cannot be checked for freshness, so the gate is
            # reduced to the spread test. The report says this out loud.
            gate = datagate.GateResult(
                datagate.OK, "OK", 0.0, spread, "modelled", 0, index + 1,
                "Replay: the feed is assumed to have been healthy at the time.",
            )
            result = engine.evaluate_prepared(
                candles[max(0, index - 1) : index + 1],
                features=features,
                gate=gate,
                regime_state=regime,
                model=model,
                weights=weights,
                source="BACKTEST",
                at=entry_stamp,
            )

            # With adaptive horizon the hold length is chosen per bar, so the
            # label and the exit must both follow that bar's horizon. Using one
            # fixed horizon here would train the model on a question the trade
            # is not asking.
            bar_horizon = bar_horizons[index] if adaptive else horizon
            if bar_horizon is None or index + bar_horizon >= len(candles):
                continue

            label = _label(candles, index, bar_horizon)
            if label is not None:
                try:
                    model_points.append((model.predict_up(features), label))
                except ValueError:
                    pass

            if not result.actionable or result.signal is None:
                continue
            if (candles[index + bar_horizon].timestamp - candles[index + 1].timestamp) > timedelta(
                minutes=bar_horizon + 5
            ):
                continue

            awaiting_shadow = False
            if loss_streak_cap > 0 and settled_outcomes:
                realised = [(ts, bad) for ts, bad in settled_outcomes if ts <= entry_stamp]
                lookback_span = max(loss_lookback_m, 720) if shadow_rec else loss_lookback_m
                cutoff = entry_stamp - timedelta(minutes=lookback_span)
                realised_recent = [(ts, bad) for ts, bad in realised if ts >= cutoff]
                streak = 0
                last_bad_ts = None
                for ts, bad in reversed(realised_recent):
                    if bad:
                        streak += 1
                        if last_bad_ts is None:
                            last_bad_ts = ts
                    else:
                        break
                if (
                    streak >= loss_streak_cap
                    and last_bad_ts is not None
                    and len(realised_recent) >= loss_streak_cap
                    and (last_bad_ts - realised_recent[-loss_streak_cap][0]) <= timedelta(minutes=loss_lookback_m)
                ):
                    unlock_ts = last_bad_ts + timedelta(minutes=loss_cd_m)
                    if unlock_ts > loss_pause_until_ts:
                        loss_pause_until_ts = unlock_ts
                    if (
                        shadow_rec
                        and (shadow_cleared_ts is None or shadow_cleared_ts < last_bad_ts)
                        and entry_stamp < (last_bad_ts + timedelta(minutes=720))
                    ):
                        awaiting_shadow = True

            if entry_stamp < loss_pause_until_ts or awaiting_shadow:
                if shadow_rec and shadow_trade is None:
                    sh_move = candles[index + bar_horizon].close - candles[index + 1].open
                    sh_gross = (sh_move if result.signal.side == "BUY" else -sh_move) / instrument.pip
                    sh_net = sh_gross - spread
                    shadow_trade = (candles[index + bar_horizon].timestamp, sh_net > 0.0)
                continue

            if entry_stamp < next_allowed_ts:
                continue

            entry = candles[index + 1].open
            exit_price = candles[index + bar_horizon].close
            move = exit_price - entry
            gross_pips = (move if result.signal.side == "BUY" else -move) / instrument.pip
            net_pips = gross_pips - spread

            if cooldown_mins > 0:
                next_allowed_ts = entry_stamp + timedelta(minutes=cooldown_mins)
            if loss_streak_cap > 0 and not math.isclose(gross_pips, 0.0, abs_tol=1e-12):
                settled_at = candles[index + bar_horizon].timestamp
                settled_outcomes.append((settled_at, net_pips < 0.0))

            fold_signals += 1
            fold_net += net_pips
            report.trade_net_pips.append(net_pips)
            report.chosen_horizons.append(int(bar_horizon))
            if math.isclose(gross_pips, 0.0, abs_tol=1e-12):
                report.ties += 1
            elif gross_pips > 0:
                report.wins += 1
                fold_wins += 1
            else:
                report.losses += 1
                fold_losses += 1
            report.gross_pips += gross_pips

            # Learn the council exactly the way the live runner does: every
            # settled, non-tied outcome credits or blames the prototypes that
            # spoke. Without this the replay measures a bot that never learns,
            # so no prototype can ever earn a voice in a backtest — the scalping
            # family would be invisible here while fully live in production.
            winning_side = "BUY" if move > 0 else "SELL" if move < 0 else None
            if result.council is not None and winning_side is not None:
                weights.observe(spoken_votes(result.council.votes), winning_side)

            # Post-mortem, opt-in. Passing a memory is the caller asserting the
            # replay is over real bars: synthetic data trains nothing, and this
            # is the one mechanism that can permanently ban a setup, so it must
            # never be fed a random walk.
            if mistakes is not None and result.signal is not None and not math.isclose(gross_pips, 0.0, abs_tol=1e-12):
                mistakes.observe(
                    MistakeMemory.fingerprint(
                        result.regime or "UNKNOWN",
                        session_of(candles[index].timestamp, instrument)
                        if config.session_filter
                        else None,
                        result.signal.side,
                    ),
                    gross_pips > 0,
                )

        report.folds += 1
        report.signals += fold_signals
        report.net_pips += fold_net
        report.per_fold.append(
            {
                "fold": fold_index,
                "train_bars": len(train_idx),
                "test_bars": len(test_idx),
                "signals": fold_signals,
                "wins": fold_wins,
                "losses": fold_losses,
                "net_pips": round(fold_net, 2),
            }
        )

    decided = report.wins + report.losses
    if decided:
        report.win_rate = report.wins / decided
        report.wilson_low, report.wilson_high = charter.wilson_interval(report.wins, decided)

    # Judge the expectancy, not just the win rate. A 55% win rate with a bad
    # payoff is still a losing system, and an average over one path is a single
    # draw, so the verdict is what gets reported rather than the point estimate.
    verdict = edge_verdict(report.trade_net_pips)
    report.expectancy_low = verdict.ci_low
    report.expectancy_high = verdict.ci_high
    report.edge_status = verdict.status
    report.samples_needed = verdict.samples_needed
    report.edge = verdict
    # Break-even hit rate, computed from the volatility actually present and the
    # hold length actually used. This is the number the achieved win rate has to
    # beat, and it is usually far higher than intuition expects.
    atr_pips = [value / instrument.pip for value in atr_values if value is not None]
    if atr_pips and report.chosen_horizons:
        mean_atr = sum(atr_pips) / len(atr_pips)
        mean_hold = report.mean_horizon
        if mean_hold > 0:
            required = costs.break_even_win_rate(mean_atr, int(round(mean_hold)), spread)
            report.required_win_rate = (None if required == float("inf") else required)

    report.coverage = (report.signals / len(usable)) if usable else 0.0
    report.learned_weights = dict(weights.active())
    if model_points:
        report.model_points = len(model_points)
        report.model_accuracy = (
            sum(1 for probability, label in model_points if (probability >= 0.5) == bool(label))
            / len(model_points)
        )
    report.note = _note(report, spread)
    return report


def _note(report: BacktestReport, spread: float) -> str:
    parts = [
        f"{report.signals} signals replayed across {report.folds} purged folds; "
        f"every one charged {spread:.2f} pips of spread."
    ]
    if not report.enough_for_a_view:
        parts.append(
            f"Fewer than {MIN_SIGNALS_FOR_A_VIEW} signals: the interval is too wide to call this evidence."
        )

    # The verdict, not the average, is the finding. An average over one path is
    # one draw from a distribution and says nothing about the next path.
    if report.edge is not None:
        parts.append(report.edge.describe())
        if report.edge.inconclusive and report.samples_needed > 0:
            parts.append(
                f"Resolving a 0.25 pip/trade edge at this volatility would take about "
                f"{report.samples_needed} settled trades."
            )
    # Say what the win rate needed to be. Without this the achieved rate is just
    # a number with no yardstick, and a 56% hit rate sounds like skill.
    if report.win_rate is not None and report.required_win_rate is not None:
        gap = report.win_rate - report.required_win_rate
        parts.append(
            f"Win rate needed to break even at this volatility and hold length is "
            f"{report.required_win_rate * 100:.1f}%; achieved {report.win_rate * 100:.1f}% "
            f"({gap * 100:+.1f} points)."
        )
    elif report.win_rate is not None and report.required_win_rate is None:
        parts.append(
            "No hit rate can break even at this volatility and hold length — the "
            "projected move cannot cover the spread at any accuracy."
        )
    if report.net_pips <= 0:
        parts.append("Net expectancy is not positive after costs — the gate should stay in VETO mode.")
    parts.append(
        "Replay assumes the feed was healthy (freshness cannot be checked historically) and "
        "measures one path through one history, not a distribution: run --paths to see the spread."
    )
    return " ".join(parts)


@dataclass
class MultiPathReport:
    """The same system replayed over several independent synthetic price paths.

    One path is one draw. Reporting it as though it were a measurement is how a
    losing system gets described as a 55% win rate. This is the distribution the
    single number was hiding inside.
    """

    paths: int
    bars: int
    reports: list[BacktestReport] = field(default_factory=list)
    summary: MultiPathSummary | None = None

    @property
    def signals(self) -> int:
        return sum(report.signals for report in self.reports)

    def to_dict(self) -> dict:
        return {
            "paths": self.paths,
            "bars": self.bars,
            "signals": self.signals,
            "summary": self.summary.to_dict() if self.summary else None,
            "per_path": [report.to_dict() for report in self.reports],
        }


def run_multi_path(
    config: Config,
    *,
    paths: int = 8,
    bars: int = 3000,
    first_seed: int = 1,
    horizon: int | None = None,
    spread_pips: float | None = None,
    folds: int = 4,
    volatility: float | None = None,
    drift: float = 0.0,
) -> MultiPathReport:
    """Replay the system over ``paths`` independent synthetic price paths.

    Synthetic series are random walks, so the true directional edge on them is
    zero by construction. A system that reports a positive expectancy on one
    path of a random walk is reporting noise; replaying several paths shows
    how loud that noise is.

    ``drift`` defaults to zero so the paths carry no directional bias of their
    own — otherwise a long-biased model could appear skilful simply by trading
    the generator's drift.
    """
    from .simulate import synthetic_series

    report = MultiPathReport(paths=int(paths), bars=int(bars))
    seeds = [int(first_seed) + offset for offset in range(max(1, int(paths)))]
    for seed in seeds:
        kwargs = {"count": int(bars), "seed": seed, "drift": drift}
        if volatility is not None:
            kwargs["volatility"] = float(volatility)
        candles = synthetic_series(**kwargs)
        report.reports.append(
            run_backtest(
                candles,
                config,
                folds=folds,
                horizon=horizon,
                spread_pips=spread_pips,
            )
        )

    expectancies = [
        value
        for value in (item.expectancy_pips for item in report.reports)
        if value is not None
    ]
    report.summary = summarise_paths(expectancies)
    return report
