"""Search scalp-filter conjunctions for an edge that survives out of sample.

The question this answers
-------------------------

``scalp_research.py`` shows what each scalping member does on its own: about a
50% gross hit rate, which is what a coin flip does, and then minus the spread.
This script asks the only question that can rescue that: is there a *condition*
under which the same rules are better than a coin flip by enough to pay 1.2 pips?

The search is deliberate about not fooling itself:

* every candidate is a **conjunction** of conditions that were specified before
  the numbers were seen -- regime, session, volatility band, trend alignment,
  higher-timeframe agreement -- not a threshold tuned per bar;
* candidates are scored on a **train** split and then re-scored on a **held-out**
  split from independent paths, and only the held-out column is evidence;
* the reported ``p`` is the probability of seeing that hit rate by chance given
  the number of trades, so 60% over 30 trades is visibly not 60% over 3000;
* the required rate is printed next to the achieved one, because the spread sets
  the bar and 55% is a losing rate here.

If nothing clears the bar, the honest conclusion is that this family has no
scalp-length edge on this data, and the module docstring of ``scalping.py``
should say so rather than implying the rules are merely unproven.

Usage
-----

    python3 scripts/scalp_edge_search.py --bars 4000 --paths 4
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion import charter, scalping  # noqa: E402
from fusion.features import compute_features  # noqa: E402
from fusion.instruments import for_symbol  # noqa: E402
from fusion.simulate import synthetic_series  # noqa: E402

HORIZONS = (1, 2, 3)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _p_value(wins: int, trades: int) -> float:
    """One-sided p for a hit rate at or above 50%, normal approximation."""
    if trades <= 0:
        return 1.0
    p_hat = wins / trades
    se = math.sqrt(0.25 / trades)
    return 1.0 - _norm_cdf((p_hat - 0.5) / se)


def _session(hour: int) -> str:
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 17:
        return "OVERLAP"
    if 17 <= hour < 21:
        return "NEW_YORK"
    return "ASIA_OR_OFF"


def _conditions(features: dict[str, float], candle) -> dict[str, bool]:
    """The conditions a candidate conjunction may be built from.

    Fixed up front. Nothing here reads the outcome.
    """
    adx = float(features.get("adx", 0.0) or 0.0)
    atr_pips = float(features.get("atr_pips", 0.0) or 0.0)
    ema_fast = float(features.get("ema_fast", 0.0) or 0.0)
    ema_slow = float(features.get("ema_slow", 0.0) or 0.0)
    close_pos = float(features.get("close_position", 0.5) or 0.5)
    gap15 = float(features.get("mtf_gap_15", 0.0) or 0.0)
    slope60 = float(features.get("mtf_slope_60", 0.0) or 0.0)
    return {
        "always": True,
        "adx>=20": adx >= 20.0,
        "adx>=25": adx >= 25.0,
        "adx<20": adx < 20.0,
        "atr>=3": atr_pips >= 3.0,
        "atr>=5": atr_pips >= 5.0,
        "ema_up": ema_fast > ema_slow,
        "ema_down": ema_fast < ema_slow,
        "bar_bull": candle.close > candle.open,
        "bar_bear": candle.close < candle.open,
        "close_extreme": close_pos >= 0.8 or close_pos <= 0.2,
        "close_middle": 0.35 <= close_pos <= 0.65,
        "mtf15_flat": abs(gap15) <= 2.0,
        "mtf15_trending": abs(gap15) >= 5.0,
        "macro_calm": abs(slope60) <= 0.5,
    }


def _aligned(features: dict[str, float], side: str) -> bool:
    """Does the higher-timeframe tilt agree with the side?"""
    gap5 = float(features.get("mtf_gap_5", 0.0) or 0.0)
    gap15 = float(features.get("mtf_gap_15", 0.0) or 0.0)
    if side == "BUY":
        return gap5 >= 0.0 and gap15 >= 0.0
    return gap5 <= 0.0 and gap15 <= 0.0


def _observe(candles, rows, rows_out: list[dict]) -> None:
    """Record every scalp signal with the conditions present on its bar."""
    pip = for_symbol("EUR/USD").pip
    last = len(candles) - 1
    for index in range(1, last):
        features = rows[index]
        if features is None:
            continue
        candle = candles[index]
        conditions = _conditions(features, candle)
        conditions["mtf_aligned"] = False  # filled per side below
        stamp = candle.timestamp.astimezone(timezone.utc)
        session = _session(stamp.hour)
        for member, fn in scalping.SCALPING_REGISTRY.items():
            decision = fn(candle, candles[index - 1].close, features, None)
            if decision.side not in ("BUY", "SELL"):
                continue
            for horizon in HORIZONS:
                exit_index = index + horizon
                if exit_index > last:
                    continue
                entry = candles[index + 1].open
                move = candles[exit_index].close - entry
                gross = (move if decision.side == "BUY" else -move) / pip
                rows_out.append(
                    {
                        "member": member,
                        "horizon": horizon,
                        "session": session,
                        "side": decision.side,
                        "aligned": _aligned(features, decision.side),
                        "conditions": conditions,
                        "gross": gross,
                    }
                )


class Cell:
    __slots__ = ("n", "wins", "net", "abs_sum")

    def __init__(self) -> None:
        self.n = 0
        self.wins = 0
        self.net = 0.0
        self.abs_sum = 0.0

    def add(self, gross: float, spread: float) -> None:
        if abs(gross) < 1e-12:
            return
        self.n += 1
        if gross > 0:
            self.wins += 1
        self.net += gross - spread
        self.abs_sum += abs(gross)

    @property
    def mean_net(self) -> float:
        return self.net / self.n if self.n else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def required(self) -> float | None:
        if not self.n:
            return None
        mean_abs = self.abs_sum / self.n
        if mean_abs <= 0:
            return None
        needed = 0.5 + 0.0 / 1  # replaced by caller with the spread
        return needed


def score(observations: list[dict], spread: float) -> dict[tuple, Cell]:
    """Aggregate observations into (member, horizon, session, filter) cells."""
    cells: dict[tuple, Cell] = defaultdict(Cell)
    for row in observations:
        for name, present in row["conditions"].items():
            if not present:
                continue
            cells[(row["member"], row["horizon"], row["session"], name)].add(row["gross"], spread)
            cells[(row["member"], row["horizon"], "*", name)].add(row["gross"], spread)
            if row["aligned"]:
                cells[(row["member"], row["horizon"], row["session"], name + "+MTF")].add(row["gross"], spread)
                cells[(row["member"], row["horizon"], "*", name + "+MTF")].add(row["gross"], spread)
    return cells


def report(cells: dict[tuple, Cell], spread: float, label: str, minimum: int) -> list[tuple]:
    rows = []
    for key, cell in cells.items():
        if cell.n < minimum:
            continue
        required = (0.5 + spread / (2.0 * (cell.abs_sum / cell.n))) if cell.abs_sum else None
        rows.append((cell.mean_net, key, cell, required))
    rows.sort(reverse=True, key=lambda item: item[0])
    print(f"\n{label}: best {min(15, len(rows))} conjunctions with n >= {minimum}")
    print(f"{'mean net':>9s} {'n':>6s} {'win%':>6s} {'need%':>6s} {'p':>6s}  member / horizon / session / filter")
    for mean_net, key, cell, required in rows[:15]:
        need_txt = f"{required * 100:6.1f}" if required is not None else "   n/a"
        member, horizon, session, name = key
        print(
            f"{mean_net:+9.3f} {cell.n:6d} {cell.win_rate * 100:6.1f} {need_txt} "
            f"{_p_value(cell.wins, cell.n):6.3f}  {member} h={horizon} {session} {name}"
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=4000)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--spread", type=float, default=charter.DEFAULT_SPREAD_PIPS)
    parser.add_argument("--volatility", type=float, default=0.00024)
    parser.add_argument("--minimum", type=int, default=200)
    args = parser.parse_args(argv)

    instrument = for_symbol("EUR/USD")
    train: list[dict] = []
    test: list[dict] = []
    # Odd paths train, even paths hold out. Independent synthetic series, so the
    # held-out set is genuinely unseen rather than a later slice of the same walk.
    for index in range(args.paths):
        seed = 500 + 7 * index
        candles = synthetic_series(args.bars, seed=seed, volatility=args.volatility)
        rows = compute_features(candles, instrument=instrument)
        bucket = train if index % 2 == 0 else test
        _observe(candles, rows, bucket)
        print(f"  path {index + 1}/{args.paths} seed={seed} -> {'train' if bucket is train else 'test'}", flush=True)

    train_cells = score(train, args.spread)
    test_cells = score(test, args.spread)
    train_rows = report(train_cells, args.spread, "TRAIN (in-sample, not evidence)", args.minimum)

    # The only rows that count: the same conjunction, on the held-out paths.
    print(f"\nHELD-OUT, restricted to TRAIN's best 20 (n >= {args.minimum})")
    print(f"{'mean net':>9s} {'n':>6s} {'win%':>6s} {'need%':>6s} {'p':>6s}  member / horizon / session / filter")
    survivors = 0
    for _train_net, key, _cell, _required in train_rows[:20]:
        cell = test_cells.get(key)
        if cell is None or cell.n < args.minimum:
            continue
        required = (0.5 + args.spread / (2.0 * (cell.abs_sum / cell.n))) if cell.abs_sum else None
        need_txt = f"{required * 100:6.1f}" if required is not None else "   n/a"
        member, horizon, session, name = key
        print(
            f"{cell.mean_net:+9.3f} {cell.n:6d} {cell.win_rate * 100:6.1f} {need_txt} "
            f"{_p_value(cell.wins, cell.n):6.3f}  {member} h={horizon} {session} {name}"
        )
        if cell.mean_net > 0 and cell.win_rate > 0.5:
            survivors += 1

    print()
    if survivors:
        print(f"{survivors} of the train's best 20 stayed positive on the held-out paths.")
        print("Re-run with more paths before believing any of them: 20 candidates")
        print("scored at p=0.05 will produce one false positive about every time.")
    else:
        print("None of the train's best 20 stayed positive out of sample.")
        print("There is no scalp-length edge in this family on this data: at a")
        print(f"{args.spread:.2f} pip round turn the members need a hit rate the")
        print("patterns do not deliver, and the spreads they must clear are larger")
        print("than the moves they aim at.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
