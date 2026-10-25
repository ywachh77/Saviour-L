"""Configuration from environment variables (and a local .env file).

Secrets are read from the environment only. Nothing in this module ever asks a
human to paste a key into a chat window.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import charter, instruments
from .timeutil import UTC


def _load_dotenv(path: str | Path = ".env") -> None:
    """Populate os.environ from a .env file without overwriting real env vars.

    Deliberately tiny: no dependency, no shell evaluation.
    """
    target = Path(path)
    if not target.is_file():
        return
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _keys_from_env() -> list[str]:
    """Collect Twelve Data keys from TWELVE_DATA_KEY plus slot list.

    Slots exist so a user who has legitimately provisioned several keys can use
    them. The bot does not create accounts and does not help bypass rate
    limits; see README.
    """
    found: list[str] = []
    single = os.environ.get("TWELVE_DATA_API_KEY", "").strip()
    if single:
        found.append(single)
    for index in range(1, 31):
        value = os.environ.get(f"TWELVE_DATA_KEY_{index}", "").strip()
        if value:
            found.append(value)
    # Keep order, drop duplicates.
    unique: list[str] = []
    for key in found:
        if key not in unique:
            unique.append(key)
    return unique


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    value = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return value in ("1", "true", "yes", "on")


@dataclass
class Config:
    # --- Telegram (owner-only) ---
    telegram_token: str = ""
    telegram_owner_chat_id: str = ""
    telegram_signal_style: str = "standard"
    telegram_signal_template: str = ""
    telegram_result_style: str = "standard"
    telegram_result_template: str = ""
    telegram_apply_to_signals_cmd: bool = False

    # --- Supabase ---
    supabase_url: str = ""
    supabase_service_key: str = ""

    # --- provider ---
    twelve_data_keys: list[str] = field(default_factory=list)
    provider_base_url: str = "https://api.twelvedata.com"
    request_timeout_seconds: float = 8.0

    # --- market ---
    pair: str = charter.ALLOWED_PAIR
    pairs: tuple[str, ...] = ()
    timeframe: str = charter.ALLOWED_TIMEFRAME
    candle_history: int = 320
    # Bars the nightly job trains on. Separate from candle_history, which is the
    # rolling window the live cycle keeps in memory for signal generation.
    training_history: int = 200_000

    # --- signal shape ---
    horizon_minutes: int = 5
    # Hold long enough to pay for the trip. Off by default because it changes
    # which bars produce signals, and the default path stays the one that is
    # already measured. See fusion/costs.py for why 5 minutes at 1.2 pips is
    # usually not enough: at a realistic EUR/USD M1 ATR of 1.5 pips the same
    # trade needs about 36 minutes to clear three times its cost.
    adaptive_horizon: bool = False
    # Narrow which strategies may speak by FX session. On by default: the
    # mapping is standard domain knowledge, and on a synthetic replay it is
    # inert anyway, because prototypes need 30 real outcomes to earn a voice
    # and synthetic data grants none. See fusion/session.py.
    session_filter: bool = True
    # Only take a trade that goes with the higher-timeframe tilt, resampled
    # locally from one-minute bars at zero extra API cost.
    #
    # Validated on 101,730 real EUR/USD M1 bars (Aug-Oct 2026): a 15-minute
    # tilt (MTF_MINUTES=15) lifted Train win rate from 51.49% to 52.21%
    # (+0.201 net pips/trade) and Out-of-Sample October win rate from 57.47% to
    # 58.51% (-0.116 -> +0.338 net pips/trade after 1.20-pip spread). Kept
    # opt-in on Config() so synthetic random-walk unit replays stay unmoved,
    # and enabled in the live workflow via MTF_FILTER=1.
    mtf_filter: bool = False
    mtf_minutes: int = 15
    # Deterministic macro release blackout (US NFP first Friday 08:30 ET plus
    # any explicit UTC windows in NEWS_BLACKOUT_WINDOWS). See fusion/news.py.
    news_filter: bool = True
    news_blackout_windows: str = ""
    reward_to_cost: float = 3.0
    max_horizon_minutes: int = 120
    max_seal_latency_seconds: int = charter.MAX_SEAL_LATENCY_SECONDS
    max_candle_age_seconds: int = 180
    max_spread_pips: float = 2.5
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS

    def __post_init__(self) -> None:
        inst = instruments.for_symbol(self.pair)
        if (
            self.pair.strip().upper() != charter.ALLOWED_PAIR
            and self.spread_pips == charter.DEFAULT_SPREAD_PIPS
            and self.max_spread_pips == 2.5
        ):
            self.spread_pips = inst.spread_pips
            self.max_spread_pips = round(inst.spread_pips * charter.MAX_SPREAD_MULTIPLE, 4)

    @property
    def active_pairs(self) -> tuple[str, ...]:
        """Ordered unique markets configured for this run."""
        if not self.pairs:
            return (self.pair,)
        seen: list[str] = []
        for item in self.pairs:
            symbol = item.strip().upper()
            if symbol and symbol not in seen:
                seen.append(symbol)
        return tuple(seen) if seen else (self.pair,)

    def for_pair(self, symbol: str) -> "Config":
        """Derive a per-market Config carrying that instrument's spread scale."""
        clean = symbol.strip().upper()
        if clean == self.pair.strip().upper():
            return replace(self, pair=clean, pairs=(clean,))
        inst = instruments.for_symbol(clean)
        spread = inst.spread_pips
        return replace(
            self,
            pair=clean,
            pairs=(clean,),
            spread_pips=spread,
            max_spread_pips=round(spread * charter.MAX_SPREAD_MULTIPLE, 4),
        )

    @property
    def instrument(self) -> "instruments.Instrument":
        """The market this config trades, resolved from ``pair``.

        A property rather than a field so it can never drift out of step with
        the pair: set the pair and the pip, the precision and the assumed
        spread all follow. Storing it separately would let a config carry
        EUR/USD's pip alongside BTC/USD's name, which is the exact mistake
        that makes a pips figure meaningless.
        """
        return instruments.for_symbol(self.pair)

    # --- gates ---
    max_uncertainty: float = 0.10
    min_confluence_checks: int = 5
    min_calibration_points: int = charter.MIN_CALIBRATION_POINTS

    # --- protections (Freqtrade-style guards, opt-in on bare Config()) ---
    signal_cooldown_minutes: int = 0
    loss_guard_streak: int = 0
    loss_guard_lookback_minutes: int = 180
    loss_guard_cooldown_minutes: int = 60
    fx_rollover_guard: bool = False

    # --- learning ---
    nightly_min_samples: int = 400
    nightly_train_fraction: float = 0.70
    online_learning_rate: float = 0.02

    # --- runtime ---
    tick_seconds: int = 15
    heartbeat_stale_seconds: int = charter.HEARTBEAT_STALE_SECONDS
    digest_hour_utc: int = 0
    digest_minute_utc: int = 5
    nightly_hour_utc: int = 23
    nightly_minute_utc: int = 55
    timezone_name: str = "UTC"
    local_state_dir: str = "var"
    runner: str = "github-actions"

    # --- switches ---
    allow_synthetic_learning: bool = False  # charter: never flip this on
    require_owner_commands: bool = True

    @property
    def telegram_ready(self) -> bool:
        return bool(self.telegram_token and self.telegram_owner_chat_id)

    @property
    def supabase_ready(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_key)

    @property
    def provider_ready(self) -> bool:
        return bool(self.twelve_data_keys)

    def redacted(self) -> dict:
        return {
            "telegram_ready": self.telegram_ready,
            "supabase_ready": self.supabase_ready,
            "provider_keys": len(self.twelve_data_keys),
            "pair": self.pair,
            "pairs": list(self.active_pairs),
            "horizon_minutes": self.horizon_minutes,
            "adaptive_horizon": self.adaptive_horizon,
            "mtf_filter": self.mtf_filter,
            "mtf_minutes": self.mtf_minutes,
            "news_filter": self.news_filter,
            "reward_to_cost": self.reward_to_cost,
            "signal_cooldown_minutes": self.signal_cooldown_minutes,
            "loss_guard_streak": self.loss_guard_streak,
            "fx_rollover_guard": self.fx_rollover_guard,
            "telegram_signal_style": self.telegram_signal_style,
            "telegram_result_style": self.telegram_result_style,
            "synthetic_learning_allowed": self.allow_synthetic_learning,
        }


def _parse_pairs(raw: str, default_pair: str) -> tuple[str, ...]:
    if not raw.strip():
        return (default_pair,)
    parsed: list[str] = []
    for part in raw.split(","):
        symbol = part.strip().upper()
        if not symbol:
            continue
        if not instruments.is_allowed(symbol):
            raise ValueError(
                f"PAIRS entry {symbol!r} is not a market this bot knows about. "
                f"Supported: {', '.join(instruments.symbols())}."
            )
        if symbol not in parsed:
            parsed.append(symbol)
    return tuple(parsed) if parsed else (default_pair,)


def load_config(dotenv_path: str | Path = ".env") -> Config:
    _load_dotenv(dotenv_path)

    pair = (os.environ.get("PAIR", charter.ALLOWED_PAIR).strip() or charter.ALLOWED_PAIR).upper()
    # Refuse an unknown market at startup rather than trading it. for_symbol()
    # falls back to EUR/USD's pip for anything it does not recognise, which is
    # the right behaviour when reading an old row and the wrong behaviour when
    # configuring a run: a typo would scale every pip figure by thousands and
    # still produce plausible-looking output.
    if not instruments.is_allowed(pair):
        raise ValueError(
            f"PAIR={pair!r} is not a market this bot knows about. "
            f"Supported: {', '.join(instruments.symbols())}."
        )
    pairs = _parse_pairs(os.environ.get("PAIRS", ""), pair)
    if os.environ.get("PAIRS", "").strip() and not os.environ.get("PAIR", "").strip():
        pair = pairs[0]
    instrument = instruments.for_symbol(pair)

    # The assumed spread follows the instrument unless it is set explicitly.
    # A bitcoin round trip costs about eight dollars, not the 1.2 pips EUR/USD
    # pays, and hardcoding the FX figure would have charged a crypto trade
    # roughly a seventh of its true cost -- flattering every backtest ever run
    # on a crypto series.
    spread = _float("SPREAD_PIPS", instrument.spread_pips)
    # The gate defaults to a multiple of the assumed spread rather than to a
    # fixed number. An absolute cap decoupled from the assumed cost meant the
    # two drifted apart: the modelled off-hours spread (2.1x) could land 0.02
    # pips over the cap purely by coincidence, and raising SPREAD_PIPS changed
    # what the bot charged without changing what it would trade.
    max_spread = _float("MAX_SPREAD_PIPS", 0.0)
    if max_spread <= 0:
        max_spread = spread * charter.MAX_SPREAD_MULTIPLE

    return Config(
        telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_owner_chat_id=os.environ.get("TELEGRAM_OWNER_CHAT_ID", "").strip(),
        telegram_signal_style=os.environ.get("TELEGRAM_SIGNAL_STYLE", "standard").strip() or "standard",
        telegram_signal_template=os.environ.get("TELEGRAM_SIGNAL_TEMPLATE", "").replace("\\n", "\n"),
        telegram_result_style=os.environ.get("TELEGRAM_RESULT_STYLE", "standard").strip() or "standard",
        telegram_result_template=os.environ.get("TELEGRAM_RESULT_TEMPLATE", "").replace("\\n", "\n"),
        telegram_apply_to_signals_cmd=_bool("TELEGRAM_APPLY_TO_SIGNALS_CMD", False),
        supabase_url=os.environ.get("SUPABASE_URL", "").strip().rstrip("/"),
        supabase_service_key=os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip(),
        twelve_data_keys=_keys_from_env(),
        provider_base_url=os.environ.get("TWELVE_DATA_BASE_URL", "https://api.twelvedata.com").strip(),
        request_timeout_seconds=_float("REQUEST_TIMEOUT_SECONDS", 8.0),
        pair=pair,
        pairs=pairs,
        timeframe=os.environ.get("TIMEFRAME", charter.ALLOWED_TIMEFRAME).strip(),
        candle_history=_int("CANDLE_HISTORY", 320),
        training_history=_int("TRAINING_HISTORY", 200_000),
        horizon_minutes=_int("SIGNAL_HORIZON_MINUTES", 5),
        adaptive_horizon=_bool("ADAPTIVE_HORIZON", False),
        session_filter=_bool("SESSION_FILTER", True),
        mtf_filter=_bool("MTF_FILTER", False),
        mtf_minutes=_int("MTF_MINUTES", 15),
        news_filter=_bool("NEWS_FILTER", True),
        news_blackout_windows=os.environ.get("NEWS_BLACKOUT_WINDOWS", "").strip(),
        reward_to_cost=_float("REWARD_TO_COST", 3.0),
        max_horizon_minutes=_int("MAX_HORIZON_MINUTES", 120),
        max_seal_latency_seconds=_int("MAX_SEAL_LATENCY_SECONDS", charter.MAX_SEAL_LATENCY_SECONDS),
        max_candle_age_seconds=_int("MAX_CANDLE_AGE_SECONDS", 180),
        max_spread_pips=max_spread,
        spread_pips=spread,
        max_uncertainty=_float("MAX_UNCERTAINTY", 0.10),
        min_confluence_checks=_int("MIN_CONFLUENCE_CHECKS", 5),
        min_calibration_points=_int("MIN_CALIBRATION_POINTS", charter.MIN_CALIBRATION_POINTS),
        signal_cooldown_minutes=_int("SIGNAL_COOLDOWN_MINUTES", 0),
        loss_guard_streak=_int("LOSS_GUARD_STREAK", 0),
        loss_guard_lookback_minutes=_int("LOSS_GUARD_LOOKBACK_MINUTES", 180),
        loss_guard_cooldown_minutes=_int("LOSS_GUARD_COOLDOWN_MINUTES", 60),
        fx_rollover_guard=_bool("FX_ROLLOVER_GUARD", False),
        nightly_min_samples=_int("NIGHTLY_MIN_SAMPLES", 400),
        nightly_train_fraction=_float("NIGHTLY_TRAIN_FRACTION", 0.70),
        online_learning_rate=_float("ONLINE_LEARNING_RATE", 0.02),
        tick_seconds=_int("TICK_SECONDS", 15),
        heartbeat_stale_seconds=_int("HEARTBEAT_STALE_SECONDS", charter.HEARTBEAT_STALE_SECONDS),
        digest_hour_utc=_int("DIGEST_HOUR_UTC", 0),
        digest_minute_utc=_int("DIGEST_MINUTE_UTC", 5),
        nightly_hour_utc=_int("NIGHTLY_HOUR_UTC", 23),
        nightly_minute_utc=_int("NIGHTLY_MINUTE_UTC", 55),
        timezone_name=os.environ.get("TZ_NAME", "UTC").strip(),
        local_state_dir=os.environ.get("LOCAL_STATE_DIR", "var").strip(),
        runner=os.environ.get("RUNNER", "github-actions").strip() or "github-actions",
        allow_synthetic_learning=False,
        require_owner_commands=_bool("REQUIRE_OWNER_COMMANDS", True),
    )


UTC  # re-exported for callers that import it from here
