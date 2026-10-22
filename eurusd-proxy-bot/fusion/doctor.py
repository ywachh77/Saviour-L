"""Setup doctor: tell the operator exactly which step is broken, and why.

Running the bot for the first time involves four external services. When one of
them is wrong the failure is usually silent, so this module checks each one for
real (not just "is the variable set?") and prints a single next action per
problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .keypool import mask_key
from .store import TABLES, SupabaseStore
from .telegram import TelegramClient

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
WARN = "WARN"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    action: str = ""

    def line(self) -> str:
        text = f"  [{self.status:4}] {self.name}: {self.detail}"
        if self.action:
            text += f"\n         → {self.action}"
        return text


def check_env(config, env_path: Path) -> Check:
    if not env_path.is_file():
        return Check(
            ".env file",
            FAIL,
            f"no {env_path} found",
            f"cp .env.example {env_path} then fill in your values",
        )
    return Check(".env file", PASS, f"found {env_path}")


def check_provider(config, *, offline: bool = False) -> Check:
    if not config.provider_ready:
        return Check(
            "Twelve Data",
            FAIL,
            "no API key configured",
            "get a free key at twelvedata.com and set TWELVE_DATA_API_KEY in .env",
        )
    labels = ", ".join(mask_key(key) for key in config.twelve_data_keys[:3])
    suffix = " ..." if len(config.twelve_data_keys) > 3 else ""
    if offline:
        return Check("Twelve Data", SKIP, f"{len(config.twelve_data_keys)} slot(s): {labels}{suffix}")

    from .provider import TwelveDataProvider

    provider = TwelveDataProvider(
        config.twelve_data_keys, base_url=config.provider_base_url, timeout=config.request_timeout_seconds
    )
    try:
        candles = provider.time_series(config.pair, config.timeframe, 1)
    except Exception as error:  # noqa: BLE001 - the doctor reports, it does not crash
        return Check(
            "Twelve Data",
            FAIL,
            f"{len(config.twelve_data_keys)} slot(s) configured, but the test call failed: {error}",
            "check the key, your plan's quota, and that EUR/USD is enabled on it",
        )
    if not candles:
        return Check(
            "Twelve Data",
            FAIL,
            "the test call returned no candles",
            "the key works but returned nothing for EUR/USD; check the symbol on your plan",
        )
    pool = provider.pool.summary()
    enabled = pool["enabled"]
    return Check(
        "Twelve Data",
        PASS,
        f"{enabled}/{len(config.twelve_data_keys)} slot(s) usable · latest {config.pair} bar "
        f"{candles[-1].timestamp.strftime('%Y-%m-%d %H:%M')}Z (costs 1 credit per run)",
    )


def check_supabase(config) -> Check:
    if not config.supabase_ready:
        return Check(
            "Supabase",
            FAIL,
            "SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is empty",
            "Supabase → Project Settings → API: copy the Project URL and the service_role "
            "key (not the anon key) into .env",
        )
    store = SupabaseStore(config.supabase_url, config.supabase_service_key, timeout=config.request_timeout_seconds)
    missing: list[str] = []
    unreachable: str | None = None
    for table in TABLES:
        error = store.probe(table)
        if error is None:
            continue
        if "does not exist" in error or "42P01" in error or "404" in error:
            missing.append(table)
        elif unreachable is None:
            unreachable = error
    if unreachable is not None:
        return Check(
            "Supabase",
            FAIL,
            f"could not reach the project: {unreachable[:160]}",
            "check SUPABASE_URL and that you used the service_role key, not the anon key",
        )
    if missing:
        return Check(
            "Supabase",
            FAIL,
            f"{len(missing)} of {len(TABLES)} tables missing: {', '.join(missing)}",
            "Supabase → SQL Editor → paste supabase/migrations/0001_init.sql and run it",
        )
    return Check(
        "Supabase",
        PASS,
        f"all {len(TABLES)} tables reachable with row-level security on",
    )


def check_telegram(config, *, send_test: bool = False) -> Check:
    if not config.telegram_ready:
        return Check(
            "Telegram",
            FAIL,
            "TELEGRAM_BOT_TOKEN or TELEGRAM_OWNER_CHAT_ID is empty",
            "BotFather gives the token; @userinfobot gives your numeric chat id",
        )
    client = TelegramClient(config.telegram_token, config.telegram_owner_chat_id)
    result, error = client._call("getMe", {})
    if error:
        # "Wrong token" and "no route to host" need different advice. Telling
        # someone to regenerate a working token because their wifi dropped sends
        # them chasing a problem they do not have.
        if error.startswith("network:"):
            return Check(
                "Telegram",
                FAIL,
                f"could not reach Telegram: {error[:160]}",
                "the token was not tested — check the connection and run this again",
            )
        if "HTTP 401" in error or "HTTP 404" in error:
            return Check(
                "Telegram",
                FAIL,
                f"getMe failed: {error[:160]}",
                "the token is wrong or revoked; ask BotFather for a new one",
            )
        return Check(
            "Telegram",
            FAIL,
            f"getMe failed: {error[:160]}",
            "Telegram refused the call; the token and the network both look present",
        )
    username = ((result or {}).get("result") or {}).get("username", "?")
    if not send_test:
        return Check(
            "Telegram",
            PASS,
            f"bot @{username} is reachable; owner chat id {config.telegram_owner_chat_id}",
            "run with --send to put a test message in the chat",
        )
    sent = client.send("FUSION BOT setup check — if you can read this, delivery works.")
    if sent.confirmed:
        return Check("Telegram", PASS, f"test message delivered to @{username} (message {sent.message_id})")
    return Check(
        "Telegram",
        FAIL,
        f"bot @{username} is reachable but the message was {sent.state}: {sent.error[:120]}",
        "open a chat with your bot and press /start — a bot cannot message you first",
    )


def check_storage(config) -> Check:
    try:
        path = Path(config.local_state_dir)
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".doctor"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        return Check("Local state", FAIL, f"cannot write to {config.local_state_dir}: {error}")
    return Check("Local state", PASS, f"writable at {Path(config.local_state_dir).resolve()}")


def run_checks(config, *, env_path: Path, offline: bool = False, send_test: bool = False) -> list[Check]:
    return [
        check_env(config, env_path),
        check_provider(config, offline=offline),
        check_supabase(config),
        check_telegram(config, send_test=send_test),
        check_storage(config),
    ]


def summarise(checks: list[Check]) -> tuple[int, int]:
    failed = sum(1 for check in checks if check.status == FAIL)
    passed = sum(1 for check in checks if check.status == PASS)
    return passed, failed
