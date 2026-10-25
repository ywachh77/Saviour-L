"""Twelve Data HTTP client, quota-aware through the key pool.

Standard library ``urllib`` only. Every request reserves quota first and
refunds it if the call never reached the provider.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from . import instruments
from .domain import Candle
from .keypool import KeyPool
from .timeutil import parse_datetime


class ProviderError(Exception):
    """A provider call that did not return usable data."""

    def __init__(self, message: str, status: int | None = None, exhausted: bool = False):
        super().__init__(message)
        self.status = status
        self.exhausted = exhausted


class TwelveDataProvider:
    def __init__(
        self,
        keys: list[str],
        *,
        base_url: str = "https://api.twelvedata.com",
        timeout: float = 8.0,
        pool: KeyPool | None = None,
        opener=urllib.request.urlopen,
    ):
        self.keys = [key for key in keys if key]
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.pool = pool or KeyPool(self.keys)
        self._opener = opener
        self.last_status: int | None = None
        self.calls = 0
        self.failures = 0

    @property
    def configured(self) -> bool:
        return bool(self.keys)

    # ---------------------------------------------------------------- plumbing
    def _request(self, path: str, params: dict[str, str], cost: int = 1) -> dict[str, Any]:
        if not self.configured:
            raise ProviderError("No Twelve Data key is configured", status=None, exhausted=True)

        reservation = self.pool.reserve(cost)
        if not reservation.ok:
            raise ProviderError(f"Key pool exhausted: {reservation.reason}", status=None, exhausted=True)

        query = dict(params)
        query["apikey"] = self.keys[reservation.key_index]
        url = f"{self.base_url}/{path.lstrip('/')}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(url, headers={"User-Agent": "fusion-bot/2.0", "Accept": "application/json"})

        try:
            with self._opener(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", 200) or 200)
                body = response.read().decode("utf-8")
            self.last_status = status
            self.calls += 1
            payload = json.loads(body) if body else {}
            if isinstance(payload, dict) and payload.get("status") == "error":
                code = int(payload.get("code") or status or 500)
                self.failures += 1
                self.pool.report_failure(reservation, code, str(payload.get("message", "")))
                raise ProviderError(str(payload.get("message", "provider error")), status=code)
            if status >= 400:
                self.failures += 1
                self.pool.report_failure(reservation, status, body[:200])
                raise ProviderError(f"HTTP {status}", status=status)
            self.pool.report_success(reservation)
            return payload
        except urllib.error.HTTPError as error:  # noqa: PERF203 - clarity over micro-optimisation
            self.failures += 1
            self.last_status = int(error.code)
            self.pool.report_failure(reservation, int(error.code), str(error))
            raise ProviderError(f"HTTP {error.code}", status=int(error.code)) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            self.failures += 1
            # Refunded because the call never reached the provider, so it is not
            # charged against the key either. Cooling a slot for our own network
            # being unreachable would empty the pool over a fault no key caused.
            self.pool.report_network_error(reservation)
            self.pool.refund(reservation)
            raise ProviderError(f"Network error: {error}", status=None) from error
        except json.JSONDecodeError as error:
            self.failures += 1
            self.pool.report_failure(reservation, None, str(error))
            raise ProviderError("Provider returned a non-JSON response", status=None) from error

    # ------------------------------------------------------------------ market
    def time_series(self, symbol: str, interval: str = "1min", outputsize: int = 60) -> list[Candle]:
        payload = self._request(
            "time_series",
            {
                "symbol": symbol,
                "interval": interval,
                "outputsize": str(int(outputsize)),
                "order": "ASC",
                "timezone": "UTC",
                "format": "JSON",
            },
        )
        values = payload.get("values") if isinstance(payload, dict) else None
        if not values:
            raise ProviderError("Provider returned no candles", status=self.last_status)

        candles: list[Candle] = []
        for row in values:
            try:
                candles.append(Candle.from_mapping(row))
            except (ValueError, KeyError, TypeError):
                # One malformed row must not poison the whole series.
                continue
        candles.sort(key=lambda candle: candle.timestamp)
        return candles

    def quote(self, symbol: str) -> dict[str, Any] | None:
        """Best-effort bid/ask. Free plans often lack it; absence is not fatal."""
        try:
            payload = self._request("quote", {"symbol": symbol, "format": "JSON"})
        except ProviderError:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            bid = float(payload["bid"])
            ask = float(payload["ask"])
        except (KeyError, TypeError, ValueError):
            return None
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        pip = instruments.pip_for(symbol)
        return {"bid": bid, "ask": ask, "spread_pips": (ask - bid) / pip, "source": "provider"}


class SyntheticProvider:
    """Deterministic fake feed used by the smoke test and offline demos.

    Anything produced here is tagged synthetic and is never allowed to train
    the model (see charter.is_synthetic).
    """

    def __init__(self, candles: list[Candle]):
        self.candles = list(candles)
        self.calls = 0
        self.failures = 0

    @property
    def configured(self) -> bool:
        return bool(self.candles)

    def time_series(self, symbol: str, interval: str = "1min", outputsize: int = 60) -> list[Candle]:
        self.calls += 1
        return list(self.candles[-int(outputsize) :])

    def quote(self, symbol: str) -> dict[str, Any] | None:
        return None


def candles_from_rows(rows: list[dict[str, Any]]) -> list[Candle]:
    """Parse provider rows into candles, skipping anything malformed."""
    candles = []
    for row in rows:
        try:
            candles.append(Candle.from_mapping(row))
        except (ValueError, KeyError, TypeError):
            continue
    candles.sort(key=lambda candle: parse_datetime(candle.timestamp))
    return candles
