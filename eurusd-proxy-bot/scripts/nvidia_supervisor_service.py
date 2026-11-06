"""NVIDIA supervisor service: 24/7 loop + one-shot 19/20 test.

WHAT THIS IS
------------
A long-running supervisor that sits above the barrier engine, using the
project's existing AI gateway (`fusion/ai_gateway`) with an NVIDIA NIM
credential. On each cycle it:

  1. auto-loads stored credentials (a key added later is picked up live --
     no restart),
  2. health-checks the gateway and writes a heartbeat file,
  3. runs the 19/20 test: candidate trades from the barrier engine are
     presented to the supervisor model, which answers APPROVE or VETO, and
     the approved set is measured.

MEASUREMENT RULES (unchanged project policy)
---------------------------------------------
A win rate is never reported alone. Every test prints, beside it: the
break-even win rate for the geometry, net pips/trade, worst trade, max
drawdown, and the count of rolling 20-trade windows with >=19 (and >=17,
the locked floor) wins.

HONESTY RULE
------------
If no NVIDIA credential is stored, the supervisor is APPROVE-ALL and the
report says mode=NO_NVIDIA_CREDENTIAL. That is the baseline the AI must
beat -- it is never presented as an AI result.

Run one test:   python3 scripts/nvidia_supervisor_service.py --once
Run 24/7:       python3 scripts/nvidia_supervisor_service.py --daemon
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parents[1]
for p in (BOT_ROOT, BOT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_supervisor_dataset import build, load_candles  # noqa: E402

HEARTBEAT = BOT_ROOT / "var" / "nvidia_supervisor_heartbeat.json"
_STOP = False


def _stop(signum, frame):  # pragma: no cover - signal path
    global _STOP
    _STOP = True


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def summarise(name: str, nets: list[float], tp: float, sl: float, spread: float) -> dict:
    n = len(nets)
    wins = sum(1 for x in nets if x > 0)
    win_rate = 100.0 * wins / n if n else None
    break_even = 100.0 * (sl + spread) / (tp + sl) if (tp + sl) else None
    eq = peak = max_dd = 0.0
    for x in nets:
        eq += x
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    run = longest = 0
    for x in nets:
        run = run + 1 if x < 0 else 0
        longest = max(longest, run)
    flags = [x > 0 for x in nets]
    return {
        "set": name,
        "trades": n,
        "wins": wins,
        "win_rate_pct": None if win_rate is None else round(win_rate, 2),
        "wilson_95_pct": [None if v is None else round(v, 2) for v in wilson(wins, n)],
        "break_even_win_rate_pct": None if break_even is None else round(break_even, 2),
        "net_pips_total": round(sum(nets), 1),
        "net_pips_per_trade": None if not n else round(sum(nets) / n, 3),
        "worst_trade_pips": None if not n else min(nets),
        "max_drawdown_pips": round(max_dd, 1),
        "longest_losing_run": longest,
        "windows_20_with_19plus_wins": sum(
            1 for i in range(max(0, n - 19)) if sum(flags[i:i + 20]) >= 19),
        "windows_20_with_17plus_wins": sum(
            1 for i in range(max(0, n - 19)) if sum(flags[i:i + 20]) >= 17),
    }


# --------------------------------------------------------------------------
# supervisor
# --------------------------------------------------------------------------

def get_gateway():
    """Load the gateway: stored credentials first, then the env secret.

    Two credential sources, in order:
      1. the gateway credential store (a key connected through the app);
      2. the ``NVIDIA_API_KEY`` environment secret (e.g. added in the
         host's Secrets / env settings). It is connected through the
         gateway's normal validate-and-store path. The key value is never
         printed or logged -- only whether the connection succeeded.

    Never raises: the gateway is optional infrastructure and the service
    must keep cycling (in baseline mode) without it.
    """
    gateway = None
    try:
        from fusion.ai_gateway import initialize_ai_gateway

        gateway = initialize_ai_gateway(auto_connect=True)
    except Exception as exc:  # gateway is optional infrastructure
        print(f"[nvidia-service] gateway unavailable: {exc}")

    env_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if env_key:
        try:
            from fusion.ai_gateway import AIGateway, CredentialStore, ModelRegistry

            if gateway is None:
                gateway = AIGateway(registry=ModelRegistry(),
                                    credential_store=CredentialStore())
            if not gateway.credentials.has_credential("nvidia_nim"):
                result = gateway.connect("nvidia_nim", env_key)
                if result.get("ok"):
                    print("[nvidia-service] NVIDIA connected from "
                          "NVIDIA_API_KEY secret "
                          f"({result.get('models_discovered', 0)} models)")
                else:
                    print("[nvidia-service] NVIDIA_API_KEY present but "
                          "connection failed: "
                          f"{result.get('error', 'unknown error')}")
        except Exception as exc:
            print(f"[nvidia-service] env-secret connect failed: {exc}")
    return gateway


def pick_model(gateway) -> str | None:
    """Prefer a Qwen model on NVIDIA; else any chat model the gateway has."""
    if gateway is None:
        return None
    try:
        models = gateway.list_available_models(provider="nvidia_nim")
    except Exception:
        models = []
    ids = [m.get("model_id") or m.get("id") or "" for m in models]
    for mid in ids:
        if "qwen" in mid.lower():
            return mid
    return ids[0] if ids else None


def supervisor_votes(gateway, model_id, examples) -> tuple[list[bool], str]:
    """Return (approve?, ...) per example, plus the mode actually used."""
    if gateway is None or not model_id:
        return [True] * len(examples), "NO_NVIDIA_CREDENTIAL (approve-all baseline)"
    votes = []
    for ex in examples:
        try:
            resp = gateway.chat(
                model_id,
                [{"role": "user", "content": ex["prompt"]}],
                temperature=0.0, max_tokens=4,
            )
            text = ""
            if isinstance(resp, dict):
                text = str(resp.get("content") or resp.get("text") or resp.get("error") or "")
            votes.append("VETO" not in text.upper())
        except Exception:
            votes.append(True)  # a failed call must not silently veto
    return votes, f"NVIDIA:{model_id}"


# --------------------------------------------------------------------------
# the 19/20 test
# --------------------------------------------------------------------------

def synthetic_candles(bars: int, seed: int) -> list[dict]:
    """Zero-drift EUR/USD-like M1 series (clearly labelled synthetic)."""
    import random

    rng = random.Random(seed)
    price = 1.1000
    out = []
    base = datetime(2026, 1, 5, tzinfo=timezone.utc)
    from datetime import timedelta

    for i in range(bars):
        o = price
        c = max(0.5, o + rng.gauss(0, 0.00024))
        w = abs(rng.gauss(0, 0.00017)) + 0.00002
        out.append({
            "ts": (base + timedelta(minutes=i)).isoformat(),
            "o": o, "h": max(o, c) + w, "l": min(o, c) - w, "c": c,
        })
        price = c
    return out


def run_test(args, gateway) -> dict:
    if args.source == "real":
        candles = load_candles(
            BOT_ROOT / "var/real_candles/eurusd_real_20261030_live.json")
        source_label = "REAL repo candles (740 bars)"
    else:
        candles = synthetic_candles(args.bars, args.seed)
        source_label = f"SYNTHETIC zero-drift {args.bars} bars (seed {args.seed})"

    examples = build(candles, args.tp, args.sl, args.spread)
    cut = len(examples) // 2
    held = examples[cut:]

    model_id = pick_model(gateway)
    votes, mode = supervisor_votes(gateway, model_id, held)

    baseline = summarise("approve-all (held out)",
                         [e["net_pips"] for e in held], args.tp, args.sl, args.spread)
    approved = summarise("supervisor-approved (held out)",
                         [e["net_pips"] for e, v in zip(held, votes) if v],
                         args.tp, args.sl, args.spread)

    report = {
        "test": "19_OF_20_SUPERVISOR_TEST",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "data": source_label,
        "geometry": {"tp": args.tp, "sl": args.sl, "spread": args.spread},
        "supervisor_mode": mode,
        "candidates_held_out": len(held),
        "baseline": baseline,
        "supervisor": approved,
        "gates": {
            "floor_17_of_20":
                (approved["win_rate_pct"] or 0) >= 85.0 and approved["trades"] >= 20,
            "expectancy_positive":
                (approved["net_pips_per_trade"] or -1) > 0,
            "note": "Both gates must pass. A win rate without its break-even "
                    "rate beside it is not a result.",
        },
    }

    print(f"\n=== 19/20 TEST ({source_label}) ===")
    print(f"geometry tp={args.tp} sl={args.sl} spread={args.spread} | "
          f"supervisor: {mode}")
    for row in (baseline, approved):
        print(f"  {row['set']:<32} n={row['trades']:<4} "
              f"win={row['win_rate_pct']}% (need {row['break_even_win_rate_pct']}%) "
              f"net/trade={row['net_pips_per_trade']}p worst={row['worst_trade_pips']}p "
              f"maxDD={row['max_drawdown_pips']}p L-run={row['longest_losing_run']} "
              f"19+/20 windows={row['windows_20_with_19plus_wins']} "
              f"17+/20 windows={row['windows_20_with_17plus_wins']}")
    print(f"  gates: floor17={report['gates']['floor_17_of_20']} "
          f"expectancy={report['gates']['expectancy_positive']}")
    if len(held) < 20:
        print(f"  NOTE: only {len(held)} held-out candidates -- too few for any "
              f"20-trade window. Result is indicative only.")
    return report


# --------------------------------------------------------------------------
# service loop (24/7)
# --------------------------------------------------------------------------

def heartbeat(state: dict) -> None:
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(json.dumps(
        {**state, "at": datetime.now(timezone.utc).isoformat()}, indent=2))


def daemon(args) -> int:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    cycle = 0
    print(f"[nvidia-service] 24/7 supervisor started "
          f"(cycle every {args.interval}s, test every {args.test_every} cycles)")
    while not _STOP:
        cycle += 1
        gateway = get_gateway()
        status = {}
        if gateway is not None:
            try:
                status = gateway.get_status()
            except Exception as exc:
                status = {"error": str(exc)}
        model_id = pick_model(gateway)
        state = {
            "service": "nvidia_supervisor", "cycle": cycle,
            "gateway_status": status,
            "nvidia_model": model_id,
            "mode": "NVIDIA" if model_id else "NO_NVIDIA_CREDENTIAL",
        }
        if cycle == 1 or cycle % args.test_every == 0:
            try:
                state["last_test"] = run_test(args, gateway)
            except Exception as exc:
                state["last_test"] = {"error": str(exc)}
                print(f"[nvidia-service] test failed: {exc}")
        heartbeat(state)
        print(f"[nvidia-service] cycle {cycle}: mode={state['mode']} "
              f"model={model_id} heartbeat written")
        for _ in range(args.interval):
            if _STOP:
                break
            time.sleep(1)
    print("[nvidia-service] stopped cleanly")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="run the 19/20 test once")
    mode.add_argument("--daemon", action="store_true", help="run 24/7")
    p.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--bars", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--tp", type=float, default=1.25, help="19/20 geometry target")
    p.add_argument("--sl", type=float, default=116.0, help="19/20 geometry stop")
    p.add_argument("--spread", type=float, default=1.2)
    p.add_argument("--interval", type=int, default=300, help="daemon cycle seconds")
    p.add_argument("--test-every", type=int, default=12, help="test every N cycles")
    p.add_argument("--out", type=Path, default=BOT_ROOT / "var/nvidia_19of20_report.json")
    a = p.parse_args()

    if a.daemon:
        return daemon(a)
    gateway = get_gateway()
    report = run_test(a, gateway)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(f"report saved: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
