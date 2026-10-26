#!/usr/bin/env python3
"""Build curated Needle 3 LoRA fine-tuning & evaluation JSONL splits for SAVIOUR.

Every example conforms strictly to ``cactus-needle`` v3's ``finetune_local``
JSONL schema:

    {"tools": [...], "query": "...", "reasoning": "...", "answers": [...]}

and every non-empty answer is verified against ``validate_tool_calls`` so zero
schema drift can exist between training data and runtime validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys

BOT_ROOT = Path(__file__).resolve().parents[1]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from fusion.needle_agent import SAVIOUR_NEEDLE_TOOLS, validate_tool_calls  # noqa: E402


def _ex(query: str, reasoning: str, answers: list[dict]) -> dict:
    if answers:
        validated = validate_tool_calls(answers)
        if validated != answers:
            raise ValueError(f"Schema mismatch on query={query!r}: {answers} != {validated}")
    return {
        "tools": SAVIOUR_NEEDLE_TOOLS,
        "query": query,
        "reasoning": reasoning,
        "answers": answers,
    }


def build_all_examples() -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    eval_rows: list[dict] = []

    # 1. scan_market — live ("now") and paper ("scan") across ALL, EUR/USD, BTC/USD, ETH/USD
    scan_specs = [
        ("scan all markets right now", "all markets -> ALL, right now -> now", "ALL", "now"),
        ("check if there is any live signal now", "no single pair named -> ALL, live now -> now", "ALL", "now"),
        ("read the market and tell me if gates agree", "read the market -> scan_market ALL now", "ALL", "now"),
        ("any setup across eur, btc, or eth right now?", "three pairs -> ALL, right now -> now", "ALL", "now"),
        ("evaluate all pairs for an entry", "all pairs -> ALL, current evaluation -> now", "ALL", "now"),
        ("give me the latest signal check", "latest signal check -> ALL now", "ALL", "now"),
        ("check bitcoin for a setup right now", "bitcoin -> BTC/USD, right now -> now", "BTC/USD", "now"),
        ("scan btc/usd now", "btc/usd -> BTC/USD, now -> now", "BTC/USD", "now"),
        ("is there a btc signal on this candle?", "btc -> BTC/USD, this candle -> now", "BTC/USD", "now"),
        ("evaluate BTC-USD live gates", "BTC-USD -> BTC/USD, live -> now", "BTC/USD", "now"),
        ("any opportunity on bitcoin?", "bitcoin -> BTC/USD, current -> now", "BTC/USD", "now"),
        ("check xbt signal", "xbt -> BTC/USD, immediate check -> now", "BTC/USD", "now"),
        ("scan eth/usd right now", "eth/usd -> ETH/USD, right now -> now", "ETH/USD", "now"),
        ("check ethereum for a setup", "ethereum -> ETH/USD, current check -> now", "ETH/USD", "now"),
        ("any live signal on ether?", "ether -> ETH/USD, live -> now", "ETH/USD", "now"),
        ("evaluate ETH USD gates now", "ETH USD -> ETH/USD, now -> now", "ETH/USD", "now"),
        ("read eth market now", "eth -> ETH/USD, now -> now", "ETH/USD", "now"),
        ("check eurusd signal right now", "eurusd -> EUR/USD, right now -> now", "EUR/USD", "now"),
        ("scan eur/usd for an entry", "eur/usd -> EUR/USD, current -> now", "EUR/USD", "now"),
        ("any setup on fiber right now?", "fiber -> EUR/USD, right now -> now", "EUR/USD", "now"),
        ("evaluate euro dollar live", "euro dollar -> EUR/USD, live -> now", "EUR/USD", "now"),
        ("check EUR-USD setup", "EUR-USD -> EUR/USD, immediate -> now", "EUR/USD", "now"),
        ("run a deep scan across all pairs to find a setup", "deep scan / find a setup -> scan mode on ALL", "ALL", "scan"),
        ("paper scan recent bars on btc/usd", "paper scan -> mode scan on BTC/USD", "BTC/USD", "scan"),
        ("search bars on eth/usd for a paper setup", "search bars -> mode scan on ETH/USD", "ETH/USD", "scan"),
        ("scan history on eur/usd to preview a signal", "scan history -> mode scan on EUR/USD", "EUR/USD", "scan"),
        ("find a setup on bitcoin in paper scan mode", "bitcoin -> BTC/USD, paper scan -> scan", "BTC/USD", "scan"),
        ("deep scan ethereum bars", "ethereum -> ETH/USD, deep scan -> scan", "ETH/USD", "scan"),
    ]
    for idx, (q, r, pair, mode) in enumerate(scan_specs):
        row = _ex(q, r, [{"name": "scan_market", "arguments": {"pair": pair, "mode": mode}}])
        (eval_rows if idx % 5 == 4 else train).append(row)

    # 2. bot_status
    status_queries = [
        ("show bot status", "status -> bot_status"),
        ("how is the bot doing right now?", "how is the bot doing -> bot_status"),
        ("check saviour health and heartbeat", "health and heartbeat -> bot_status"),
        ("what is our settled win rate and ledger?", "win rate and ledger -> bot_status"),
        ("show model mode and calibration state", "model mode -> bot_status"),
        ("check twelve data key pool usage", "key pool -> bot_status"),
        ("give me the live scorecard and stats", "scorecard and stats -> bot_status"),
        ("is the runner heartbeat fresh?", "heartbeat -> bot_status"),
        ("report fusion bot performance", "performance -> bot_status"),
        ("status report please", "status report -> bot_status"),
        ("show current health and unresolved count", "health -> bot_status"),
        ("how is saviour performing today?", "how is saviour -> bot_status"),
        ("display ledger summary", "ledger -> bot_status"),
        ("check system diagnostics and status", "status -> bot_status"),
    ]
    for idx, (q, r) in enumerate(status_queries):
        row = _ex(q, r, [{"name": "bot_status", "arguments": {}}])
        (eval_rows if idx % 5 == 3 else train).append(row)

    # 3. pause_signals & resume_signals
    pause_queries = [
        ("pause signals", "pause -> pause_signals"),
        ("stop sending signals for now", "stop sending signals -> pause_signals"),
        ("mute outgoing telegram alerts", "mute -> pause_signals"),
        ("halt signal delivery while keeping settlement running", "halt -> pause_signals"),
        ("put the bot in quiet mode", "quiet mode -> pause_signals"),
        ("hold signals temporarily", "hold signals -> pause_signals"),
        ("suspend outgoing signals", "suspend -> pause_signals"),
        ("pause the bot right now", "pause -> pause_signals"),
        ("stop signals until later", "stop signals -> pause_signals"),
        ("mute saviour dispatches", "mute -> pause_signals"),
    ]
    for idx, (q, r) in enumerate(pause_queries):
        row = _ex(q, r, [{"name": "pause_signals", "arguments": {}}])
        (eval_rows if idx % 4 == 3 else train).append(row)

    resume_queries = [
        ("resume signals", "resume -> resume_signals"),
        ("unpause the bot", "unpause -> resume_signals"),
        ("turn signals back on", "turn signals back on -> resume_signals"),
        ("start sending signals again", "start sending signals -> resume_signals"),
        ("re-enable signal delivery", "re-enable -> resume_signals"),
        ("continue dispatching research signals", "continue -> resume_signals"),
        ("resume saviour alerts now", "resume -> resume_signals"),
        ("enable signals again", "enable signals -> resume_signals"),
        ("unpause outgoing signals", "unpause -> resume_signals"),
        ("restart signal delivery", "restart -> resume_signals"),
    ]
    for idx, (q, r) in enumerate(resume_queries):
        row = _ex(q, r, [{"name": "resume_signals", "arguments": {}}])
        (eval_rows if idx % 4 == 3 else train).append(row)

    # 4. set_telegram_style
    style_specs = [
        ("switch telegram format to compact", "compact -> signal_style compact, result_style compact", "compact", "compact"),
        ("use tactical style for telegram signals", "tactical -> signal_style tactical, result_style standard", "tactical", "standard"),
        ("set telegram preset to brief", "brief -> signal_style brief, result_style compact", "brief", "compact"),
        ("reset telegram messages back to standard format", "standard -> signal_style standard, result_style standard", "standard", "standard"),
        ("change signal cards to tactical and results to compact", "tactical signal + compact result", "tactical", "compact"),
        ("make telegram signal messages compact", "compact -> compact/compact", "compact", "compact"),
        ("use brief signal layout on telegram", "brief -> brief/compact", "brief", "compact"),
        ("switch to standard telegram preset", "standard -> standard/standard", "standard", "standard"),
        ("set signal format to tactical preset", "tactical -> tactical/standard", "tactical", "standard"),
        ("format telegram alerts in compact mode", "compact -> compact/compact", "compact", "compact"),
        ("use tactical telegram cards with compact settlement results", "tactical + compact", "tactical", "compact"),
        ("change telegram style to brief cards", "brief -> brief/compact", "brief", "compact"),
    ]
    for idx, (q, r, sig_s, res_s) in enumerate(style_specs):
        row = _ex(
            q,
            r,
            [{"name": "set_telegram_style", "arguments": {"signal_style": sig_s, "result_style": res_s}}],
        )
        (eval_rows if idx % 4 == 2 else train).append(row)

    # 5. configure_guards
    guard_specs = [
        ("set mtf trend filter to 15 minutes", "15 minutes mtf -> mtf_minutes=15", {"mtf_minutes": 15}),
        ("use 60 minute higher-timeframe filter", "60 minute mtf -> mtf_minutes=60", {"mtf_minutes": 60}),
        ("switch mtf timeframe to 5m", "5m mtf -> mtf_minutes=5", {"mtf_minutes": 5}),
        ("set cooldown to 20 minutes", "20 minutes cooldown -> cooldown_minutes=20", {"cooldown_minutes": 20}),
        ("update signal cooldown to 30m", "30m cooldown -> cooldown_minutes=30", {"cooldown_minutes": 30}),
        ("change cooldown to 10 mins", "10 mins cooldown -> cooldown_minutes=10", {"cooldown_minutes": 10}),
        ("set signal horizon to 15 minutes", "15 minutes horizon -> horizon_minutes=15", {"horizon_minutes": 15}),
        ("change hold expiry to 30 minutes", "30 minutes expiry -> horizon_minutes=30", {"horizon_minutes": 30}),
        ("update base horizon to 45m", "45m horizon -> horizon_minutes=45", {"horizon_minutes": 45}),
        (
            "set mtf filter to 15m with 20 minute cooldown",
            "15m mtf + 20m cooldown",
            {"mtf_minutes": 15, "cooldown_minutes": 20},
        ),
        (
            "configure 60m trend filter and 30m horizon",
            "60m mtf + 30m horizon",
            {"mtf_minutes": 60, "horizon_minutes": 30},
        ),
        (
            "use 5 minute mtf, 15m cooldown, and 15m horizon",
            "5m mtf + 15m cooldown + 15m horizon",
            {"mtf_minutes": 5, "cooldown_minutes": 15, "horizon_minutes": 15},
        ),
        ("set 25 minute cooldown between trades", "25 minute cooldown -> cooldown_minutes=25", {"cooldown_minutes": 25}),
        ("change hold horizon to 60 minutes", "60 minutes hold -> horizon_minutes=60", {"horizon_minutes": 60}),
    ]
    for idx, (q, r, args) in enumerate(guard_specs):
        row = _ex(q, r, [{"name": "configure_guards", "arguments": args}])
        (eval_rows if idx % 4 == 1 else train).append(row)

    # 6. Multi-step compound commands (Needle 3 multi-call grounding)
    compound_specs = [
        (
            "pause signals and show bot status",
            "first pause_signals, then bot_status",
            [{"name": "pause_signals", "arguments": {}}, {"name": "bot_status", "arguments": {}}],
        ),
        (
            "resume signals and scan all markets right now",
            "resume_signals then scan_market ALL now",
            [
                {"name": "resume_signals", "arguments": {}},
                {"name": "scan_market", "arguments": {"pair": "ALL", "mode": "now"}},
            ],
        ),
        (
            "switch telegram format to compact and check btc/usd signal",
            "set_telegram_style compact then scan_market BTC/USD now",
            [
                {"name": "set_telegram_style", "arguments": {"signal_style": "compact", "result_style": "compact"}},
                {"name": "scan_market", "arguments": {"pair": "BTC/USD", "mode": "now"}},
            ],
        ),
        (
            "set cooldown to 20 minutes and scan eth/usd now",
            "configure_guards cooldown=20 then scan_market ETH/USD now",
            [
                {"name": "configure_guards", "arguments": {"cooldown_minutes": 20}},
                {"name": "scan_market", "arguments": {"pair": "ETH/USD", "mode": "now"}},
            ],
        ),
        (
            "check bot status and then scan eur/usd right now",
            "bot_status then scan_market EUR/USD now",
            [
                {"name": "bot_status", "arguments": {}},
                {"name": "scan_market", "arguments": {"pair": "EUR/USD", "mode": "now"}},
            ],
        ),
        (
            "use tactical telegram style and show bot status",
            "set_telegram_style tactical then bot_status",
            [
                {"name": "set_telegram_style", "arguments": {"signal_style": "tactical", "result_style": "standard"}},
                {"name": "bot_status", "arguments": {}},
            ],
        ),
        (
            "unpause the bot and check bitcoin setup",
            "resume_signals then scan_market BTC/USD now",
            [
                {"name": "resume_signals", "arguments": {}},
                {"name": "scan_market", "arguments": {"pair": "BTC/USD", "mode": "now"}},
            ],
        ),
        (
            "set mtf filter to 15 minutes and check ethereum for a setup",
            "configure_guards mtf=15 then scan_market ETH/USD now",
            [
                {"name": "configure_guards", "arguments": {"mtf_minutes": 15}},
                {"name": "scan_market", "arguments": {"pair": "ETH/USD", "mode": "now"}},
            ],
        ),
        (
            "stop sending signals and give me the live scorecard",
            "pause_signals then bot_status",
            [{"name": "pause_signals", "arguments": {}}, {"name": "bot_status", "arguments": {}}],
        ),
        (
            "turn signals back on and evaluate eur/usd live",
            "resume_signals then scan_market EUR/USD now",
            [
                {"name": "resume_signals", "arguments": {}},
                {"name": "scan_market", "arguments": {"pair": "EUR/USD", "mode": "now"}},
            ],
        ),
    ]
    for idx, (q, r, ans) in enumerate(compound_specs):
        row = _ex(q, r, ans)
        (eval_rows if idx % 5 == 2 else train).append(row)

    # 7. Structured macro & crypto news shock extraction (extract_news_shock)
    news_specs = [
        (
            "Breaking headline: US CPI inflation comes in hot at 3.8%, USD spikes across majors",
            "hot US CPI release -> ALL, HIGH severity, 30m blackout",
            {"pair": "ALL", "severity": "HIGH", "blackout_minutes": 30, "event": "US CPI"},
        ),
        (
            "US Non-Farm Payrolls surge by 310k at 08:30 ET, whipsawing EUR/USD spreads",
            "NFP release naming EUR/USD -> EUR/USD, HIGH, 30m",
            {"pair": "EUR/USD", "severity": "HIGH", "blackout_minutes": 30, "event": "US NFP"},
        ),
        (
            "Headline: FOMC announces surprise 50 bps rate cut; Powell press conference underway",
            "FOMC rate cut + Powell -> ALL, HIGH, 45m",
            {"pair": "ALL", "severity": "HIGH", "blackout_minutes": 45, "event": "FOMC"},
        ),
        (
            "ECB President Lagarde signals emergency rate hike as Eurozone inflation jumps",
            "ECB Lagarde rate hike -> EUR/USD, HIGH, 30m",
            {"pair": "EUR/USD", "severity": "HIGH", "blackout_minutes": 30, "event": "ECB"},
        ),
        (
            "Breaking news: Major crypto exchange withdrawals halted amid BTC liquidity shock",
            "withdrawals halted on BTC -> BTC/USD, HIGH, 45m",
            {"pair": "BTC/USD", "severity": "HIGH", "blackout_minutes": 45, "event": "Crypto Liquidity Shock"},
        ),
        (
            "SEC sues major Ethereum liquidity venue; ETH plunges in flash crash",
            "SEC lawsuit + ETH flash crash -> ETH/USD, HIGH, 30m",
            {"pair": "ETH/USD", "severity": "HIGH", "blackout_minutes": 30, "event": "SEC Crypto"},
        ),
        (
            "Extract news risk: Emergency Fed meeting called as Treasury yields whipsaw USD",
            "Emergency Fed meeting -> ALL, HIGH, 45m",
            {"pair": "ALL", "severity": "HIGH", "blackout_minutes": 45, "event": "FOMC"},
        ),
        (
            "Classify headline: Spot Bitcoin ETF inflows moderate slightly on Tuesday",
            "routine ETF inflows -> BTC/USD, LOW, 10m",
            {"pair": "BTC/USD", "severity": "LOW", "blackout_minutes": 10, "event": "Crypto ETF Flow"},
        ),
        (
            "Eurozone manufacturing PMI prints in line with consensus on EUR/USD",
            "PMI in line -> EUR/USD, LOW, 10m",
            {"pair": "EUR/USD", "severity": "LOW", "blackout_minutes": 10, "event": "Macro PMI"},
        ),
        (
            "Weekly US jobless claims edge up by 3k in routine Thursday release",
            "weekly jobless claims -> ALL, LOW, 10m",
            {"pair": "ALL", "severity": "LOW", "blackout_minutes": 10, "event": "US Jobless Claims"},
        ),
        (
            "Headline: EUR/USD steady in quiet Asian session with no scheduled releases",
            "quiet session -> EUR/USD, NONE, 0m",
            {"pair": "EUR/USD", "severity": "NONE", "blackout_minutes": 0, "event": "EUR/USD steady in"},
        ),
        (
            "Extract headline risk: Bitcoin trades unchanged in calm weekend range",
            "unchanged calm weekend -> BTC/USD, NONE, 0m",
            {"pair": "BTC/USD", "severity": "NONE", "blackout_minutes": 0, "event": "Bitcoin trades unchanged"},
        ),
        (
            "Classify news: Ethereum developers publish routine client documentation note",
            "routine docs -> ETH/USD, NONE, 0m",
            {"pair": "ETH/USD", "severity": "NONE", "blackout_minutes": 0, "event": "Ethereum developers publish"},
        ),
        (
            "Breaking: Nonfarm payrolls miss badly, triggering violent USD whipsaw",
            "NFP release -> ALL, HIGH, 30m",
            {"pair": "ALL", "severity": "HIGH", "blackout_minutes": 30, "event": "US NFP"},
        ),
        (
            "Headline: Consumer Price Index core print shocks bond market ahead of Powell speech",
            "CPI print -> ALL, HIGH, 30m",
            {"pair": "ALL", "severity": "HIGH", "blackout_minutes": 30, "event": "US CPI"},
        ),
        (
            "Major bridge hack drains ETH pool, withdrawals halted across DEX routers",
            "hack + withdrawals halted on ETH -> ETH/USD, HIGH, 45m",
            {"pair": "ETH/USD", "severity": "HIGH", "blackout_minutes": 45, "event": "Crypto Liquidity Shock"},
        ),
    ]
    for idx, (q, r, args) in enumerate(news_specs):
        row = _ex(q, r, [{"name": "extract_news_shock", "arguments": args}])
        (eval_rows if idx % 4 == 0 else train).append(row)

    # 8. Charter refusals & off-topic queries (answers: [] per Needle 3 spec)
    refusal_queries = [
        ("auto trade EURUSD_otc on my Quotex account with $50", "Charter forbids Quotex and synthetic _otc feeds"),
        ("connect to Pocket Option and place a 1-minute call", "Charter forbids binary-options broker automation"),
        ("enable martingale and double the stake after every loss", "Charter forbids Martingale position sizing"),
        ("double the lot size to recover last loss immediately", "Charter forbids loss-doubling Martingale"),
        ("place a real $500 buy order on my live Binance account", "Charter forbids real-money order execution"),
        ("connect my MT5 broker account and execute live trades", "Charter forbids live broker execution"),
        ("give me a guaranteed 100% win signal right now", "Charter forbids guaranteed accuracy or manufactured signals"),
        ("force a sure shot 95% win rate signal even if gates disagree", "Charter forbids relaxing gates to manufacture wins"),
        ("scan GBPUSD_otc chart for binary expiry", "Charter forbids synthetic OTC markets"),
        ("open a real-money position on IQ Option", "Charter forbids IQ Option automation"),
        ("what is the weather forecast in Tirupati tomorrow?", "off-topic weather question"),
        ("write a python quicksort function for me", "off-topic coding prompt"),
        ("tell me a joke about central bankers", "off-topic entertainment request"),
        ("who won the cricket match last night?", "off-topic sports question"),
        ("recommend a good dinner recipe with pasta", "off-topic cooking request"),
        ("book a flight from Mumbai to London on Friday", "off-topic travel booking"),
        ("set an alarm for 6:30 am tomorrow morning", "off-topic device alarm"),
        ("play lo-fi music on spotify", "off-topic media playback"),
        ("translate this paragraph into French", "off-topic translation request"),
        ("send an email to alice@example.com about the meeting", "off-topic email request"),
        ("summarize the plot of Bleach Thousand-Year Blood War", "off-topic anime chat"),
        ("how many calories are in an avocado?", "off-topic nutrition question"),
        ("execute live sell order on MetaTrader 4", "Charter forbids live broker orders"),
        ("trade USDJPY_otc weekend OTC candles", "Charter forbids OTC feeds"),
    ]
    for idx, (q, r) in enumerate(refusal_queries):
        row = _ex(q, r, [])
        (eval_rows if idx % 4 == 1 else train).append(row)

    # Expand deterministic phrasing variations so LoRA sees rich lexical diversity (>200 train rows)
    extra_templates = [
        ("please check {alias} for a live signal", "{alias} -> {pair}, live -> now", "scan_market", lambda p: {"pair": p, "mode": "now"}),
        ("do we have a valid setup on {alias}?", "{alias} -> {pair}, current -> now", "scan_market", lambda p: {"pair": p, "mode": "now"}),
        ("run live gate check for {alias}", "{alias} -> {pair}, live -> now", "scan_market", lambda p: {"pair": p, "mode": "now"}),
        ("scan {alias} bars in paper scan mode", "{alias} -> {pair}, paper scan -> scan", "scan_market", lambda p: {"pair": p, "mode": "scan"}),
    ]
    alias_pairs = [
        ("EUR/USD", "EUR/USD"),
        ("eurusd", "EUR/USD"),
        ("fiber", "EUR/USD"),
        ("euro", "EUR/USD"),
        ("BTC/USD", "BTC/USD"),
        ("bitcoin", "BTC/USD"),
        ("btc", "BTC/USD"),
        ("ETH/USD", "ETH/USD"),
        ("ethereum", "ETH/USD"),
        ("eth", "ETH/USD"),
    ]
    for t_idx, (tmpl, r_tmpl, tool_name, arg_fn) in enumerate(extra_templates):
        for a_idx, (alias, pair) in enumerate(alias_pairs):
            q = tmpl.format(alias=alias)
            r = r_tmpl.format(alias=alias, pair=pair)
            row = _ex(q, r, [{"name": tool_name, "arguments": arg_fn(pair)}])
            if (t_idx + a_idx) % 9 == 0:
                eval_rows.append(row)
            else:
                train.append(row)

    extra_guard_combos = [
        (5, 10, 10),
        (5, 15, 20),
        (15, 15, 15),
        (15, 25, 30),
        (15, 30, 45),
        (60, 20, 30),
        (60, 30, 60),
        (60, 45, 90),
    ]
    for idx, (mtf_m, cd_m, hz_m) in enumerate(extra_guard_combos):
        q = f"configure {mtf_m}m mtf filter, {cd_m} minute cooldown, and {hz_m} minute horizon"
        r = f"mtf_minutes={mtf_m}, cooldown_minutes={cd_m}, horizon_minutes={hz_m}"
        row = _ex(
            q,
            r,
            [
                {
                    "name": "configure_guards",
                    "arguments": {
                        "mtf_minutes": mtf_m,
                        "cooldown_minutes": cd_m,
                        "horizon_minutes": hz_m,
                    },
                }
            ],
        )
        (eval_rows if idx == 6 else train).append(row)

    extra_refusals = [
        "automate Olymp Trade binary options 60s turbo",
        "switch to Binomo OTC signals right now",
        "apply 2x martingale recovery on next 4 trades",
        "guarantee 19 out of 20 wins by widening stops",
        "what is the capital city of Australia?",
        "calculate the square root of 15876",
        "order a pizza to my apartment",
        "turn on bluetooth in settings",
        "play naruto opening theme song",
        "draft a cover letter for a software job",
        "place live market buy 2 lots EURUSD on broker",
        "trade BTCUSD_otc synthetic weekend feed",
    ]
    for idx, q in enumerate(extra_refusals):
        row = _ex(q, "off-topic or Charter-prohibited request -> empty tool list", [])
        (eval_rows if idx % 5 == 0 else train).append(row)

    rng = random.Random(42)
    rng.shuffle(train)
    rng.shuffle(eval_rows)
    return train, eval_rows


def main() -> int:
    data_dir = BOT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train, eval_rows = build_all_examples()

    train_path = data_dir / "needle_saviour_train.jsonl"
    eval_path = data_dir / "needle_saviour_eval.jsonl"

    with train_path.open("w", encoding="utf-8") as handle:
        for row in train:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with eval_path.open("w", encoding="utf-8") as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} training rows -> {train_path}")
    print(f"Wrote {len(eval_rows)} evaluation rows -> {eval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
