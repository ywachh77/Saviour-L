#!/usr/bin/env python3
"""
Full Training Pipeline Orchestrator
====================================
Fetches real candles from Supabase → trains GBDT ensembles → runs walk-forward
backtest → generates summary report. Designed for GitHub Actions scheduled jobs
but works locally too.

Usage:
    python3 scripts/run_training_pipeline.py
    python3 scripts/run_training_pipeline.py --pairs EUR/USD,BTC/USD --candles 5000
    python3 scripts/run_training_pipeline.py --pair EUR/USD --candles 2000

Environment variables (set in GitHub Actions secrets):
    SUPABASE_URL:          e.g. https://abc.supabase.co
    SUPABASE_SERVICE_ROLE_KEY: service role key
    SUPABASE_PROJECT_REF:  optional, defaults to idudlcxkihqnvecwqqoj
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"

DEFAULT_PAIRS = "EUR/USD,BTC/USD,ETH/USD"
DEFAULT_CANDLESTICK_COUNT = 2000
BATCH_SIZE = 200


def pair_slug(pair: str) -> str:
    """Convert pair name to directory slug: 'EUR/USD' → 'EURUSD'."""
    return pair.replace("/", "").replace(" ", "")


def get_supabase_config() -> tuple[str, str, str]:
    """Get Supabase configuration from environment variables."""
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    project_ref = os.environ.get("SUPABASE_PROJECT_REF", "idudlcxkihqnvecwqqoj")

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")
        print("Set them as GitHub Secrets or environment variables.")
        sys.exit(1)

    return supabase_url, supabase_key, project_ref


def fetch_candles_for_pair(
    supabase_url: str,
    supabase_key: str,
    pair: str,
    num_candles: int,
    output_dir: Path,
) -> dict:
    """Fetch candles for a single pair and save to batch files.

    Returns summary dict with batch count, candle count, and timestamps.
    """
    if not HAS_REQUESTS:
        print("Error: 'requests' library not installed. Cannot fetch candles.")
        print("Install with: pip install requests")
        return {"pair": pair, "status": "error", "error": "requests not installed"}

    slug = pair_slug(pair)
    pair_dir = output_dir / slug
    pair_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    post_url = f"{supabase_url}/rest/v1/"

    print(f"\n{'='*60}")
    print(f"FETCHING {pair} — {num_candles} candles")
    print(f"{'='*60}")

    # Get total count
    count_query = f"SELECT count(*) as total FROM candles_m1 WHERE pair = '{pair}';"
    try:
        resp = requests.post(post_url, headers=headers, json={"query": count_query}, timeout=30)
        resp.raise_for_status()
        total = int(resp.json()[0]["total"])
        print(f"  Total in DB: {total}")
    except Exception as e:
        print(f"  Error checking count: {e}")
        return {"pair": pair, "status": "error", "error": str(e)}

    num_batches = (num_candles + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Fetching {num_batches} batches ({num_candles} candles)")

    fetched = 0
    batches_saved = 0
    start_ts = None
    end_ts = None

    for batch_idx in range(num_batches):
        offset = batch_idx * BATCH_SIZE
        # Use DESC order to get most recent candles first
        query = f"""
        SELECT string_agg(row_data, ';') as candles
        FROM (
            SELECT extract(epoch from timestamp)::bigint::text || ',' ||
                   open::text || ',' || high::text || ',' || low::text ||
                   ',' || close::text || ',' || volume::text as row_data
            FROM candles_m1
            WHERE pair = '{pair}'
            ORDER BY timestamp DESC
            LIMIT {BATCH_SIZE}
            OFFSET {offset}
        ) t;
        """

        try:
            resp = requests.post(post_url, headers=headers, json={"query": query}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data or not data[0].get("candles"):
                print(f"  Batch {batch_idx+1}/{num_batches}: empty")
                continue

            candles_str = data[0]["candles"]
            candles = candles_str.split(";")

            # Save to batch file
            batch_file = pair_dir / f"batch_real_{batch_idx}.txt"
            batch_file.write_text(candles_str)

            # Track timestamps
            first_parts = candles[0].split(",")
            last_parts = candles[-1].split(",")
            if start_ts is None:
                start_ts = int(float(first_parts[0]))
            end_ts = int(float(last_parts[0]))

            fetched += len(candles)
            batches_saved += 1
            print(f"  Batch {batch_idx+1}/{num_batches}: {len(candles)} candles ✓", flush=True)

        except Exception as e:
            print(f"  Batch {batch_idx+1}/{num_batches}: FAILED — {e}")

        if batch_idx < num_batches - 1:
            time.sleep(0.5)

    summary = {
        "pair": pair,
        "slug": slug,
        "status": "ok" if batches_saved > 0 else "error",
        "batches_saved": batches_saved,
        "candles_fetched": fetched,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "output_dir": str(pair_dir),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"  {pair}: {fetched} candles in {batches_saved} batches")
    return summary


def run_training(pair_slug_str: str, candle_dir: Path) -> dict:
    """Run model training for a pair using train_real_model.py logic.

    Imports and calls the training functions from train_real_model.py.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING MODELS — {pair_slug_str}")
    print(f"{'='*60}")

    # Import the training module
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from train_real_model import (
        load_real_candles,
        compute_features_all,
        build_samples,
        build_mr_samples,
        train_model,
        save_ensemble,
        HORIZON,
        HORIZON_1BAR,
        MIN_BARS,
        N_MODELS,
    )
    from fusion.features import FEATURE_NAMES, FEATURE_VERSION

    # Override CANDLE_DIR to point to the correct pair directory
    import train_real_model
    original_dir = train_real_model.CANDLE_DIR

    # If the pair subdirectory is empty (e.g. --skip-fetch with candles in parent),
    # fall back to the parent directory
    pair_candle_files = list(candle_dir.glob("batch_real_*.txt"))
    if not pair_candle_files:
        parent_candle_files = list(candle_dir.parent.glob("batch_real_*.txt"))
        if parent_candle_files:
            candle_dir = candle_dir.parent
            print(f"  Note: using candle files from parent directory ({candle_dir})")

    train_real_model.CANDLE_DIR = candle_dir

    candles = load_real_candles()
    print(f"  Loaded {len(candles)} candles")
    if len(candles) < 100:
        return {"pair": pair_slug_str, "status": "error", "error": "too few candles"}

    feat_dicts = compute_features_all(candles)
    print(f"  Computed features for {len(feat_dicts)} candles")

    samples = build_samples(candles, feat_dicts, HORIZON)
    samples_1bar = build_samples(candles, feat_dicts, HORIZON_1BAR)
    samples_mr = build_mr_samples(candles, feat_dicts)

    print(f"  Main: {len(samples)} samples, 1-bar: {len(samples_1bar)}, MR: {len(samples_mr)}")

    configs = [
        {"n_estimators": 50,  "max_depth": 3, "learning_rate": 0.10, "subsample": 0.7},
        {"n_estimators": 60,  "max_depth": 3, "learning_rate": 0.15, "subsample": 0.8},
        {"n_estimators": 40,  "max_depth": 4, "learning_rate": 0.12, "subsample": 0.6},
        {"n_estimators": 80,  "max_depth": 3, "learning_rate": 0.08, "subsample": 0.9},
        {"n_estimators": 30,  "max_depth": 2, "learning_rate": 0.20, "subsample": 0.5},
    ]

    results = {"pair": pair_slug_str, "n_candles": len(candles)}

    # Helper to convert vectors to dicts
    def vec_to_dict(vec):
        return {name: val for name, val in zip(FEATURE_NAMES, vec)}

    # Train main model
    print("  Training main model...")
    models_main, accs_main = [], []
    for i, cfg in enumerate(configs):
        m, a = train_model(samples, cfg, seed=42 + i)
        models_main.append(m)
        accs_main.append(a)
    save_ensemble(models_main, samples, configs, accs_main,
                  candle_dir / "ensemble_model.json", HORIZON, f"{pair_slug_str}_main")
    results["main_accuracy"] = round(sum(accs_main) / len(accs_main), 4)

    # Train 1-bar model
    print("  Training 1-bar model...")
    models_1bar, accs_1bar = [], []
    for i, cfg in enumerate(configs):
        m, a = train_model(samples_1bar, cfg, seed=42 + i)
        models_1bar.append(m)
        accs_1bar.append(a)
    save_ensemble(models_1bar, samples_1bar, configs, accs_1bar,
                  candle_dir / "ensemble_model_1bar.json", HORIZON_1BAR, f"{pair_slug_str}_1bar")
    results["1bar_accuracy"] = round(sum(accs_1bar) / len(accs_1bar), 4)

    # Train MR model
    print("  Training MR model...")
    models_mr, accs_mr = [], []
    for i, cfg in enumerate(configs):
        m, a = train_model(samples_mr, cfg, seed=42 + i)
        models_mr.append(m)
        accs_mr.append(a)
    save_ensemble(models_mr, samples_mr, configs, accs_mr,
                  candle_dir / "ensemble_model_mr.json", HORIZON, f"{pair_slug_str}_mr")
    results["mr_accuracy"] = round(sum(accs_mr) / len(accs_mr), 4)

    # Ensemble evaluation
    def ensemble_eval(samples_list, models_list):
        correct = 0
        for x, y in samples_list:
            x_dict = vec_to_dict(x) if isinstance(x, list) else x
            pred = sum(m.predict_up(x_dict) for m in models_list) / len(models_list)
            if (pred > 0.5) == bool(y):
                correct += 1
        return correct / len(samples_list) if samples_list else 0.0

    results["ensemble_accuracy"] = {
        "main": round(ensemble_eval(samples, models_main), 4),
        "1bar": round(ensemble_eval(samples_1bar, models_1bar), 4),
        "mr": round(ensemble_eval(samples_mr, models_mr), 4),
    }
    results["status"] = "ok"

    # Save training report
    report = {
        "pair": pair_slug_str,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_candles": len(candles),
        "ensemble_accuracy": results["ensemble_accuracy"],
        "per_model_accuracy": {
            "main": [round(a, 4) for a in accs_main],
            "1bar": [round(a, 4) for a in accs_1bar],
            "mr": [round(a, 4) for a in accs_mr],
        },
        "feature_version": FEATURE_VERSION,
        "n_features": len(FEATURE_NAMES),
    }
    (candle_dir / "training_report.json").write_text(json.dumps(report, indent=2))

    # Restore original dir
    train_real_model.CANDLE_DIR = original_dir

    print(f"  Main:  {results['ensemble_accuracy']['main']:.1%}")
    print(f"  1-bar: {results['ensemble_accuracy']['1bar']:.1%}")
    print(f"  MR:    {results['ensemble_accuracy']['mr']:.1%}")
    return results


def run_backtest(pair_slug_str: str, candle_dir: Path, candles_limit: int) -> dict:
    """Run backtest for a pair using backtest_engine.py."""
    print(f"\n{'='*60}")
    print(f"BACKTESTING — {pair_slug_str}")
    print(f"{'='*60}")

    # Fall back to parent directory if no candle files in pair subdirectory
    if not list(candle_dir.glob("batch_*.txt")):
        parent_files = list(candle_dir.parent.glob("batch_*.txt"))
        if parent_files:
            candle_dir = candle_dir.parent
            print(f"  Note: using candle files from parent directory ({candle_dir})")

    # Run backtest_engine.py as subprocess
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "backtest_engine.py"),
        "--all-expiries",
        "--candles", str(candles_limit),
        "--candle-dir", str(candle_dir),
    ]

    # Set environment to pass through
    env = os.environ.copy()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )

    # Print backtest output
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(f"  STDERR: {result.stderr[-1000:]}")
        print(f"  Backtest exited with code {result.returncode}")

    # Parse backtest report
    report_file = candle_dir / "backtest_report.json"
    if report_file.exists():
        report = json.loads(report_file.read_text())
        return {"pair": pair_slug_str, "status": "ok", "report": report}
    else:
        error_msg = "no backtest report generated"
        if result.returncode != 0:
            error_msg = f"backtest exited with code {result.returncode}"
        print(f"  {pair_slug_str} backtest FAILED: {error_msg}")
        return {
            "pair": pair_slug_str,
            "status": "error",
            "error": error_msg,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }


def generate_summary(results: list, output_dir: Path) -> dict:
    """Generate a summary report of all pairs."""
    summary = {
        "pipeline_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_pairs": len(results),
        "pairs": {},
    }

    for r in results:
        slug = r.get("pair", "unknown")

        # Determine overall status
        statuses = []
        if r.get("status") == "error":
            statuses.append("fetch_error")
        if r.get("training"):
            if r["training"].get("status") == "error":
                statuses.append("train_error")
            elif r["training"].get("status") == "ok":
                statuses.append("ok")
        if r.get("backtest"):
            if r["backtest"].get("status") == "error":
                statuses.append("backtest_error")
            elif r["backtest"].get("status") == "ok":
                statuses.append("ok")

        overall = "ok" if "ok" in statuses and "error" not in [s for s in statuses if "error" in s] else (
            "partial" if "ok" in statuses else "failed"
        )

        train = r.get("training", {}) or {}
        bt = r.get("backtest", {}) or {}
        bt_report = bt.get("report", {}) or {}

        summary["pairs"][slug] = {
            "status": overall,
            "candles": r.get("candles_fetched", r.get("n_candles", 0)) or train.get("n_candles", 0),
            "training": {
                "main_accuracy": (train.get("ensemble_accuracy", {}) or {}).get("main"),
                "1bar_accuracy": (train.get("ensemble_accuracy", {}) or {}).get("1bar"),
                "mr_accuracy": (train.get("ensemble_accuracy", {}) or {}).get("mr"),
            },
            "backtest": bt_report,
            "errors": {
                "fetch": r.get("error"),
                "backtest": bt.get("error"),
                "backtest_stderr": bt.get("stderr", "")[:500] if bt.get("stderr") else "",
            },
        }

    summary_file = output_dir / "pipeline_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*60}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*60}")
    for slug, info in summary["pairs"].items():
        status = info["status"]
        if status in ("ok", "partial"):
            print(f"  {slug}: [{status}]")
            print(f"    Candles: {info['candles']}")
            main = info["training"]["main_accuracy"]
            bar = info["training"]["1bar_accuracy"]
            mr = info["training"]["mr_accuracy"]
            if main:
                print(f"    Main: {main:.1%}")
            if bar:
                print(f"    1-bar: {bar:.1%}")
            if mr:
                print(f"    MR: {mr:.1%}")
            bt = info.get("backtest", {})
            if bt:
                for exp, data in bt.items():
                    if isinstance(data, dict) and "win_rate" in data:
                        print(f"    {exp} win rate: {data['win_rate']:.1%}")
        else:
            print(f"  {slug}: FAILED")
            errors = info.get("errors", {})
            for k, v in errors.items():
                if v:
                    print(f"    {k}: {str(v)[:100]}")
    print(f"\n  Summary: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Full training pipeline orchestrator")
    parser.add_argument("--pair", action="append", default=None,
                        help="Pair to process (can be repeated). Default: all")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Comma-separated pairs. Default: EUR/USD,BTC/USD,ETH/USD")
    parser.add_argument("--candles", type=int, default=DEFAULT_CANDLESTICK_COUNT,
                        help=f"Candles per pair (default: {DEFAULT_CANDLESTICK_COUNT})")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip candle fetching, use existing files")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip backtest")
    parser.add_argument("--output", type=str, default=str(CANDLE_DIR),
                        help="Output directory")
    args = parser.parse_args()

    # Determine pairs
    if args.pairs:
        pairs = [p.strip() for p in args.pairs.split(",")]
    elif args.pair:
        pairs = args.pair
    else:
        pairs = [p.strip() for p in DEFAULT_PAIRS.split(",")]

    print("=" * 60)
    print("FULL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Pairs: {pairs}")
    print(f"Candles per pair: {args.candles}")
    print(f"Output: {args.output}")
    print()

    # Get Supabase config (only if fetching)
    if args.skip_fetch:
        supabase_url, supabase_key, project_ref = "", "", ""
        print("Skipping fetch — using existing candle files")
    else:
        supabase_url, supabase_key, project_ref = get_supabase_config()
        print(f"Supabase: {supabase_url} (ref: {project_ref})")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for pair in pairs:
        slug = pair_slug(pair)
        pair_dir = output_dir / slug
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Fetch candles
        if args.skip_fetch:
            print(f"Skipping fetch for {pair} (using existing files)")
            fetch_summary = {"pair": slug, "status": "skipped", "candles_fetched": 0}
        else:
            fetch_summary = fetch_candles_for_pair(
                supabase_url, supabase_key, pair, args.candles, output_dir
            )
        all_results.append(fetch_summary)

        # Step 2: Train models
        if args.skip_train:
            print(f"Skipping training for {pair}")
            train_summary = {"pair": slug, "status": "skipped"}
        else:
            train_summary = run_training(slug, pair_dir)
            all_results[-1]["training"] = train_summary

        # Step 3: Run backtest
        if args.skip_backtest:
            print(f"Skipping backtest for {pair}")
            backtest_summary = {"pair": slug, "status": "skipped"}
        else:
            backtest_summary = run_backtest(slug, pair_dir, args.candles)
            all_results[-1]["backtest"] = backtest_summary

        # Small delay between pairs
        if pair != pairs[-1]:
            time.sleep(1)

    # Generate summary
    summary = generate_summary(all_results, output_dir)

    # Print final status
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Pairs processed: {len(pairs)}")
    print(f"Summary: {output_dir / 'pipeline_summary.json'}")

    # Exit with error if all pairs failed
    all_failed = all(
        r.get("status") == "error" for r in all_results
    )
    sys.exit(1 if all_failed else 0)


if __name__ == "__main__":
    main()
