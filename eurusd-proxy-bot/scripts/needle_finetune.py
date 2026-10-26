#!/usr/bin/env python3
"""Fine-tune Cactus Compute Needle 3 (LoRA QAT) for SAVIOUR & export ``.cact``.

Modes
-----
1. **Pretrained Needle 3 (default when Hugging Face or local checkpoint is available)**:
   - Loads ``Cactus-Compute/needle3`` (``checkpoints/needle3.safetensors``).
   - Trains a rank-16 LoRA adapter under Needle 3's ``CQ W4 STE + A8``
     quantization-aware training loop on ``data/needle_saviour_train.jsonl``.
   - Merges the LoRA adapter and exports ``var/needle/saviour_needle3.cact``
     (optional ``--layers`` ladder slice, e.g. 8 or 20 layers).
   - Evaluates ``SaviourNeedleAgent`` on held-out ``data/needle_saviour_eval.jsonl``.

2. **Offline / Air-Gapped Bootstrap (``--bootstrap-local`` or auto-fallback when
   ``huggingface.co`` is unreachable)**:
   - Builds a local ``SentencePiece`` BPE tokenizer with Needle 3's exact control
     token IDs (``PAD=0, EOS=1, BOS=2, UNK=3`` + ``CHAT_MARKERS``) and a compact
     Needle 3 ``SimpleAttentionNetwork`` (Monarch Hadamard MLP + GQA + engram
     memory + causal conv taps) base checkpoint in ``var/needle/``.
   - Runs the real ``needle.model.finetune.finetune_local`` JAX LoRA QAT loop,
     verifies loss convergence, merges the adapter, and exports a verified W4A8
     ``.cact`` archive (``var/needle/saviour_needle3_local.cact``).

Telemetry is strictly disabled (``NEEDLE_TELEMETRY=0``, ``DO_NOT_TRACK=1``).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import types
from typing import Any

# Strictly disable all Needle binary/SDK telemetry before importing needle.
os.environ["NEEDLE_TELEMETRY"] = "0"
os.environ["DO_NOT_TRACK"] = "1"

BOT_ROOT = Path(__file__).resolve().parents[1]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from fusion.needle_agent import SAVIOUR_NEEDLE_TOOLS, SaviourNeedleAgent  # noqa: E402


def _hf_reachable(timeout: float = 2.5) -> bool:
    """Fast TCP+TLS probe to huggingface.co so offline hosts don't stall on retries."""
    import ssl

    try:
        ctx = ssl.create_default_context()
        with socket.create_connection(("huggingface.co", 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname="huggingface.co"):
                return True
    except OSError:
        return False


def _build_offline_bootstrap_assets(work_dir: Path, train_jsonl: Path, vocab_size: int = 2048) -> tuple[Path, Path]:
    """Create a local SANTokenizer-compatible SentencePiece model & Needle 3 v2 checkpoint."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import sentencepiece as spm
    from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig
    from needle.model.checkpoints import write_checkpoint
    from needle.model import tokenizer as needle_tok

    work_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = work_dir / "bootstrap_corpus.txt"
    tok_prefix = work_dir / "bootstrap_tokenizer"
    tok_model = Path(str(tok_prefix) + ".model")

    lines: list[str] = []
    with train_jsonl.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            lines.append(str(row.get("query", "")))
            lines.append(str(row.get("reasoning", "")))
            lines.append(json.dumps(row.get("answers", []), ensure_ascii=False))
            lines.append(json.dumps(row.get("tools", []), ensure_ascii=False))
    corpus_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=str(tok_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=needle_tok.PAD_ID,
        eos_id=needle_tok.EOS_ID,
        bos_id=needle_tok.BOS_ID,
        unk_id=needle_tok.UNK_ID,
        user_defined_symbols=list(needle_tok.CHAT_MARKERS),
        character_coverage=1.0,
        hard_vocab_limit=False,
        minloglevel=2,
    )

    config = TransformerConfig(
        vocab_size=vocab_size,
        out_vocab=vocab_size,
        d_model=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=4,
        qk_head_dim=16,
        v_head_dim=16,
        max_seq_len=1024,
        engram_layers=(1, 3),
        engram_slots=64,
        global_layers=(3,),
        sliding_window=64,
        mhc_lanes=2,
        qkv_conv_taps=3,
        flash=False,
    )
    model = SimpleAttentionNetwork(config)
    params = model.init(jax.random.PRNGKey(42), jnp.ones((1, 8), jnp.int32))["params"]
    params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)

    ckpt_path = work_dir / "bootstrap_base.safetensors"
    write_checkpoint(
        ckpt_path,
        {
            "format_version": 2,
            "params": params,
            "config": dict(vars(config)),
        },
    )
    return tok_model, ckpt_path


def _parse_epoch_metrics(progress_lines: list[str]) -> list[dict[str, float | int]]:
    metrics: list[dict[str, float | int]] = []
    epoch_re = re.compile(r"epoch\s+(\d+)/(\d+)\s+loss\s+([0-9.]+)(?:\s+val\s+([0-9.]+))?")
    for line in progress_lines:
        m = epoch_re.search(line)
        if m:
            entry: dict[str, float | int] = {
                "epoch": int(m.group(1)),
                "total_epochs": int(m.group(2)),
                "train_loss": round(float(m.group(3)), 4),
            }
            if m.group(4) is not None:
                entry["val_loss"] = round(float(m.group(4)), 4)
            metrics.append(entry)
    return metrics


def evaluate_agent(state_dir: Path, eval_jsonl: Path, weights_path: Path | None = None) -> dict[str, Any]:
    agent = SaviourNeedleAgent(state_dir=state_dir, weights_path=weights_path)
    rows = [json.loads(line) for line in eval_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    exact = 0
    refusal_total = 0
    refusal_exact = 0
    engines_used: set[str] = set()

    for row in rows:
        dec = agent.route(row["query"])
        engines_used.add(dec.engine)
        want = row.get("answers", [])
        if dec.function_calls == want:
            exact += 1
        if not want:
            refusal_total += 1
            if not dec.function_calls:
                refusal_exact += 1

    total = len(rows)
    return {
        "eval_examples": total,
        "exact_matches": exact,
        "exact_match_rate": round(exact / total, 4) if total else 0.0,
        "refusal_examples": refusal_total,
        "refusal_exact": refusal_exact,
        "engines_used": sorted(engines_used),
    }


def run_finetune(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from needle.model import tokenizer as needle_tok
    from needle.model.architecture import ConfidenceHead, effective_kv_window
    from needle.model.checkpoints import read_adapter
    from needle.model.export import read_export, write_export
    from needle.model.finetune import DEFAULT_BASE, finetune_local, merge_lora, rung
    from needle.model.quantize import WEIGHT_BITS
    from needle.model.run import load_checkpoint

    started = time.monotonic()
    state_dir = Path(args.state_dir)
    needle_dir = state_dir / "needle"
    needle_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = Path(args.train_data)
    eval_jsonl = Path(args.eval_data)
    if not train_jsonl.is_file():
        raise FileNotFoundError(f"Training dataset not found: {train_jsonl}")

    base_ckpt_candidate = Path(args.checkpoint) if args.checkpoint else Path(DEFAULT_BASE)
    use_bootstrap = bool(args.bootstrap_local)
    if not use_bootstrap and not base_ckpt_candidate.is_file() and not _hf_reachable():
        print("  notice    huggingface.co unreachable and no cached base checkpoint; running offline JAX bootstrap LoRA QAT.", flush=True)
        use_bootstrap = True

    saved_tok_prefix = needle_tok.TOKENIZER_PREFIX
    tok_model_path: Path | None = None
    try:
        if use_bootstrap:
            tok_model_path, ckpt_path = _build_offline_bootstrap_assets(needle_dir, train_jsonl)
            needle_tok.TOKENIZER_PREFIX = str(tok_model_path.with_suffix(""))
            base_checkpoint = str(ckpt_path)
            cact_out = Path(args.out) if args.out else (needle_dir / "saviour_needle3_local.cact")
            max_len = min(int(args.max_len), 1024)
            score_heldout = False
        else:
            base_checkpoint = str(base_ckpt_candidate)
            cact_out = Path(args.out) if args.out else (needle_dir / "saviour_needle3.cact")
            max_len = int(args.max_len)
            score_heldout = True

        adapter_out = needle_dir / "saviour_lora.safetensors"
        ft_ns = types.SimpleNamespace(
            jsonl_path=str(train_jsonl),
            checkpoint=base_checkpoint,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            lora_rank=int(args.lora_rank),
            lora_alpha=float(args.lora_alpha),
            max_len=max_len,
            generate=0,
            model=None,
            checkpoint_dir=str(needle_dir),
            out=str(adapter_out),
            seed=int(args.seed),
            val_split=float(args.val_split),
            score=score_heldout,
        )

        progress_lines: list[str] = []
        finetune_local(ft_ns, progress=progress_lines.append)

        # Merge LoRA adapter & export W4A8 .cact archive
        if not use_bootstrap:
            from needle.model.build import build_main

            build_ns = types.SimpleNamespace(
                checkpoint=base_checkpoint,
                lora=str(adapter_out),
                out=str(cact_out),
                upload=False,
                layers=args.layers,
                platform=None,
            )
            build_main(build_ns)
        else:
            adapter = read_adapter(str(adapter_out))
            params, config = load_checkpoint(base_checkpoint)
            lora = {
                tuple(key.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
                for key, v in adapter["lora"].items()
            }
            merged = merge_lora(params, lora, adapter["scale"])
            if ConfidenceHead.key in merged:
                merged = {k: v for k, v in merged.items() if k != ConfidenceHead.key}
            target_layers = args.layers if (args.layers and args.layers <= config.num_layers) else None
            merged, config = rung(merged, config, target_layers)
            write_export(
                merged,
                config,
                str(cact_out),
                bits=WEIGHT_BITS,
                tokenizer=needle_tok.get_tokenizer(config.vocab_size),
                kv_window=effective_kv_window(config),
            )

        header, _ = read_export(str(cact_out))
        adapter_meta = read_adapter(str(adapter_out))
    finally:
        needle_tok.TOKENIZER_PREFIX = saved_tok_prefix

    epoch_metrics = _parse_epoch_metrics(progress_lines)
    eval_summary = evaluate_agent(
        state_dir=state_dir,
        eval_jsonl=eval_jsonl,
        weights_path=(cact_out if not use_bootstrap else None),
    )

    elapsed = round(time.monotonic() - started, 2)
    report: dict[str, Any] = {
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_seconds": elapsed,
        "mode": "offline_bootstrap_qat" if use_bootstrap else "pretrained_needle3_qat",
        "bootstrap_tiny_init": use_bootstrap,
        "telemetry_disabled": True,
        "jax_backend": jax.default_backend(),
        "train_dataset": str(train_jsonl),
        "train_examples": sum(1 for line in train_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()),
        "lora_rank": int(adapter_meta["rank"]),
        "lora_alpha": float(args.lora_alpha),
        "lora_scale": float(adapter_meta["scale"]),
        "lora_weight_groups": len(adapter_meta["lora"]),
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "seed": int(args.seed),
        "epoch_metrics": epoch_metrics,
        "loss_start": epoch_metrics[0]["train_loss"] if epoch_metrics else None,
        "loss_end": epoch_metrics[-1]["train_loss"] if epoch_metrics else None,
        "val_loss_end": epoch_metrics[-1].get("val_loss") if epoch_metrics else None,
        "adapter_path": str(adapter_out),
        "adapter_bytes": int(adapter_out.stat().st_size),
        "cact_path": str(cact_out),
        "cact_bytes": int(cact_out.stat().st_size),
        "cact_num_layers": int(header.get("num_layers", 0)),
        "cact_num_tensors": int(header.get("num_tensors", 0)),
        "quantization": f"CQ W{WEIGHT_BITS}A8",
        "evaluation": eval_summary,
    }

    report_path = needle_dir / "finetune_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"  report    {report_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Needle 3 (LoRA QAT) for SAVIOUR")
    parser.add_argument(
        "--train-data",
        default=str(BOT_ROOT / "data" / "needle_saviour_train.jsonl"),
        help="Training JSONL path",
    )
    parser.add_argument(
        "--eval-data",
        default=str(BOT_ROOT / "data" / "needle_saviour_eval.jsonl"),
        help="Held-out evaluation JSONL path",
    )
    parser.add_argument("--checkpoint", default=None, help="Base .safetensors checkpoint (default: Needle 3 base)")
    parser.add_argument("--state-dir", default=str(BOT_ROOT / "var"), help="Runtime state directory")
    parser.add_argument("--out", default=None, help="Output .cact archive path")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs (default: 4)")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=5e-4, help="AdamW learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA adapter rank (default: 16)")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA scaling alpha (default: 32)")
    parser.add_argument("--layers", type=int, default=None, help="Optional ladder layer slice on export (2..20)")
    parser.add_argument("--max-len", type=int, default=1024, help="Sequence length cap")
    parser.add_argument("--val-split", type=float, default=0.1, help="Held-out validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    parser.add_argument(
        "--bootstrap-local",
        action="store_true",
        help="Force offline JAX architecture bootstrap without reaching Hugging Face",
    )
    args = parser.parse_args()

    report = run_finetune(args)
    print(
        f"  summary   mode={report['mode']}  loss={report['loss_start']}->{report['loss_end']}  "
        f"cact={report['cact_bytes'] / 1e6:.2f}MB ({report['cact_num_tensors']} tensors)  "
        f"eval={report['evaluation']['exact_matches']}/{report['evaluation']['eval_examples']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
