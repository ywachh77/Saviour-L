#!/usr/bin/env python3
"""Add (or replace) your Twelve Data keys in one command.

    python scripts/set_keys.py              # paste them, blank line to finish
    python scripts/set_keys.py keys.txt     # one key per line
    python scripts/set_keys.py --show       # what is configured, masked
    python scripts/set_keys.py --clear      # remove every key

Why this exists rather than typing thirty lines by hand: editing a .env file on
a phone is miserable, and one typo in one line is a slot that silently never
rotates in. This reads them once, de-duplicates, caps at the pool maximum and
rewrites the block cleanly.

The keys land in .env, which is git-ignored, and the file is chmod 600 so other
apps on the device cannot read it. Nothing here ever prints a key back: the
summary and --show both display masked labels only.

One key is enough to run. Slots 1..30 exist for keys you have *legitimately*
provisioned — do not create multiple free accounts to work around Twelve Data
rate limits. After changing keys, push them to Actions with:
    ./scripts/push_secrets.sh
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion import envfile  # noqa: E402
from fusion.keypool import MAX_SLOTS, mask_key  # noqa: E402

ENV_PATH = ROOT / ".env"
SINGLE_VAR = "TWELVE_DATA_API_KEY"
SLOT_VAR = re.compile(r"^#?\s*TWELVE_DATA_KEY_(\d+)\s*=")
SINGLE_LINE = re.compile(r"^\s*TWELVE_DATA_API_KEY\s*=")
BOUNDARY = "Reminder: keep these to keys you have legitimately provisioned."
# The reminder this script writes must be recognised on the way back in,
# otherwise every run appends another copy and the file grows a line at a time.
REMINDER_LINE = re.compile(r"^#\s*Reminder: keep these to keys you have legitimately provisioned")

MIN_KEY_LENGTH = 8
TYPICAL_KEY_LENGTH = 32


# --------------------------------------------------------------------- parsing
def normalise_keys(raw_lines) -> tuple[list[str], list[str]]:
    """Return (keys, problems). Blank lines and # comments are skipped.

    Order is preserved and duplicates are dropped, because the pool rotates in
    the order keys were given and a duplicate would waste a slot.
    """
    keys: list[str] = []
    problems: list[str] = []
    seen: set[str] = set()

    for number, raw in enumerate(raw_lines, start=1):
        line = str(raw).strip()
        if not line or line.startswith("#"):
            continue
        value = line.strip().strip('"').strip("'").strip()
        if not value:
            continue
        if re.search(r"\s", value):
            problems.append(f"line {number}: ignored, the value contains whitespace")
            continue
        if len(value) < MIN_KEY_LENGTH:
            problems.append(f"line {number}: ignored, shorter than {MIN_KEY_LENGTH} characters")
            continue
        if len(value) != TYPICAL_KEY_LENGTH:
            problems.append(
                f"line {number}: kept, but {len(value)} characters is not the usual "
                f"{TYPICAL_KEY_LENGTH} — check it was copied whole"
            )
        if value in seen:
            problems.append(f"line {number}: ignored, duplicate of an earlier key")
            continue
        seen.add(value)
        keys.append(value)

    overflow = []
    if len(keys) > MAX_SLOTS:
        overflow = keys[MAX_SLOTS:]
        keys = keys[:MAX_SLOTS]
        problems.append(f"only the first {MAX_SLOTS} keys were kept; {len(overflow)} ignored")
    return keys, problems


def rewrite_key_block(lines: list[str], keys: list[str]) -> list[str]:
    """Return .env lines with the Twelve Data key block replaced by ``keys``.

    Pure, so it can be tested without touching a real .env. Everything outside
    the key block — comments, Telegram, Supabase, tuning — is left untouched.
    """
    kept: list[str] = []
    insert_at: int | None = None

    for index, line in enumerate(lines):
        if SLOT_VAR.match(line) or SINGLE_LINE.match(line):
            if insert_at is None:
                insert_at = len(kept)
            continue
        if REMINDER_LINE.match(line):
            # Owned by this script, so it is rewritten rather than preserved.
            if insert_at is None:
                insert_at = len(kept)
            continue
        kept.append(line)

    if insert_at is None:
        # No key block yet: put it at the end, on its own section.
        while kept and not kept[-1].strip():
            kept.pop()
        kept.append("")
        kept.append("# ------------------------------------------------------------ Twelve Data")
        insert_at = len(kept)

    block: list[str] = []
    if not keys:
        block.append(f"{SINGLE_VAR}=")
    elif len(keys) == 1:
        block.append(f"{SINGLE_VAR}={keys[0]}")
    else:
        # With several keys the numbered slots are used and the single variable
        # is left empty, so there is exactly one place keys come from.
        block.append(f"{SINGLE_VAR}=")
        for index, key in enumerate(keys, start=1):
            block.append(f"TWELVE_DATA_KEY_{index}={key}")
    block.append(f"# {BOUNDARY}")

    return kept[:insert_at] + block + kept[insert_at:]


def read_keys_from_env(path: Path) -> list[str]:
    """Return the keys currently configured in a .env file."""
    if not path.is_file():
        return []
    keys: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if SINGLE_LINE.match(line):
            value = line.partition("=")[2].strip()
            if value:
                keys.append(value)
        elif match := SLOT_VAR.match(line):
            value = line.partition("=")[2].strip()
            if value and not line.lstrip().startswith("#"):
                keys.append(value)
    # Order-independent de-duplication, matching the loader in config.py.
    unique: list[str] = []
    for key in keys:
        if key not in unique:
            unique.append(key)
    return unique


def write_env(path: Path, lines: list[str]) -> None:
    """Write .env through the shared writer: atomic, and owner-only."""
    envfile.write(path, lines)


# ------------------------------------------------------------------------ main
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=True, description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", help="a file with one key per line (omit to paste)")
    parser.add_argument("--show", action="store_true", help="show what is configured, masked")
    parser.add_argument("--clear", action="store_true", help="remove every Twelve Data key")
    args = parser.parse_args(argv)

    print()
    print("FUSION BOT — Twelve Data keys")
    print("=" * 62)

    if not ENV_PATH.is_file():
        print(f"  no .env at {ENV_PATH}")
        print("  copy .env.example to .env first, then run this again.")
        print()
        return 1

    if args.show:
        keys = read_keys_from_env(ENV_PATH)
        print(f"  {len(keys)} key(s) configured")
        for index, key in enumerate(keys, start=1):
            print(f"    {index:>2}. {mask_key(key)}")
        if not keys:
            print("    (none — the bot will run without a market feed)")
        print()
        print("  Keys are never printed in full. Run without --show to replace them.")
        print()
        return 0

    if args.clear:
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
        write_env(ENV_PATH, rewrite_key_block(lines, []))
        print("  all Twelve Data keys removed from .env")
        print("  run ./scripts/push_secrets.sh to update GitHub Actions.")
        print()
        return 0

    if args.source:
        source_path = Path(args.source).expanduser()
        if not source_path.is_file():
            print(f"  no such file: {source_path}")
            print()
            return 1
        raw = source_path.read_text(encoding="utf-8").splitlines()
    else:
        print("  Paste your keys, one per line. Blank line when done.")
        print("  (Nothing is echoed as you type, and nothing here is stored in git.)")
        print()
        raw = []
        try:
            while True:
                line = input()
                if not line.strip():
                    break
                raw.append(line)
        except EOFError:
            pass

    keys, problems = normalise_keys(raw)

    print()
    if not keys:
        print("  no usable keys found; .env was left unchanged.")
        for problem in problems:
            print(f"    - {problem}")
        print()
        return 1

    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    write_env(ENV_PATH, rewrite_key_block(lines, keys))

    print(f"  {len(keys)} key(s) written to .env")
    for index, key in enumerate(keys, start=1):
        print(f"    {index:>2}. {mask_key(key)}")
    for problem in problems:
        print(f"  note: {problem}")
    print()
    print(f"  .env permissions set to owner-only where the filesystem allows it.")
    print("  Next: ./scripts/push_secrets.sh   (copies these into GitHub Secrets)")
    print()
    print("  One path through one history: more keys raise your rate limit,")
    print("  they do not make the strategy profitable.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
