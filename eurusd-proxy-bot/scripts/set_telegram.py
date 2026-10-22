#!/usr/bin/env python3
"""Put your Telegram bot token and owner chat id into .env.

    python scripts/set_telegram.py                      # asks for both
    python scripts/set_telegram.py --token T --chat C   # non-interactive
    python scripts/set_telegram.py --show               # what is set, masked
    python scripts/set_telegram.py --clear              # remove both

Where the two values come from:

* **token** — open a chat with @BotFather, send /newbot, follow the prompts, and
  it hands you a token that looks like `123456789:AAFxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`.
* **chat id** — open a chat with @userinfobot, send /start, and it replies with
  your numeric id. That number is the only chat the bot will talk to.

Both are written to .env, which is git-ignored, and the file is chmod 600. The
token is validated for shape before it is stored and never printed back in full.

Then check it works:
    python scripts/doctor.py --send
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion import envfile  # noqa: E402
from fusion.keypool import mask_key  # noqa: E402

ENV_PATH = ROOT / ".env"
TOKEN_KEY = "TELEGRAM_BOT_TOKEN"
CHAT_KEY = "TELEGRAM_OWNER_CHAT_ID"

TOKEN_SHAPE = re.compile(r"^(\d{5,}):([A-Za-z0-9_\-]+)$")
MIN_SECRET_LENGTH = 30
CHAT_SHAPE = re.compile(r"^-?\d+$")


# ------------------------------------------------------------------ validation
def validate_token(value: str) -> tuple[bool, str]:
    """Return (usable, note). A note is either a warning or a reason to refuse."""
    token = (value or "").strip().strip('"').strip("'")
    if not token:
        return False, "no token given"
    if " " in token or "\t" in token:
        return False, "the token contains whitespace — copy it as one piece"
    match = TOKEN_SHAPE.match(token)
    if not match:
        if ":" not in token:
            return False, "a bot token has one colon, like 123456789:AAF… — this has none"
        return False, "does not look like a bot token (expected digits, a colon, then the secret)"
    secret = match.group(2)
    if len(secret) < MIN_SECRET_LENGTH:
        return True, (
            f"kept, but the secret part is {len(secret)} characters; Telegram uses about 35 — "
            f"check the copy was not truncated"
        )
    return True, ""


def validate_chat_id(value: str) -> tuple[bool, str]:
    """Return (usable, note)."""
    chat = (value or "").strip().strip('"').strip("'")
    if not chat:
        return False, "no chat id given"
    if not CHAT_SHAPE.match(chat):
        return False, "the chat id must be a number — @userinfobot gives you one"
    if chat.startswith("-"):
        return True, "kept, but it is negative: that is a group chat, not your own id"
    return True, ""


# ------------------------------------------------------------------------ main
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Set the Telegram credentials in .env.")
    parser.add_argument("--token", help="the bot token from @BotFather")
    parser.add_argument("--chat", dest="chat_id", help="your numeric chat id from @userinfobot")
    parser.add_argument("--show", action="store_true", help="show what is set, masked")
    parser.add_argument("--clear", action="store_true", help="remove both values")
    args = parser.parse_args(argv)

    print()
    print("FUSION BOT — Telegram credentials")
    print("=" * 62)

    if not envfile.ensure_exists(ENV_PATH, ENV_PATH.with_name(".env.example")):
        print(f"  neither .env nor .env.example exists at {ENV_PATH.parent}")
        print("  are you running this from inside the eurusd-proxy-bot folder?")
        print()
        return 1

    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    if args.show:
        token = envfile.get_value(lines, TOKEN_KEY)
        chat = envfile.get_value(lines, CHAT_KEY)
        print(f"  bot token : {mask_key(token) if token else '(not set)'}")
        print(f"  owner chat: {chat if chat else '(not set)'}")
        print()
        print("  The token is never printed in full. Run with --send on the doctor")
        print("  to confirm delivery:  python scripts/doctor.py --send")
        print()
        return 0

    if args.clear:
        updated = envfile.remove_key(lines, TOKEN_KEY)
        updated = envfile.remove_key(updated, CHAT_KEY)
        envfile.write(ENV_PATH, updated)
        print("  both Telegram values removed from .env")
        print()
        return 0

    token = args.token
    chat = args.chat_id

    if token is None:
        print("  Paste the token @BotFather gave you (input is not echoed twice):")
        token = input("  token > ").strip()
    if chat is None:
        print("  Paste the numeric id @userinfobot gave you:")
        chat = input("  chat id > ").strip()

    token_ok, token_note = validate_token(token)
    chat_ok, chat_note = validate_chat_id(chat)

    if not token_ok or not chat_ok:
        print()
        print("  nothing was written; fix these and run again:")
        if not token_ok:
            print(f"    token  : {token_note}")
        if not chat_ok:
            print(f"    chat id: {chat_note}")
        print()
        return 1

    updated = envfile.set_value(lines, TOKEN_KEY, token.strip().strip('"').strip("'"))
    updated = envfile.set_value(updated, CHAT_KEY, chat.strip().strip('"').strip("'"))
    envfile.write(ENV_PATH, updated)

    print()
    print(f"  bot token : {mask_key(token.strip())}  written")
    print(f"  owner chat: {chat.strip()}  written")
    for note in (token_note, chat_note):
        if note:
            print(f"  note: {note}")
    print()
    print("  .env permissions set to owner-only where the filesystem allows it.")
    print()
    print("  Next:  python scripts/doctor.py --send")
    print("         confirms the bot is reachable and puts a test message in your chat.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
