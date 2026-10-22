#!/usr/bin/env python3
"""Put your Supabase project URL and service_role key into .env.

    python scripts/set_supabase.py                        # asks for both
    python scripts/set_supabase.py --url U --key K        # non-interactive
    python scripts/set_supabase.py --show                 # what is set, masked

Where the two values come from: dashboard.supabase.com → your project →
**Project Settings → API**:

* **Project URL** — looks like `https://abcdefgh.supabase.co`
* **service_role** key — the long `eyJhbGci...` one. Not the `anon` key, and not
  the newer `sb_secret_` key: this bot talks to PostgREST as the service role,
  because RLS is enabled on all 13 tables with no policies, so the anon key can
  read and write nothing at all.

These two values cannot live in the database itself — the bot needs them before
it can open its first connection. They stay in .env, git-ignored, chmod 600.
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
URL_KEY = "SUPABASE_URL"
KEY_KEY = "SUPABASE_SERVICE_ROLE_KEY"

URL_SHAPE = re.compile(r"^https://[A-Za-z0-9\-]+\.supabase\.(co|in|net)/?$", re.IGNORECASE)
# A Supabase legacy key is a JWT: three base64url segments separated by dots.
JWT_SHAPE = re.compile(r"^eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*$")


def validate_url(value: str) -> tuple[bool, str]:
    url = (value or "").strip().strip('"').strip("'").rstrip("/")
    if not url:
        return False, "no URL given"
    if " " in url:
        return False, "the URL contains whitespace"
    if not url.lower().startswith("https://"):
        return False, "the URL must start with https://"
    if not URL_SHAPE.match(url):
        return True, (
            "kept, but it is not the usual https://<ref>.supabase.co shape — "
            "a self-hosted or custom-domain project should still work"
        )
    return True, ""


def validate_service_key(value: str) -> tuple[bool, str]:
    key = (value or "").strip().strip('"').strip("'")
    if not key:
        return False, "no key given"
    if " " in key or "\t" in key:
        return False, "the key contains whitespace — copy it as one piece"
    if key.startswith("sb_publishable_"):
        return False, "that is the publishable (anon) key — this bot needs the service_role key"
    if JWT_SHAPE.match(key):
        return True, ""
    if key.startswith("sb_secret_"):
        return True, (
            "kept, but this is a new-style secret key; if the bot cannot connect, "
            "try the legacy service_role JWT from the same page"
        )
    return True, (
        "kept, but it does not look like a Supabase key (expected eyJ… or sb_secret_…) "
        "— check the whole value was copied"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Set the Supabase credentials in .env.")
    parser.add_argument("--url", help="your project URL")
    parser.add_argument("--key", dest="service_key", help="the service_role key")
    parser.add_argument("--show", action="store_true", help="show what is set, masked")
    args = parser.parse_args(argv)

    print()
    print("FUSION BOT — Supabase credentials")
    print("=" * 62)

    if not envfile.ensure_exists(ENV_PATH, ENV_PATH.with_name(".env.example")):
        print(f"  neither .env nor .env.example exists at {ENV_PATH.parent}")
        print("  are you running this from inside the eurusd-proxy-bot folder?")
        print()
        return 1

    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    if args.show:
        url = envfile.get_value(lines, URL_KEY)
        key = envfile.get_value(lines, KEY_KEY)
        print(f"  project URL  : {url if url else '(not set)'}")
        print(f"  service key  : {mask_key(key) if key else '(not set)'}")
        print()
        print("  The key is never printed in full.")
        print("  Check the connection with:  python scripts/doctor.py")
        print()
        return 0

    url = args.url
    key = args.service_key

    if url is None:
        print("  Dashboard → Project Settings → API → Project URL:")
        url = input("  url > ").strip()
    if key is None:
        print("  Same page → service_role key (the long eyJ… one, NOT the anon key):")
        key = input("  key > ").strip()

    url_ok, url_note = validate_url(url)
    key_ok, key_note = validate_service_key(key)

    if not url_ok or not key_ok:
        print()
        print("  nothing was written; fix these and run again:")
        if not url_ok:
            print(f"    url: {url_note}")
        if not key_ok:
            print(f"    key: {key_note}")
        print()
        return 1

    updated = envfile.set_value(lines, URL_KEY, url.strip().strip('"').strip("'").rstrip("/"))
    updated = envfile.set_value(updated, KEY_KEY, key.strip().strip('"').strip("'"))
    envfile.write(ENV_PATH, updated)

    print()
    print(f"  project URL : {url.strip()}  written")
    print(f"  service key : {mask_key(key.strip())}  written")
    for note in (url_note, key_note):
        if note:
            print(f"  note: {note}")
    print()
    print("  .env permissions set to owner-only where the filesystem allows it.")
    print()
    print("  Next:  python scripts/doctor.py")
    print("         checks the schema is reachable (it queries all 13 tables).")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
