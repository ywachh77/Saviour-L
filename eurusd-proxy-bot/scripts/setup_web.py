#!/usr/bin/env python3
"""A small local web form for entering every credential at once.

    python scripts/setup_web.py            # prints a URL with a one-time token
    python scripts/setup_web.py --local    # bind to 127.0.0.1 only (Termux)
    python scripts/setup_web.py --port 9000

Why a token
-----------

This form can read and write your service_role key, your Telegram bot token and
up to thirty market-data keys. Anything that can reach the port can therefore
reach all of them. A preview URL is guessable in principle, so the form is
served at a random 32-character path generated at startup and printed once.
Without it the server answers 404 to everything.

That is the whole access control. Treat the printed URL like a password, and do
not leave the server running once you are done.

What it deliberately does not do
--------------------------------

* **Never prints a secret to the terminal.** Existing values are shown masked,
  and only in the browser, and only to someone who already had the token.
* **Never sends a secret in a URL.** Everything arrives by POST, so credentials
  do not land in browser history, proxy logs or referrer headers.
* **Writes only .env**, which is git-ignored, atomically, chmod 600 — the same
  path every other credential script uses.
* **Validates before writing.** A rejected field writes nothing at all, so a
  typo cannot clobber a working key.

Stdlib only, so it runs unchanged on Termux.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion import charter, envfile  # noqa: E402
from fusion.keypool import mask_key  # noqa: E402

ENV_PATH = ROOT / ".env"

# Off by default. The random path is what keeps this safe on a shared network;
# serving the form at / as well removes that, so it is opt-in and only sensible
# somewhere the host itself is already private (a sandbox preview, a loopback
# bind). On Termux over shared wifi, leave it off and use --local.
ALLOW_ROOT = False


class config_defaults:
    """What the form shows when .env has no opinion.

    Mirrors fusion.config. It is stated here rather than imported because
    Config is built from os.environ, which is exactly the state this page must
    not trust (see _read_current).
    """

    HORIZON_MINUTES = 5
    ADAPTIVE_HORIZON = False
    SESSION_FILTER = True


def _on(value: str | None, default: bool) -> bool:
    """Interpret a .env toggle, falling back to the default when unset."""
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")

# Reuse the validators the command-line scripts use, so the web form and the
# terminal cannot drift into disagreeing about what a valid token looks like.
def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_set_supabase = _load("set_supabase", "set_supabase.py")
_set_telegram = _load("set_telegram", "set_telegram.py")

MAX_BODY = 256 * 1024
SHUTDOWN_AFTER_SAVE = True

FIELDS = (
    ("supabase_url", "Supabase project URL"),
    ("supabase_key", "Supabase service_role key"),
    ("telegram_token", "Telegram bot token"),
    ("telegram_chat", "Telegram owner chat id"),
    ("provider_keys", "Twelve Data keys (one per line, up to 30)"),
)

PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FUSION BOT setup</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 20px; max-width: 720px; margin-inline: auto;
         background: #14161a; color: #e6e6e6; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  h2 {{ font-size: 15px; margin: 26px 0 10px; padding-bottom: 6px;
        border-bottom: 1px solid #2a2f38; }}
  p.sub {{ color: #9aa2ae; margin: 0 0 4px; font-size: 13px; }}
  label {{ display: block; margin: 14px 0 5px; font-weight: 600; font-size: 14px; }}
  .hint {{ color: #8b93a0; font-weight: 400; font-size: 12px; }}
  input, textarea {{ width: 100%%; box-sizing: border-box; padding: 10px;
        background: #1c2027; color: #e6e6e6; border: 1px solid #2f3540;
        border-radius: 6px; font: inherit; }}
  textarea {{ min-height: 120px; font-family: ui-monospace, monospace; font-size: 13px; }}
  input:focus, textarea:focus {{ outline: 2px solid #4a90d9; border-color: #4a90d9; }}
  button {{ margin-top: 22px; padding: 12px 22px; font: inherit; font-weight: 600;
        background: #2f6feb; color: #fff; border: 0; border-radius: 6px; cursor: pointer; }}
  button.secondary {{ background: #2a2f38; }}
  .current {{ font-family: ui-monospace, monospace; font-size: 12px;
        color: #7fd08a; margin-top: 4px; }}
  .current.unset {{ color: #8b93a0; }}
  .ok {{ background: #14351f; border: 1px solid #2d6b3f; padding: 14px;
        border-radius: 6px; margin-top: 18px; }}
  .bad {{ background: #3a1a1a; border: 1px solid #7a2f2f; padding: 14px;
        border-radius: 6px; margin-top: 18px; }}
  ul {{ margin: 8px 0 0; padding-left: 20px; }}
  li {{ margin: 3px 0; }}
  code {{ background: #1c2027; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
</style>
<h1>FUSION BOT &mdash; credentials</h1>
<p class="sub">Saved to <code>.env</code>, which is git-ignored and never committed.
Values already set are shown masked. Leave a box blank to keep what is there.</p>
{result}
<form method="post" action="/{token}">
  <h2>Supabase</h2>
  <label>Project URL <span class="hint">&mdash; https://xxxx.supabase.co</span></label>
  <input name="supabase_url" value="{supabase_url}" placeholder="https://abcdefgh.supabase.co" autocomplete="off">
  <div class="current {s1}">{s1text}</div>
  <label>service_role key <span class="hint">&mdash; the long eyJ&hellip; one, not the anon key</span></label>
  <input type="password" name="supabase_key" placeholder="eyJhbGci&hellip;" autocomplete="off">
  <div class="current {s2}">{s2text}</div>

  <h2>Telegram</h2>
  <label>Bot token <span class="hint">&mdash; from @BotFather</span></label>
  <input type="password" name="telegram_token" placeholder="123456789:AAF&hellip;" autocomplete="off">
  <div class="current {s3}">{s3text}</div>
  <label>Owner chat id <span class="hint">&mdash; the number from @userinfobot</span></label>
  <input name="telegram_chat" value="{telegram_chat}" placeholder="987654321" autocomplete="off">
  <div class="current {s4}">{s4text}</div>

  <h2>Market data</h2>
  <label>Twelve Data keys <span class="hint">&mdash; one per line, up to 30</span></label>
  <textarea name="provider_keys" placeholder="one key per line" autocomplete="off"></textarea>
  <div class="current {s5}">{s5text}</div>

  <h2>Optional tuning</h2>
  <label>Round-turn spread in pips <span class="hint">&mdash; default 1.2</span></label>
  <input name="spread_pips" value="{spread_pips}" autocomplete="off">
  <label>Signal horizon in minutes <span class="hint">&mdash; default 5</span></label>
  <input name="horizon_minutes" value="{horizon_minutes}" autocomplete="off">
  <label><input type="checkbox" name="adaptive_horizon" {adaptive_horizon}
         style="width:auto;margin-right:8px"> Adaptive horizon (stretch holds that cannot pay their cost)</label>
  <label><input type="checkbox" name="session_filter" {session_filter}
         style="width:auto;margin-right:8px"> Session filter (Asia ranges, London trends)</label>

  <input type="hidden" name="_present" value="adaptive_horizon,session_filter">
  <button type="submit">Save to .env</button>
</form>
<p class="sub" style="margin-top:24px">When you are finished, stop this server
with Ctrl-C in the terminal it is running in.</p>
"""


def _mask_or_unset(value: str) -> tuple[str, str]:
    """(css class, display text) for a currently-stored secret."""
    if not value:
        return "unset", "not set"
    return "", f"current: {mask_key(value)}"


def _read_current() -> dict:
    """Read what is on disk.

    Deliberately does NOT go through fusion.config.load_config. That function
    seeds os.environ from .env but never overwrites a key that is already set,
    so in a long-lived process the first render would fix the values in
    os.environ and every later render would ignore whatever was just saved.
    """
    if not ENV_PATH.is_file():
        return {}
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    return {
        "supabase_url": envfile.get_value(lines, "SUPABASE_URL"),
        "supabase_key": envfile.get_value(lines, "SUPABASE_SERVICE_ROLE_KEY"),
        "telegram_token": envfile.get_value(lines, "TELEGRAM_BOT_TOKEN"),
        "telegram_chat": envfile.get_value(lines, "TELEGRAM_OWNER_CHAT_ID"),
        "spread_pips": envfile.get_value(lines, "SPREAD_PIPS"),
        "horizon_minutes": envfile.get_value(lines, "SIGNAL_HORIZON_MINUTES"),
        "adaptive_horizon": envfile.get_value(lines, "ADAPTIVE_HORIZON"),
        "session_filter": envfile.get_value(lines, "SESSION_FILTER"),
    }


class Handler(BaseHTTPRequestHandler):
    token = ""
    server_version = "fusion-setup"
    # Nothing a browser or scanner does should print a credential.
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, body: str, status: int = 200) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(payload)

    def _authorised(self) -> bool:
        path = self.path.split("?")[0].strip("/")
        if not path:
            return ALLOW_ROOT
        return path == self.token

    def do_GET(self) -> None:
        if not self._authorised():
            self._send("<meta name=viewport content='width=device-width,initial-scale=1'>"
                       "<p style='font-family:sans-serif'>Not found.</p>", 404)
            return
        self._send(self._render("", {}))

    def do_POST(self) -> None:
        if not self._authorised():
            self._send("Not found.", 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length > MAX_BODY:
            self._send("Too large.", 413)
            return
        raw = self.rfile.read(length).decode("utf-8")
        form = {key: values[0] for key, values in parse_qs(raw, keep_blank_values=True).items()}
        ok, message = self._save(form)
        self._send(self._render(message, form, ok=ok))
        if ok and SHUTDOWN_AFTER_SAVE:
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    # ------------------------------------------------------------------ saving
    def _save(self, form: dict) -> tuple[bool, str]:
        # Nothing touches the disk until the input is known to be good. Creating
        # .env up front would mean a fully rejected submission still left a file
        # behind, which makes "nothing was saved" a small lie.
        plan: list[tuple[str, str]] = []
        clears: list[str] = []
        saved: list[str] = []
        problems: list[str] = []

        # --- Supabase -------------------------------------------------------
        url = (form.get("supabase_url") or "").strip()
        if url:
            good, note = _set_supabase.validate_url(url)
            if not good:
                problems.append(f"Supabase URL: {note}")
            else:
                plan.append(("SUPABASE_URL", url.rstrip("/")))
                saved.append(f"Supabase URL &rarr; {html.escape(url)}")
                if note:
                    problems.append(f"Supabase URL kept, but: {note}")

        key = (form.get("supabase_key") or "").strip()
        if key:
            good, note = _set_supabase.validate_service_key(key)
            if not good:
                problems.append(f"service_role key: {note}")
            else:
                plan.append(("SUPABASE_SERVICE_ROLE_KEY", key))
                saved.append(f"service_role key &rarr; {html.escape(mask_key(key))}")
                if note:
                    problems.append(f"service_role key kept, but: {note}")

        # --- Telegram -------------------------------------------------------
        token = (form.get("telegram_token") or "").strip()
        if token:
            good, note = _set_telegram.validate_token(token)
            if not good:
                problems.append(f"bot token: {note}")
            else:
                plan.append(("TELEGRAM_BOT_TOKEN", token))
                saved.append(f"bot token &rarr; {html.escape(mask_key(token))}")
                if note:
                    problems.append(f"bot token kept, but: {note}")

        chat = (form.get("telegram_chat") or "").strip()
        if chat:
            good, note = _set_telegram.validate_chat_id(chat)
            if not good:
                problems.append(f"chat id: {note}")
            else:
                plan.append(("TELEGRAM_OWNER_CHAT_ID", chat))
                saved.append(f"owner chat id &rarr; {html.escape(chat)}")
                if note:
                    problems.append(f"chat id kept, but: {note}")

        # --- market data keys ------------------------------------------------
        raw_keys = (form.get("provider_keys") or "").strip()
        if raw_keys:
            entries = [line.strip() for line in raw_keys.splitlines()]
            entries = [entry for entry in entries if entry]
            if len(entries) > 30:
                problems.append(f"{len(entries)} keys given; the pool holds 30. None were saved.")
            else:
                # Clear the old pool first so a removed key does not linger.
                clears.extend(f"TWELVE_DATA_KEY_{i}" for i in range(1, 31))
                plan.append(("TWELVE_DATA_API_KEY", entries[0]))
                for index, entry in enumerate(entries[1:], start=1):
                    plan.append((f"TWELVE_DATA_KEY_{index}", entry))
                saved.append(f"{len(entries)} market-data key(s) &rarr; "
                             + ", ".join(html.escape(mask_key(entry)) for entry in entries[:4])
                             + ("&hellip;" if len(entries) > 4 else ""))

        # --- optional tuning --------------------------------------------------
        for field, env_name, low, high in (
            ("spread_pips", "SPREAD_PIPS", 0.0, 50.0),
            ("horizon_minutes", "SIGNAL_HORIZON_MINUTES", 1, 240),
        ):
            given = (form.get(field) or "").strip()
            if not given:
                continue
            try:
                value = float(given)
            except ValueError:
                problems.append(f"{field}: must be a number")
                continue
            if not (low <= value <= high):
                problems.append(f"{field}: must be between {low} and {high}")
                continue
            text = str(int(value)) if env_name.endswith("MINUTES") else str(value)
            plan.append((env_name, text))
            saved.append(f"{env_name} &rarr; {html.escape(text)}")

        # An unchecked checkbox submits nothing at all, so "off" and "not in
        # this form" look identical. Without a marker, saving one field would
        # silently switch every toggle the user left alone. _present lists the
        # toggles the page actually rendered.
        present = {
            item.strip()
            for item in (form.get("_present") or "").split(",")
            if item.strip()
        }
        for field, env_name in (("adaptive_horizon", "ADAPTIVE_HORIZON"),
                                ("session_filter", "SESSION_FILTER")):
            if field not in present:
                continue
            plan.append((env_name, "1" if form.get(field) else "0"))
            saved.append(f"{env_name} &rarr; {'1' if form.get(field) else '0'}")

        if problems:
            # A rejected field writes nothing: a typo must never clobber a
            # credential that already works.
            return False, ("<b>Nothing was saved. Fix these and try again:</b><ul>"
                           + "".join(f"<li>{html.escape(p)}</li>" for p in problems) + "</ul>")

        if not saved:
            return False, "Nothing to save &mdash; every box was blank."

        # Only now, with valid input in hand, do we touch the file.
        if not envfile.ensure_exists(ENV_PATH, ENV_PATH.with_name(".env.example")):
            return False, ("Could not find .env or .env.example. "
                           "Run this from inside the eurusd-proxy-bot folder.")
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
        for name in clears:
            lines = envfile.remove_key(lines, name)
        for name, value in plan:
            lines = envfile.set_value(lines, name, value)
        envfile.write(ENV_PATH, lines)
        return True, ("<b>Saved to .env</b><ul>"
                      + "".join(f"<li>{item}</li>" for item in saved) + "</ul>")

    # ----------------------------------------------------------------- render
    def _render(self, result: str, form: dict, ok: bool | None = None) -> str:
        current = _read_current()
        fields = {}
        for name, _label in FIELDS:
            stored = current.get(name, "")
            shown = ""
            if name in ("supabase_key", "telegram_token") and stored:
                shown = ""  # never pre-fill a secret into the page
            elif name == "provider_keys":
                shown = ""
            else:
                shown = form.get(name) or (stored if name in ("supabase_url", "telegram_chat") else "")
            fields[name] = html.escape(shown, quote=True)

        pairs = []
        for name in ("supabase_url", "supabase_key", "telegram_token",
                     "telegram_chat", "provider_keys"):
            if name == "provider_keys":
                count = _count_keys()
                pairs.append(("unset" if count == 0 else "",
                              f"{count} key(s) configured" if count else "not set"))
                continue
            pairs.append(_mask_or_unset(current.get(name, "")))

        banner = ""
        if result:
            cls = "ok" if ok else "bad"
            banner = f'<div class="{cls}">{result}</div>'

        return PAGE.format(
            token=self.token,
            result=banner,
            supabase_url=fields["supabase_url"],
            telegram_chat=fields["telegram_chat"],
            spread_pips=html.escape(form.get("spread_pips")
                                    or current.get("spread_pips")
                                    or str(charter.DEFAULT_SPREAD_PIPS), quote=True),
            horizon_minutes=html.escape(form.get("horizon_minutes")
                                        or current.get("horizon_minutes")
                                        or str(config_defaults.HORIZON_MINUTES), quote=True),
            adaptive_horizon="checked" if _on(current.get("adaptive_horizon"),
                                              config_defaults.ADAPTIVE_HORIZON) else "",
            session_filter="checked" if _on(current.get("session_filter"),
                                            config_defaults.SESSION_FILTER) else "",
            s1=pairs[0][0], s1text=html.escape(pairs[0][1]),
            s2=pairs[1][0], s2text=html.escape(pairs[1][1]),
            s3=pairs[2][0], s3text=html.escape(pairs[2][1]),
            s4=pairs[3][0], s4text=html.escape(pairs[3][1]),
            s5=pairs[4][0], s5text=html.escape(pairs[4][1]),
        )


def _count_keys() -> int:
    if not ENV_PATH.is_file():
        return 0
    lines = ENV_PATH.read_text().splitlines()
    count = 0
    if envfile.get_value(lines, "TWELVE_DATA_API_KEY"):
        count += 1
    for index in range(1, 31):
        if envfile.get_value(lines, f"TWELVE_DATA_KEY_{index}"):
            count += 1
    return count


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Local web form for credentials.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--local", action="store_true",
                        help="bind to 127.0.0.1 only; use this on Termux or shared wifi")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--allow-root", action="store_true",
                        help="also serve the form at / — only safe where the "
                             "host itself is private (sandbox preview, loopback)")
    args = parser.parse_args(argv)

    global ALLOW_ROOT
    ALLOW_ROOT = args.allow_root

    host = "127.0.0.1" if args.local else args.host
    token = secrets.token_urlsafe(24)

    handler = type("BoundHandler", (Handler,), {"token": token})
    server = ThreadingHTTPServer((host, args.port), handler)
    server.daemon_threads = True

    def say(text: str = "") -> None:
        # Flushed on purpose. Under a process manager stdout is a pipe and
        # Python buffers it, which would hide the one-time URL until exit.
        print(text, flush=True)

    say()
    say("FUSION BOT — credential setup")
    say("=" * 62)
    say()
    if ALLOW_ROOT:
        say("  open   http://localhost:%d/  (root is serving the form)" % args.port)
        say()
        say("  Root access is ON. Turn it off with --allow-root omitted")
        say("  before running this anywhere the host is reachable by others.")
        say()
    say(f"  path   /{token}")
    say()
    say(f"  local  http://localhost:{args.port}/{token}")
    if host not in ("127.0.0.1", "localhost"):
        # 0.0.0.0 is a bind address, not something a browser can open. In a
        # hosted preview the host is assigned from outside, so print the path
        # to append rather than a URL that cannot possibly work.
        say(f"  hosted https://<your-preview-host>/{token}")
        say("         (same path, on whatever host this sandbox is served from)")
    say()
    say("  That URL is the access control. Anyone who has it can read and")
    say("  overwrite your credentials, so do not share it, and stop the")
    say("  server with Ctrl-C once you are done.")
    say()
    if host not in ("127.0.0.1", "localhost"):
        say("  WARNING: bound to all interfaces. On shared wifi or a public")
        say("  network, restart with --local so only this device can reach it.")
        say()
    say("  Nothing you type is printed here. Existing values are shown")
    say("  masked in the browser only.")
    say()
    say("  Waiting for one save, then the server stops itself.")
    say()

    if not args.no_browser:
        try:
            webbrowser.open(f"http://localhost:{args.port}/{token}")
        except Exception:  # noqa: BLE001 - headless boxes have no browser
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
