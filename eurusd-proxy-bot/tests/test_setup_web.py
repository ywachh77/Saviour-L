"""The credential web form: access control and what it must never do.

This form can read and overwrite every secret the bot has, so these tests are
mostly about the things it must refuse or avoid rather than the things it does.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "setup_web.py"


def _load():
    spec = importlib.util.spec_from_file_location("setup_web", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup_web = _load()

FAKE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSJ9." + "s" * 43
FAKE_TOKEN = "123456789:AAF" + "k" * 32
GOOD = {
    "supabase_url": "https://abcdefghijklmnop.supabase.co",
    "supabase_key": FAKE_KEY,
    "telegram_token": FAKE_TOKEN,
    "telegram_chat": "987654321",
    "_present": "adaptive_horizon,session_filter",
}


class Harness:
    """Runs the real handler on a real socket against a temporary .env."""

    def __init__(self, env_path: Path) -> None:
        from http.server import ThreadingHTTPServer

        # The handler resolves .env.example next to whatever ENV_PATH points at,
        # so the temp directory needs one for the first save to have a template.
        shutil.copyfile(ROOT / ".env.example", env_path.with_name(".env.example"))
        setup_web.ENV_PATH = env_path
        setup_web.SHUTDOWN_AFTER_SAVE = False  # keep it alive across requests
        self.token = "testtoken"
        handler = type("H", (setup_web.Handler,), {"token": self.token})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str | None = None) -> str:
        if path is None:
            path = self.token
        return f"http://127.0.0.1:{self.port}/{path.lstrip('/')}"

    def get(self, path: str | None = None) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(self.url(path)) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def post(self, fields: dict) -> tuple[int, str]:
        body = urllib.parse.urlencode(fields).encode()
        request = urllib.request.Request(self.url(), data=body)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class WebFormTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.env = self.tmp / ".env"
        self.web = Harness(self.env)

    def tearDown(self) -> None:
        self.web.stop()
        self._tmp.cleanup()


class AccessControlTests(WebFormTests):
    def test_the_root_path_is_a_404(self) -> None:
        self.assertEqual(404, self.web.get("/")[0])

    def test_a_guessed_path_is_a_404(self) -> None:
        self.assertEqual(404, self.web.get("admin")[0])
        self.assertEqual(404, self.web.get("setup")[0])

    def test_the_token_path_serves_the_form(self) -> None:
        status, body = self.web.get()
        self.assertEqual(200, status)
        self.assertIn("<form", body)

    def test_root_is_refused_by_default(self) -> None:
        """On a shared network the random path is the only thing keeping this
        safe, so / must not serve the form unless it is explicitly allowed."""
        self.assertEqual(404, self.web.get("/")[0])

    def test_root_serves_the_form_only_when_allowed(self) -> None:
        setup_web.ALLOW_ROOT = True
        try:
            status, body = self.web.get("/")
            self.assertEqual(200, status)
            self.assertIn("<form", body)
        finally:
            setup_web.ALLOW_ROOT = False

    def test_the_landing_page_never_reveals_the_token(self) -> None:
        """A safe landing page must stay safe: no token anywhere in the 404."""
        _, body = self.web.get("/")
        self.assertNotIn(self.web.token, body)


class LeakTests(WebFormTests):
    def test_the_page_never_contains_a_stored_secret(self) -> None:
        self.web.post(GOOD)
        _, body = self.web.get()
        self.assertNotIn(FAKE_KEY, body)
        self.assertNotIn(FAKE_TOKEN, body)

    def test_secrets_are_shown_masked(self) -> None:
        self.web.post(GOOD)
        _, body = self.web.get()
        self.assertIn("eyJh...ss", body)
        self.assertIn("1234...kk", body)

    def test_secret_inputs_are_never_prefilled(self) -> None:
        """A prefilled password field would put the live key in the page."""
        self.web.post(GOOD)
        _, body = self.web.get()
        for match in re.finditer(r'<input[^>]*name="(supabase_key|telegram_token)"[^>]*>', body):
            self.assertNotIn("value=", match.group(0))

    def test_the_page_sets_no_referrer(self) -> None:
        """Without this, a browser could leak the token to a linked resource."""
        connection = HTTPConnection("127.0.0.1", self.web.port)
        connection.request("GET", f"/{self.web.token}")
        response = connection.getresponse()
        self.assertEqual("no-referrer", response.getheader("Referrer-Policy"))
        response.read()
        connection.close()


class ValidationTests(WebFormTests):
    def test_a_rejected_submission_creates_nothing(self) -> None:
        """Creating .env up front would make 'nothing was saved' a small lie."""
        status, body = self.web.post({"supabase_url": "nope"})
        self.assertEqual(200, status)
        self.assertIn("Nothing was saved", body)
        self.assertFalse(self.env.exists())

    def test_a_rejected_submission_keeps_the_existing_file(self) -> None:
        self.web.post(GOOD)
        before = self.env.read_text()
        self.web.post({"telegram_chat": "@myhandle"})
        self.assertEqual(before, self.env.read_text())

    def test_the_anon_key_is_refused(self) -> None:
        _, body = self.web.post({"supabase_key": "sb_publishable_AAAAAAAAAAAAAAAA"})
        self.assertIn("publishable", body)

    def test_a_malformed_bot_token_is_refused(self) -> None:
        _, body = self.web.post({"telegram_token": "nocolon"})
        self.assertIn("colon", body)

    def test_valid_input_is_written(self) -> None:
        self.web.post(GOOD)
        from fusion import envfile

        lines = self.env.read_text().splitlines()
        self.assertEqual("https://abcdefghijklmnop.supabase.co",
                         envfile.get_value(lines, "SUPABASE_URL"))
        self.assertEqual(FAKE_KEY, envfile.get_value(lines, "SUPABASE_SERVICE_ROLE_KEY"))
        self.assertEqual(FAKE_TOKEN, envfile.get_value(lines, "TELEGRAM_BOT_TOKEN"))
        self.assertEqual("987654321", envfile.get_value(lines, "TELEGRAM_OWNER_CHAT_ID"))

    def test_a_non_numeric_spread_is_refused(self) -> None:
        _, body = self.web.post({"spread_pips": "abc"})
        self.assertIn("must be a number", body)

    def test_an_out_of_range_horizon_is_refused(self) -> None:
        _, body = self.web.post({"horizon_minutes": "9999"})
        self.assertIn("must be between", body)


class KeyPoolTests(WebFormTests):
    def test_several_keys_land_in_their_own_slots(self) -> None:
        keys = "\n".join(f"{chr(97 + i)}" * 32 for i in range(5))
        self.web.post({"provider_keys": keys})
        from fusion import envfile

        lines = self.env.read_text().splitlines()
        self.assertEqual("a" * 32, envfile.get_value(lines, "TWELVE_DATA_API_KEY"))
        self.assertEqual("e" * 32, envfile.get_value(lines, "TWELVE_DATA_KEY_4"))

    def test_replacing_the_pool_clears_old_slots(self) -> None:
        self.web.post({"provider_keys": "\n".join("b" * 32 for _ in range(6))})
        self.web.post({"provider_keys": "c" * 32})
        from fusion import envfile

        lines = self.env.read_text().splitlines()
        self.assertEqual("c" * 32, envfile.get_value(lines, "TWELVE_DATA_API_KEY"))
        self.assertEqual("", envfile.get_value(lines, "TWELVE_DATA_KEY_1"))

    def test_more_than_thirty_keys_is_refused(self) -> None:
        keys = "\n".join("d" * 32 for _ in range(31))
        _, body = self.web.post({"provider_keys": keys})
        self.assertIn("30", body)
        self.assertFalse(self.env.exists())


class StaleEnvironmentTests(WebFormTests):
    """The server is long-lived, so it must never trust os.environ.

    fusion.config seeds os.environ from .env but skips any key already present.
    A page that rendered before a save would therefore keep showing the old
    value forever. The form reads the file instead.
    """

    def test_a_save_is_visible_after_the_page_was_rendered_first(self) -> None:
        from fusion import envfile

        envfile.ensure_exists(self.env, self.tmp / ".env.example")
        lines = envfile.set_value(self.env.read_text().splitlines(),
                                  "ADAPTIVE_HORIZON", "0")
        envfile.write(self.env, lines)

        self.web.get()  # seeds os.environ with the old value
        self.web.post(dict(GOOD, adaptive_horizon="on"))
        _, body = self.web.get()
        self.assertRegex(body, r'name="adaptive_horizon"[^>]*checked')


class ToggleTests(WebFormTests):
    def test_checkboxes_render_the_stored_state(self) -> None:
        self.web.post(dict(GOOD, adaptive_horizon="on"))
        _, body = self.web.get()
        self.assertRegex(body, r'name="adaptive_horizon"[^>]*checked')

    def test_unticking_a_box_turns_it_off(self) -> None:
        self.web.post(dict(GOOD, adaptive_horizon="on"))
        self.web.post(dict(_present="adaptive_horizon,session_filter"))
        from fusion import envfile

        value = envfile.get_value(self.env.read_text().splitlines(), "ADAPTIVE_HORIZON")
        self.assertEqual("0", value)

    def test_a_submission_without_the_marker_leaves_toggles_alone(self) -> None:
        """Protects against a stale cached page silently turning everything off."""
        self.web.post(dict(GOOD, adaptive_horizon="on"))
        self.web.post({"provider_keys": "e" * 32})  # no _present
        from fusion import envfile

        value = envfile.get_value(self.env.read_text().splitlines(), "ADAPTIVE_HORIZON")
        self.assertEqual("1", value)


if __name__ == "__main__":
    unittest.main()
