"""The Supabase credential helper, and the .env auto-create it relies on."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fusion import envfile
from fusion.keypool import mask_key

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "set_supabase.py"


def _load():
    spec = importlib.util.spec_from_file_location("set_supabase", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


set_supabase = _load()

# A realistic-looking service_role JWT: header, payload, signature.
SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJvbGUiOiJzZXJ2aWNlX3JvbGUifQ." + "s" * 43
GOOD_URL = "https://abcdefghijklmnop.supabase.co"


class UrlValidationTests(unittest.TestCase):
    def test_accepts_a_project_url(self) -> None:
        ok, note = set_supabase.validate_url(GOOD_URL)
        self.assertTrue(ok)
        self.assertEqual("", note)

    def test_accepts_a_trailing_slash(self) -> None:
        ok, _ = set_supabase.validate_url(GOOD_URL + "/")
        self.assertTrue(ok)

    def test_rejects_plain_http(self) -> None:
        ok, note = set_supabase.validate_url("http://abcdefgh.supabase.co")
        self.assertFalse(ok)
        self.assertIn("https", note)

    def test_rejects_an_empty_value(self) -> None:
        ok, _ = set_supabase.validate_url("")
        self.assertFalse(ok)

    def test_rejects_whitespace(self) -> None:
        ok, note = set_supabase.validate_url(GOOD_URL + " extra")
        self.assertFalse(ok)
        self.assertIn("whitespace", note)

    def test_warns_but_keeps_a_custom_domain(self) -> None:
        ok, note = set_supabase.validate_url("https://db.mycompany.internal")
        self.assertTrue(ok)
        self.assertIn("custom-domain", note)


class ServiceKeyValidationTests(unittest.TestCase):
    def test_accepts_a_service_role_jwt(self) -> None:
        ok, note = set_supabase.validate_service_key(SERVICE_KEY)
        self.assertTrue(ok)
        self.assertEqual("", note)

    def test_rejects_the_anon_publishable_key(self) -> None:
        """The anon key can read nothing here: RLS is on with no policies."""
        ok, note = set_supabase.validate_service_key("sb_publishable_yEZ6hFlXmmRTUVFPMNnTqQ_xWE2NFnH")
        self.assertFalse(ok)
        self.assertIn("service_role", note)

    def test_rejects_an_empty_value(self) -> None:
        ok, _ = set_supabase.validate_service_key("")
        self.assertFalse(ok)

    def test_rejects_whitespace_inside(self) -> None:
        ok, note = set_supabase.validate_service_key("eyJabc.def ghi")
        self.assertFalse(ok)
        self.assertIn("whitespace", note)

    def test_warns_on_a_new_style_secret_key(self) -> None:
        ok, note = set_supabase.validate_service_key("sb_secret_KX9Irabcdefghijklmnop")
        self.assertTrue(ok)
        self.assertIn("legacy", note)

    def test_warns_on_an_unrecognised_shape(self) -> None:
        ok, note = set_supabase.validate_service_key("some-other-value")
        self.assertTrue(ok)
        self.assertIn("does not look like", note)


class EnsureExistsTests(unittest.TestCase):
    """A fresh clone has no .env; the scripts must create it, not refuse to run."""

    def test_creates_env_from_the_example(self) -> None:
        with TemporaryDirectory() as tmp:
            example = Path(tmp) / ".env.example"
            example.write_text("PAIR=EUR/USD\n", encoding="utf-8")
            target = Path(tmp) / ".env"
            self.assertTrue(envfile.ensure_exists(target, example))
            self.assertEqual("EUR/USD", envfile.get_value(target.read_text().splitlines(), "PAIR"))

    def test_does_not_overwrite_an_existing_env(self) -> None:
        with TemporaryDirectory() as tmp:
            example = Path(tmp) / ".env.example"
            example.write_text("PAIR=EUR/USD\n", encoding="utf-8")
            target = Path(tmp) / ".env"
            target.write_text("PAIR=EUR/GBP\nSECRET=keep\n", encoding="utf-8")
            self.assertTrue(envfile.ensure_exists(target, example))
            self.assertEqual("keep", envfile.get_value(target.read_text().splitlines(), "SECRET"))
            self.assertEqual("EUR/GBP", envfile.get_value(target.read_text().splitlines(), "PAIR"))

    def test_reports_failure_when_neither_file_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertFalse(
                envfile.ensure_exists(Path(tmp) / ".env", Path(tmp) / ".env.example")
            )


class StorageTests(unittest.TestCase):
    def test_both_values_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("PAIR=EUR/USD\n", encoding="utf-8")
            lines = path.read_text(encoding="utf-8").splitlines()
            lines = envfile.set_value(lines, set_supabase.URL_KEY, GOOD_URL)
            lines = envfile.set_value(lines, set_supabase.KEY_KEY, SERVICE_KEY)
            envfile.write(path, lines)

            final = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(GOOD_URL, envfile.get_value(final, set_supabase.URL_KEY))
            self.assertEqual(SERVICE_KEY, envfile.get_value(final, set_supabase.KEY_KEY))
            self.assertEqual("EUR/USD", envfile.get_value(final, "PAIR"))

    def test_the_masked_form_never_contains_the_key(self) -> None:
        shown = mask_key(SERVICE_KEY)
        self.assertNotIn(SERVICE_KEY, shown)
        self.assertNotIn(SERVICE_KEY.split(".", 1)[1], shown)


if __name__ == "__main__":
    unittest.main()
