"""Delivery honesty: SENT means a message id, never 'the request went out'."""

import unittest
from urllib.error import URLError

from fusion import charter
from fusion.telegram import COMMANDS, TelegramClient as Client


class FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.status = 200

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ScriptedTransport:
    """Returns canned responses; records what was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, request, timeout=None):
        self.calls.append(
            {
                "url": request.full_url,
                "data": (request.data or b"").decode(),
                "timeout": timeout,
            }
        )
        item = self.responses.pop(0) if self.responses else b'{"ok":true,"result":{"message_id":1}}'
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


class TestDelivery(unittest.TestCase):
    def setUp(self):
        self.transport = ScriptedTransport()
        self.client = Client("token", "12345", opener=self.transport)

    def test_sent_only_with_a_message_id(self):
        self.transport.responses = [b'{"ok":true,"result":{"message_id":991}}']
        result = self.client.send("hello")
        self.assertTrue(result.confirmed)
        self.assertEqual(result.state, charter.DELIVERY_SENT)
        self.assertEqual(result.message_id, "991")

    def test_api_rejection_is_a_failure_not_a_success(self):
        self.transport.responses = [b'{"ok":false,"description":"chat not found"}']
        result = self.client.send("hello")
        self.assertFalse(result.confirmed)
        self.assertEqual(result.state, charter.DELIVERY_FAILED)

    def test_no_message_id_is_unresolved_not_sent(self):
        self.transport.responses = [b'{"ok":true,"result":{}}']
        result = self.client.send("hello")
        self.assertFalse(result.confirmed)
        self.assertEqual(result.state, charter.DELIVERY_UNKNOWN)

    def test_timeout_is_unresolved_because_we_cannot_know(self):
        self.transport.responses = [URLError("timed out")]
        result = self.client.send("hello")
        self.assertFalse(result.confirmed)
        self.assertEqual(result.state, charter.DELIVERY_UNKNOWN)

    def test_http_error_is_a_clean_failure(self):
        from urllib.error import HTTPError

        self.transport.responses = [HTTPError("u", 403, "Forbidden", {}, None)]
        result = self.client.send("hello")
        self.assertEqual(result.state, charter.DELIVERY_FAILED)

    def test_unconfigured_client_sends_nothing(self):
        client = Client("", "")
        self.assertFalse(client.configured)
        result = client.send("hello")
        self.assertEqual(result.state, charter.DELIVERY_FAILED)
        self.assertEqual(client.poll_commands(), [])

    def test_messages_go_to_the_owner(self):
        self.client.send("hello")
        self.assertIn("chat_id=12345", self.transport.calls[0]["data"])

    def test_the_token_is_never_logged_in_the_message_body(self):
        self.client.send("hello")
        self.assertNotIn("token", self.transport.calls[0]["data"])


class TestCommands(unittest.TestCase):
    def setUp(self):
        self.transport = ScriptedTransport()
        self.client = Client("token", "12345", opener=self.transport)

    def _updates(self, chat_id: str, text: str) -> bytes:
        return (
            '{"ok":true,"result":[{"update_id":7,"message":{"text":"%s",'
            '"chat":{"id":%s}}}]}' % (text, chat_id)
        ).encode()

    def test_owner_commands_are_recognised(self):
        self.transport.responses = [self._updates("12345", "/status")]
        commands = self.client.poll_commands()
        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0].from_owner)
        self.assertEqual(commands[0].command, "/status")

    def test_strangers_are_ignored_but_still_acked(self):
        self.transport.responses = [self._updates("999", "/status")]
        commands = self.client.poll_commands()
        self.assertEqual(len(commands), 1)
        self.assertFalse(commands[0].from_owner)

    def test_offset_advances_so_a_command_is_never_handled_twice(self):
        self.transport.responses = [self._updates("12345", "/pause")]
        self.client.poll_commands()
        self.assertEqual(self.client.offset, 8)

    def test_non_command_messages_are_ignored(self):
        self.transport.responses = [self._updates("12345", "hello there")]
        self.assertEqual(self.client.poll_commands(), [])

    def test_the_command_set_is_exactly_what_the_owner_approved(self):
        """Five for months; a sixth by explicit request.

        The original brief fixed the surface at five commands and this test
        enforced it. The owner then asked for /signals by name, so the set is
        now six. It stays a closed list on purpose: every command is a way to
        reach the bot, so the list should grow only when asked, never drift.
        """
        self.assertEqual(
            set(COMMANDS),
            {"/start", "/help", "/signals", "/status", "/pause", "/resume"},
        )
        self.assertEqual(len(COMMANDS), 6)


class TestSilentFailureModes(unittest.TestCase):
    """A bot that cannot hear its owner must not look like a healthy one.

    Each of these failures used to end in ``return []``, so from the database
    a bot that was permanently deaf was indistinguishable from a bot that
    simply had nothing to say. That is what made a silent bot undiagnosable.
    """

    def setUp(self):
        self.transport = ScriptedTransport()
        self.client = Client("token", "12345", opener=self.transport)

    def _updates(self, chat_id: str, text: str) -> bytes:
        return (
            '{"ok":true,"result":[{"update_id":7,"message":{"text":"%s",'
            '"chat":{"id":%s}}}]}' % (text, chat_id)
        ).encode()

    def test_the_socket_outlives_the_poll_window(self):
        """A 20s poll killed by a 10s socket timeout never heard anything."""
        self.transport.responses = [self._updates("12345", "/status")]
        self.client.poll_commands(timeout_seconds=20)
        self.assertGreaterEqual(
            self.transport.calls[0]["timeout"],
            20.0,
            "the socket must stay open longer than the poll window it waits on",
        )

    def test_an_active_webhook_is_cleared_and_polling_continues(self):
        """Telegram refuses getUpdates while a webhook is set.

        There is no webhook handler here, so without clearing it the bot is
        deaf for as long as the webhook exists and reports nothing at all.
        """
        self.transport.responses = [
            b'{"ok":false,"description":"Conflict: terminated by other getUpdates"}',
            b'{"ok":true,"result":true}',
            self._updates("12345", "/status"),
        ]
        commands = self.client.poll_commands(timeout_seconds=5)
        self.assertTrue(self.client.webhook_cleared)
        self.assertIn("deleteWebhook", self.transport.calls[1]["url"])
        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0].from_owner)

    def test_webhook_recovery_is_attempted_only_once(self):
        conflict = b'{"ok":false,"description":"Conflict"}'
        self.transport.responses = [conflict, b'{"ok":false,"description":"nope"}', conflict]
        self.client.poll_commands(timeout_seconds=1)
        self.client.poll_commands(timeout_seconds=1)
        deletes = [c for c in self.transport.calls if "deleteWebhook" in c["url"]]
        self.assertEqual(len(deletes), 1)

    def test_a_poll_error_is_recorded_not_swallowed(self):
        self.transport.responses = [URLError("timed out")]
        self.assertEqual(self.client.poll_commands(timeout_seconds=5), [])
        self.assertTrue(self.client.last_poll_error)

    def test_a_stranger_is_recorded_so_a_wrong_owner_id_is_diagnosable(self):
        """Seeing the id that arrived is how a wrong secret gets found."""
        self.transport.responses = [self._updates("999", "/status")]
        self.client.poll_commands()
        self.assertEqual(len(self.client.ignored_senders), 1)
        self.assertEqual(self.client.ignored_senders[0]["chat_id"], "999")

    def test_the_owner_is_never_recorded_as_a_stranger(self):
        self.transport.responses = [self._updates("12345", "/status")]
        self.client.poll_commands()
        self.assertEqual(self.client.ignored_senders, [])

    def test_a_malformed_payload_records_an_error_instead_of_raising(self):
        """A dict where a list belongs used to raise inside the loop."""
        self.transport.responses = [b'{"ok":true,"result":{"message_id":1}}']
        self.assertEqual(self.client.poll_commands(), [])
        self.assertTrue(self.client.last_poll_error)

    def test_the_token_is_never_written_into_the_recorded_error(self):
        """The error is stored in the database, so it must stay clean."""
        client = Client("secret-token-abc", "12345", opener=ScriptedTransport(URLError("boom")))
        client.poll_commands(timeout_seconds=1)
        self.assertTrue(client.last_poll_error)
        self.assertNotIn("secret-token-abc", client.last_poll_error)


if __name__ == "__main__":
    unittest.main()
