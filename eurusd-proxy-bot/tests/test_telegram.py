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
        self.calls.append({"url": request.full_url, "data": (request.data or b"").decode()})
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

    def test_only_five_commands_exist(self):
        self.assertEqual(set(COMMANDS), {"/start", "/help", "/status", "/pause", "/resume"})


if __name__ == "__main__":
    unittest.main()
