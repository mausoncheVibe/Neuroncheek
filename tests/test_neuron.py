"""Offline tests: real SDK/Telethon types, fake Telegram I/O, no credentials."""
import asyncio
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from telethon import errors, types

import neuron as n

TARGET, OWNER = 12345678, 87654321


def event(mid=10, text="привет", sender=TARGET, chat=TARGET, private=True, out=False):
    return NS(id=mid, raw_text=text, message=text, sender_id=sender,
              chat_id=chat, is_private=private, out=out)


def message(mid, text="привет", sender=TARGET, chat=TARGET, out=False):
    return NS(id=mid, message=text, sender_id=sender, chat_id=chat, out=out)


def response(text="Привет! Как день?"):
    return NS(choices=[NS(message=NS(content=text))])


def api_error(status, headers=None):
    req = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return APIStatusError("PRIVATE_SERVER_BODY", response=httpx.Response(
        status, request=req, headers=headers), body={"secret": "PRIVATE_SERVER_BODY"})


class ConfigAndStateTests(unittest.TestCase):
    def test_invalid_id_fail_closed(self):
        for bad in (None, 0, -1, "all", "123", True, 1.0, "@name", "", [], 2**63):
            with self.subTest(bad=bad), self.assertRaises(n.ConfigError):
                n.Settings.parse({"allowed_user_id": bad})
        with self.assertRaises(n.ConfigError):
            n.Settings.parse({})

    def test_other_settings_fail_closed(self):
        for key, bad in (("enabled", "true"), ("show_typing", 1),
                         ("initiate_first", "false"), ("debounce_seconds", float("nan")),
                         ("history_messages", False), ("max_output_tokens", 100000),
                         ("api_timeout_seconds", -1), ("model", ""), ("unknown", True)):
            with self.subTest(key=key), self.assertRaises(n.ConfigError):
                n.Settings.parse({"allowed_user_id": TARGET, key: bad})

    def test_disabled_needs_no_id(self):
        self.assertFalse(n.Settings.parse({"enabled": False}).enabled)

    def test_shipped_settings_intentionally_fail_closed(self):
        with self.assertRaises(n.ConfigError):
            n.Settings.load(n.ROOT / "settings.json")

    def test_state_roundtrip_atomic_and_no_chat_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = n.State(OWNER, TARGET, awaiting_reply=True, last_proactive=10)
            with patch.object(n.os, "replace", wraps=n.os.replace) as replace_file:
                state.save(path)
                replace_file.assert_called_once()
                self.assertEqual(Path(replace_file.call_args.args[0]).parent, path.parent)
            self.assertEqual(n.State.load(path, OWNER, TARGET), state)
            self.assertFalse(list(path.parent.glob("*.tmp")))
            self.assertNotIn("content", path.read_text())
            self.assertNotIn("key", path.read_text())

    def test_atomic_failure_preserves_old_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = n.State(OWNER, TARGET)
            state.save(path)
            before = path.read_bytes()
            state.awaiting_reply = True
            with patch.object(n.os, "replace", side_effect=OSError), self.assertRaises(OSError):
                state.save(path)
            self.assertEqual(before, path.read_bytes())
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_state_corrupt_or_wrong_identity_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            n.State(OWNER, TARGET).save(path)
            with self.assertRaises(n.ConfigError):
                n.State.load(path, OWNER, TARGET + 1)
            for text in ("{", "[]", '{"account_id": 1}'):
                path.write_text(text)
                with self.assertRaises(n.ConfigError):
                    n.State.load(path, OWNER, TARGET)

    def test_session_path_and_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "account.session").write_bytes(b"fake-session-no-login")
            def env(session):
                (root / "secrets.env").write_text(
                    f"TELEGRAM_API_ID=123\nTELEGRAM_API_HASH={'a' * 32}\n"
                    f"TELEGRAM_SESSION_PATH={session}\nDEEPSEEK_API_KEY=fake-test-key\n")
            for path in ("account", "account.session", str(root / "account.session")):
                env(path)
                self.assertEqual(n.load_secrets(root)["session"], str(root / "account.session"))
            env("absent")
            with self.assertRaises(n.ConfigError):
                n.load_secrets(root)
            env("account")
            with (root / "secrets.env").open("a") as stream:
                stream.write("DEEPSEEK_BASE_URL=http://unsafe.invalid\n")
            with self.assertRaises(n.ConfigError):
                n.load_secrets(root)

    def test_single_instance_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "secrets.env"
            path.write_text("fake-content")
            with n.instance_lock(path):
                with self.assertRaises(n.ConfigError), n.instance_lock(path):
                    pass
            with n.instance_lock(path):
                pass


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "persona.txt").write_text("Тестовый вымышленный собеседник", encoding="utf-8")
        self.cfg = n.Settings.parse({"allowed_user_id": TARGET, "debounce_seconds": 0.01,
                                    "reply_delay_seconds": 0, "show_typing": False})
        self.state = n.State(OWNER, TARGET)
        self.tg = NS(get_messages=AsyncMock(return_value=[]),
                     send_message=AsyncMock(return_value=NS(id=100)), action=MagicMock())
        self.create = AsyncMock(return_value=response())
        self.ai = NS(chat=NS(completions=NS(create=self.create)))
        self.app = n.Neuron(self.tg, self.ai, self.cfg, self.state, self.root, TARGET)

    async def stop_worker(self, worker):
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker

    async def wait_until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0.002)
        await asyncio.wait_for(wait(), 1)

    def age(self, hours=9):
        self.app.started = self.state.last_activity = time.time() - hours * 3600
        self.state.last_incoming_id = 5
        self.app.cfg = replace(self.cfg, initiate_first=True)
        self.tg.get_messages.return_value = [message(5, "Как поездка?")]

    async def test_foreign_groups_channels_self_and_media_are_ignored(self):
        bad = [event(sender=44), event(chat=44), event(private=False),
               event(chat=-100555, private=False), event(out=True, sender=OWNER),
               event(sender=OWNER), event(chat=OWNER, sender=OWNER, out=True),
               event(text=""), event(text=None), event(text="  ")]
        for incoming in bad:
            await self.app.incoming(incoming)
        await self.app.respond()
        self.create.assert_not_awaited()
        self.tg.send_message.assert_not_awaited()
        self.tg.get_messages.assert_not_awaited()

    async def test_one_message_exactly_one_reply_correct_chat(self):
        await self.app.incoming(event())
        await self.app.respond()
        await self.app.respond()
        self.create.assert_awaited_once()
        self.tg.send_message.assert_awaited_once_with(
            TARGET, "Привет! Как день?", parse_mode=None, link_preview=False)
        kwargs = self.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-flash")
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(kwargs["max_tokens"], 220)
        self.assertNotIn("tools", kwargs)
        self.assertEqual(self.state.last_handled_incoming_id, 10)

    async def test_duplicate_event_is_not_sent_twice(self):
        await self.app.incoming(event())
        await self.app.incoming(event())
        await self.app.respond()
        await self.app.incoming(event())
        await self.app.respond()
        self.create.assert_awaited_once()
        self.tg.send_message.assert_awaited_once()

    async def test_worker_coalesces_three_inputs(self):
        worker = asyncio.create_task(self.app.worker())
        try:
            for mid in (10, 11, 12):
                await self.app.incoming(event(mid, f"текст {mid}"))
                await asyncio.sleep(0.002)
            await self.wait_until(lambda: self.tg.send_message.await_count == 1)
            self.create.assert_awaited_once()
            contents = [m["content"] for m in self.create.call_args.kwargs["messages"][1:]]
            self.assertEqual(contents, ["текст 10", "текст 11", "текст 12"])
        finally:
            await self.stop_worker(worker)

    async def test_new_input_during_api_discards_stale_generation(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed(**kwargs):
            entered.set()
            await release.wait()
            return response("Устаревший ответ")
        self.create.side_effect = delayed
        await self.app.incoming(event())
        first = asyncio.create_task(self.app.respond())
        await entered.wait()
        await self.app.incoming(event(11, "дополнение"))
        release.set()
        await first
        self.tg.send_message.assert_not_awaited()
        self.create.side_effect = None
        self.create.return_value = response("Актуальный ответ")
        await self.app.respond()
        self.tg.send_message.assert_awaited_once()
        self.assertEqual(self.create.call_args.kwargs["messages"][-1]["content"], "дополнение")

    async def test_history_order_roles_scope_captions_and_pending(self):
        self.tg.get_messages.return_value = [
            message(9, "последний"), message(8, "наш", OWNER, out=True),
            message(7, "чужая утечка", chat=55), message(6, ""),
            message(5, "подпись к фото"), message(4, "неизвестный", sender=77),
            message(3, "старый"), message(500, "будущий")]
        await self.app.incoming(event())
        history = await self.app.history(10)
        self.assertEqual([m["content"] for m in history],
                         ["старый", "подпись к фото", "наш", "последний", "привет"])
        self.assertEqual([m["role"] for m in history], ["user", "user", "assistant", "user", "user"])
        self.tg.get_messages.assert_awaited_once_with(TARGET, limit=30, max_id=11)

    async def test_typing_context_used(self):
        self.app.cfg = replace(self.cfg, show_typing=True)
        self.tg.action.return_value = n.no_typing()
        await self.app.incoming(event())
        await self.app.respond()
        self.tg.action.assert_called_once_with(TARGET, "typing")

    async def test_persona_reloaded_between_requests(self):
        await self.app.incoming(event())
        await self.app.respond()
        (self.root / "persona.txt").write_text("Новая манера речи", encoding="utf-8")
        await self.app.incoming(event(101))
        await self.app.respond()
        self.assertIn("Новая манера", self.create.call_args.kwargs["messages"][0]["content"])
        self.assertIn("честно", self.create.call_args.kwargs["messages"][0]["content"])

    async def test_missing_persona_sends_nothing(self):
        (self.root / "persona.txt").unlink()
        await self.app.incoming(event())
        await self.app.respond()
        self.create.assert_not_awaited()
        self.tg.send_message.assert_not_awaited()

    async def test_initiative_off_and_restart_grace(self):
        self.age()
        self.app.cfg = self.cfg
        await self.app.maybe_initiate()
        self.create.assert_not_awaited()
        self.app.cfg = replace(self.cfg, initiate_first=True)
        self.app.started = time.time()
        await self.app.maybe_initiate()
        self.create.assert_not_awaited()

    async def test_one_initiative_until_incoming_even_across_restart(self):
        self.age()
        await self.app.maybe_initiate()
        self.tg.send_message.assert_awaited_once()
        self.assertTrue(self.state.awaiting_reply)
        self.state.last_activity = self.app.started = time.time() - 86400 * 2
        await self.app.maybe_initiate()
        restored = n.State.load(self.root / "state.json", OWNER, TARGET)
        restarted = n.Neuron(self.tg, self.ai, self.app.cfg, restored, self.root, TARGET)
        restarted.started = restored.last_activity = time.time() - 86400 * 3
        await restarted.maybe_initiate()
        self.tg.send_message.assert_awaited_once()
        await restarted.incoming(event(200, ""))
        self.assertFalse(restored.awaiting_reply)
        restarted.started = restored.last_activity = time.time() - 9 * 3600
        await restarted.maybe_initiate()
        self.assertEqual(self.tg.send_message.await_count, 2)

    async def test_initiative_requires_incoming_and_day_budget(self):
        self.age()
        self.state.last_incoming_id = 0
        await self.app.maybe_initiate()
        self.create.assert_not_awaited()
        self.state.last_incoming_id = 5
        self.state.proactive_day = datetime.now(timezone.utc).date().isoformat()
        self.state.proactive_count = 2
        await self.app.maybe_initiate()
        self.create.assert_not_awaited()
        self.state.proactive_day = "2000-01-01"
        await self.app.maybe_initiate()
        self.create.assert_awaited_once()
        self.assertEqual(self.state.proactive_count, 1)

    async def test_no_initiative_for_empty_history(self):
        self.age()
        self.tg.get_messages.return_value = []
        await self.app.maybe_initiate()
        self.create.assert_not_awaited()

    async def test_incoming_during_initiative_cancels_it(self):
        self.age()
        async def race(**kwargs):
            await self.app.incoming(event())
            return response()
        self.create.side_effect = race
        await self.app.maybe_initiate()
        self.tg.send_message.assert_not_awaited()
        self.assertFalse(self.state.awaiting_reply)
        await self.app.respond()
        self.tg.send_message.assert_awaited_once()

    async def test_manual_send_suppresses_waiting_bot_and_resets_silence(self):
        await self.app.incoming(event())
        await self.app.outgoing(event(11, "Я отвечу сам", OWNER, out=True))
        await self.app.respond()
        self.create.assert_not_awaited()
        self.assertLess(time.time() - self.state.last_activity, 1)

    async def test_manual_send_during_generation_discards_bot(self):
        async def manual(**kwargs):
            await self.app.outgoing(event(11, "Я тут", OWNER, out=True))
            return response()
        self.create.side_effect = manual
        await self.app.incoming(event())
        await self.app.respond()
        self.tg.send_message.assert_not_awaited()

    async def test_bot_outgoing_echo_before_and_after_send_return_is_ignored(self):
        async def send(*args, **kwargs):
            await self.app.outgoing(event(100, "бот", OWNER, out=True))
            return NS(id=100)
        self.tg.send_message.side_effect = send
        await self.app.incoming(event())
        revision = self.app.revision
        await self.app.respond()
        await self.app.outgoing(event(100, "бот", OWNER, out=True))
        self.assertEqual(revision, self.app.revision)

    async def test_auth_balance_access_model_errors_pause_api_without_leak(self):
        for status in (401, 402, 403, 404):
            with self.subTest(status=status):
                self.app.api_disabled = False
                self.create.reset_mock()
                self.create.side_effect = api_error(status)
                with self.assertLogs("neuron", level="WARNING") as logs:
                    await self.app.incoming(event(1000 + status))
                    await self.app.respond()
                    await self.app.incoming(event(2000 + status))
                    await self.app.respond()
                self.assertNotIn("PRIVATE_SERVER_BODY", "\n".join(logs.output))
                self.create.assert_awaited_once()
        self.tg.send_message.assert_not_awaited()

    async def test_rate_limit_long_retry_after_does_not_storm(self):
        self.create.side_effect = api_error(429, {"Retry-After": "180"})
        await self.app.incoming(event())
        await self.app.respond()
        await self.app.incoming(event(11))
        await self.app.respond()
        self.create.assert_awaited_once()
        self.tg.send_message.assert_not_awaited()

    async def test_api_transient_retry_bounded_and_success_sends_once(self):
        self.create.side_effect = [api_error(500), response()]
        await self.app.incoming(event())
        with patch.object(n.asyncio, "sleep", new=AsyncMock()) as sleep:
            await self.app.respond()
        self.assertEqual(self.create.await_count, 2)
        self.tg.send_message.assert_awaited_once()
        self.assertIn(2.0, [call.args[0] for call in sleep.call_args_list])

    async def test_network_failure_retries_once_no_reply(self):
        self.create.side_effect = APIConnectionError(request=httpx.Request("POST", "https://api.deepseek.com"))
        await self.app.incoming(event())
        with patch.object(n.asyncio, "sleep", new=AsyncMock()):
            await self.app.respond()
        self.assertEqual(self.create.await_count, 2)
        self.tg.send_message.assert_not_awaited()
        self.assertEqual(self.state.last_handled_incoming_id, 10)

    async def test_empty_refusal_and_bad_request_never_send(self):
        for mid, result in enumerate((response(""), response(None), response(" "), NS(choices=[])), 10):
            self.create.return_value = result
            await self.app.incoming(event(mid))
            await self.app.respond()
        self.create.side_effect = api_error(400)
        await self.app.incoming(event(20))
        await self.app.respond()
        self.tg.send_message.assert_not_awaited()

    async def test_telegram_floodwait_and_ambiguous_send_not_retried(self):
        for mid, exc in ((10, errors.FloodWaitError(None, capture=60)), (11, OSError("PRIVATE"))):
            self.app.tg_blocked_until = 0
            self.tg.send_message.side_effect = exc
            await self.app.incoming(event(mid))
            await self.app.respond()
            await self.app.respond()
            self.assertEqual(self.state.last_handled_incoming_id, mid)
        self.assertEqual(self.tg.send_message.await_count, 2)
        self.assertGreater(self.app.tg_blocked_until, time.monotonic())

    async def test_ambiguous_initiative_stays_reserved_across_restart(self):
        self.age()
        self.tg.send_message.side_effect = OSError()
        await self.app.maybe_initiate()
        state = n.State.load(self.root / "state.json", OWNER, TARGET)
        self.assertTrue(state.awaiting_reply)
        self.app.started = self.state.last_activity = time.time() - 86400
        self.app.tg_blocked_until = 0
        await self.app.maybe_initiate()
        self.tg.send_message.assert_awaited_once()

    async def test_state_restart_skips_already_handled_message(self):
        await self.app.incoming(event())
        await self.app.respond()
        state = n.State.load(self.root / "state.json", OWNER, TARGET)
        app = n.Neuron(self.tg, self.ai, self.cfg, state, self.root, TARGET)
        await app.incoming(event())
        await app.respond()
        self.create.assert_awaited_once()

    async def test_history_failure_sends_nothing(self):
        self.tg.get_messages.side_effect = errors.FloodWaitError(None, capture=120)
        await self.app.incoming(event())
        await self.app.respond()
        self.create.assert_not_awaited()
        self.tg.send_message.assert_not_awaited()

    async def test_long_unicode_output_fits_one_telegram_message(self):
        self.create.return_value = response("😎" * 5000)
        await self.app.incoming(event())
        await self.app.respond()
        text = self.tg.send_message.call_args.args[1]
        self.assertLessEqual(len(text.encode("utf-16-le")) // 2, 4000)
        self.tg.send_message.assert_awaited_once()

    async def test_sdk_real_serialization_over_mock_http_transport(self):
        bodies = []
        async def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "test", "object": "chat.completion", "created": 1,
                "model": "deepseek-flash", "choices": [{"index": 0,
                    "finish_reason": "stop", "message": {"role": "assistant", "content": "Работает"}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with AsyncOpenAI(api_key="fake-test-key", base_url="https://api.deepseek.com",
                                   http_client=http, max_retries=0) as ai:
                self.app.ai = ai
                await self.app.incoming(event())
                await self.app.respond()
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0]["thinking"], {"type": "disabled"})
        self.assertEqual(bodies[0]["max_tokens"], 220)
        self.assertNotIn("extra_body", bodies[0])
        self.tg.send_message.assert_awaited_once()


class StartupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = n.Settings.parse({"allowed_user_id": TARGET})
        self.secret = dict(session="fake.session", api_id=123, api_hash="a" * 32,
                           key="fake-test-key", base="https://api.deepseek.com")
        self.disconnected = asyncio.get_running_loop().create_future()
        self.disconnected.set_result(None)
        self.tg = NS(connect=AsyncMock(), disconnect=AsyncMock(),
                     is_user_authorized=AsyncMock(return_value=True),
                     get_me=AsyncMock(return_value=NS(id=OWNER, bot=False)),
                     get_entity=AsyncMock(return_value=types.User(id=TARGET)),
                     get_messages=AsyncMock(return_value=[message(50), message(49, "наш", OWNER, out=True)]),
                     add_event_handler=MagicMock(), set_receive_updates=AsyncMock(),
                     disconnected=self.disconnected, send_message=AsyncMock())

    async def test_disabled_exits_without_reading_secrets_or_connecting(self):
        (self.root / "settings.json").write_text('{"enabled": false}')
        with patch.object(n, "TelegramClient") as client:
            await n.run(self.root)
        client.assert_not_called()

    async def test_invalid_id_never_connects(self):
        (self.root / "settings.json").write_text('{"allowed_user_id": 0}')
        with patch.object(n, "TelegramClient") as client, self.assertRaises(n.ConfigError):
            await n.run(self.root)
        client.assert_not_called()

    async def test_unauthed_session_never_prompts_or_listens(self):
        self.tg.is_user_authorized.return_value = False
        with patch.object(n, "TelegramClient", return_value=self.tg), self.assertRaises(n.ConfigError):
            await n.serve(self.root, self.cfg, self.secret)
        self.tg.add_event_handler.assert_not_called()
        self.tg.disconnect.assert_awaited_once()

    async def test_saved_messages_and_bot_target_rejected(self):
        for me, target in ((NS(id=TARGET, bot=False), types.User(id=TARGET)),
                           (NS(id=OWNER, bot=False), types.User(id=TARGET, bot=True))):
            self.tg.get_me.return_value = me
            self.tg.get_entity.return_value = target
            with patch.object(n, "TelegramClient", return_value=self.tg), self.assertRaises(n.ConfigError):
                await n.serve(self.root, self.cfg, self.secret)
        self.tg.add_event_handler.assert_not_called()

    async def test_startup_baselines_backlog_no_generation_and_closes_clients(self):
        ai = NS(close=AsyncMock())
        with patch.object(n, "TelegramClient", return_value=self.tg) as ctor, \
                patch.object(n, "AsyncOpenAI", return_value=ai) as ai_ctor:
            await n.serve(self.root, self.cfg, self.secret)
        state = n.State.load(self.root / "state.json", OWNER, TARGET)
        self.assertEqual(state.last_handled_incoming_id, 50)
        self.assertEqual(state.last_incoming_id, 50)
        self.tg.send_message.assert_not_awaited()
        self.tg.set_receive_updates.assert_awaited_once_with(True)
        self.assertFalse(ctor.call_args.kwargs["catch_up"])
        self.assertFalse(ctor.call_args.kwargs["receive_updates"])
        self.assertEqual(ai_ctor.call_args.kwargs["max_retries"], 0)
        ai.close.assert_awaited_once()
        self.tg.disconnect.assert_awaited_once()
        incoming_handler = self.tg.add_event_handler.call_args_list[0].args[0]
        await incoming_handler(event(50))
        self.tg.send_message.assert_not_awaited()

    async def test_full_service_wiring_receives_new_event_and_stops_cleanly(self):
        (self.root / "persona.txt").write_text("Тестовая личность", encoding="utf-8")
        self.tg.disconnected = asyncio.get_running_loop().create_future()
        self.tg.send_message.return_value = NS(id=70)
        ai = NS(chat=NS(completions=NS(create=AsyncMock(return_value=response()))), close=AsyncMock())
        cfg = replace(self.cfg, debounce_seconds=0, reply_delay_seconds=0, show_typing=False)
        with patch.object(n, "TelegramClient", return_value=self.tg), patch.object(n, "AsyncOpenAI", return_value=ai):
            service = asyncio.create_task(n.serve(self.root, cfg, self.secret))
            try:
                for _ in range(100):
                    if self.tg.set_receive_updates.await_count:
                        break
                    await asyncio.sleep(0.002)
                handler = self.tg.add_event_handler.call_args_list[0].args[0]
                await handler(event(51, "Новое сообщение после запуска"))
                for _ in range(100):
                    if self.tg.send_message.await_count:
                        break
                    await asyncio.sleep(0.002)
                self.tg.send_message.assert_awaited_once()
                ai.chat.completions.create.assert_awaited_once()
                self.tg.disconnected.set_result(None)
                await asyncio.wait_for(service, 1)
            finally:
                service.cancel()
                await asyncio.gather(service, return_exceptions=True)
        ai.close.assert_awaited_once()
        self.tg.disconnect.assert_awaited_once()

    async def test_cancellation_closes_both_clients(self):
        self.tg.disconnected = asyncio.get_running_loop().create_future()
        ai = NS(close=AsyncMock())
        with patch.object(n, "TelegramClient", return_value=self.tg), patch.object(n, "AsyncOpenAI", return_value=ai):
            service = asyncio.create_task(n.serve(self.root, self.cfg, self.secret))
            for _ in range(100):
                if self.tg.set_receive_updates.await_count:
                    break
                await asyncio.sleep(0.002)
            service.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await service
        ai.close.assert_awaited_once()
        self.tg.disconnect.assert_awaited_once()

    async def test_storage_failure_in_handler_stops_service(self):
        self.tg.disconnected = asyncio.get_running_loop().create_future()
        ai = NS(close=AsyncMock())
        with patch.object(n, "TelegramClient", return_value=self.tg), patch.object(n, "AsyncOpenAI", return_value=ai):
            service = asyncio.create_task(n.serve(self.root, self.cfg, self.secret))
            try:
                for _ in range(100):
                    if self.tg.set_receive_updates.await_count:
                        break
                    await asyncio.sleep(0.002)
                handler = self.tg.add_event_handler.call_args_list[0].args[0]
                with patch.object(n.State, "save", side_effect=OSError("PRIVATE")):
                    await handler(event(51))
                    with self.assertRaises(n.ConfigError):
                        await asyncio.wait_for(service, 1)
                self.tg.send_message.assert_not_awaited()
            finally:
                service.cancel()
                await asyncio.gather(service, return_exceptions=True)
        ai.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
