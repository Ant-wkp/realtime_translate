from __future__ import annotations

import contextlib
import io
import queue
import signal
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from realtime_translate import main as runtime
from realtime_translate.asr import AsrResult
from realtime_translate.transcriber import TranscriberWorker


class TestDispatcher(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.messages = []
        self.server = SimpleNamespace(broadcast=self.messages.append)
        self.translation = Mock()
        self.dispatcher = runtime.SubtitleDispatcher(
            self.server, self.translation, clock=lambda: self.now,
            wall_clock=lambda: -100.0, log=Mock())
        self.dispatcher.open("seg", seed_secs=1.0)

    def final(self, text="你好", task_id="seg"):
        return runtime.SubtitleEvent("asr_result", task_id, result=AsrResult(
            text=text, meta={"final_replaces_partial": True, "task_id": task_id}))

    def finish(self):
        self.dispatcher.finish("seg", speech_end_at=11.0, seg_seconds=1.6)

    def test_discard_rejects_all_late_events(self):
        self.dispatcher.discard("seg")
        for event in (runtime.SubtitleEvent("partial", "seg", text="迟到"),
                      self.final(), runtime.SubtitleEvent("translation", "seg", text="late")):
            self.dispatcher.handle(event)
        self.assertEqual([m["type"] for m in self.messages], ["discard"])
        self.translation.submit.assert_not_called()
        self.assertEqual(self.dispatcher.segments, {})

    def test_finishing_accepts_tail_partial_and_counts_first_once(self):
        self.finish()
        self.now = 11.4
        self.dispatcher.handle(runtime.SubtitleEvent("partial", "seg", text="尾字"))
        self.now = 11.5
        self.dispatcher.handle(runtime.SubtitleEvent("partial", "seg", text="尾字补齐"))
        self.assertEqual(self.dispatcher.first_char_stats["n"], 1)
        self.assertAlmostEqual(self.dispatcher.first_char_stats["sum"], 1.4)
        self.assertEqual(self.dispatcher.first_char_stats["miss"], 0)

    def test_final_rejects_partial_and_duplicate_final(self):
        self.finish()
        self.now = 12.0
        self.dispatcher.handle(self.final())
        self.dispatcher.handle(runtime.SubtitleEvent("partial", "seg", text="旧文本"))
        self.dispatcher.handle(self.final("重复终稿"))
        self.assertEqual([m["type"] for m in self.messages], ["final"])
        self.translation.submit.assert_called_once_with("seg", "你好", "zh")
        self.assertEqual(self.dispatcher.first_char_stats["miss"], 1)

    def test_source_and_translation_share_timestamp_until_completion(self):
        self.finish()
        self.now = 12.0
        self.dispatcher.handle(self.final())
        self.assertEqual(self.dispatcher.segments["seg"].speech_end_at, 11.0)
        self.assertAlmostEqual(self.dispatcher.source_stats["sum"], 1.0)
        self.now = 13.0
        self.dispatcher.handle(runtime.SubtitleEvent(
            "translation", "seg", text="hello", target_lang="en"))
        self.assertAlmostEqual(self.dispatcher.translation_stats["sum"], 2.0)
        self.assertEqual(self.dispatcher.translation_stats["n"], 1)
        self.assertEqual(self.dispatcher.segments, {})
        self.assertTrue(all(msg["ts"] == -100.0 for msg in self.messages))
        self.dispatcher.handle(self.final())
        self.assertEqual(len(self.messages), 2)

    def test_translation_failure_cleans_state_without_success_sample(self):
        self.finish()
        self.dispatcher.handle(self.final())
        self.dispatcher.handle(runtime.SubtitleEvent("translation", "seg", error="GPU failure"))
        self.assertEqual(self.messages[-1]["error"], "GPU failure")
        self.assertEqual(self.dispatcher.translation_stats, {"n": 0, "sum": 0.0, "failed": 1})
        self.assertEqual(self.dispatcher.segments, {})

    def test_empty_final_and_asr_error_are_terminal(self):
        self.finish()
        self.dispatcher.handle(self.final(""))
        self.assertEqual(self.dispatcher.segments, {})
        self.assertEqual(self.dispatcher.source_stats["n"], 0)
        self.dispatcher.open("other")
        self.dispatcher.handle(runtime.SubtitleEvent("asr_error", "other", error="broken"))
        self.assertEqual(self.dispatcher.segments, {})
        self.assertEqual([m["type"] for m in self.messages], ["discard", "discard"])

    def test_unknown_and_premature_translation_are_ignored(self):
        self.dispatcher.handle(runtime.SubtitleEvent("partial", "unknown", text="bad"))
        self.dispatcher.handle(runtime.SubtitleEvent("translation", "seg", text="premature"))
        self.assertEqual(self.messages, [])

    def test_zero_timestamp_is_valid(self):
        self.dispatcher.finish("seg", speech_end_at=0.0, seg_seconds=1.0)
        self.dispatcher.handle(self.final())
        self.assertEqual(self.dispatcher.source_stats["sum"], 10.0)


class TestWorkers(unittest.TestCase):
    def test_translation_reports_success_and_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                output = queue.Queue()
                translator = Mock()
                if failure:
                    translator.translate.side_effect = RuntimeError()
                else:
                    translator.translate.return_value = ("hello", "en")
                worker = runtime.TranslationThread(translator, output.put)
                worker.start()
                try:
                    worker.submit("seg", "你好", "zh")
                    event = output.get(timeout=1)
                    self.assertEqual(event.kind, "translation")
                    self.assertEqual(event.task_id, "seg")
                    self.assertEqual(bool(event.error), failure)
                finally:
                    worker.stop()
                    worker.stop()
                    worker.join(1)
                self.assertFalse(worker.is_alive())

    def test_final_decode_failure_keeps_partial_hint(self):
        output = queue.Queue()
        provider = SimpleNamespace(name="fake", transcribe=Mock(side_effect=RuntimeError("failed")))
        worker = runtime.FinalDecodeThread(provider, output.put)
        worker.start()
        try:
            worker.submit("seg", np.zeros(1600), partial_hint="保留尾字")
            result = output.get(timeout=1)
            self.assertEqual(result.text, "保留尾字")
            self.assertEqual(result.meta["mode"], "2pass-final-partial")
        finally:
            worker.stop()
            worker.join(1)
        self.assertFalse(worker.is_alive())

    def test_delayed_feed_after_discard_cannot_recreate_subtitle(self):
        entered, release = threading.Event(), threading.Event()
        output = queue.Queue()

        def feed(*_args):
            entered.set()
            release.wait(1)
            return "迟到文本"

        provider = SimpleNamespace(feed=feed, abort=Mock())
        worker = TranscriberWorker(provider, on_partial=lambda sid, text:
                                   output.put(runtime.SubtitleEvent("partial", sid, text=text)))
        messages = []
        dispatcher = runtime.SubtitleDispatcher(SimpleNamespace(broadcast=messages.append), Mock(), log=Mock())
        dispatcher.open("seg")
        worker.start()
        try:
            worker.stream_feed("seg", np.zeros(512), "seg")
            self.assertTrue(entered.wait(1))
            dispatcher.discard("seg")
            worker.stream_abort("seg")
            release.set()
            dispatcher.handle(output.get(timeout=1))
            self.assertEqual([msg["type"] for msg in messages], ["discard"])
        finally:
            release.set()
            worker.stop()
            worker.join(1)

    def test_transcriber_failure_reports_terminal_error(self):
        output = queue.Queue()
        provider = SimpleNamespace(feed=Mock(side_effect=RuntimeError("bad audio")), abort=Mock())
        worker = TranscriberWorker(provider, on_error=lambda sid, error: output.put((sid, error)))
        worker.start()
        try:
            worker.stream_feed("seg", np.zeros(512), "seg")
            self.assertEqual(output.get(timeout=1), ("seg", "bad audio"))
            provider.abort.assert_called_once_with("seg")
        finally:
            worker.stop()
            worker.join(1)

    def test_usb_poll_sleep_is_interruptible(self):
        read = threading.Event()

        def probe(*_args, **_kwargs):
            read.set()
            return [90, 1]

        worker = runtime.VadDoaReader(SimpleNamespace(read=probe), interval=30)
        worker.start()
        try:
            self.assertTrue(read.wait(1))
            worker.stop()
            worker.join(0.3)
            self.assertFalse(worker.is_alive())
        finally:
            worker.stop()
            worker.join(1)

    def test_shutdown_timeout_keeps_inflight_resource_and_cancels_pending(self):
        entered, release = threading.Event(), threading.Event()
        provider = Mock()

        def feed(*_args):
            entered.set()
            release.wait(2)
            return ""

        provider.feed.side_effect = feed
        worker = TranscriberWorker(provider)
        worker.start()
        try:
            worker.stream_feed("first", np.zeros(512), "first")
            self.assertTrue(entered.wait(1))
            worker.stream_feed("pending", np.zeros(512), "pending")
            started = time.monotonic()
            self.assertFalse(runtime.shutdown_resources(
                [worker], [("provider", provider, (worker,))], timeout=0.03))
            self.assertLess(time.monotonic() - started, 0.3)
            provider.close.assert_not_called()
            release.set()
            worker.join(1)
            self.assertEqual(provider.feed.call_count, 1)
        finally:
            release.set()
            worker.stop()
            worker.join(1)

    def test_shutdown_idle_and_unstarted_workers(self):
        provider = Mock()
        active = TranscriberWorker(provider)
        unstarted = runtime.FinalDecodeThread(provider, Mock())
        active.start()
        self.assertTrue(runtime.shutdown_resources(
            [active, unstarted], [("provider", provider, (active, unstarted))], timeout=1))
        provider.close.assert_called_once()
        self.assertFalse(active.is_alive())

    def test_all_workers_share_one_shutdown_deadline(self):
        now = 0.0
        joins = []

        def join(timeout):
            nonlocal now
            joins.append(timeout)
            now += timeout

        workers = [SimpleNamespace(name=f"blocked-{i}", stop=Mock(),
                                   is_alive=lambda: True, join=join) for i in range(2)]
        self.assertFalse(runtime.shutdown_resources(workers, [], timeout=5, clock=lambda: now))
        self.assertEqual(joins, [5.0, 0.0])
        self.assertEqual(now, 5.0)
        for worker in workers:
            worker.stop.assert_called_once()


class TestMain(unittest.TestCase):
    def test_hardware_failure_precedes_model_loading(self):
        for error in (RuntimeError("not found"), PermissionError("access denied"), TimeoutError("probe timeout")):
            with self.subTest(error=error), patch.object(runtime, "connect_control", side_effect=error), \
                    patch.dict("sys.modules", {"realtime_translate.funasr_provider": None}), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(runtime.main([]), 1)
                self.assertIn(str(error), output.getvalue())

    def test_partial_initialization_is_cleaned_and_signals_restored(self):
        device = Mock()
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        with patch.object(runtime, "connect_control", return_value=device), \
                patch.object(runtime, "resolve_input_device", side_effect=RuntimeError("audio missing")), \
                patch.dict("sys.modules", {"sounddevice": SimpleNamespace()}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.main([]), 1)
        device.close.assert_called_once()
        for sig, previous in handlers.items():
            self.assertEqual(signal.getsignal(sig), previous)

    def test_model_initialization_failure_closes_prior_resources(self):
        device, provider = Mock(), Mock()
        modules = {
            "sounddevice": SimpleNamespace(query_devices=lambda *args:
                {"name": "fake", "max_input_channels": 6}),
            "realtime_translate.funasr_provider": SimpleNamespace(FunasrProvider=Mock(
                return_value=provider, CHUNK_SIZE_480=[5, 8, 4], CHUNK_SIZE_600=[5, 10, 5])),
            "realtime_translate.translator": SimpleNamespace(DEFAULT_BEAM_SIZE=1,
                NllbCTranslate2Provider=Mock(side_effect=RuntimeError("model missing"))),
        }
        with patch.dict("sys.modules", modules), patch.object(runtime, "connect_control", return_value=device), \
                patch.object(runtime, "resolve_input_device", return_value=0), \
                patch.object(runtime, "describe_platform", return_value="fake"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.main([]), 1)
        provider.close.assert_called_once()
        device.close.assert_called_once()

    def test_sigint_during_startup_cleans_up_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGINT)

        def interrupt(*_args):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

        with patch.object(runtime, "connect_control", side_effect=interrupt), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.main([]), 0)
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)

    def test_software_mode_runs_full_pipeline_without_control_reader(self):
        self.run_pipeline(no_hw_vad=True)

    def test_hardware_mode_runs_full_pipeline(self):
        self.run_pipeline(no_hw_vad=False)

    def run_pipeline(self, no_hw_vad):
        messages = []
        callback = None
        mono = 0.0
        ticks = 0
        device = Mock()
        reader_queue = queue.Queue()
        reader = SimpleNamespace(
            name="FakeReader", vad_queue=reader_queue,
            start=lambda: reader_queue.put({"hw_vad": 1, "doa": 90}),
            stop=Mock(), is_alive=lambda: False)
        provider, translator = Mock(), Mock()
        provider.name = "fake"
        provider.describe_device.return_value = "cpu"
        provider.feed.return_value = "你好"
        provider.finish_online.return_value = ("你好啊", np.full(16000, 0.2, np.float32))
        provider.transcribe.return_value = AsrResult(text="你好啊。")
        translator.model_path, translator.device, translator.compute_type = "fake", "cpu", "int8"
        translator.beam_size, translator.zh_to, translator.other_to = 1, "en", "zh"
        translator.translate.return_value = ("Hello.", "en")
        vad = SimpleNamespace(device="cpu", last_probability=0.0, reset=Mock(), close=Mock())
        vad_calls = 0

        def vad_feed(audio):
            nonlocal vad_calls
            vad_calls += 1
            vad.last_probability = 0.9 if vad_calls <= 10 else 0.02
            return vad.last_probability

        vad.feed = vad_feed

        class FakeServer:
            def __init__(self, **kwargs):
                self.port = kwargs["port"]
                self.cmd_queue = queue.Queue()
                self.name = "FakeServer"
            def start(self):
                pass
            def stop(self):
                pass
            def is_alive(self):
                return False
            def broadcast(self, msg):
                messages.append(msg)

        class FakeInput:
            def __init__(self, **kwargs):
                nonlocal callback
                callback = kwargs["callback"]
            def __enter__(self):
                callback(np.full((1600, 6), 0.2, np.float32), 1600, None, None)
                return self
            def __exit__(self, *_args):
                pass

        def tick(_delay):
            nonlocal mono, ticks
            time.sleep(0.002)
            mono += 0.1
            ticks += 1
            if ticks >= 25 and any(msg["type"] == "translation" for msg in messages):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            if ticks >= 60:
                raise AssertionError("pipeline failed to finish")
            if not no_hw_vad:
                reader_queue.put({"hw_vad": 1, "doa": 90})
            callback(np.full((1600, 6), 0.2, np.float32), 1600, None, None)

        modules = {
            "sounddevice": SimpleNamespace(InputStream=FakeInput,
                query_devices=lambda *args: {"name": "fake", "max_input_channels": 6}),
            "realtime_translate.funasr_provider": SimpleNamespace(FunasrProvider=Mock(
                return_value=provider, CHUNK_SIZE_480=[5, 8, 4], CHUNK_SIZE_600=[5, 10, 5])),
            "realtime_translate.translator": SimpleNamespace(DEFAULT_BEAM_SIZE=1,
                NllbCTranslate2Provider=Mock(return_value=translator)),
            "realtime_translate.sw_vad": SimpleNamespace(SileroVad=Mock(return_value=vad)),
            "realtime_translate.ws_server": SimpleNamespace(BroadcastServer=FakeServer),
        }
        with patch.dict("sys.modules", modules), patch.object(runtime, "connect_control", \
                return_value=None if no_hw_vad else device), \
                patch.object(runtime, "VadDoaReader", return_value=reader) as make_reader, \
                patch.object(runtime, "resolve_input_device", return_value=0), \
                patch.object(runtime, "describe_platform", return_value="fake"), \
                patch.object(runtime, "time", SimpleNamespace(
                    monotonic=lambda: mono, time=lambda: -1000.0, sleep=tick)), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runtime.main(["--no-hw-vad"] if no_hw_vad else []), 0, output.getvalue())
        if no_hw_vad:
            make_reader.assert_not_called()
            device.close.assert_not_called()
        else:
            make_reader.assert_called_once()
            device.close.assert_called_once()
        types = [msg["type"] for msg in messages]
        self.assertIn("partial", types)
        self.assertIn("final", types)
        self.assertIn("translation", types)
        self.assertLess(types.index("final"), types.index("translation"))
        states = [msg for msg in messages if msg["type"] == "vad.state"]
        self.assertEqual(states[0]["segment_state"], "SPEAKING")
        self.assertTrue(all(msg["gate_mode"] == ("software" if no_hw_vad else "and") for msg in states))
        self.assertTrue(any(msg["segment_state"] == "IDLE" for msg in states))
        translator.translate.assert_called_once_with("你好啊。", "zh")
        provider.close.assert_called_once()
        translator.close.assert_called_once()
        vad.close.assert_called_once()
        self.assertNotIn("上屏", output.getvalue())


if __name__ == "__main__":
    unittest.main()
