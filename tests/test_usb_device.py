from __future__ import annotations

import contextlib
import io
import struct
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from realtime_translate import main as runtime
from realtime_translate import usb_device


class TestUsbReads(unittest.TestCase):
    def device(self, transfer):
        usb = SimpleNamespace(ctrl_transfer=transfer)
        with patch.object(usb_device, "find", return_value=usb):
            return usb_device.ReSpeaker()

    def test_read_protocol_is_unchanged(self):
        transfer = Mock(return_value=bytes([0]) + struct.pack("<HH", 90, 1))
        device = self.device(transfer)
        self.assertEqual(device.read("DOA_VALUE"), [90, 1])
        self.assertEqual(transfer.call_args.args, (0xC0, 0, 0x80 | 18, 20, 5, 1000))

    def test_stop_before_transfer_and_between_retries(self):
        stop = threading.Event()
        transfer = Mock(return_value=bytes([64]))
        device = self.device(transfer)
        stop.set()
        with self.assertRaises(InterruptedError):
            device.read("DOA_VALUE", stop_event=stop)
        transfer.assert_not_called()
        stop.clear()

        def cancel(*_args):
            stop.set()
            return bytes([64])

        transfer.side_effect = cancel
        with self.assertRaises(InterruptedError):
            device.read("DOA_VALUE", stop_event=stop)
        self.assertEqual(transfer.call_count, 1)

    def test_probe_deadline_caps_total_retry_time_and_transfer_timeout(self):
        now = 0.0
        timeouts = []

        def transfer(*args):
            nonlocal now
            timeout = args[-1]
            timeouts.append(timeout)
            now += min(0.4, timeout / 1000)
            return bytes([64])

        def sleep(delay):
            nonlocal now
            now += delay

        device = self.device(transfer)
        with patch.object(usb_device, "time", SimpleNamespace(monotonic=lambda: now, sleep=sleep)):
            with self.assertRaises(TimeoutError):
                device.read("DOA_VALUE", deadline=2.0)
        self.assertAlmostEqual(now, 2.0, places=6)
        self.assertTrue(all(0 < ms <= 1000 for ms in timeouts))
        self.assertLess(timeouts[-1], 1000)

    def test_expired_deadline_does_not_start_transfer(self):
        transfer = Mock()
        device = self.device(transfer)
        with self.assertRaises(TimeoutError):
            device.read("DOA_VALUE", deadline=time.monotonic() - 1)
        transfer.assert_not_called()

    def test_reader_cancels_real_usb_retry_loop(self):
        entered = threading.Event()

        def transfer(*_args):
            entered.set()
            return bytes([64])

        device = self.device(transfer)
        worker = runtime.VadDoaReader(device)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            worker.stop()
            worker.join(0.3)
            self.assertFalse(worker.is_alive())
            self.assertTrue(worker.vad_queue.empty())
        finally:
            worker.stop()
            worker.join(1)


class TestControlStartup(unittest.TestCase):
    def test_default_missing_device_returns_error_before_models(self):
        with patch.object(usb_device, "ReSpeaker", side_effect=RuntimeError("not found")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runtime.main([]), 1)
        self.assertIn("--no-hw-vad", output.getvalue())

    def test_permission_timeout_and_bad_payload_close_device(self):
        for failure in (PermissionError("denied"), TimeoutError("timeout"), None):
            with self.subTest(failure=failure):
                device = Mock()
                device.read.side_effect = failure
                device.read.return_value = []
                with patch.object(usb_device, "ReSpeaker", return_value=device):
                    with self.assertRaisesRegex(RuntimeError, "--no-hw-vad"):
                        runtime.connect_control(False)
                device.close.assert_called_once()
                self.assertIn("deadline", device.read.call_args.kwargs)

    def test_software_mode_allows_missing_or_unreadable_device(self):
        with patch.object(usb_device, "ReSpeaker", side_effect=RuntimeError("not found")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(runtime.connect_control(True))
        device = Mock()
        device.read.side_effect = PermissionError("denied")
        with patch.object(usb_device, "ReSpeaker", return_value=device), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(runtime.connect_control(True))
        device.close.assert_called_once()

    def test_verified_control_device_remains_open(self):
        device = Mock()
        device.read.return_value = [90, 1]
        with patch.object(usb_device, "ReSpeaker", return_value=device), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(runtime.connect_control(False), device)
        device.close.assert_not_called()

    def test_probe_interrupt_closes_device_before_reraising(self):
        device = Mock()
        device.read.side_effect = KeyboardInterrupt
        with patch.object(usb_device, "ReSpeaker", return_value=device):
            with self.assertRaises(KeyboardInterrupt):
                runtime.connect_control(False)
        device.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
