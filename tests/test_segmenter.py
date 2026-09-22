"""Deterministic tests for the segmentation state machine.

These need no audio hardware and no models: the software VAD is scripted and
the audio blocks are synthetic, so every filter boundary and every diagnostic
line can be asserted exactly. The point is to pin the behaviour that P1 must
not change - the wording of the logs, the order of the Actions, and the fact
that a discarded or flushed segment never seeds the next one.

    python -m unittest tests.test_segmenter -v
    # or simply
    python tests/test_segmenter.py
"""

from __future__ import annotations

import io
import contextlib
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realtime_subtitle.segmenter import (    # noqa: E402
    SegmentCfg, Segmenter, trim_tail,
)

SR = 16000
TICK = 0.1
BLOCK = int(SR * TICK)
LOUD = (np.sin(np.arange(BLOCK) * 0.1) * 0.2).astype(np.float32)
QUIET = (np.sin(np.arange(BLOCK) * 0.1) * 8e-4).astype(np.float32)


class ScriptedVad:
    """Returns the scripted probabilities in order, repeating the last."""

    def __init__(self, probs):
        self.probs = list(probs)
        self.i = 0
        self.last_probability = 0.0
        self.resets = 0

    def feed(self, audio):
        p = self.probs[min(self.i, len(self.probs) - 1)]
        self.i += 1
        self.last_probability = p
        return p

    def reset(self):
        self.resets += 1


def make(probs, **cfg):
    cfg.setdefault("max_seconds", 20.0)
    seg = Segmenter(SegmentCfg(**cfg), ScriptedVad(probs))
    return seg


def drive(seg, probs, loud_at=None, auto_stream=False):
    """Feed one tick per probability, gate = prob >= threshold."""
    actions = []
    for i, p in enumerate(probs):
        gate = p >= 0.5
        block = LOUD if (p >= 0.2) else QUIET
        if loud_at is not None:
            block = LOUD if i in loud_at else QUIET
        batch = seg.feed([block], lambda prob, g=gate: g,
                         1 if p >= 0.2 else 0, i * TICK)
        actions += batch
        if auto_stream and any(a.kind == "open" for a in batch):
            seg.mark_stream_open()
    return actions


def kinds(actions):
    return [a.kind for a in actions]


class TestOpen(unittest.TestCase):
    def test_open_seeds_from_pre_roll(self):
        # 10 ticks of quiet first: the ring buffer should be at full depth
        seg = make([0.02] * 10 + [0.9] * 5, pre_roll=1.0)
        acts = drive(seg, [0.02] * 10 + [0.9] * 5)
        opens = [a for a in acts if a.kind == "open"]
        self.assertEqual(len(opens), 1)
        # the whole 1s window was buffered, so the seed is a full second
        self.assertAlmostEqual(opens[0].seed_secs, 1.0, places=3)
        self.assertAlmostEqual(opens[0].audio.size / SR, 1.0, places=3)

    def test_closed_gate_does_nothing(self):
        seg = make([0.02] * 10)
        acts = drive(seg, [0.02] * 10)
        self.assertEqual(acts, [])

    def test_open_then_feed(self):
        seg = make([0.9] * 8)
        acts = drive(seg, [0.9] * 8)
        self.assertEqual(kinds(acts)[0], "open")
        # streaming is not open yet, so the opening tick owes no feed
        self.assertNotIn("feed", kinds(acts))

    def test_feed_only_while_stream_open(self):
        seg = make([0.9] * 8)
        acts = drive(seg, [0.9] * 8)
        seg.mark_stream_open()
        acts += drive(seg, [0.9] * 3)
        self.assertIn("feed", kinds(acts))


class TestFilters(unittest.TestCase):
    def test_min_length_discard(self):
        # three voiced ticks = 0.3s, just below the 0.3s floor once the
        # comparison is strict
        seg = make([0.9] * 2 + [0.02] * 8, min_seconds=0.31)
        acts = drive(seg, [0.9] * 2 + [0.02] * 8)
        disc = [a for a in acts if a.kind == "discard"]
        self.assertEqual(len(disc), 1)
        self.assertFalse(disc[0].delivered)
        self.assertIn("duration", disc[0].reason)
        self.assertEqual(seg.counters.discarded, 1)

    def test_flush_delivers(self):
        seg = make([0.9] * 12 + [0.02] * 8, min_seconds=0.3)
        acts = drive(seg, [0.9] * 12 + [0.02] * 8)
        fl = [a for a in acts if a.kind == "flush"]
        self.assertEqual(len(fl), 1)
        self.assertTrue(fl[0].delivered)
        self.assertGreater(fl[0].audio.size, 0)
        self.assertEqual(seg.counters.flushed, 1)

    def test_energy_filter_rejects_quiet_speech(self):
        # voiced for long enough, and loud enough to pass the ratio filter
        # (gate open every voiced tick), but the audio itself is near silence
        probs = [0.9] * 12 + [0.02] * 8
        seg = Segmenter(SegmentCfg(min_seconds=0.3, max_seconds=20.0),
                        ScriptedVad(probs))
        acts = []
        for i, p in enumerate(probs):
            gate = p >= 0.5
            acts += seg.feed([QUIET], lambda prob, g=gate: g,
                             1 if p >= 0.2 else 0, i * TICK)
        disc = [a for a in acts if a.kind == "discard"]
        self.assertEqual(len(disc), 1)
        self.assertIn("energy", disc[0].reason)

    def test_ratio_filter_rejects_sparse_speech(self):
        # Voice on one tick in six, with the gaps held at 0.3 - above the 0.2
        # "clearly non-speech" floor, so the segment is not ended early - and
        # the short-speech exemption switched off. Voiced frames are then a
        # small fraction of the span, which is what the ratio floor exists for.
        probs = ([0.9] + [0.3] * 5) * 4 + [0.02] * 8
        seg = Segmenter(SegmentCfg(min_seconds=0.3, max_seconds=20.0,
                                   short_exempt_seconds=0.0,
                                   silence_hold=99.0),
                        ScriptedVad(probs))
        acts = drive(seg, probs)
        disc = [a for a in acts if a.kind == "discard"]
        self.assertEqual(len(disc), 1)
        self.assertIn("ratio", disc[0].reason)


class TestEndOfUtterance(unittest.TestCase):
    def test_silence_ends_early(self):
        # A trailing silence is normally cut at the 0.3s early-end mark, well
        # before silence_hold; this is the common path for real utterances.
        seg = make([0.9] * 10 + [0.02] * 20, silence_hold=99.0)
        acts = drive(seg, [0.9] * 10 + [0.02] * 20)
        fl = [a for a in acts if a.kind == "flush"]
        self.assertEqual(len(fl), 1)
        self.assertAlmostEqual(fl[0].tail_wait, 0.3, places=2)

    def test_silence_hold_flush(self):
        # Gaps held at 0.3 - above the 0.2 confirmation floor, so the early-end
        # path never fires - leave the hold timer as what ends the segment.
        gap = [0.3] * 20
        seg = make([0.9] * 10 + gap, silence_hold=0.6)
        acts = drive(seg, [0.9] * 10 + gap)
        fl = [a for a in acts if a.kind == "flush"]
        self.assertEqual(len(fl), 1)
        self.assertAlmostEqual(fl[0].tail_wait, 0.6, places=2)

    def test_max_seconds_force_cut(self):
        # A continuous talker is cut every max_seconds and re-opened on the
        # next tick, so 4s of speech at a 1s cap yields several segments.
        seg = make([0.9] * 40, max_seconds=1.0, silence_hold=99.0)
        acts = drive(seg, [0.9] * 40)
        fl = [a for a in acts if a.kind == "flush"]
        self.assertGreaterEqual(len(fl), 3)
        self.assertAlmostEqual(fl[0].seg_seconds, 1.0, delta=0.15)

    def test_early_end_after_consecutive_quiet(self):
        # three consecutive quiet ticks (~0.3s) plus 0.3s since speech ends it
        # without waiting for the full silence_hold
        seg = make([0.9] * 10 + [0.02] * 20, silence_hold=99.0)
        acts = drive(seg, [0.9] * 10 + [0.02] * 20)
        fl = [a for a in acts if a.kind == "flush"]
        self.assertEqual(len(fl), 1)
        self.assertLess(fl[0].tail_wait, 0.6)

    def test_mid_utterance_dips_do_not_cut(self):
        # 0.4 is above the 0.2 confirmation floor, so these must not end it
        seg = make([0.9] * 6 + [0.4] * 4 + [0.9] * 6 + [0.02] * 8,
                   silence_hold=0.6)
        acts = drive(seg, [0.9] * 6 + [0.4] * 4 + [0.9] * 6 + [0.02] * 8)
        fl = [a for a in acts if a.kind == "flush"]
        self.assertEqual(len(fl), 1)
        self.assertGreater(fl[0].seg_seconds, 1.0)


def loud_samples(audio: np.ndarray) -> int:
    """Count samples carrying speech-level energy (LOUD peaks at 0.2)."""
    return int((np.abs(audio) > 0.05).sum())


class TestSeedIsolation(unittest.TestCase):
    def test_flushed_speech_does_not_seed_next_segment(self):
        """A delivered sentence's audio must not reappear in the next seed.

        This is the duplicate-text bug in disguise: if the pre-roll survives a
        flush, the next stream re-decodes the previous sentence's words. The
        one speech-level block that is expected is the tick that opened the
        new segment - anything more means the old tail leaked through.
        """
        probs = [0.9] * 12 + [0.02] * 20 + [0.9] * 12 + [0.02] * 20
        seg = make(probs, silence_hold=0.6, pre_roll=1.0)
        acts = drive(seg, probs)
        opens = [a for a in acts if a.kind == "open"]
        self.assertEqual(len(opens), 2)
        self.assertLessEqual(loud_samples(opens[1].audio), BLOCK)

    def test_discarded_speech_does_not_seed_next_segment(self):
        probs = [0.9] * 2 + [0.02] * 20 + [0.9] * 12 + [0.02] * 20
        seg = make(probs, min_seconds=0.3, silence_hold=0.6, pre_roll=1.0)
        acts = drive(seg, probs)
        self.assertEqual(len([a for a in acts if a.kind == "discard"]), 1)
        opens = [a for a in acts if a.kind == "open"]
        self.assertEqual(len(opens), 2)
        self.assertLessEqual(loud_samples(opens[1].audio), BLOCK)

    def test_reset_resets_software_vad(self):
        seg = make([0.9] * 12 + [0.02] * 20)
        drive(seg, [0.9] * 12 + [0.02] * 20)
        self.assertGreaterEqual(seg.vad.resets, 1)


class TestDiagnostics(unittest.TestCase):
    def test_flush_line_wording(self):
        seg = make([0.9] * 12 + [0.02] * 8, silence_hold=0.6)
        acts = drive(seg, [0.9] * 12 + [0.02] * 8)
        fl = [a for a in acts if a.kind == "flush"][0]
        self.assertRegex(fl.flush_line,
                         r"^\[seg\] flush after \d+\.\d{2}s \(tail wait "
                         r"\d+\.\d{2}s\)$")

    def test_flush_debug_wording(self):
        seg = make([0.9] * 12 + [0.02] * 8, debug=True)
        acts = drive(seg, [0.9] * 12 + [0.02] * 8)
        fl = [a for a in acts if a.kind == "flush"][0]
        self.assertRegex(fl.debug_line, r"^\[flush\] dur=\d+\.\d{2}s "
                                        r"ratio=\d\.\d{2} avg_prob=\d\.\d{2} "
                                        r"max_prob=\d\.\d{2} exempt=\w+ "
                                        r"hw_ticks=\d+$")

    def test_discard_wording_and_stream_flag(self):
        seg = make([0.9] * 2 + [0.02] * 8, min_seconds=0.31, debug=True)
        # the stream is opened by the caller right after the open Action, the
        # way apply_actions() does it
        acts = drive(seg, [0.9] * 2 + [0.02] * 8, auto_stream=True)
        d = [a for a in acts if a.kind == "discard"][0]
        # the caller needs this to know an abort is owed to the worker
        self.assertTrue(d.stream_opened)
        # debug_line carries both lines, in the order the inline loop printed
        # them: the [flush] stats first, then the [VAD] discard verdict
        lines = d.debug_line.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[0], r"^\[flush\] dur=")
        self.assertRegex(lines[1], r"^\[VAD\] discard \w+ \(duration .*\): "
                                   r"dur=\d+\.\d{2}s ratio=")

    def test_diagnostic_line_wording(self):
        # wording and order of the [seg]/[flush] lines, as the inline loop
        # printed them
        seg = Segmenter(SegmentCfg(debug=True),
                        ScriptedVad([0.9] * 12 + [0.02] * 8))
        acts = []
        probs = [0.9] * 12 + [0.02] * 8
        for i, p in enumerate(probs):
            gate = p >= 0.5
            acts += seg.feed([LOUD], lambda prob, g=gate: g,
                             1 if p >= 0.2 else 0, i * TICK)
        fl = [a for a in acts if a.kind == "flush"][0]
        self.assertTrue(fl.flush_line.startswith("[seg] flush"))
        self.assertTrue(fl.debug_line.startswith("[flush]"))


class TestTrimTail(unittest.TestCase):
    def test_trims_trailing_silence(self):
        audio = np.concatenate([LOUD, QUIET * 4])
        out = trim_tail(audio, 1e-4)
        self.assertLess(out.size, audio.size)
        self.assertGreaterEqual(out.size, 0)

    def test_keeps_all_speech(self):
        audio = np.concatenate([QUIET, LOUD])
        self.assertEqual(trim_tail(audio, 1e-4).size, audio.size)

    def test_empty(self):
        self.assertEqual(trim_tail(np.zeros(0, np.float32), 1e-4).size, 0)


class TestSummary(unittest.TestCase):
    def test_empty_when_idle(self):
        self.assertEqual(make([0.02] * 4).summary(), "")

    def test_counts_reasons(self):
        seg = make([0.9] * 2 + [0.02] * 8, min_seconds=0.31)
        drive(seg, [0.9] * 2 + [0.02] * 8)
        s = seg.summary()
        self.assertIn("丢弃 1", s)
        self.assertIn("duration", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)